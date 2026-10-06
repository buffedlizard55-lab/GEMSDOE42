"""Cross-scale stability pipeline: multiscale worming x topological persistence.

Stage A  worming      Hornby/Boschetti/Horowitz multiscale edges on the magnetic and gravity
                      layers, counting how many upward-continuation steps each edge survives.
Stage B  persistence  dim-0 persistent homology of the same layers' gradient-magnitude surfaces
                      across a dyadic sweep of Gaussian smoothing scales.
Stage C  score        continuation survival x topological birth-death range, normalised to [0,1].
Stage D  emission     the metric-optimal sparse read-out of that score (see the docstring of
                      ``select_emission`` for why a sparse binary read-out is what the
                      distance-weighted Tversky index actually rewards).

Every stage caches to ``.cache/stage`` so a parameter change in a later stage does not re-run the
earlier ones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .layers import Grid, band_names, fill, read_band, valid_mask
from .persistence import DEFAULT_SIGMAS, topological_persistence
from .worming import DEFAULT_HEIGHTS_M, worm_levels

CACHE = Path(__file__).resolve().parents[2] / ".cache" / "stage"

#: Layers the two independent scale notions are computed on.  "magnetic and gravity layers" per
#: the brief: rtp is the reduced-to-pole magnetic anomaly (the standard layer for magnetic edge
#: analysis because it removes the dipolar asymmetry), iso_grav_anom the isostatic gravity anomaly.
WORM_LAYERS: tuple[str, ...] = ("rtp", "iso_grav_anom")


@dataclass
class StageConfig:
    heights_m: tuple[float, ...] = DEFAULT_HEIGHTS_M
    #: 88th percentile, not 95th: worming decides *where* candidate structures are, the
    #: topological factor decides *which* of them are real.  A tighter edge threshold makes the
    #: worm mask so thin that the emission budget cannot be spent at all (measured: 12,659 dots
    #: was the ceiling at p95 with a 2-px separation, against a modelled optimum of ~42k).
    edge_percentile: float = 88.0
    migrate_px: int = 2
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS
    n_levels: int = 32
    pct: float = 99.5
    which: str = "prominence"
    worm_combine: str = "max"
    topo_combine: str = "max"


def _save(name: str, a: np.ndarray) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"{name}.npy", a)


def _load(name: str) -> np.ndarray | None:
    p = CACHE / f"{name}.npy"
    return np.load(p, mmap_mode="r") if p.exists() else None


def _grad_magnitude(a: np.ndarray, valid: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """|grad| of a nodata-filled layer, smoothed at ``sigma`` (px)."""
    f = fill(a, valid)
    gx = gaussian_filter(f, sigma, order=(0, 1), mode="nearest")
    gy = gaussian_filter(f, sigma, order=(1, 0), mode="nearest")
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _unit(a: np.ndarray, valid: np.ndarray, pct: float) -> np.ndarray:
    out = np.zeros(a.shape, dtype=np.float32)
    hi = float(np.percentile(a[valid], pct)) if valid.any() else 0.0
    if hi > 0:
        out[valid] = np.clip(a[valid] / hi, 0.0, 1.0)
    return out


def stage_a_worming(features: Path, grid: Grid, cfg: StageConfig,
                    verbose: bool = True) -> tuple[np.ndarray, dict]:
    """Continuation-step survival, combined across the magnetic and gravity layers."""
    key = f"A_wormsurv_{'-'.join(str(int(h)) for h in cfg.heights_m)}_p{cfg.edge_percentile}"
    cached = _load(key)
    if cached is not None:
        return np.asarray(cached, np.float32), {"cached": True}

    names = band_names(features)
    survs, per_layer = [], {}
    for nm in WORM_LAYERS:
        if nm not in names:
            raise KeyError(f"worming layer {nm!r} missing from {features}")
        a = read_band(features, nm, grid)
        v = valid_mask(a, grid)
        t0 = time.time()
        res = worm_levels(fill(a, v).astype(np.float32), v, heights_m=cfg.heights_m,
                          edge_percentile=cfg.edge_percentile, migrate_px=cfg.migrate_px,
                          layer=nm)
        survs.append(res.survival)
        per_layer[nm] = {
            "edge_px_per_level": [int(m.sum()) for m in res.edge_masks],
            "mean_steps_survived": float(res.steps[v].mean()),
            "max_steps_survived": int(res.steps.max()),
            "seconds": round(time.time() - t0, 1),
        }
        if verbose:
            print(f"  [A] {nm}: mean steps {per_layer[nm]['mean_steps_survived']:.3f} "
                  f"max {per_layer[nm]['max_steps_survived']}  ({per_layer[nm]['seconds']}s)",
                  flush=True)
    surv = _combine(survs, cfg.worm_combine)
    surv[~grid.footprint] = 0.0
    _save(key, surv)
    return surv, {"cached": False, "layers": per_layer, "combine": cfg.worm_combine}


def _combine(maps: list[np.ndarray], how: str) -> np.ndarray:
    """Combine per-layer maps across the two physics."""
    if not maps:
        raise ValueError("no layers to combine")
    if how == "max":
        return np.max(np.stack(maps), axis=0).astype(np.float32)
    if how == "mean":
        return np.mean(np.stack(maps), axis=0).astype(np.float32)
    raise ValueError(f"unknown combine={how!r}")


def stage_b_persistence(features: Path, grid: Grid, cfg: StageConfig,
                        verbose: bool = True) -> tuple[np.ndarray, dict]:
    """Topological birth-death range of ridge features, swept over smoothing scales."""
    key = f"B_topo_{cfg.which}_{'-'.join(str(s) for s in cfg.sigmas)}_L{cfg.n_levels}_p{cfg.pct}"
    cached = _load(key)
    if cached is not None:
        return np.asarray(cached, np.float32), {"cached": True}

    names = band_names(features)
    maps, info = [], {}
    for nm in WORM_LAYERS:
        a = read_band(features, nm, grid)
        v = valid_mask(a, grid)
        G = _grad_magnitude(a, v, sigma=1.0)
        t0 = time.time()
        topo, inf = topological_persistence(G, grid.footprint, sigmas=cfg.sigmas,
                                            n_levels=cfg.n_levels, pct=cfg.pct,
                                            which=cfg.which, verbose=verbose)
        maps.append(topo)
        info[nm] = {"n_features": inf["n_features"], "seconds": round(time.time() - t0, 1)}
        if verbose:
            print(f"  [B] {nm}: features/sigma {inf['n_features']}  ({info[nm]['seconds']}s)",
                  flush=True)
    if cfg.topo_combine == "max":
        topo = np.max(np.stack(maps), axis=0)
    elif cfg.topo_combine == "mean":
        topo = np.mean(np.stack(maps), axis=0).astype(np.float32)
    else:
        raise ValueError(f"unknown topo_combine={cfg.topo_combine!r}")
    topo[~grid.footprint] = 0.0
    _save(key, topo)
    return topo, {"cached": False, "layers": info, "combine": cfg.topo_combine}


def stage_c_score(worm_surv: np.ndarray, topo: np.ndarray, grid: Grid) -> np.ndarray:
    """Continuation steps survived x topological birth-death range, in [0, 1].

    Both factors are already in [0, 1] by construction (a survival *fraction* and a
    quantile-normalised range), so the product is in [0, 1] without further rescaling.  A pixel
    needs corroboration from both notions of scale: an edge that survives deep continuation but
    sits on a topologically ephemeral ridge scores 0, and so does a persistent ridge that no
    continuation level marks as an edge.
    """
    score = (worm_surv.astype(np.float32) * topo.astype(np.float32)).astype(np.float32)
    score[~grid.footprint] = 0.0
    return np.clip(score, 0.0, 1.0)


def select_emission(score: np.ndarray, grid: Grid, budget: int,
                    min_separation_px: int = 2,
                    exclude: np.ndarray | None = None) -> np.ndarray:
    """Greedy non-maximum-suppressed top-``budget`` read-out of the score.

    Why a sparse binary read-out rather than the raw continuous score: with the official metric
    ``DTI = TP / (0.2*(TP+FP) + 0.8*|G|)``, a pixel at distance ``d`` from truth contributes
    ``p*k(d)`` to TP and ``p*(1-k(d))`` to FP.  Raising ``p`` from ``v`` to 1 helps whenever
    ``k(d) > 0.2*DTI`` - true for anything inside the 300 m kernel once DTI is above ~0.1 - so the
    optimum puts full mass on the pixels it does select and none anywhere else.  Spreading a small
    probability over the whole footprint instead adds ~5.2M false-positive terms.

    ``min_separation_px`` enforces spatial concentration: the selected dots must be at least that
    far apart, so the budget buys coverage of distinct structures rather than a cluster on one.
    """
    score = np.asarray(score, np.float32)
    valid = grid.footprint & (score > 0)
    if exclude is not None:
        valid &= ~exclude
    if not valid.any() or budget <= 0:
        return np.zeros(score.shape, bool)

    idx = np.flatnonzero(valid.ravel())
    vals = score.ravel()[idx]
    order = idx[np.argsort(-vals, kind="stable")]

    H, W = score.shape
    yy = order // W
    xx = order % W
    r = max(int(min_separation_px), 1)

    taken = np.zeros((H, W), bool)
    chosen_y: list[int] = []
    chosen_x: list[int] = []
    # Pre-filter with a strided pass on the coarse grid for speed, then verify exactly.
    for y, x in zip(yy.tolist(), xx.tolist()):
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        if taken[y0:y1, x0:x1].any():
            continue
        taken[y, x] = True
        chosen_y.append(y)
        chosen_x.append(x)
        if len(chosen_y) >= budget:
            break

    out = np.zeros(score.shape, bool)
    if chosen_y:
        out[np.asarray(chosen_y), np.asarray(chosen_x)] = True
    return out
