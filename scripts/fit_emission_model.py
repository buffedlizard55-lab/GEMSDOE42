#!/usr/bin/env python3
"""Fit an emission model to the 11 restored prior submissions whose official scores are known.

Why
---
The spatially-blocked catalogue holdout is uncorrelated with the real leaderboard
(``registry/holdout_calibration.json``: Spearman +0.087, p = 0.80, n = 11).  It cannot select a
candidate.  What *can* be used is the set of prior submissions for which the official public
leaderboard score is known: 11 rasters with a known emitted-pixel count and a known score.  That
is a real regression dataset.

Model
-----
Assume the mean kernel credit a dot earns falls as a power law in the emission size,

    h(N) = A * N ** (-b)

which is the standard precision-decay assumption for a ranked candidate list.  Then, with the
official metric and ``|G|`` the hidden truth size,

    DTI(N) = N h / ( 0.2 * N * (h + f) + 0.8 * |G| ),    f = 1 - 0.5 h

Three parameters (``A``, ``b``, ``|G|``) are fitted by least squares to the 11 observed
(official DTI, N) pairs.  ``|G|`` is not free-floating guesswork: it is constrained by the fit and
can then be compared against the region's plausible unmapped-fault budget.

Outputs the modelled optimum emission size and the DTI it implies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems42.layers import Grid, load_grid                    # noqa: E402
from gems42.emission_model import ALPHA, BETA, dti_from_parts  # noqa: E402

OBS = {
    "scored_h19_5": 0.1922,
    "scored_h19_4": 0.1894,
    "scored_h16_1": 0.1855,
    "scored_d15_scored": 0.2477,
    "scored_gems27_tgc_v2_d15": 0.2449,
    "calib_13gems_20261001_r13-lattice-s5_v2_nan-ou": 0.0904,
    "calib_8GEMSDOE_Hedge-v2_submission": 0.1563,
    "calib_gems10-h25-ctx-ridge-20260927T2329477041": 0.1280,
    "calib_gems10-h28-dotted-ridge-20260928T0202562": 0.1839,
    "calib_gemsdoe-ens12-adopted-7f00890a": 0.1563,
    "calib_gemsdoe9-PLACEHOLDER-2314b599": 0.0107,
}


def model_dti(N, A, b, G, h_cap: float = 1.0):
    """Modelled DTI.  ``h`` is capped at 1 because the triangular kernel is bounded by 1, so the
    mean credit a dot can earn cannot exceed 1 - without the cap the fitted power law
    extrapolates to h(2000) ~ 4.1, which is not a number the metric can produce."""
    N = np.asarray(N, dtype=np.float64)
    h = np.minimum(A * np.power(np.maximum(N, 1.0), -b), h_cap)
    f = 1.0 - 0.5 * h
    return np.asarray([dti_from_parts(n * hh, n * ff, G)
                       for n, hh, ff in zip(np.atleast_1d(N), np.atleast_1d(h),
                                            np.atleast_1d(f))])


def main() -> int:
    from gems42.paths import data_dir
    ddir = data_dir()
    grid = load_grid(ddir / "sample_submission.tif")
    man = json.loads((ROOT / "registry" / "data_manifest.json").read_text())
    dest = {f["id"]: f["dest"] for f in man["files"]}

    rows = []
    for pid, official in OBS.items():
        rel = dest.get(pid)
        if not rel or not (ddir / rel).exists():
            continue
        with rasterio.open(ddir / rel) as ds:
            a = ds.read(1)
        if a.shape != grid.shape:
            continue
        n = int((np.nan_to_num(a, nan=0.0) > 0).sum())
        rows.append({"id": pid, "official": official, "emitted_px": n})
    rows.sort(key=lambda r: r["emitted_px"])

    N = np.array([r["emitted_px"] for r in rows], dtype=np.float64)
    y = np.array([r["official"] for r in rows], dtype=np.float64)
    print(f"[fit] n={len(rows)} observations, emitted px {int(N.min()):,} .. {int(N.max()):,}",
          flush=True)

    def resid(p):
        A, b, G = p
        return model_dti(N, A, b, G) - y

    best = None
    for A0 in (0.5, 2.0, 8.0):
        for b0 in (0.2, 0.4, 0.6):
            for G0 in (6_000.0, 15_000.0, 40_000.0):
                try:
                    s = least_squares(resid, [A0, b0, G0],
                                      bounds=([1e-3, 0.0, 1_000.0], [1e4, 2.0, 500_000.0]),
                                      xtol=1e-12, ftol=1e-12)
                except Exception:
                    continue
                if best is None or s.cost < best.cost:
                    best = s
    A, b, G = (float(v) for v in best.x)
    rmse = float(np.sqrt(2 * best.cost / len(y)))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((best.fun ** 2).sum()) / ss_tot

    print(f"[fit] h(N) = {A:.4f} * N^(-{b:.4f});   |G| = {G:,.0f} px", flush=True)
    print(f"[fit] RMSE = {rmse:.4f}   R^2 = {r2:.3f}", flush=True)
    print(f"[fit] per-observation fit:", flush=True)
    for r, pred in zip(rows, model_dti(N, A, b, G)):
        print(f"    {r['id'][:52]:<52} N={r['emitted_px']:>7,} "
              f"official={r['official']:.4f} model={pred:.4f} resid={pred - r['official']:+.4f}",
              flush=True)

    grid_N = np.unique(np.concatenate([
        np.array([r["emitted_px"] for r in rows]),
        np.arange(int(N.min()), int(N.max()) + 1, 2_000)]))
    curve = model_dti(grid_N, A, b, G)
    k = int(np.argmax(curve))
    print(f"[fit] modelled optimum INSIDE the observed range "
          f"[{int(N.min()):,} .. {int(N.max()):,}] = {int(grid_N[k]):,} px "
          f"-> DTI {curve[k]:.4f}", flush=True)
    h_at = lambda n: min(A * n ** -b, 1.0)
    def _worth(n: float) -> bool:
        h = h_at(n)
        return h > 0.2 * (1.0 - 0.5 * h)
    breakeven = int(next((n for n in grid_N if not _worth(n)), 0))
    print(f"[fit] breakeven (largest N where one more dot still pays, h > 0.2 f): "
          f"{breakeven:,}", flush=True)
    print(f"[fit] h at the observed minimum N={int(N.min()):,} is {h_at(N.min()):.3f}; "
          f"h at the observed maximum N={int(N.max()):,} is {h_at(N.max()):.3f} "
          f"(capped at 1.0)", flush=True)

    out = {
        "n_observations": len(rows),
        "parameters": {"A": A, "b": b, "hidden_truth_px_G": G,
                       "alpha": ALPHA, "beta": BETA, "fp_charge_rule": "f = 1 - 0.5 h"},
        "rmse": rmse, "r2": r2,
        "observations": [{**r, "modelled": float(p)} for r, p in zip(rows, model_dti(N, A, b, G))],
        "optimum": {"emitted_px": int(grid_N[k]), "modelled_dti": float(curve[k]),
                    "observed_range_px": [int(N.min()), int(N.max())],
                    "breakeven_px": breakeven},
        "curve": [{"emitted_px": int(n), "modelled_dti": float(d),
                   "mean_credit_h": float(A * n ** -b)}
                  for n, d in zip(grid_N.tolist(), curve.tolist())],
        "caveat": ("h(N) = A*N^-b is an assumed precision-decay law, not a measurement; |G| is "
                   "fitted from 11 public-leaderboard scores and inherits their noise. The model "
                   "is used to choose an emission SIZE, not to predict a score."),
    }
    (ROOT / "registry" / "emission_model.json").write_text(json.dumps(out, indent=1))
    print("[write] registry/emission_model.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
