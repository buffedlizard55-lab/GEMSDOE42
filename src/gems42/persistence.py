"""Dim-0 persistent homology of gradient-magnitude surfaces, across a sweep of smoothing scales.

What is computed
----------------
For a scalar surface ``G`` (a gradient magnitude) and the *superlevel-set* filtration

    X_t = { x : G(x) >= t },        t descending from high to low,

dimension-0 persistent homology pairs each connected component's **birth** level ``b`` (the value
of the local maximum that seeded it) with its **death** level ``d`` (the level at which it merges
into an older, still-alive component).  ``b - d`` is the component's *persistence*: how far the
threshold has to fall before that ridge stops being an independent structure.  A tall, isolated
gradient ridge has a large range; a speckle of gradient noise has a tiny one.  This is the
standard merge-tree / level-set-tree construction (Edelsbrunner & Harer, *Computational
Topology*, ch. III).

Two per-pixel quantities are returned, because they answer slightly different questions and the
pipeline should not have to guess:

``persistence[p] = birth[p] - death[p]``
    The birth-death range of the H0 feature pixel ``p`` belongs to.  Constant along a feature, so
    it scores *which ridges are structures*.

``prominence[p] = G[p] - death[p]``
    How far the level must fall from ``p``'s own height before ``p``'s component is absorbed.
    Peaks on ridge crests and decays into their flanks, so it scores *where on the structure*.

The component that is still alive at the lowest threshold is the global structure, not a feature:
it is closed out at ``t_low``, which sends its background pixels to ~0 prominence rather than
handing the whole footprint the largest value in the map.

Why a smoothing sweep on top
----------------------------
A ridge that is topologically persistent at one smoothing scale but vanishes at the next is an
artefact of that scale.  The surface is therefore pre-smoothed at a dyadic series of Gaussian
scales and the filtration is recomputed at each; ``topological_persistence`` returns the mean
normalised value over the sweep, so a pixel scores highly only if its ridge is persistent *and*
stable under smoothing.

Implementation
--------------
Exact merge tree from nested labelings: ``scipy.ndimage.label`` at ``n_levels`` quantised
thresholds, plus the fact that superlevel sets are nested, so a component at a higher threshold
lies inside exactly one component at the next lower threshold (recovered with a single
``ndimage.maximum`` pass).  The only approximation is the ``n_levels`` quantisation of the
threshold grid.  In a *superlevel* filtration the sweep descends, so the oldest component - the
survivor of a merge - is the one with the HIGHEST birth value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.ndimage import maximum as ndi_maximum

STRUCT8 = np.ones((3, 3), bool)

__all__ = ["persistence_map_h0", "topological_persistence", "PersistenceResult", "DEFAULT_SIGMAS"]

#: Dyadic Gaussian smoothing sweep, in pixels (1 px = 100 m): 80 m .. 640 m characteristic width.
DEFAULT_SIGMAS: tuple[float, ...] = (0.8, 1.6, 3.2, 6.4)


@dataclass
class PersistenceResult:
    persistence: np.ndarray    # float32: birth - death of the feature each pixel belongs to
    prominence: np.ndarray     # float32: G - death, the per-pixel crest significance
    birth: np.ndarray          # float32: birth level of that feature
    death: np.ndarray          # float32: death level of that feature
    n_features: int            # number of H0 features paired off by a merge
    levels: np.ndarray         # the quantised threshold levels used, descending


def _descending(levels: np.ndarray) -> np.ndarray:
    """Force strict monotone decrease (ties would make a level a no-op in the merge tree)."""
    out = np.asarray(levels, dtype=np.float64).copy()
    for i in range(1, out.size):
        if out[i] >= out[i - 1]:
            out[i] = np.nextafter(out[i - 1], -np.inf)
    return out


def persistence_map_h0(G: np.ndarray, valid: np.ndarray, n_levels: int = 48,
                       q_hi: float = 99.95, q_lo: float = 0.5,
                       levels: np.ndarray | None = None) -> PersistenceResult:
    """Exact per-pixel dim-0 persistence of the superlevel filtration of ``G``.

    Every pixel receives the birth and death level of the H0 feature it belongs to; see the module
    docstring for the two derived maps.  Pixels whose value is below the lowest threshold are
    outside the filtration and receive 0 for everything.

    ``levels`` overrides the quantile grid (the tests use it to feed every distinct value, which
    removes the quantisation and makes the result directly comparable to a brute-force reference).
    """
    G = np.asarray(G, dtype=np.float32)
    valid = np.asarray(valid, bool)
    if G.shape != valid.shape:
        raise ValueError("G and valid must share a shape")

    vals = G[valid & np.isfinite(G)]
    if vals.size == 0:
        raise ValueError("no finite values inside the footprint")

    if levels is None:
        levels = np.percentile(vals, np.linspace(q_hi, q_lo, n_levels))
    levels = _descending(levels)

    birth_px = np.zeros(G.shape, dtype=np.float32)
    death_px = np.zeros(G.shape, dtype=np.float32)
    done = np.zeros(G.shape, dtype=bool)

    node_birth: list[float] = []       # node id -> birth level
    n_nodes = 0
    n_features = 0

    prev_lab: np.ndarray | None = None
    prev_node: np.ndarray | None = None    # prev label -> node id
    last_lab: np.ndarray | None = None
    last_node: np.ndarray | None = None

    for t in levels:
        mask = valid & (G >= t)
        lab, n = label(mask, structure=STRUCT8)
        node_of = np.zeros(n + 1, dtype=np.int64)     # current label -> node id

        if n == 0:
            last_lab, last_node = lab, node_of
            prev_lab, prev_node = lab, node_of
            continue

        comp_birth = np.asarray(ndi_maximum(G, labels=lab, index=np.arange(1, n + 1)),
                                dtype=np.float64)
        n_prev = 0 if prev_node is None else int(prev_node.size) - 1

        if n_prev > 0:
            # Nested superlevel sets => each previous component lies inside exactly one current
            # component, so `maximum` of the current labels over a previous component IS its
            # parent.  ndi_maximum returns one value per requested index: entry i is label i+1.
            kid_labels = np.arange(1, n_prev + 1)
            parent_of = np.asarray(ndi_maximum(lab, labels=prev_lab, index=kid_labels),
                                   dtype=np.int64)
            order = np.argsort(parent_of, kind="stable")
            parents_sorted = parent_of[order]
            children_sorted = kid_labels[order]
            idx = np.arange(1, n + 1)
            lo_b = np.searchsorted(parents_sorted, idx, side="left")
            hi_b = np.searchsorted(parents_sorted, idx, side="right")

            die_birth = np.zeros(n_prev + 1, dtype=np.float32)
            dying = np.zeros(n_prev + 1, dtype=bool)
            for p in range(1, n + 1):
                lo, hi = int(lo_b[p - 1]), int(hi_b[p - 1])
                if hi - lo == 1:                       # 1:1 growth -> the child continues
                    node_of[p] = prev_node[int(children_sorted[lo])]
                elif hi - lo == 0:                     # component born at this level
                    node_of[p] = n_nodes
                    n_nodes += 1
                    node_birth.append(float(comp_birth[p - 1]))
                else:                                  # merge: all but the oldest child die here
                    kids = children_sorted[lo:hi]
                    kb = np.asarray([node_birth[int(prev_node[int(c)])] for c in kids])
                    keep = int(kids[int(np.argmax(kb))])   # highest birth = oldest = survivor
                    node_of[p] = prev_node[keep]
                    for c in kids:
                        c = int(c)
                        if c != keep:
                            dying[c] = True
                            die_birth[c] = np.float32(node_birth[int(prev_node[c])])

            if dying.any():
                sel = (prev_lab > 0) & ~done & dying[prev_lab]
                if sel.any():
                    birth_px[sel] = die_birth[prev_lab][sel]
                    death_px[sel] = np.float32(t)
                    done |= sel
                n_features += int(dying.sum())
        else:
            for p in range(1, n + 1):
                node_of[p] = n_nodes
                n_nodes += 1
                node_birth.append(float(comp_birth[p - 1]))

        last_lab, last_node = lab, node_of
        prev_lab, prev_node = lab, node_of

    # Close the filtration: whatever is still alive at the lowest level is closed out there.
    if last_lab is not None and last_node is not None and last_node.size > 1:
        t_low = np.float32(levels[-1])
        alive_b = np.zeros(int(last_node.size), dtype=np.float32)
        for lb in range(1, last_node.size):
            alive_b[lb] = np.float32(node_birth[int(last_node[lb])])
        sel = (last_lab > 0) & ~done & (alive_b[last_lab] > 0)
        if sel.any():
            birth_px[sel] = alive_b[last_lab][sel]
            death_px[sel] = t_low
            done |= sel
            n_features += int((alive_b > 0).sum())

    inside = valid & done
    persistence = np.zeros(G.shape, dtype=np.float32)
    prominence = np.zeros(G.shape, dtype=np.float32)
    persistence[inside] = np.maximum(birth_px[inside] - death_px[inside], 0.0)
    prominence[inside] = np.maximum(G[inside] - death_px[inside], 0.0)
    birth_px[~inside] = 0.0
    death_px[~inside] = 0.0
    return PersistenceResult(persistence=persistence, prominence=prominence, birth=birth_px,
                             death=death_px, n_features=n_features, levels=levels)


def topological_persistence(G: np.ndarray, valid: np.ndarray,
                            sigmas: tuple[float, ...] = DEFAULT_SIGMAS,
                            n_levels: int = 48, pct: float = 99.5,
                            which: str = "prominence",
                            verbose: bool = False,
                            keep_per_sigma: bool = False) -> tuple[np.ndarray, dict]:
    """Ridge persistence averaged across a sweep of smoothing scales; result in [0, 1].

    ``which`` selects the per-pixel quantity that is swept: ``'prominence'`` (G - death, peaks on
    crests) or ``'persistence'`` (birth - death, constant along a feature).  At each sigma the
    chosen map is normalised by its own ``pct`` quantile inside the footprint and the sweep
    average is returned.  ``info['per_sigma']`` keeps the individual maps for auditing.
    """
    if which not in ("prominence", "persistence"):
        raise ValueError(f"which must be 'prominence' or 'persistence', got {which!r}")

    G = np.asarray(G, dtype=np.float32)
    valid = np.asarray(valid, bool)
    filled = np.where(valid & np.isfinite(G), G, 0.0).astype(np.float64)

    acc = np.zeros(G.shape, dtype=np.float32)
    per_sigma: dict[float, np.ndarray] = {}
    feats: dict[float, int] = {}
    for s in sigmas:
        Gs = gaussian_filter(filled, s, mode="nearest").astype(np.float32)
        res = persistence_map_h0(Gs, valid, n_levels=n_levels)
        raw = res.prominence if which == "prominence" else res.persistence
        v = valid & (raw > 0)
        hi = float(np.percentile(raw[v], pct)) if v.any() else 0.0
        norm = np.zeros(G.shape, dtype=np.float32)
        if hi > 0:
            norm[valid] = np.clip(raw[valid] / hi, 0.0, 1.0)
        if keep_per_sigma:
            per_sigma[float(s)] = norm
        feats[float(s)] = res.n_features
        n_feat = res.n_features
        acc += norm
        if verbose:
            print(f"    sigma={s:>4}: features={n_feat:>8}  p{pct}({which})={hi:.4g}", flush=True)
        del res, raw, Gs
    topo = (acc / float(len(sigmas))).astype(np.float32)
    topo[~valid] = 0.0
    return topo, {"per_sigma": per_sigma, "n_features": feats, "sigmas": tuple(sigmas),
                  "which": which}
