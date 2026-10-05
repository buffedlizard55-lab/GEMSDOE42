"""Fusion of worming x persistence into one submission-ready emission.

Per-pixel construction (H42-1):
  worm_joint = worm_mag + worm_grav        # continuation steps, 0..12
  pers_joint = pers_mag + pers_grav        # topo birth-death steps, 0..12
  raw        = worm_joint * pers_joint     # 0..144; corroboration of two
                                             # independent notions of scale
  score01    = raw / max(raw)              # normalised to [0, 1]

Layers are summed (a fault may be magnetic-only or gravity-only) while the
two scale notions are multiplied (a candidate must survive both). Catalogue
pixels are then zeroed: the hidden test set is faults NOT in the catalogue,
so on-catalogue mass cannot earn round-1 credit.

DTI decision theory (metric identity: adding unit mass at weight k is a gain
iff k > 0.2*DTI) plus the group's measured result that thinning a surface to
dots raises mean credit per pixel motivates the dotted emission: greedy
score-descending acceptance with a euclidean exclusion disk of `spacing_px`.
Greedy prefixes are nested, so one run to the max budget yields every
smaller budget by slicing.
"""
from __future__ import annotations

import numpy as np

SPACING_PX = 2.8  # matches the proven D2.8 dotted family spacing
BUDGET = 40_000   # near the H33-2-B2 operating point (37,654 dots)


def fuse_scores(worm_mag, worm_grav, pers_mag, pers_grav):
    """Return (raw, components dict). All inputs array-like, same shape."""
    wm = np.asarray(worm_mag, dtype=np.float64)
    wg = np.asarray(worm_grav, dtype=np.float64)
    pm = np.asarray(pers_mag, dtype=np.float64)
    pg = np.asarray(pers_grav, dtype=np.float64)
    worm_joint = wm + wg
    pers_joint = pm + pg
    raw = worm_joint * pers_joint
    return raw, dict(worm_joint=worm_joint, pers_joint=pers_joint)


def normalize01(raw, footprint) -> np.ndarray:
    """Normalise in-footprint raw scores to [0,1]; outside -> 0 (finite)."""
    r = np.asarray(raw, dtype=np.float64)
    foot = np.asarray(footprint, bool)
    mx = float(r[foot].max(initial=0.0))
    out = np.zeros(r.shape, dtype=np.float64)
    if mx > 0:
        out[foot] = r[foot] / mx
    return out


def _disk_offsets(spacing_px: float):
    r = int(np.ceil(spacing_px))
    offs = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            if np.hypot(dy, dx) <= spacing_px + 1e-9:
                offs.append((dy, dx))
    return offs


def thin_dots(score01, footprint, labels, spacing_px: float = SPACING_PX,
              budget: int = BUDGET):
    """Greedy score-descending dotted emission.

    Candidates: score01 > 0 & footprint & ~labels, ties broken by (y, x) for
    determinism. Returns (accept_ordered_coords (K,2), mask (H,W) bool).
    """
    s = np.asarray(score01, dtype=np.float64)
    foot = np.asarray(footprint, bool)
    lab = np.asarray(labels, bool)
    cand = (s > 0) & foot & (~lab)
    yy, xx = np.nonzero(cand)
    vals = s[yy, xx]
    order = np.lexsort((xx, yy, -vals))
    ys, xs = yy[order], xx[order]
    H, W = s.shape
    blocked = np.zeros((H, W), dtype=bool)
    offs = _disk_offsets(spacing_px)
    accepted = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        if blocked[y, x]:
            continue
        accepted.append((y, x))
        blocked[y, x] = True
        for dy, dx in offs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                blocked[ny, nx] = True
        if len(accepted) >= budget:
            break
    acc = np.array(accepted, dtype=np.int64).reshape(-1, 2)
    mask = np.zeros((H, W), dtype=bool)
    if len(acc):
        mask[acc[:, 0], acc[:, 1]] = True
    return acc, mask


def mask_from_prefix(accept_ordered_coords: np.ndarray, shape, k: int):
    """Nested budget-k mask from a greedy acceptance sequence."""
    m = np.zeros(shape, dtype=bool)
    k = min(int(k), len(accept_ordered_coords))
    if k > 0:
        sel = accept_ordered_coords[:k]
        m[sel[:, 0], sel[:, 1]] = True
    return m
