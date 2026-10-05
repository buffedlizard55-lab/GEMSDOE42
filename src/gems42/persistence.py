"""Scale-space topological persistence of gradient-magnitude ridges.

Theory. Zero-dimensional persistent homology of superlevel sets tracks how
connected components (local maxima) of a scalar field appear (birth) and merge
(death) as the level sweeps down; persistence = death - birth measures
feature significance (Edelsbrunner, Letscher & Zomorodian, "Topological
persistence and simplification", Discrete & Computational Geometry 28, 2002).
Applied across a Gaussian *smoothing* sweep instead of a level sweep, the
same birth/death bookkeeping measures which ridge maxima survive
simplification of the surface - i.e. cross-scale topological stability,
independent of the upward-continuation physics used by worming.

Implementation (exact elder-rule tracking of discrete maxima):
  1. smooth the h=0 gradient-magnitude surface with Gaussians
     sigmas_px = (0, 1, 2, 4, 8, 16)  (dyadic, 100 m px)
  2. detect 3x3 local maxima above a permissive per-scale threshold
  3. link maxima across adjacent scales by nearest neighbour within
     link_radius px; oldest track wins ties (elder rule); unmatched coarse
     maxima are born at that scale
  4. per-track persistence_steps = death_idx - birth_idx + 1 (1..6),
     persistence_sigma = sigmas[death] - sigmas[birth] (px)
  5. splat persistence at each track's birth (finest, best-localised) pixel

The output is sparse by construction: one pixel per tracked maximum.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

SIGMAS_PX = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
LINK_RADIUS_PX = 3.0


def detect_maxima(g: np.ndarray, footprint: np.ndarray,
                  top_frac: float = 0.50):
    """Local maxima of `g` inside `footprint` above the top-fraction cutoff.

    Returns (coords (N,2) int array sorted by (-value, y, x), values (N,)).
    """
    g = np.asarray(g, dtype=np.float64)
    foot = np.asarray(footprint, bool)
    mx = maximum_filter(g, size=3, mode="reflect")
    thr = float(np.percentile(g[foot], 100.0 * (1.0 - top_frac)))
    mask = (g >= mx) & (g >= thr) & foot
    yy, xx = np.nonzero(mask)
    vals = g[yy, xx]
    order = np.lexsort((xx, yy, -vals))  # deterministic: value desc, then y, x
    coords = np.column_stack([yy[order], xx[order]]).astype(np.int64)
    return coords, vals[order]


def track_maxima(scales, link_radius: float = LINK_RADIUS_PX):
    """Link per-scale maxima into tracks with the elder rule.

    `scales`: list of (coords, values) per smoothing scale, finest first.
    Returns list of track dicts with keys birth, death, birth_pos (y,x),
    pos (latest y,x), birth_val, steps (= death-birth+1).
    """
    tracks = []
    alive = []  # indices into tracks, oldest birth first
    for i, (coords, vals) in enumerate(scales):
        claimed = np.zeros(len(coords), dtype=bool)
        if i == 0 or len(coords) == 0:
            # every maximum starts a track (scale 0, or empty coarse scale)
            for j in range(len(coords)):
                tracks.append(dict(birth=i, death=i,
                                   birth_pos=(int(coords[j, 0]), int(coords[j, 1])),
                                   pos=(int(coords[j, 0]), int(coords[j, 1])),
                                   birth_val=float(vals[j])))
                alive.append(len(tracks) - 1)
                claimed[j] = True
            # tracks alive but no maxima to match: all die at previous scale
            if len(coords) == 0:
                alive = []
            continue
        tree = cKDTree(coords.astype(np.float64))
        # elder rule: oldest birth first; ties by birth value desc, then pos
        alive.sort(key=lambda t: (tracks[t]["birth"],
                                  -tracks[t]["birth_val"],
                                  tracks[t]["pos"]))
        still = []
        for t in alive:
            y, x = tracks[t]["pos"]
            dist, j = tree.query([y, x], k=1,
                                 distance_upper_bound=link_radius)
            j = int(j)
            if np.isfinite(dist) and j < len(coords) and not claimed[j]:
                claimed[j] = True
                tracks[t]["death"] = i
                tracks[t]["pos"] = (int(coords[j, 0]), int(coords[j, 1]))
                still.append(t)
            # else: track dies at its current death (no extension)
        # unmatched coarse maxima are born here
        for j in np.nonzero(~claimed)[0]:
            tracks.append(dict(birth=i, death=i,
                               birth_pos=(int(coords[j, 0]), int(coords[j, 1])),
                               pos=(int(coords[j, 0]), int(coords[j, 1])),
                               birth_val=float(vals[j])))
            still.append(len(tracks) - 1)
        alive = still
    for t in tracks:
        t["steps"] = int(t["death"] - t["birth"] + 1)
    return tracks


def scale_space_persistence(gradmag0: np.ndarray, footprint: np.ndarray,
                            sigmas_px=SIGMAS_PX,
                            top_frac: float = 0.50,
                            link_radius: float = LINK_RADIUS_PX):
    """Full persistence sweep on one layer's h=0 gradient magnitude.

    Returns (pers_steps_img, pers_sigma_img, tracks, n_maxima_per_scale).
    Images are float64, nonzero only at track birth pixels.
    """
    g0 = np.asarray(gradmag0, dtype=np.float64)
    foot = np.asarray(footprint, bool)
    scales = []
    counts = []
    for s in sigmas_px:
        gs = g0 if float(s) == 0.0 else gaussian_filter(g0, float(s),
                                                       mode="reflect")
        coords, vals = detect_maxima(gs, foot, top_frac)
        scales.append((coords, vals))
        counts.append(int(len(coords)))
    tracks = track_maxima(scales, link_radius)
    sig = list(sigmas_px)
    img_steps = np.zeros(g0.shape, dtype=np.float64)
    img_sigma = np.zeros(g0.shape, dtype=np.float64)
    for t in tracks:
        y, x = t["birth_pos"]
        # birth pixels are unique per scale but two tracks born at different
        # scales could share a pixel in theory; keep the max (elder wins).
        if t["steps"] > img_steps[y, x]:
            img_steps[y, x] = float(t["steps"])
            img_sigma[y, x] = float(sig[t["death"]] - sig[t["birth"]])
    return img_steps, img_sigma, tracks, counts
