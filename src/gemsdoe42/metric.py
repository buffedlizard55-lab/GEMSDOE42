"""Vectorized distance-weighted Tversky metric for modest/full rasters."""

from __future__ import annotations


def distance_weighted_tversky(
    prediction,
    truth,
    valid=None,
    *,
    pixel_size_m: float = 100.0,
    radius_m: float = 300.0,
    alpha: float = 0.2,
    beta: float = 0.8,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Compute the official competition DTI using exact discrete offsets.

    Raster values outside ``valid`` are ignored. For each truth pixel, TP coverage is
    the maximum nearby ``p(x) * k(d)``; FP weight uses the nearest truth kernel; FN is
    the uncovered residual. The function returns the three weighted sums and DTI.
    """
    import math

    import numpy as np
    from scipy.ndimage import distance_transform_edt

    p = np.asarray(prediction, dtype=np.float32)
    g = np.asarray(truth, dtype=bool)
    if p.ndim != 2 or p.shape != g.shape:
        raise ValueError("prediction and truth must be same-shaped 2D arrays")
    if valid is None:
        mask = np.ones(p.shape, dtype=bool)
    else:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != p.shape:
            raise ValueError("valid mask shape must match prediction")
    if pixel_size_m <= 0 or radius_m <= 0 or alpha < 0 or beta < 0 or epsilon < 0:
        raise ValueError("pixel/radius must be positive; metric coefficients non-negative")
    if not mask.any():
        raise ValueError("valid mask has no pixels")
    if np.any(~np.isfinite(p[mask])) or np.any((p[mask] < 0.0) | (p[mask] > 1.0)):
        raise ValueError("valid prediction values must be finite and in [0,1]")

    g = g & mask
    p = np.where(mask, p, 0.0).astype(np.float32, copy=False)
    truth_coordinates = np.argwhere(g)
    if truth_coordinates.size:
        distance_to_truth = distance_transform_edt(~g, sampling=(pixel_size_m, pixel_size_m))
        truth_weight = np.clip(1.0 - distance_to_truth / radius_m, 0.0, 1.0)
    else:
        truth_weight = np.zeros(p.shape, dtype=np.float32)

    fp_weighted = float(np.sum(p[mask] * (1.0 - truth_weight[mask]), dtype=np.float64))

    best_cover = np.zeros(p.shape, dtype=np.float32)
    max_offset = math.floor(radius_m / pixel_size_m)
    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            distance = math.hypot(dy, dx) * pixel_size_m
            if distance > radius_m:
                continue
            kernel = 1.0 - distance / radius_m
            # Destination cells (truth sites) receive nearby prediction values.
            y0 = max(0, -dy)
            y1 = min(p.shape[0], p.shape[0] - dy)
            x0 = max(0, -dx)
            x1 = min(p.shape[1], p.shape[1] - dx)
            src_y0, src_y1 = y0 + dy, y1 + dy
            src_x0, src_x1 = x0 + dx, x1 + dx
            candidate = p[src_y0:src_y1, src_x0:src_x1] * kernel
            target = best_cover[y0:y1, x0:x1]
            np.maximum(target, candidate, out=target)

    best_cover[~mask] = 0.0
    tp_weighted = float(np.sum(best_cover[g], dtype=np.float64))
    fn_weighted = float(np.sum(1.0 - best_cover[g], dtype=np.float64))
    denominator = tp_weighted + alpha * fp_weighted + beta * fn_weighted + epsilon
    score = tp_weighted / denominator if denominator > 0.0 else 0.0
    return {
        "tp_weighted": tp_weighted,
        "fp_weighted": fp_weighted,
        "fn_weighted": fn_weighted,
        "dti": float(score),
    }
