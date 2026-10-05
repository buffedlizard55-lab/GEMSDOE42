"""Pre-registered concentration statistics and single-scale comparator."""

from __future__ import annotations


def single_scale_gradient_baseline(gravity, magnetic, footprint, *, pixel_size_m: float = 100.0):
    """Create an unsupervised equal-weight single-scale gradient-magnitude comparator."""
    import numpy as np

    from .pipeline import _nearest_fill, _positive_robust_scale
    from .worming import horizontal_gradient_magnitude

    footprint = np.asarray(footprint, dtype=bool)
    layers = []
    for values in (gravity, magnetic):
        filled, valid = _nearest_fill(values, footprint)
        gradient = horizontal_gradient_magnitude(filled, pixel_size_m)
        normalized, _ = _positive_robust_scale(gradient, valid)
        layers.append(normalized)
    comparator = (layers[0] + layers[1]) * 0.5
    comparator[~footprint] = 0.0
    return comparator.astype(np.float32, copy=False)


def concentration_report(score, valid) -> dict:
    """Report support, top-rank score-mass concentration and four-neighbor Moran I."""
    import math

    import numpy as np

    values = np.asarray(score, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("score and valid mask must be same-shaped 2D arrays")
    if not mask.any():
        raise ValueError("no valid score cells")
    in_footprint = values[mask]
    if np.any((in_footprint < 0.0) | (in_footprint > 1.0)):
        raise ValueError("scores must lie in [0,1]")
    mass = float(in_footprint.sum(dtype=np.float64))
    report = {
        "valid_pixels": int(mask.sum()),
        "nonzero_pixels": int(np.count_nonzero(in_footprint)),
        "nonzero_support_fraction": float(np.count_nonzero(in_footprint) / in_footprint.size),
        "score_mass": mass,
    }
    flat = values[mask]
    for percent in (1, 5):
        count = max(1, math.ceil(flat.size * percent / 100.0))
        if mass <= 0.0:
            fraction = 0.0
        elif count >= flat.size:
            fraction = 1.0
        else:
            top = np.argpartition(flat, -count)[-count:]
            fraction = float(flat[top].sum(dtype=np.float64) / mass)
        report[f"top_{percent}_percent_mass_fraction"] = fraction

    mean = float(in_footprint.mean())
    centered = np.where(mask, values - mean, 0.0)
    variance_sum = float(np.sum(centered[mask] ** 2, dtype=np.float64))
    horizontal = mask[:, 1:] & mask[:, :-1]
    vertical = mask[1:, :] & mask[:-1, :]
    edge_count = int(horizontal.sum() + vertical.sum())
    neighbor_product = float(
        np.sum(centered[:, 1:][horizontal] * centered[:, :-1][horizontal], dtype=np.float64)
        + np.sum(centered[1:, :][vertical] * centered[:-1, :][vertical], dtype=np.float64)
    )
    if variance_sum > 0.0 and edge_count > 0:
        moran = float(mask.sum() / edge_count * neighbor_product / variance_sum)
    else:
        moran = 0.0
    report["four_neighbor_moran_i"] = moran
    report["moran_definition"] = "n / undirected-valid-neighbor-pairs * sum(centered_i*centered_j) / sum(centered_i^2)"
    return report
