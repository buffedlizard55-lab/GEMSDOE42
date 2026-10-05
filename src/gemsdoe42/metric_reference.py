"""Dependency-free reference implementation for the official DTI arithmetic."""

from __future__ import annotations

import math
from collections.abc import Sequence


def dti_from_counts(
    tp_weighted: float,
    fp_weighted: float,
    fn_weighted: float,
    *,
    alpha: float = 0.2,
    beta: float = 0.8,
    epsilon: float = 1e-12,
) -> float:
    """Compute the distance-weighted Tversky ratio from its three weighted terms."""
    if alpha < 0 or beta < 0 or epsilon < 0:
        raise ValueError("alpha, beta and epsilon must be non-negative")
    denominator = tp_weighted + alpha * fp_weighted + beta * fn_weighted + epsilon
    if denominator == 0:
        return 0.0
    return float(tp_weighted / denominator)


def distance_weighted_tversky_reference(
    prediction: Sequence[Sequence[float]],
    truth: Sequence[Sequence[bool]],
    valid: Sequence[Sequence[bool]] | None = None,
    *,
    pixel_size_m: float = 100.0,
    radius_m: float = 300.0,
    alpha: float = 0.2,
    beta: float = 0.8,
) -> dict[str, float]:
    """Slow exact reference for small arrays, following DrivenData's definition.

    The true-positive term is summed over truth pixels using the best nearby
    prediction confidence multiplied by the triangular distance kernel. The false
    positive term is predicted mass times one minus the nearest-truth kernel weight.
    The false-negative term is the residual uncovered fraction at each truth pixel.
    This O(N×|truth|) implementation is for tests and audits, not full-scene scoring.
    """
    height = len(prediction)
    if height == 0 or len(truth) != height:
        raise ValueError("prediction and truth must be non-empty and same-shaped")
    width = len(prediction[0])
    if width == 0 or any(len(row) != width for row in prediction) or any(
        len(row) != width for row in truth
    ):
        raise ValueError("prediction and truth must be rectangular and same-shaped")
    if valid is not None and (
        len(valid) != height or any(len(row) != width for row in valid)
    ):
        raise ValueError("valid mask shape must match prediction")
    if pixel_size_m <= 0 or radius_m <= 0:
        raise ValueError("pixel_size_m and radius_m must be positive")

    truth_points = []
    prediction_points = []
    for y in range(height):
        for x in range(width):
            keep = valid is None or bool(valid[y][x])
            value = float(prediction[y][x])
            if keep:
                if not math.isfinite(value) or value < 0.0 or value > 1.0:
                    raise ValueError("valid prediction values must be finite and in [0,1]")
                if bool(truth[y][x]):
                    truth_points.append((y, x))
                if value > 0.0:
                    prediction_points.append((y, x, value))

    def kernel(distance_m: float) -> float:
        return max(1.0 - distance_m / radius_m, 0.0)

    fp_weighted = 0.0
    for y, x, value in prediction_points:
        nearest_weight = 0.0
        for gy, gx in truth_points:
            distance = math.hypot(y - gy, x - gx) * pixel_size_m
            nearest_weight = max(nearest_weight, kernel(distance))
        fp_weighted += value * (1.0 - nearest_weight)

    tp_weighted = 0.0
    fn_weighted = 0.0
    for gy, gx in truth_points:
        best_cover = 0.0
        for y, x, value in prediction_points:
            distance = math.hypot(y - gy, x - gx) * pixel_size_m
            if distance <= radius_m:
                best_cover = max(best_cover, value * kernel(distance))
        tp_weighted += best_cover
        fn_weighted += 1.0 - best_cover

    score = dti_from_counts(tp_weighted, fp_weighted, fn_weighted, alpha=alpha, beta=beta)
    return {
        "tp_weighted": tp_weighted,
        "fp_weighted": fp_weighted,
        "fn_weighted": fn_weighted,
        "dti": score,
    }
