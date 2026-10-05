"""Spatially blocked, matched-support proxy evaluation utilities."""

from __future__ import annotations


def four_spatial_fold_masks(valid, *, guard_pixels: int = 3):
    """Split a raster into four contiguous quadrants and remove each block edge guard."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(valid, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("valid must be a non-empty 2D mask")
    if guard_pixels < 0:
        raise ValueError("guard_pixels must be non-negative")

    row_edges = np.linspace(0, mask.shape[0], 3, dtype=int)
    col_edges = np.linspace(0, mask.shape[1], 3, dtype=int)
    folds = []
    for row_block in range(2):
        for col_block in range(2):
            block = np.zeros(mask.shape, dtype=bool)
            block[
                row_edges[row_block] : row_edges[row_block + 1],
                col_edges[col_block] : col_edges[col_block + 1],
            ] = True
            padded = np.pad(block, 1, mode="constant", constant_values=False)
            inside_distance = distance_transform_edt(padded)[1:-1, 1:-1]
            evaluation = block & (inside_distance > guard_pixels) & mask
            if evaluation.any():
                folds.append(evaluation)
    if len(folds) != 4:
        raise ValueError(
            f"expected four non-empty spatial folds after guarding, got {len(folds)}"
        )
    return folds


def _top_k_binary(scores, mask, count):
    import numpy as np

    output = np.zeros(scores.shape, dtype=np.float32)
    if count <= 0:
        return output
    flat_mask = np.flatnonzero(mask.ravel() & np.isfinite(scores.ravel()))
    flat_scores = scores.ravel()[flat_mask]
    positive_positions = np.flatnonzero(flat_scores > 0.0)
    if positive_positions.size < count:
        raise ValueError(
            f"candidate has only {positive_positions.size} positive cells but matched "
            f"support requires {count}; do not fill the score map with arbitrary zeros"
        )
    # Partition first to keep memory/time bounded, then deterministically order the K
    # selected values by descending score and ascending global index for tie-breaking.
    local = np.argpartition(flat_scores, -count)[-count:]
    chosen_scores = flat_scores[local]
    chosen_indices = flat_mask[local]
    order = np.lexsort((chosen_indices, -chosen_scores))
    selected = chosen_indices[order]
    output.ravel()[selected] = 1.0
    return output


def _bootstrap_mean_interval(deltas, *, iterations: int = 20000, seed: int = 42):
    import numpy as np

    values = np.asarray(deltas, dtype=np.float64)
    if values.size == 0:
        raise ValueError("at least one fold delta is required")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(iterations, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def evaluate_matched_support_holdout(
    candidate,
    incumbent,
    truth,
    valid,
    *,
    pixel_size_m: float = 100.0,
    radius_m: float = 300.0,
    guard_pixels: int = 3,
    alpha: float = 0.2,
    beta: float = 0.8,
    bootstrap_iterations: int = 20000,
    bootstrap_seed: int = 42,
    mass_area_fractions=(0.005, 0.01, 0.02),
    fold_domain=None,
):
    """Compare candidate/holdout incumbent on four spatial blocks at equal support.

    The continuous candidate and incumbent are converted to binary unit-confidence
    top-K layouts within each held-out block, where K is the incumbent's count of
    positive pixels. This implements the registered same-count geometry comparison.
    A fixed 0.5%, 1% and 2% of valid-fold-area support sensitivity sweep and
    unthresholded DTI are also reported. If the candidate cannot supply K positive
    pixels for the primary comparison, evaluation fails closed.
    """
    import numpy as np

    from .metric import distance_weighted_tversky

    candidate = np.asarray(candidate, dtype=np.float32)
    incumbent = np.asarray(incumbent, dtype=np.float32)
    truth = np.asarray(truth, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if candidate.ndim != 2 or any(
        array.shape != candidate.shape for array in (incumbent, truth, valid)
    ):
        raise ValueError("candidate, incumbent, truth and valid must have equal 2D shapes")
    if np.any(~np.isfinite(candidate[valid])) or np.any(~np.isfinite(incumbent[valid])):
        raise ValueError("candidate/incumbent must be finite within the valid footprint")
    if np.any((candidate[valid] < 0) | (candidate[valid] > 1)) or np.any(
        (incumbent[valid] < 0) | (incumbent[valid] > 1)
    ):
        raise ValueError("candidate/incumbent values must be in [0,1]")

    domain = valid if fold_domain is None else np.asarray(fold_domain, dtype=bool)
    if domain.shape != candidate.shape or not domain.any():
        raise ValueError("fold_domain must be a non-empty mask matching candidate shape")
    folds = four_spatial_fold_masks(domain, guard_pixels=guard_pixels)
    folds = [fold & valid for fold in folds]
    if any(not fold.any() for fold in folds):
        raise ValueError("one or more guarded folds has no valid labeled evaluation pixels")

    area_fractions = tuple(float(value) for value in mass_area_fractions)
    if not area_fractions or any(value <= 0.0 or value > 1.0 for value in area_fractions):
        raise ValueError("mass_area_fractions must be in (0,1]")
    rows = []
    for index, fold_mask in enumerate(folds, start=1):
        incumbent_support = int(np.count_nonzero((incumbent > 0.0) & fold_mask))
        if incumbent_support == 0:
            raise ValueError(f"incumbent has no positive prediction mass in fold {index}")
        base_prediction = _top_k_binary(incumbent, fold_mask, incumbent_support)
        candidate_prediction = _top_k_binary(candidate, fold_mask, incumbent_support)
        base_terms = distance_weighted_tversky(
            base_prediction,
            truth,
            fold_mask,
            pixel_size_m=pixel_size_m,
            radius_m=radius_m,
            alpha=alpha,
            beta=beta,
        )
        candidate_terms = distance_weighted_tversky(
            candidate_prediction,
            truth,
            fold_mask,
            pixel_size_m=pixel_size_m,
            radius_m=radius_m,
            alpha=alpha,
            beta=beta,
        )
        continuous_base = distance_weighted_tversky(
            incumbent,
            truth,
            fold_mask,
            pixel_size_m=pixel_size_m,
            radius_m=radius_m,
            alpha=alpha,
            beta=beta,
        )
        continuous_candidate = distance_weighted_tversky(
            candidate,
            truth,
            fold_mask,
            pixel_size_m=pixel_size_m,
            radius_m=radius_m,
            alpha=alpha,
            beta=beta,
        )
        sweep = []
        candidate_support = int(np.count_nonzero((candidate > 0.0) & fold_mask))
        for area_fraction in area_fractions:
            requested = max(1, int(np.ceil(fold_mask.sum() * area_fraction)))
            if incumbent_support < requested:
                sweep.append({
                    "area_fraction": area_fraction,
                    "support": requested,
                    "status": f"incumbent has only {incumbent_support} positive cells",
                })
                continue
            if candidate_support < requested:
                sweep.append({
                    "area_fraction": area_fraction,
                    "support": requested,
                    "status": f"candidate has only {candidate_support} positive cells",
                })
                continue
            sweep_base = _top_k_binary(incumbent, fold_mask, requested)
            sweep_candidate = _top_k_binary(candidate, fold_mask, requested)
            sweep_base_dti = distance_weighted_tversky(
                sweep_base, truth, fold_mask, pixel_size_m=pixel_size_m,
                radius_m=radius_m, alpha=alpha, beta=beta,
            )["dti"]
            sweep_candidate_dti = distance_weighted_tversky(
                sweep_candidate, truth, fold_mask, pixel_size_m=pixel_size_m,
                radius_m=radius_m, alpha=alpha, beta=beta,
            )["dti"]
            sweep.append({
                "area_fraction": area_fraction,
                "support": requested,
                "incumbent_dti": sweep_base_dti,
                "candidate_dti": sweep_candidate_dti,
                "delta_dti": sweep_candidate_dti - sweep_base_dti,
                "status": "evaluated",
            })
        rows.append(
            {
                "fold": index,
                "matched_positive_pixels": incumbent_support,
                "incumbent": base_terms,
                "candidate": candidate_terms,
                "delta_dti": candidate_terms["dti"] - base_terms["dti"],
                "unthresholded_incumbent": continuous_base,
                "unthresholded_candidate": continuous_candidate,
                "mass_sweep": sweep,
                "evaluation_pixels": int(fold_mask.sum()),
                "truth_pixels": int((truth & fold_mask).sum()),
            }
        )

    deltas = [row["delta_dti"] for row in rows]
    mean_delta = float(np.mean(deltas))
    interval = _bootstrap_mean_interval(
        deltas, iterations=bootstrap_iterations, seed=bootstrap_seed
    )
    positive_folds = int(sum(delta > 0.0 for delta in deltas))
    passed = mean_delta > 0.0 and positive_folds >= 3 and interval[0] > 0.0
    return {
        "protocol": "four contiguous quadrants; 300 m guarded; matched incumbent positive-support count; official DTI",
        "fold_domain_pixels": int(domain.sum()),
        "evaluation_valid_pixels": int(valid.sum()),
        "mass_sweep_area_fractions": list(area_fractions),
        "folds": rows,
        "mean_paired_delta": mean_delta,
        "positive_folds": positive_folds,
        "bootstrap_95_percent_interval": [interval[0], interval[1]],
        "promotion_gate_passed": bool(passed),
        "truth_caveat": "Known-fault labels are an imperfect proxy and cannot establish recall on newly mapped faults.",
    }
