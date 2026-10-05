"""WPH-01 unsupervised inference pipeline."""

from __future__ import annotations

from collections.abc import Sequence


def _nearest_fill(values, valid):
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    data = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(data)
    if data.ndim != 2 or mask.shape != data.shape:
        raise ValueError("layer and validity mask must be same-shaped 2D arrays")
    if not mask.any():
        raise ValueError("layer has no valid cells in the official footprint")
    nearest = distance_transform_edt(
        ~mask, return_distances=False, return_indices=True
    )
    return data[tuple(nearest)].astype(np.float32, copy=False), mask


def _positive_robust_scale(values, valid, quantile: float = 0.995):
    import numpy as np

    data = np.asarray(values, dtype=np.float32)
    positive = data[np.asarray(valid, dtype=bool) & np.isfinite(data) & (data > 0)]
    if positive.size == 0:
        return np.zeros(data.shape, dtype=np.float32), 0.0
    scale = float(np.quantile(positive, quantile))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(positive.max())
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros(data.shape, dtype=np.float32), 0.0
    normalized = np.clip(data / scale, 0.0, 1.0).astype(np.float32, copy=False)
    normalized[~np.asarray(valid, dtype=bool)] = 0.0
    return normalized, scale


def wph01_score(
    gravity,
    magnetic,
    footprint,
    *,
    pixel_size_m: float = 100.0,
    continuation_heights_m: Sequence[float] = (0, 100, 200, 400, 800, 1600, 3200),
    smoothing_sigmas_pixels: Sequence[float] = (0.5, 1, 2, 4, 8, 16, 32),
    worm_match_radius_pixels: int = 2,
    topology_match_radius_pixels: int = 2,
):
    """Compute the WPH-01 cross-scale score for gravity and RTP magnetic layers.

    No labels are read or used. Each field contributes the product of its worm
    continuation survival and smoothing-scale 0D topological persistence. The two
    products are robustly normalized separately and averaged. Output is a continuous
    confidence score, not a calibrated probability.
    """
    import numpy as np

    from .topology import multiscale_topology_score
    from .worming import horizontal_gradient_magnitude, worm_survival_score

    footprint = np.asarray(footprint, dtype=bool)
    if footprint.ndim != 2 or not footprint.any():
        raise ValueError("official footprint must be a non-empty 2D mask")

    layer_scores = []
    layer_reports = {}
    for label, raw_layer in (("gravity", gravity), ("magnetic_rtp", magnetic)):
        layer, layer_valid = _nearest_fill(raw_layer, footprint)
        gradient_magnitude = horizontal_gradient_magnitude(layer, pixel_size_m)
        worm, worm_report = worm_survival_score(
            layer,
            layer_valid,
            continuation_heights_m,
            pixel_size_m=pixel_size_m,
            match_radius_pixels=worm_match_radius_pixels,
        )
        topology = multiscale_topology_score(
            gradient_magnitude,
            layer_valid,
            smoothing_sigmas_pixels,
            match_radius_pixels=topology_match_radius_pixels,
        )
        product = worm * topology
        product[~layer_valid] = 0.0
        normalized, scale = _positive_robust_scale(product, layer_valid)
        layer_scores.append(normalized)
        nonzero = int(np.count_nonzero(normalized[layer_valid]))
        layer_reports[label] = {
            **worm_report,
            "valid_pixels": int(layer_valid.sum()),
            "nonzero_score_pixels": nonzero,
            "nonzero_fraction": float(nonzero / max(1, int(layer_valid.sum()))),
            "robust_product_scale": scale,
        }

    score = (layer_scores[0] + layer_scores[1]) * 0.5
    score[~footprint] = 0.0
    score = np.clip(score, 0.0, 1.0).astype(np.float32, copy=False)
    positive = score[footprint & (score > 0.0)]
    report = {
        "hypothesis_id": "WPH-01",
        "score_semantics": "continuous ranking/confidence, not calibrated probability",
        "footprint_pixels": int(footprint.sum()),
        "nonzero_score_pixels": int(np.count_nonzero(score[footprint])),
        "nonzero_fraction": float(np.count_nonzero(score[footprint]) / max(1, int(footprint.sum()))),
        "max_score": float(score[footprint].max(initial=0.0)),
        "positive_score_p50": float(np.quantile(positive, 0.5)) if positive.size else 0.0,
        "positive_score_p99": float(np.quantile(positive, 0.99)) if positive.size else 0.0,
        "layers": layer_reports,
        "settings": {
            "pixel_size_m": float(pixel_size_m),
            "continuation_heights_m": [float(x) for x in continuation_heights_m],
            "smoothing_sigmas_pixels": [float(x) for x in smoothing_sigmas_pixels],
            "worm_match_radius_pixels": int(worm_match_radius_pixels),
            "topology_match_radius_pixels": int(topology_match_radius_pixels),
        },
    }
    if not positive.size:
        raise RuntimeError(
            "WPH-01 produced no positive scores on the valid footprint; do not write or "
            "submit an all-zero artifact"
        )
    return score, report
