"""Production 0D persistence and smoothing-scale peak tracking.

The topology term is computed on gradient-magnitude surfaces, not directly on the
labels. A 4-neighbor superlevel filtration yields H0 peak lifetimes (birth minus
merge/death level). Peak locations are then matched locally between Gaussian scales.
This remains a geological proxy and needs spatially blocked validation.
"""

from __future__ import annotations

from collections.abc import Sequence

try:  # The reference fallback keeps small tests usable without Numba.
    import numba as _numba
except ImportError:  # pragma: no cover - depends on the user's environment
    _numba = None

if _numba is not None:  # pragma: no cover - exercised in the full research environment
    import numpy as _np

    @_numba.njit(cache=True)
    def _find_root(parent, index):
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != index:
            following = parent[index]
            parent[index] = root
            index = following
        return root

    @_numba.njit(cache=True)
    def _push_unique(roots, count, root):
        for position in range(count):
            if roots[position] == root:
                return count
        roots[count] = root
        return count + 1

    @_numba.njit(cache=True)
    def _h0_numba(values, valid, height, width):
        n = height * width
        order = _np.argsort(-values, kind="mergesort")
        parent = _np.full(n, -1, dtype=_np.int32)
        birth = _np.zeros(n, dtype=_np.float32)
        birth_pixel = _np.full(n, -1, dtype=_np.int32)
        persistence = _np.zeros(n, dtype=_np.float32)
        roots = _np.empty(5, dtype=_np.int32)

        for sequence_index in range(n):
            pixel = order[sequence_index]
            if not valid[pixel]:
                continue
            level = values[pixel]
            parent[pixel] = pixel
            birth[pixel] = level
            birth_pixel[pixel] = pixel
            roots[0] = pixel
            root_count = 1
            y = pixel // width
            x = pixel - y * width

            if x > 0 and parent[pixel - 1] >= 0:
                root = _find_root(parent, pixel - 1)
                root_count = _push_unique(roots, root_count, root)
            if x + 1 < width and parent[pixel + 1] >= 0:
                root = _find_root(parent, pixel + 1)
                root_count = _push_unique(roots, root_count, root)
            if y > 0 and parent[pixel - width] >= 0:
                root = _find_root(parent, pixel - width)
                root_count = _push_unique(roots, root_count, root)
            if y + 1 < height and parent[pixel + width] >= 0:
                root = _find_root(parent, pixel + width)
                root_count = _push_unique(roots, root_count, root)

            winner = roots[0]
            for position in range(1, root_count):
                candidate = roots[position]
                if birth[candidate] > birth[winner] or (
                    birth[candidate] == birth[winner]
                    and birth_pixel[candidate] < birth_pixel[winner]
                ):
                    winner = candidate

            for position in range(root_count):
                root = roots[position]
                if root == winner:
                    continue
                lifetime = birth[root] - level
                if lifetime > 0.0:
                    peak = birth_pixel[root]
                    persistence[peak] = max(persistence[peak], lifetime)
                parent[root] = winner
            parent[pixel] = winner

        return persistence.reshape((height, width))


def h0_persistence_map(
    surface,
    valid,
    *,
    max_python_cells: int = 100_000,
):
    """Compute H0 superlevel lifetimes at the pixels where finite peaks were born.

    The compiled path is intended for the full 100 m competition raster. Without
    Numba, only small arrays are allowed through the transparent Python reference;
    large arrays fail rather than silently run for an impractical time.
    """
    import numpy as np

    values = np.asarray(surface, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("surface and valid mask must be same-shaped 2D arrays")
    mask = mask & np.isfinite(values)
    if not mask.any():
        raise ValueError("persistence surface has no valid pixels")

    if _numba is not None:
        return _h0_numba(
            np.ascontiguousarray(values).ravel(),
            np.ascontiguousarray(mask).ravel(),
            values.shape[0],
            values.shape[1],
        )

    if values.size > max_python_cells:
        raise RuntimeError(
            "Numba is required for production-sized persistence rasters; install the "
            "declared project dependencies instead of using the slow Python fallback"
        )

    from .topology_reference import h0_superlevel_persistence

    nested = h0_superlevel_persistence(values.tolist(), mask.tolist())
    return np.asarray(nested, dtype=np.float32)


def multiscale_topology_score(
    gradient_magnitude,
    valid,
    sigmas_pixels: Sequence[float],
    *,
    match_radius_pixels: int = 2,
):
    """Return a pixel map of peak lifetime × consecutive smoothing-scale survival.

    At each sigma, a normalized masked Gaussian filter is applied to the gradient-
    magnitude field. H0 persistence lifetimes are normalized by that surface's robust
    dynamic range. A persistence maximum is linked to a maximum at the next sigma if
    one lies within ``match_radius_pixels``. The longest consecutive track fraction
    multiplies the normalized lifetime. The returned map is bounded by [0,1].
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter, maximum_filter

    sigmas = tuple(float(sigma) for sigma in sigmas_pixels)
    if not sigmas or any(sigma <= 0 for sigma in sigmas):
        raise ValueError("sigmas_pixels must contain positive values")
    if tuple(sorted(sigmas)) != sigmas:
        raise ValueError("sigmas_pixels must be in ascending order")
    if match_radius_pixels < 0:
        raise ValueError("match_radius_pixels must be non-negative")

    gradient = np.asarray(gradient_magnitude, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(gradient)
    if gradient.ndim != 2 or mask.shape != gradient.shape:
        raise ValueError("gradient and valid mask must have the same 2D shape")
    if not mask.any():
        raise ValueError("gradient surface has no valid pixels")

    source = np.where(mask, gradient, 0.0).astype(np.float32, copy=False)
    weights = mask.astype(np.float32)
    previous_track = np.zeros(gradient.shape, dtype=np.uint8)
    best_score = np.zeros(gradient.shape, dtype=np.float32)
    footprint_size = 2 * match_radius_pixels + 1
    n_scales = len(sigmas)

    for sigma in sigmas:
        numerator = gaussian_filter(source, sigma=sigma, mode="nearest")
        denominator = gaussian_filter(weights, sigma=sigma, mode="nearest")
        smooth = np.zeros(gradient.shape, dtype=np.float32)
        np.divide(numerator, denominator, out=smooth, where=denominator > 1e-6)
        smooth[~mask] = 0.0

        lifetime_map = h0_persistence_map(smooth, mask)
        values = smooth[mask]
        low, high = np.quantile(values, [0.005, 0.995])
        span = float(high - low)
        if not np.isfinite(span) or span <= 0:
            normalized_lifetime = np.zeros(gradient.shape, dtype=np.float32)
        else:
            normalized_lifetime = np.clip(lifetime_map / span, 0.0, 1.0)

        peaks = (lifetime_map > 0.0) & mask
        if previous_track.any():
            nearby_previous = maximum_filter(
                previous_track, size=footprint_size, mode="nearest"
            )
            track = np.where(
                peaks,
                np.minimum(nearby_previous.astype(np.uint16) + 1, n_scales),
                0,
            ).astype(np.uint8)
        else:
            track = peaks.astype(np.uint8)

        score = normalized_lifetime * (track.astype(np.float32) / n_scales)
        best_score = np.maximum(best_score, score)
        previous_track = track

    best_score[~mask] = 0.0
    return np.clip(best_score, 0.0, 1.0).astype(np.float32, copy=False)
