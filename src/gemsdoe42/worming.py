"""FFT upward continuation and cross-level gradient-edge persistence."""

from __future__ import annotations

from collections.abc import Iterator, Sequence


def upward_continuation_levels(
    field,
    valid,
    heights_m: Sequence[float],
    *,
    pixel_size_m: float = 100.0,
) -> Iterator[tuple[float, object]]:
    """Yield upward-continued scalar potential fields at requested heights.

    Uses the Fourier-domain Poisson continuation factor ``exp(-|k| h)`` with
    ``|k|=2π sqrt(fx²+fy²)``. Nodata cells are nearest-neighbor-filled only to make
    the FFT well-defined; downstream edge scores are masked back to ``valid``. A
    reflected border is padded to reduce the circular-wrap artifact of a finite FFT.
    This does not perform magnetic-to-pseudogravity conversion.
    """
    import math

    import numpy as np
    from scipy import fft
    from scipy.ndimage import distance_transform_edt

    values = np.asarray(field, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("field and valid mask must have the same 2D shape")
    if not mask.any():
        raise ValueError("potential field has no valid pixels")
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")

    heights = tuple(float(height) for height in heights_m)
    if not heights or any(height < 0 for height in heights):
        raise ValueError("heights_m must contain non-negative heights")
    if tuple(sorted(heights)) != heights:
        raise ValueError("heights_m must be in ascending order")

    # Distance-transform indices are the nearest valid cell for every invalid cell.
    # This intentionally affects only the FFT calculation; the returned field is
    # interpreted only at original valid cells.
    nearest_indices = distance_transform_edt(
        ~mask, return_distances=False, return_indices=True
    )
    filled = values[tuple(nearest_indices)]
    maximum_height = max(heights)
    padding = max(8, math.ceil(3.0 * maximum_height / pixel_size_m) + 4)
    padded = np.pad(filled, ((padding, padding), (padding, padding)), mode="reflect")

    spectrum = fft.rfft2(padded, workers=1)
    fy = fft.fftfreq(padded.shape[0], d=pixel_size_m).astype(np.float32)
    fx = fft.rfftfreq(padded.shape[1], d=pixel_size_m).astype(np.float32)
    radial_wavenumber = (2.0 * np.pi) * np.sqrt(
        fy[:, None] * fy[:, None] + fx[None, :] * fx[None, :]
    )
    crop = (
        slice(padding, padding + values.shape[0]),
        slice(padding, padding + values.shape[1]),
    )

    for height in heights:
        transfer = np.exp(-radial_wavenumber * height).astype(np.float32)
        continued_padded = fft.irfft2(
            spectrum * transfer, s=padded.shape, workers=1
        ).astype(np.float32, copy=False)
        continued = continued_padded[crop].copy()
        yield height, continued
        del continued, continued_padded, transfer


def horizontal_gradient_magnitude(field, pixel_size_m: float = 100.0):
    """Compute the Euclidean horizontal gradient magnitude in field-units/metre."""
    import numpy as np

    values = np.asarray(field, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("field must be a 2D raster with at least two rows and columns")
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    gy, gx = np.gradient(values, pixel_size_m, pixel_size_m)
    return np.hypot(gx, gy).astype(np.float32, copy=False)


def worm_survival_score(
    field,
    valid,
    heights_m: Sequence[float],
    *,
    pixel_size_m: float = 100.0,
    match_radius_pixels: int = 2,
):
    """Return a [0,1] edge map weighted by consecutive continuation-level survival.

    At each height, local maxima of the horizontal-gradient magnitude are picked by
    a 3×3 maximum filter. A peak is continued when a peak from the previous height
    occurs within the configured Chebyshev neighborhood. The maximum consecutive
    run at each detected edge pixel is divided by the number of continuation levels.
    This is an auditable pixel-path proxy for a worm trajectory, not a line-vector
    worm-linking implementation.
    """
    import numpy as np
    from scipy.ndimage import maximum_filter

    mask = np.asarray(valid, dtype=bool)
    heights = tuple(float(height) for height in heights_m)
    if not heights:
        raise ValueError("heights_m cannot be empty")
    if match_radius_pixels < 0:
        raise ValueError("match_radius_pixels must be non-negative")

    level_size = 2 * match_radius_pixels + 1
    previous_run = np.zeros(mask.shape, dtype=np.uint8)
    best_run = np.zeros(mask.shape, dtype=np.uint8)
    edge_count = 0

    for _, continued in upward_continuation_levels(
        field, mask, heights, pixel_size_m=pixel_size_m
    ):
        gradient = horizontal_gradient_magnitude(continued, pixel_size_m)
        local_max = maximum_filter(gradient, size=3, mode="nearest")
        edge = (gradient > 0.0) & (gradient >= local_max) & mask
        edge_count += int(edge.sum())

        if previous_run.any():
            nearby_run = maximum_filter(previous_run, size=level_size, mode="nearest")
            current_run = np.where(
                edge,
                np.minimum(nearby_run.astype(np.uint16) + 1, len(heights)),
                0,
            ).astype(np.uint8)
        else:
            current_run = edge.astype(np.uint8)

        best_run = np.maximum(best_run, current_run)
        previous_run = current_run

    score = best_run.astype(np.float32) / float(len(heights))
    score[~mask] = 0.0
    return score, {"n_levels": len(heights), "edge_observations": edge_count}
