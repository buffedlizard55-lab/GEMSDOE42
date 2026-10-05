"""Multiscale worming after Hornby, Boschetti & Horowitz (1999).

"Analysis of potential field data in the wavelet domain",
Geophysical Journal International 137, 175-196.

Worms = loci of maxima of the horizontal gradient of a potential field,
tracked across upward-continuation heights. Upward continuation to height h
is a low-pass filter exp(-h*|k|) in the Fourier domain (Blakely, "Potential
Theory in Gravity and Magnetic Applications", Cambridge Univ. Press, 1996,
ch. 11); Hornby et al. show it is exactly the wavelet scale-change
operation, so edges that persist across heights mark real source
discontinuities ("rocks have edges") rather than single-scale noise.

Pipeline per layer:
  1. upward_continue_fft: field -> continued fields at heights_m
  2. horizontal gradient magnitude of each continued field (central differences)
  3. pick_worms: local maxima (3x3) above the per-scale top-fraction threshold
  4. worm_survival: per-pixel count of heights where the pixel is a worm
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

# Continuation heights in metres. h=0 is the observed field; doubling above
# 200 m follows the dyadic scale ladder of Hornby et al. (1999, figs. 1-3).
# At 100 m pixels these are 0, 2, 4, 8, 16, 32 px.
HEIGHTS_M = (0.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)
PIXEL_M = 100.0
PAD = 256  # mirror pad to suppress FFT wraparound (power-of-two friendly)


def _wavenumbers(nr: int, nc: int, pixel_m: float = PIXEL_M):
    """Radial wavenumber |k| in rad/m for an rfft2 of shape (nr, nc)."""
    fy = np.fft.fftfreq(nr, d=pixel_m) * 2.0 * np.pi
    fx = np.fft.rfftfreq(nc, d=pixel_m) * 2.0 * np.pi
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    return np.sqrt(kx ** 2 + ky ** 2)


def upward_continue_fft(field: np.ndarray, heights_m=HEIGHTS_M,
                        pixel_m: float = PIXEL_M, pad: int = PAD):
    """Upward-continue `field` (2-D, pre-filled, no NaN) to each height.

    Returns dict height -> continued field cropped to the input shape.
    The h=0 entry is the input itself (no FFT round-trip error).
    """
    f = np.asarray(field, dtype=np.float64)
    if not np.isfinite(f).all():
        raise ValueError("field must be finite everywhere (fill sentinel first)")
    out = {float(heights_m[0]): f} if float(heights_m[0]) == 0.0 else {}
    todo = [h for h in heights_m if float(h) != 0.0]
    if not todo:
        return {float(h): f.copy() for h in heights_m}
    fp = np.pad(f, pad, mode="reflect")
    nr, nc = fp.shape
    F = np.fft.rfft2(fp)
    k = _wavenumbers(nr, nc, pixel_m)
    for h in todo:
        Fup = F * np.exp(-float(h) * k)
        up = np.fft.irfft2(Fup, s=(nr, nc))
        out[float(h)] = up[pad:pad + f.shape[0], pad:pad + f.shape[1]]
    if float(heights_m[0]) == 0.0:
        out[float(0.0)] = f.copy()
    return out


def gradmag(field: np.ndarray, pixel_m: float = PIXEL_M) -> np.ndarray:
    """Horizontal gradient magnitude |dF/dx, dF/dy| by central differences."""
    gy, gx = np.gradient(np.asarray(field, dtype=np.float64), pixel_m)
    return np.sqrt(gx ** 2 + gy ** 2)


def pick_worms(g: np.ndarray, footprint: np.ndarray,
               top_frac: float = 0.15) -> np.ndarray:
    """Binary worm mask: 3x3 local maxima of `g` within the footprint whose
    value clears the per-scale top-fraction threshold.

    `top_frac=0.15` keeps pixels above the 85th percentile of the in-footprint
    gradient magnitude at that continuation height.
    """
    g = np.asarray(g, dtype=np.float64)
    foot = np.asarray(footprint, bool)
    if g.shape != foot.shape:
        raise ValueError("shape mismatch")
    mx = maximum_filter(g, size=3, mode="reflect")
    is_max = (g >= mx) & foot
    thr = float(np.percentile(g[foot], 100.0 * (1.0 - top_frac)))
    return is_max & (g >= thr)


def worm_survival(field: np.ndarray, footprint: np.ndarray,
                  heights_m=HEIGHTS_M, top_frac: float = 0.15,
                  pixel_m: float = PIXEL_M):
    """Run the full worming ladder on one layer.

    Returns (survival, masks, gradmag0):
      survival: uint8 per-pixel count of heights survived (0..len(heights))
      masks:    dict height -> bool worm mask
      gradmag0: float64 h=0 gradient magnitude (only this scale is retained;
                keeping all scales would hold ~6 full grids in RAM)
    """
    foot = np.asarray(footprint, bool)
    cont = upward_continue_fft(field, heights_m, pixel_m)
    masks = {}
    gradmag0 = None
    surv = np.zeros(np.asarray(field).shape, dtype=np.uint8)
    for h in heights_m:
        gh = gradmag(cont[float(h)], pixel_m)
        if float(h) == 0.0:
            gradmag0 = gh
        m = pick_worms(gh, foot, top_frac)
        masks[float(h)] = m
        surv += m.astype(np.uint8)
        del gh
    return surv, masks, gradmag0
