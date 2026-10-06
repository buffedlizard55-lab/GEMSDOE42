"""Multiscale "worming" of potential-field layers.

Method
------
Operational recipe, verified against Horowitz (2018), *Potential Field Poisson Wavelet Multiscale
Edge Analysis*, Stanford Geothermal Workshop paper SGP-TR-212,
https://pangea.stanford.edu/ERE/pdf/IGAstandard/SGW/2018/Horowitz.pdf  (quoting the original):

    "the worm technique from original Hornby et al. (1999) paper takes a Bouguer gravity grid,
     upward continues it to a suite of heights, then detects local maxima in the horizontal
     gradient of the results at each height, marking those locations as multiscale edges.
     Hornby et al. (1999) show that - to within a factor - upward continuation *is* a wavelet
     scale change ... The depths of the dipoles are equal to the negative of the height of
     upward continuation.  The multiscale edges then correspond to local peaks of the probability
     density of the horizontal dipole source distribution induced by the wavelet transform."

Primary references (both verified live 2026-10-05):
  * Hornby, P., Boschetti, F. & Horowitz, F. G. (1999). Analysis of potential field data in the
    wavelet domain.  Geophysical Journal International 137(1), 175-196.
    https://doi.org/10.1046/j.1365-246x.1999.00788.x
  * Archibald, N., Gow, P. & Boschetti, F. (1999). Multiscale edge analysis of potential field
    data.  Exploration Geophysics 30(1-2), 38-44.  https://doi.org/10.1071/EG999038
    (the "worm map" paper; also records that the trace of a gradient maximum *migrates in the
    direction of dip* as the data are continued upward - which is why ``edge_survival`` below
    tolerates a 1-pixel lateral migration between levels rather than demanding a fixed pixel.)

Upward continuation is implemented exactly, in the Fourier domain, with the Poisson kernel
``exp(-h |k|)`` (Blakely, *Potential Theory in Gravity and Magnetic Applications*, eq. 10-1:
the upward-continuation operator's Fourier transform is ``exp(-h k)``).  Reflection padding
suppresses the periodic wrap-around of the DFT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft2, rfft2
from scipy.ndimage import binary_dilation, gaussian_filter, maximum_filter

__all__ = ["upward_continue", "worm_levels", "edge_survival", "WormResult", "DEFAULT_HEIGHTS_M"]

#: Suite of upward-continuation heights, metres.  1 px = 100 m, so 250 m..4000 m = 2.5..40 px.
#: Deep levels retain only the longest-wavelength (deepest-source) edges - exactly the
#: "increasingly upward continued data surfaces" of Hornby et al. (1999).
DEFAULT_HEIGHTS_M: tuple[float, ...] = (0.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)


def _pad(a: np.ndarray, frac: float) -> tuple[np.ndarray, tuple[slice, slice]]:
    H, W = a.shape
    ph, pw = int(round(H * frac)), int(round(W * frac))
    out = np.pad(a, ((ph, ph), (pw, pw)), mode="reflect")
    return out, (slice(ph, ph + H), slice(pw, pw + W))


def upward_continue(field: np.ndarray, height_m: float, pixel_m: float = 100.0,
                    pad_frac: float = 0.25) -> np.ndarray:
    """Continue a potential field upward by ``height_m`` metres using the exact Poisson kernel.

    ``F_h(k) = F_0(k) * exp(-h |k|)`` with ``|k|`` in radians per metre.  ``height_m == 0`` is a
    no-op (returns a copy) because the kernel is then identically 1.
    """
    field = np.asarray(field, dtype=np.float64)
    if height_m <= 0.0:
        return field.copy()
    padded, crop = _pad(field, pad_frac)
    H, W = padded.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(H, d=pixel_m)[:, None]
    kx = 2.0 * np.pi * np.fft.rfftfreq(W, d=pixel_m)[None, :]
    k = np.sqrt(kx * kx + ky * ky)
    spec = rfft2(padded)
    spec *= np.exp(-height_m * k)
    out = irfft2(spec, s=(H, W))
    return np.ascontiguousarray(out[crop])


def _unit(a: np.ndarray, valid: np.ndarray, pct: float = 99.5) -> np.ndarray:
    """Robust [0, 1] normalisation inside ``valid`` (median / IQR z, clipped at the ``pct`` quantile)."""
    out = np.zeros(a.shape, dtype=np.float32)
    v = valid & np.isfinite(a)
    if not v.any():
        return out
    vals = a[v]
    med = float(np.median(vals))
    q25, q75 = (float(x) for x in np.percentile(vals, [25.0, 75.0]))
    scale = max((q75 - q25) / 1.349, 1e-9)
    z = (a - med) / scale
    hi = max(float(np.percentile(z[v], pct)), 1e-9)
    out[v] = np.clip(z[v] / hi, 0.0, 1.0)
    return out


@dataclass
class WormResult:
    """Per-pixel worming output for one potential-field layer."""
    survival: np.ndarray       # float32 in [0, 1]: fraction of continuation levels the edge survives
    steps: np.ndarray          # uint8: raw count of levels survived (0 .. n_levels)
    amplitude: np.ndarray      # float32 in [0, 1]: mean normalised horizontal-gradient modulus
    edge_masks: list[np.ndarray]   # bool per level: the multiscale edge (worm) at that height
    heights_m: tuple[float, ...]
    layer: str


def worm_levels(field: np.ndarray, valid: np.ndarray,
                heights_m: tuple[float, ...] = DEFAULT_HEIGHTS_M,
                pixel_m: float = 100.0,
                edge_percentile: float = 95.0,
                migrate_px: int = 1,
                layer: str = "layer") -> WormResult:
    """Run the Hornby et al. multiscale-edge ("worm") analysis and count cross-level survival.

    For every continuation height:
      1. continue the field upward (exact Poisson kernel);
      2. take the horizontal-gradient modulus ``|grad f_z|`` (the Poisson-wavelet modulus, up to
         the scale factor ``s = z/z0`` of Hornby et al. eq. 2.22a - the factor is a per-level
         constant, so it does not move the maxima and is normalised away below);
      3. mark the multiscale edges as the 3x3 local maxima of that modulus that also clear a
         per-level robust threshold (the ``edge_percentile`` quantile inside the footprint), so
         that every level contributes a comparable number of worms;
      4. dilate the worm mask by ``migrate_px`` to allow for the documented lateral migration of
         a gradient maximum with continuation height (dip).

    ``steps`` is then the number of levels at which the pixel lies on a (migrated) worm.
    """
    field = np.asarray(field, dtype=np.float32)
    valid = np.asarray(valid, bool)
    if not valid.any():
        raise ValueError("empty footprint")

    filled = np.where(valid & np.isfinite(field), field, np.float32(np.nanmedian(field[valid])))
    filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    steps = np.zeros(field.shape, dtype=np.uint8)
    amp_acc = np.zeros(field.shape, dtype=np.float32)
    masks: list[np.ndarray] = []
    struct = np.ones((3, 3), bool)

    for h in heights_m:
        cont = upward_continue(filled, h, pixel_m=pixel_m)
        gx = gaussian_filter(cont, 1.0, order=(0, 1), mode="nearest")
        gy = gaussian_filter(cont, 1.0, order=(1, 0), mode="nearest")
        modulus = np.sqrt(gx * gx + gy * gy)
        amp_acc += _unit(modulus, valid)

        thr = float(np.percentile(modulus[valid], edge_percentile))
        loc_max = modulus >= maximum_filter(modulus, footprint=struct, mode="nearest")
        edge = valid & loc_max & (modulus >= thr)
        if migrate_px > 0:
            edge = binary_dilation(edge, structure=struct, iterations=migrate_px) & valid
        masks.append(edge)
        steps += edge.astype(np.uint8)

    n = len(heights_m)
    survival = (steps.astype(np.float32) / float(n)).astype(np.float32)
    amplitude = (amp_acc / float(n)).astype(np.float32)
    amplitude[~valid] = 0.0
    survival[~valid] = 0.0
    return WormResult(survival=survival, steps=steps, amplitude=amplitude,
                      edge_masks=masks, heights_m=tuple(heights_m), layer=layer)


def edge_survival(results: list[WormResult], combine: str = "max") -> tuple[np.ndarray, np.ndarray]:
    """Combine per-layer worm survival into one cross-physics field.

    ``combine='max'``  : an edge counts if it survives deep in *either* physics (sensitive).
    ``combine='mean'`` : an edge must survive in *both* physics (corroborative).
    """
    if not results:
        raise ValueError("need at least one WormResult")
    if combine == "max":
        surv = np.max(np.stack([r.survival for r in results]), axis=0)
        steps = np.max(np.stack([r.steps for r in results]), axis=0)
    elif combine == "mean":
        surv = np.mean(np.stack([r.survival for r in results]), axis=0).astype(np.float32)
        steps = np.mean(np.stack([r.steps.astype(np.float32) for r in results]), axis=0).astype(np.float32)
    else:
        raise ValueError(f"unknown combine={combine!r}")
    return surv.astype(np.float32), steps
