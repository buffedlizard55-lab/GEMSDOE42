# Multiscale worming (Hornby–Boschetti–Horowitz 1999)

## Source (verified 2026-10-05)
- Hornby, P., Boschetti, F. & Horowitz, F.G. (1999). "Analysis of potential
  field data in the wavelet domain." *Geophysical Journal International* 137,
  175–196. Theory record via ResearchGate compilation of the paper's
  equations: <https://www.researchgate.net/publication/216837915_Wavelet_Based_Inversion_of_Gravity_Data>
- Boschetti et al. follow-up on edge detection / depth-to-top (PDF, CSIRO):
  <https://www.per.marine.csiro.au/staff/Fabio.Boschetti/papers/fvd_worms.PDF>
- Dyke-dip synthetic demo (worms migrate with dip): Holden et al. 2000,
  reproduced in the 3D-worming expanded abstract:
  <https://sbgf.org.br/mysbgf/eventos/expanded_abstracts/13th_CISBGf/Inversion%20of%20gravity%20gradiometry%20for%20a%20basin%20fault%20network%20in%20an%20oil%20application%20in%20Brazil,%20using%203D%20%E2%80%9Cworming%E2%80%9D.pdf>

## Key results used here
1. **Worms = maxima of horizontal gradients** of a potential field ("Worms are
   representations of the maxima of potential field horizontal gradients. They
   are calculated at different upward continuation levels" — ESPOUY et al.,
   northern Fennoscandia gravity worms).
2. **Upward continuation = wavelet scale change** (Hornby et al. 1999): the
   Green's function of the Poisson equation and its derivatives form a wavelet
   family; continuation to height h multiplies the spectrum by exp(−h|k|).
3. **Lower continuation → shallow sources; higher → deeper** (generally true,
   treat with caution near interfering anomalies).
4. **Depth rule of thumb:** ~half the continuation height (Holden guideline);
   refined by intersecting gravity-field worms with 1st-vertical-derivative
   worms continued downward (Boschetti CSIRO PDF, figs. 6–8).
5. **Magnetic application:** via the pseudo-gravity (RTP-like) transform of
   the observed field — hence our use of the RTP band, the closest available
   to a centered-source field.

## Our reduction to practice (`src/gems42/worming.py`)
- FFT continuation at 0/200/400/800/1600/3200 m, 100 m pixels, 256-px mirror
  pad (Blakely 1996 ch. 11 for the filter form).
- Gradient magnitude by central differences; worms = 3×3 local maxima above
  the per-scale top-15% in-footprint cutoff.
- Survival = count of heights survived per pixel (0–6 per layer).
