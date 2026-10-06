"""Distance-weighted Tversky index (DTI) — the official GEMS Prize Challenge metric.

Transcribed from the competition problem description, fetched live on 2026-10-05:
  https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric

Published equations (verbatim symbols):
    k(d)   = max(1 - d / R, 0),                                    R = 300 m = 3 px at 100 m
    TP_w   = sum_{g in G}  max_{x : d(x,g) <= R}  p(x) * k(d(x,g))
    FP_w   = sum_{x : p(x) > 0}  p(x) * [1 - max_{g in G} k(d(x,g))]
    FN_w   = sum_{g in G} [1 - max_{x : d(x,g) <= R} p(x) * k(d(x,g))]   == |G| - TP_w
    DTI    = TP_w / (TP_w + alpha*FP_w + beta*FN_w + eps),   alpha = 0.2, beta = 0.8

Scoring-domain clarification from the organiser (DrivenData staff user `chrisk-dd`, 2026-09-16,
https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516/2):
  "Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from evaluation, so
   they do not count towards penalty terms."  and  "Re-evaluation will also mask/exclude the
   existing USGS/INGENIOUS faults."
That is implemented here as the ``known`` mask: masked pixels are removed from *both* the
prediction support and the truth support before any term is accumulated.

``dti_exact`` is O(|G| * |kernel|) via shifted maxima; ``dti_bruteforce`` is the literal
double loop over every (g, x) pair and exists only so the fast path can be unit-tested against
the published definition.  ``tests/test_metric.py`` asserts the two agree.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

ALPHA: float = 0.2
BETA: float = 0.8
RADIUS_M: float = 300.0
PIXEL_M: float = 100.0
RADIUS_PX: float = RADIUS_M / PIXEL_M   # 3.0
EPS: float = 1e-12


def kernel(d, radius: float = RADIUS_PX) -> np.ndarray:
    """Triangular kernel k(d) = max(1 - d/R, 0); d in pixels (1 px = 100 m)."""
    return np.maximum(1.0 - np.asarray(d, dtype=np.float64) / radius, 0.0)


def _offsets(radius: float = RADIUS_PX) -> list[tuple[int, int, float]]:
    r = int(np.ceil(radius))
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            k = float(kernel(np.hypot(dy, dx), radius))
            if k > 0.0:
                out.append((dy, dx, k))
    return out


_OFFS = _offsets()


def _prepare(pred: np.ndarray, truth: np.ndarray, valid, known):
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth)
    if pred.ndim != 2 or pred.shape != truth.shape:
        raise ValueError("prediction and truth must be equal-shaped 2-D grids")
    valid = np.ones(pred.shape, bool) if valid is None else np.asarray(valid, bool)
    known = np.zeros(pred.shape, bool) if known is None else np.asarray(known, bool)
    if valid.shape != pred.shape or known.shape != pred.shape:
        raise ValueError("mask grids must match the prediction grid")
    active = valid & ~known
    vals = pred[active]
    if not np.isfinite(vals).all():
        raise ValueError("predictions inside the scored domain must be finite")
    if (vals < 0.0).any() or (vals > 1.0).any():
        raise ValueError("predictions inside the scored domain must lie in [0, 1]")
    p = np.where(active, np.nan_to_num(pred, nan=0.0), 0.0)
    g = active & (truth > 0)
    return p, g


def dti_exact(pred, truth, valid=None, known=None,
              alpha: float = ALPHA, beta: float = BETA) -> dict:
    """Exact DTI for soft or binary predictions in [0, 1].  Vectorised, no approximation."""
    p, g = _prepare(pred, truth, valid, known)
    H, W = p.shape
    yy, xx = np.nonzero(g)
    n = int(yy.size)
    if n == 0:
        return dict(tp=0.0, fp=float(p.sum()), fn=0.0, n_truth=0, dti=0.0, coverage=0.0)

    # TP_w = sum over truth pixels of max_x p(x) k(d(x,g)); k has support 3 px, so the max is
    # taken over the 7x7 offset stencil with each offset's own kernel weight.
    credit = np.zeros(n, dtype=np.float64)
    for dy, dx, k in _OFFS:
        ny, nx = yy + dy, xx + dx
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        credit[ok] = np.maximum(credit[ok], p[ny[ok], nx[ok]] * k)
    tp = float(credit.sum())
    fn = float(n) - tp

    # FP_w = sum over predicted pixels of p(x) [1 - max_g k(d(x,g))]; max_g k(d) = k(EDT to truth).
    d_to_truth = distance_transform_edt(~g)
    fp = float((p * (1.0 - kernel(d_to_truth))).sum())

    dti = tp / (tp + alpha * fp + beta * fn + EPS)
    return dict(tp=tp, fp=fp, fn=fn, n_truth=n, dti=float(dti), coverage=tp / n)


def dti_bruteforce(pred, truth, valid=None, known=None, alpha: float = ALPHA,
                   beta: float = BETA, radius: float = RADIUS_PX) -> dict:
    """Literal double-loop transcription of the published equations.  Test-only."""
    p, g = _prepare(pred, truth, valid, known)
    gs = np.argwhere(g)
    xs = np.argwhere(p > 0)
    tp = 0.0
    for gy, gx in gs:
        best = 0.0
        for py, px in xs:
            d = float(np.hypot(py - gy, px - gx))
            if d <= radius:
                best = max(best, float(p[py, px]) * float(max(1.0 - d / radius, 0.0)))
        tp += best
    n = int(gs.shape[0])
    fn = float(n) - tp
    fp = 0.0
    for py, px in xs:
        best = 0.0
        for gy, gx in gs:
            d = float(np.hypot(py - gy, px - gx))
            best = max(best, float(max(1.0 - d / radius, 0.0)))
        fp += float(p[py, px]) * (1.0 - best)
    dti = tp / (tp + alpha * fp + beta * fn + EPS)
    return dict(tp=tp, fp=fp, fn=fn, n_truth=n, dti=float(dti), coverage=tp / max(n, 1))


def dti_binary(pred_bool, truth, valid=None, known=None,
               alpha: float = ALPHA, beta: float = BETA) -> dict:
    """Exact DTI for binary predictions, using two Euclidean distance transforms."""
    pred_bool = np.asarray(pred_bool, bool)
    valid = np.ones(pred_bool.shape, bool) if valid is None else np.asarray(valid, bool)
    known = np.zeros(pred_bool.shape, bool) if known is None else np.asarray(known, bool)
    active = valid & ~known
    p = pred_bool & active
    g = (np.asarray(truth) > 0) & active
    n = int(g.sum())
    if n == 0:
        return dict(tp=0.0, fp=float(p.sum()), fn=0.0, n_truth=0, dti=0.0, coverage=0.0)
    if not p.any():
        return dict(tp=0.0, fp=0.0, fn=float(n), n_truth=n, dti=0.0, coverage=0.0)
    tp = float(kernel(distance_transform_edt(~p)[g]).sum())
    fn = float(n) - tp
    fp = float((1.0 - kernel(distance_transform_edt(~g)[p])).sum())
    dti = tp / (tp + alpha * fp + beta * fn + EPS)
    return dict(tp=tp, fp=fp, fn=fn, n_truth=n, dti=float(dti), coverage=tp / n)
