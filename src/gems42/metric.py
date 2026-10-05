"""Official Distance-Weighted Tversky Index (DTI) for the DOE GEMS Prize.

Source: competition problem description (DrivenData page 967, performance-metric
section), verified 2026-10-05:
  k(d) = max(1 - d/R, 0), R = 300 m = 3 px at 100 m resolution
  TPw = sum_{g in G} max_{x: d(x,g)<=R} p(x) k(d(x,g))
  FPw = sum_{x: p(x)>0}  p(x) [1 - max_{g in G} k(d(x,g))]
  FNw = sum_{g in G} [1 - max_{x: d(x,g)<=R} p(x) k(d(x,g))] = |G| - TPw
  DTI(a=0.2, b=0.8) = TPw / (TPw + 0.2 FPw + 0.8 FNw + eps)

Published worked example: TPw=3.00, FPw=1.89, FNw=2.00 -> 0.60.
Identity: DTI = T / (0.2 (T + S - M) + 0.8 |G|), so adding unit mass at
weight k is a gain iff k > 0.2 * DTI (the "credit bar").
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

ALPHA = 0.2
BETA = 0.8
RADIUS_PX = 3.0
EPS = 1e-12


def kernel(d, radius: float = RADIUS_PX):
    return np.maximum(1.0 - np.asarray(d, dtype=np.float64) / radius, 0.0)


def _offsets(radius: float = RADIUS_PX):
    r = int(np.ceil(radius))
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            k = float(kernel(np.hypot(dy, dx), radius))
            if k > 0.0:
                out.append((dy, dx, k))
    return out


_OFFS = _offsets()


def dti_from_components(tp: float, fp: float, fn: float,
                        alpha: float = ALPHA, beta: float = BETA) -> float:
    return float(tp / (tp + alpha * fp + beta * fn + EPS))


def dti_exact(pred, truth, valid=None) -> dict:
    """Exact DTI for soft predictions p in [0,1] (offset enumeration for TPw,
    distance transform for FPw). `truth` is boolean ground truth, `valid` the
    scored domain (default: everywhere)."""
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(truth, bool)
    if p.shape != g.shape or p.ndim != 2:
        raise ValueError("pred and truth must be equal-shaped 2-D grids")
    v = np.ones(p.shape, bool) if valid is None else np.asarray(valid, bool)
    pv = np.where(v, p, 0.0)
    inside = pv[v]
    if inside.size and ((not np.isfinite(inside).all()) or (inside < 0).any()
                        or (inside > 1).any()):
        raise ValueError("predictions in the scored domain must be finite in [0, 1]")
    gv = g & v
    H, W = p.shape
    yy, xx = np.nonzero(gv)
    n = int(yy.size)
    if n == 0:
        return dict(tp=0.0, fp=float(pv.sum()), fn=0.0, n_truth=0,
                    dti=0.0, coverage=0.0)
    credit = np.zeros(n, dtype=np.float64)
    for dy, dx, k in _OFFS:
        ny, nx = yy + dy, xx + dx
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        credit[ok] = np.maximum(credit[ok], pv[ny[ok], nx[ok]] * k)
    tp = float(credit.sum())
    fn = float(n) - tp
    dist_to_truth = distance_transform_edt(~gv)
    fp = float((pv * (1.0 - kernel(dist_to_truth))).sum())
    return dict(tp=tp, fp=fp, fn=fn, n_truth=n,
                dti=dti_from_components(tp, fp, fn), coverage=tp / n)


def dti_binary(pred_bool, truth, valid=None) -> dict:
    """Fast exact DTI for binary {0,1} predictions via distance transforms."""
    pb = np.asarray(pred_bool, bool)
    g = np.asarray(truth, bool)
    v = np.ones(pb.shape, bool) if valid is None else np.asarray(valid, bool)
    p = pb & v
    gv = g & v
    n = int(gv.sum())
    if n == 0:
        return dict(tp=0.0, fp=float(p.sum()), fn=0.0, n_truth=0,
                    dti=0.0, coverage=0.0)
    if not p.any():
        return dict(tp=0.0, fp=0.0, fn=float(n), n_truth=n,
                    dti=0.0, coverage=0.0)
    dp = distance_transform_edt(~p)
    tp = float(kernel(dp[gv]).sum())
    fn = float(n) - tp
    dg = distance_transform_edt(~gv)
    fp = float((1.0 - kernel(dg[p])).sum())
    return dict(tp=tp, fp=fp, fn=fn, n_truth=n,
                dti=dti_from_components(tp, fp, fn), coverage=tp / n)


def dti_bruteforce(pred, truth, radius: float = RADIUS_PX) -> dict:
    """Literal O(|G|*|P|) transcription of the published equations (small grids
    only; used to verify the fast implementations in tests)."""
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(truth, bool)
    H, W = p.shape
    gy, gx = np.nonzero(g)
    n = int(gy.size)
    if n == 0:
        return dict(tp=0.0, fp=float(p[p > 0].sum()), fn=0.0,
                    n_truth=0, dti=0.0)
    tp = 0.0
    for j in range(n):
        best = 0.0
        for y in range(H):
            for x in range(W):
                if p[y, x] <= 0:
                    continue
                d = np.hypot(y - gy[j], x - gx[j])  # px; R=3px
                if d <= radius:
                    best = max(best, p[y, x] * (1.0 - d / radius))
        tp += best
    fn = float(n) - tp
    fp = 0.0
    for y in range(H):
        for x in range(W):
            if p[y, x] <= 0:
                continue
            bestk = 0.0
            for j in range(n):
                d = np.hypot(y - gy[j], x - gx[j])
                if d <= radius:
                    bestk = max(bestk, 1.0 - d / radius)
            fp += p[y, x] * (1.0 - bestk)
    return dict(tp=tp, fp=fp, fn=fn, n_truth=n,
                dti=dti_from_components(tp, fp, fn))
