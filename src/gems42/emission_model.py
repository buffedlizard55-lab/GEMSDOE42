"""Analytic emission-size model for the distance-weighted Tversky index.

The holdout in :mod:`gems42.holdout` measures against *catalogue* faults, and
``scripts/compare_candidates.py`` shows it is uncorrelated with the real leaderboard
(Spearman +0.087, p = 0.80, n = 11).  It is therefore not a usable selection instrument, and
this module is the replacement for the part of the decision it was being asked to make.

Model
-----
With the official metric, and writing ``h`` for the expected kernel credit a dot earns from the
hidden truth and ``f`` for its expected false-positive charge,

    DTI(N) = N h / ( 0.2 * (N h + N f) + 0.8 * |G| )

Differentiating in ``N``:

    d(DTI)/dN > 0   <=>   0.8 * |G| * (h - 0.2 f) > 0   <=>   h > 0.2 f

so, because ``f <= 1`` always, **any dot with expected credit above 0.2 should be emitted** and
DTI is maximised by taking the whole set of them.  Precision matters far less than the intuition
from a plain F1 score suggests: the 0.2 weight on false positives is a fourfold discount.

``best_size_from_precision`` inverts that rule given a precision curve measured from the scored
priors, and ``predicted_dti`` reports the DTI a given size implies at a stated hit rate.

Everything here is a model.  Its inputs are recorded, its arithmetic is exact, and its
assumptions are stated where they are used.
"""

from __future__ import annotations

import numpy as np

ALPHA = 0.2
BETA = 0.8


def dti_from_parts(tp: float, fp: float, n_truth: float) -> float:
    """DTI given weighted TP, weighted FP and the number of truth pixels."""
    return tp / (tp + ALPHA * fp + BETA * max(n_truth - tp, 0.0) + 1e-12)


def predicted_dti(n_emit: float, mean_credit: float, n_truth: float,
                  mean_fp_charge: float = 1.0) -> float:
    """DTI implied by emitting ``n_emit`` dots that each earn ``mean_credit`` on average.

    ``mean_credit`` is ``E[k(d)]`` per emitted dot, averaged over hits and misses (a miss earns 0).
    ``mean_fp_charge`` is ``E[1 - max_g k(d)]`` per dot; it is < 1 for dots that land near truth
    and ~1 for dots that do not.
    """
    tp = n_emit * mean_credit
    fp = n_emit * mean_fp_charge
    return dti_from_parts(tp, fp, n_truth)


def marginal_dot_worth_it(mean_credit: float, mean_fp_charge: float = 1.0) -> bool:
    """Is one more dot at this expected credit a net gain?  True iff ``h > alpha * f``."""
    return mean_credit > ALPHA * mean_fp_charge


def best_size_from_precision(hit_rate: np.ndarray, budgets: np.ndarray,
                             n_truth: float, credit_if_hit: float = 0.5,
                             fp_charge_if_hit: float = 0.35) -> dict:
    """Pick the emission size that maximises the modelled DTI.

    ``hit_rate[i]`` is the fraction of the top ``budgets[i]`` candidate pixels that fall within
    the 300 m kernel of hidden truth.  A hit earns ``credit_if_hit`` (the mean of the triangular
    kernel over a hit's distance distribution) and is charged ``fp_charge_if_hit``; a miss earns
    nothing and is charged 1.
    """
    hit_rate = np.asarray(hit_rate, dtype=np.float64)
    budgets = np.asarray(budgets, dtype=np.float64)
    if hit_rate.shape != budgets.shape:
        raise ValueError("hit_rate and budgets must have the same shape")
    tp = budgets * hit_rate * credit_if_hit
    fp = budgets * (hit_rate * fp_charge_if_hit + (1.0 - hit_rate) * 1.0)
    dti = np.array([dti_from_parts(t, f, n_truth) for t, f in zip(tp, fp)])
    best = int(np.argmax(dti))
    return {
        "best_budget": float(budgets[best]),
        "best_dti": float(dti[best]),
        "hit_rate_at_best": float(hit_rate[best]),
        "curve": [{"budget": float(b), "hit_rate": float(h), "modelled_dti": float(d)}
                  for b, h, d in zip(budgets, hit_rate, dti)],
    }


def breakeven_hit_rate(credit_if_hit: float = 0.5,
                       fp_charge_if_hit: float = 0.35) -> float:
    """Hit rate at which one more dot stops being worth it (``h = alpha * f``)."""
    from scipy.optimize import brentq

    def g(q: float) -> float:
        h = q * credit_if_hit
        f = q * fp_charge_if_hit + (1.0 - q) * 1.0
        return h - ALPHA * f

    return float(brentq(g, 0.0, 1.0))
