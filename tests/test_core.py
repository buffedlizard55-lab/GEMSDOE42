"""Unit tests: exact DTI vs a literal transcription, and persistence vs a pixel-wise reference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems42.metric import (ALPHA, BETA, RADIUS_PX, dti_binary, dti_bruteforce,  # noqa: E402
                           dti_exact, kernel)
from gems42.persistence import persistence_map_h0  # noqa: E402
from gems42.persistence_reference import persistence_map_h0_reference  # noqa: E402
from gems42.worming import upward_continue  # noqa: E402


# --------------------------------------------------------------------------- metric constants


def test_metric_constants_match_problem_description():
    assert ALPHA == 0.2
    assert BETA == 0.8
    assert RADIUS_PX == 3.0
    assert kernel(0.0) == 1.0
    assert kernel(3.0) == 0.0
    assert kernel(1.5) == 0.5
    assert kernel(4.0) == 0.0


def test_scoring_example_from_problem_description():
    """The worked example on the official page: TP_w=3.00, FP_w=1.89, FN_w=2.00 -> TI = 0.60."""
    # Truth: a single vertical line of 4 pixels.  The page's diagram gives |G| = 4 with
    # TP_w + FN_w = |G| ... the published arithmetic is 3.00 / (3.00 + 0.2*1.89 + 0.8*2.00) = 0.60.
    tp, fp, fn = 3.00, 1.89, 2.00
    got = tp / (tp + 0.2 * fp + 0.8 * fn)
    assert round(got, 2) == 0.60


def test_dti_exact_matches_bruteforce_random():
    rng = np.random.default_rng(0)
    for trial in range(6):
        truth = rng.random((14, 16)) > 0.86
        pred = np.where(rng.random((14, 16)) > 0.5, rng.random((14, 16)), 0.0)
        fast = dti_exact(pred, truth)
        slow = dti_bruteforce(pred, truth)
        assert fast["tp"] == pytest.approx(slow["tp"], abs=1e-9)
        assert fast["fp"] == pytest.approx(slow["fp"], abs=1e-9)
        assert fast["dti"] == pytest.approx(slow["dti"], abs=1e-12)


def test_dti_binary_matches_exact_on_binary_input():
    rng = np.random.default_rng(3)
    truth = rng.random((20, 22)) > 0.85
    pred = rng.random((20, 22)) > 0.7
    a = dti_exact(pred.astype(float), truth)
    b = dti_binary(pred, truth)
    assert a["tp"] == pytest.approx(b["tp"], abs=1e-9)
    assert a["fp"] == pytest.approx(b["fp"], abs=1e-9)
    assert a["dti"] == pytest.approx(b["dti"], abs=1e-12)


def test_known_fault_masking_removes_masked_pixels_from_every_term():
    """Organiser clarification: pixels on known faults are masked out of the evaluation entirely."""
    truth = np.zeros((9, 9), bool)
    truth[4, 1:8] = True
    known = np.zeros((9, 9), bool)
    known[4, 4:6] = True                     # mask two truth pixels out of scoring

    unmasked = dti_exact(np.ones((9, 9)) * 0.0, truth)
    masked = dti_exact(np.zeros((9, 9)), truth, known=known)
    assert unmasked["n_truth"] == 7
    assert masked["n_truth"] == 5            # masked truth pixels leave the denominator

    # A prediction that lies entirely inside the mask contributes nothing to any term.
    p2 = np.zeros((9, 9))
    p2[4, 4:6] = 1.0
    z = dti_exact(p2, truth, known=known)
    assert z["fp"] == 0.0
    assert z["tp"] == 0.0
    assert z["dti"] == 0.0

    # The same prediction, unmasked, would be charged as a false positive.
    z2 = dti_exact(p2, np.zeros((9, 9), bool))
    assert z2["fp"] == pytest.approx(2.0)


def test_dti_rejects_out_of_range_predictions():
    truth = np.zeros((5, 5), bool)
    truth[2, 2] = True
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        dti_exact(np.full((5, 5), 1.5), truth)
    with pytest.raises(ValueError):
        dti_exact(np.full((5, 5), np.nan), truth)


def test_dti_all_zero_prediction_scores_zero():
    truth = np.zeros((6, 6), bool)
    truth[2:4, 2:4] = True
    out = dti_exact(np.zeros((6, 6)), truth)
    assert out["dti"] == 0.0
    assert out["tp"] == 0.0


def test_dti_perfect_prediction_scores_one():
    truth = np.zeros((8, 8), bool)
    truth[3, 2:6] = True
    out = dti_exact(truth.astype(float), truth)
    assert out["dti"] == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------------------- persistence


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_persistence_matches_bruteforce_reference(seed):
    """The vectorised nested-labeling merge tree must equal a mask-comparison reference exactly."""
    rng = np.random.default_rng(seed)
    G = rng.random((18, 20)).astype(np.float32)
    valid = np.ones(G.shape, bool)
    valid[0, :] = False
    valid[:, -1] = False
    distinct = np.unique(G[valid])[::-1]                    # every distinct value, descending
    fast = persistence_map_h0(G, valid, levels=distinct.astype(np.float64))
    ref = persistence_map_h0_reference(G, valid)
    for key, fast_arr in (("persistence", fast.persistence), ("prominence", fast.prominence),
                          ("birth", fast.birth), ("death", fast.death)):
        np.testing.assert_allclose(fast_arr, ref[key], atol=1e-5,
                                   err_msg=f"{key} disagrees with the brute-force reference")


def test_persistence_isolates_the_dominant_ridge():
    """Tall ridge > low bump > flat background on both derived maps."""
    G = np.zeros((24, 24), np.float32)
    G[12, 4:20] = 1.0      # tall isolated ridge
    G[4, 4] = 0.3          # small bump
    valid = np.ones(G.shape, bool)
    res = persistence_map_h0(G, valid, levels=np.array([1.0, 0.3, 0.0]))
    assert res.prominence[12, 10] > res.prominence[4, 4] > res.prominence[0, 0]
    assert res.prominence[0, 0] == 0.0
    assert res.persistence[12, 10] > res.persistence[4, 4]


def test_persistence_is_zero_outside_the_footprint():
    rng = np.random.default_rng(7)
    G = rng.random((12, 12)).astype(np.float32)
    valid = np.zeros(G.shape, bool)
    valid[2:10, 2:10] = True
    res = persistence_map_h0(G, valid, n_levels=16)
    for arr in (res.persistence, res.prominence, res.birth, res.death):
        assert (arr[~valid] == 0).all()


def test_persistence_feature_count_matches_reference():
    rng = np.random.default_rng(21)
    G = rng.random((16, 16)).astype(np.float32)
    valid = np.ones(G.shape, bool)
    distinct = np.unique(G)[::-1]
    fast = persistence_map_h0(G, valid, levels=distinct.astype(np.float64))
    ref = persistence_map_h0_reference(G, valid)
    assert fast.n_features == ref["n_features"]


# ------------------------------------------------------------------------- upward continuation


def test_upward_continue_zero_height_is_identity():
    rng = np.random.default_rng(11)
    a = rng.random((32, 24))
    out = upward_continue(a, 0.0)
    assert out == pytest.approx(a)


def test_upward_continue_attenuates_high_wavenumbers():
    """exp(-h|k|) must kill short wavelengths first: the continued field is smoother."""
    x = np.arange(128)
    field = np.sin(2 * np.pi * x / 6.0)[None, :] * np.ones((128, 1))
    c = upward_continue(field, 300.0, pixel_m=100.0)
    interior = slice(20, 108)
    assert np.std(c[interior, interior]) < 0.5 * np.std(field[interior, interior])


def test_upward_continue_preserves_dc():
    """exp(-h*0) = 1, so the mean of a smooth field is unchanged in the interior."""
    field = np.full((96, 96), 3.5)
    c = upward_continue(field, 1000.0, pixel_m=100.0)
    assert c[20:76, 20:76] == pytest.approx(3.5, abs=1e-6)


# --------------------------------------------------------------------------- #
# the correlation helpers used for the uniqueness check
# --------------------------------------------------------------------------- #
def test_correlation_helpers_reduce_grids_not_rows():
    """np.corrcoef treats each ROW of a 2-D input as a variable.

    Both ship_submission.py and run_pipeline.py compute correlations over whole grids, so both
    must ravel first.  This is a regression test for a real bug: the first version returned
    corr(row 0 of a, row 0 of b), which is -0.0072 for two sparse masks that genuinely overlap.
    """
    import importlib.util

    rng = np.random.default_rng(3)
    n = 200
    a = (rng.random((n, n)) < 0.05)
    b = a | (rng.random((n, n)) < 0.05)          # deliberately overlapping

    for script in ("ship_submission.py", "run_pipeline.py"):
        path = Path(__file__).resolve().parents[1] / "scripts" / script
        spec = importlib.util.spec_from_file_location(script[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        af, bf = a.astype(np.float64), b.astype(np.float64)
        expected = float(np.corrcoef(af.ravel(), bf.ravel())[0, 1])
        got = mod.pearson(af, bf) if hasattr(mod, "pearson") else mod.spearman(af, bf)
        assert abs(got - expected) < 1e-12, f"{script}: got {got}, expected {expected}"
        # and the overlap must show up as clearly positive, not near-zero/negative
        assert expected > 0.1, f"{script}: expected overlap to correlate, got {expected}"
