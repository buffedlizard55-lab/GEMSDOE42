"""Metric verification: published worked example + fast-vs-bruteforce agreement."""
import numpy as np

from gems42.metric import (dti_from_components, dti_exact, dti_binary,
                           dti_bruteforce)


def test_worked_example():
    # Published: TPw=3.00, FPw=1.89, FNw=2.00 -> 0.60 (page rounds to 2 dp).
    # Exact: 3.00 / (3.00 + 0.2*1.89 + 0.8*2.00) = 3/4.978 = 0.60265...
    v = dti_from_components(3.00, 1.89, 2.00)
    assert abs(v - 3.0 / 4.978) < 1e-12
    assert round(v, 2) == 0.60


def test_fast_matches_bruteforce_soft():
    rng = np.random.default_rng(0)
    p = np.round(rng.random((14, 14)), 2)
    p[rng.random((14, 14)) < 0.7] = 0.0
    g = np.zeros((14, 14), bool)
    g[3, 2:9] = True
    g[9, 5] = True
    a = dti_exact(p, g)
    b = dti_bruteforce(p, g)
    assert abs(a["tp"] - b["tp"]) < 1e-9, (a, b)
    assert abs(a["fp"] - b["fp"]) < 1e-9, (a, b)
    assert abs(a["fn"] - b["fn"]) < 1e-9, (a, b)
    assert abs(a["dti"] - b["dti"]) < 1e-12


def test_fast_matches_bruteforce_binary():
    rng = np.random.default_rng(1)
    pb = rng.random((16, 12)) < 0.15
    g = np.zeros((16, 12), bool)
    g[4:12, 6] = True
    a = dti_binary(pb, g)
    b = dti_bruteforce(pb.astype(float), g)
    assert abs(a["tp"] - b["tp"]) < 1e-9
    assert abs(a["fp"] - b["fp"]) < 1e-9
    assert abs(a["dti"] - b["dti"]) < 1e-12


def test_identity_fn_is_complement():
    rng = np.random.default_rng(2)
    p = (rng.random((12, 12)) < 0.2).astype(float)
    g = np.zeros((12, 12), bool)
    g[5, 3:10] = True
    r = dti_exact(p, g)
    assert abs(r["fn"] - (r["n_truth"] - r["tp"])) < 1e-12
