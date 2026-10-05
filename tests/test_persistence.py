"""Persistence unit checks on synthetic blobs."""
import numpy as np

from gems42.persistence import scale_space_persistence


def _blobs(n=128):
    y, x = np.mgrid[0:n, 0:n]
    # one broad strong ridge + one narrow weak bump
    strong = 10.0 * np.exp(-((x - 40) ** 2) / (2 * 12.0 ** 2))
    weak = 3.0 * np.exp(-((x - 90) ** 2 + (y - 64) ** 2) / (2 * 1.5 ** 2))
    rng = np.random.default_rng(3)
    return strong + weak + rng.normal(0, 0.05, size=(n, n))


def test_strong_ridge_outlives_weak_bump():
    g = _blobs()
    foot = np.ones(g.shape, bool)
    steps, sigma, tracks, counts = scale_space_persistence(
        g, foot, sigmas_px=(0.0, 1.0, 2.0, 4.0, 8.0), top_frac=0.60)
    assert len(tracks) > 0
    # the best track near x=40 should persist longer than the best near (64,90)
    near_strong = [t for t in tracks if abs(t["birth_pos"][1] - 40) <= 4]
    near_weak = [t for t in tracks
                 if abs(t["birth_pos"][1] - 90) <= 4
                 and abs(t["birth_pos"][0] - 64) <= 6]
    assert near_strong and near_weak
    assert max(t["steps"] for t in near_strong) >= max(
        t["steps"] for t in near_weak)


def test_output_sparse_and_bounded():
    g = _blobs(n=64)
    foot = np.ones(g.shape, bool)
    steps, sigma, tracks, counts = scale_space_persistence(
        g, foot, sigmas_px=(0.0, 2.0, 8.0), top_frac=0.60)
    assert steps.shape == g.shape
    assert steps.max() <= 3
    assert (steps > 0).mean() < 0.25  # sparse
