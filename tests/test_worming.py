"""Worming unit checks on a synthetic dyke (fault-like contact)."""
import numpy as np

from gems42.worming import upward_continue_fft, gradmag, pick_worms


def _dyke(n=128, x0=64, width=6, amp=500.0):
    y, x = np.mgrid[0:n, 0:n]
    # vertical contact: step in magnetisation -> anomaly-like ridge
    f = amp * np.tanh((x - x0) / width)
    rng = np.random.default_rng(7)
    f = f + rng.normal(0, 3.0, size=f.shape)
    return f


def test_upward_continuation_smooths():
    f = _dyke()
    out = upward_continue_fft(f, heights_m=(0.0, 800.0, 3200.0))
    g0 = gradmag(out[0.0]).max()
    g1 = gradmag(out[800.0]).max()
    g2 = gradmag(out[3200.0]).max()
    assert g0 > g1 > g2  # continuation attenuates short wavelengths


def test_worms_run_along_contact():
    f = _dyke()
    g = gradmag(f)
    foot = np.ones(f.shape, bool)
    w = pick_worms(g, foot, top_frac=0.10)
    assert w.any()
    ys, xs = np.nonzero(w)
    # worms should concentrate near the contact x0=64, not spread uniformly
    near = (np.abs(xs - 64) <= 6).mean()
    assert near > 0.5, near


def test_h0_is_identity():
    f = _dyke(n=64)
    out = upward_continue_fft(f, heights_m=(0.0,))
    assert np.array_equal(out[0.0], f)
