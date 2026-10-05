"""Submission artifact verification: re-read every published TIF from disk."""
import json
from pathlib import Path

import numpy as np
import rasterio

from gems42.grid import read_footprint, read_labels, assert_grid_matches
from gems42.submission import audit_geotiff, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "docs" / "downloads"


def _bridge():
    foot = read_footprint(ROOT / "data/bridge/sample_submission.tif")
    labels = read_labels(ROOT / "data/bridge/labels.tif")
    return foot, labels


def test_published_pair_passes_audit():
    foot, labels = _bridge()
    audits = sorted(DL.glob("*-audit.json"))
    assert audits, "no audit sidecar in docs/downloads"
    for sidecar in audits:
        bundle = json.loads(sidecar.read_text())
        for key, mode in (("zeros_tif", "zeros"), ("nan_tif", "nan")):
            tif = DL / bundle[key]["filename"]
            assert tif.exists(), tif
            assert_grid_matches(tif)
            rep = audit_geotiff(tif, foot, labels, mode=mode)
            assert rep["all_checks_passed"]
            assert rep["sha256"] == bundle[key]["sha256"]
            assert rep["emitted_positive_pixels"] > 10_000
            assert rep["on_catalogue_positive_pixels"] == 0


def test_zeros_nan_share_footprint_values():
    foot, _ = _bridge()
    audits = sorted(DL.glob("*-audit.json"))
    bundle = json.loads(audits[0].read_text())
    with rasterio.open(DL / bundle["zeros_tif"]["filename"]) as s:
        z = s.read(1)
    with rasterio.open(DL / bundle["nan_tif"]["filename"]) as s:
        n = s.read(1)
    assert np.array_equal(z[foot], n[foot])
    assert (z[~foot] == 0.0).all()
    assert np.isnan(n[~foot]).all()


def test_values_in_range_everywhere_zeros():
    foot, _ = _bridge()
    audits = sorted(DL.glob("*-audit.json"))
    bundle = json.loads(audits[0].read_text())
    with rasterio.open(DL / bundle["zeros_tif"]["filename"]) as s:
        z = s.read(1)
    assert np.isfinite(z).all()
    assert (z >= 0.0).all() and (z <= 1.0).all()
