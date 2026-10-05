"""Competition grid geometry and raster IO for GEMSDOE42.

Grid facts (measured from the competition rasters themselves, verified in this
repo by scripts working on data/bridge/*):
  - CRS EPSG:32611 (UTM 11N), 100 m pixels
  - height 3730 x width 3292 (12,279,160 cells)
  - Affine transform: (100, 0, 243350, 0, -100, 4508550)
  - feature nodata sentinel: float32 -3.4028234663852886e-38 -> use -3.4028235e+38
  - scoring footprint: finite pixels of sample_submission.tif = 5,167,373 px
  - catalogue positives: labels.tif >= 1 = 60,988 px

Band map of training_features.tif (19 bands, from raster descriptions):
  1 mag_anom | 2 rtp | 3 tmi_hg | 4 geod_2ndinv | 5 iso_grav_anom_slope |
  6 tc | 7 geod_shearrate | 8 geod_dilaterate | 9 tmi_vg | 10 deq_n100a15 |
  11 iso_grav_anom_vg | 12 det_elev | 13 iso_grav_anom | 14 tmi |
  15 depth_to_base_surf | 16 ieq_n100a15 | 17 cond_surf |
  18 iso_grav_anom_hg | 19 det_elev_slope
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine

HEIGHT = 3730
WIDTH = 3292
TOTAL_CELLS = HEIGHT * WIDTH  # 12,279,160
CRS = "EPSG:32611"
PIXEL_M = 100.0
TRANSFORM = Affine(PIXEL_M, 0.0, 243350.0, 0.0, -PIXEL_M, 4508550.0)
NODATA_F32 = np.float32(-3.4028234663852886e38)
FOOTPRINT_PX = 5_167_373  # finite pixels of sample_submission.tif
CATALOGUE_PX = 60_988  # labels.tif >= 1

BANDS: Dict[int, str] = {
    1: "mag_anom", 2: "rtp", 3: "tmi_hg", 4: "geod_2ndinv",
    5: "iso_grav_anom_slope", 6: "tc", 7: "geod_shearrate",
    8: "geod_dilaterate", 9: "tmi_vg", 10: "deq_n100a15",
    11: "iso_grav_anom_vg", 12: "det_elev", 13: "iso_grav_anom",
    14: "tmi", 15: "depth_to_base_surf", 16: "ieq_n100a15",
    17: "cond_surf", 18: "iso_grav_anom_hg", 19: "det_elev_slope",
}
BAND_RTP = 2
BAND_ISO_GRAV = 13
BAND_TMI = 14


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_bridge() -> Path:
    return repo_root() / "data" / "bridge"


def is_valid_feature_value(a: np.ndarray) -> np.ndarray:
    """True where a feature pixel is usable (finite and not the float32 sentinel)."""
    a = np.asarray(a)
    return np.isfinite(a) & (a > NODATA_F32 * np.float32(0.5))


def read_footprint(sample_submission_path: str | Path) -> np.ndarray:
    """Authoritative scoring footprint: finite pixels of the sample submission."""
    with rasterio.open(sample_submission_path) as s:
        a = s.read(1)
    foot = np.isfinite(a)
    assert foot.shape == (HEIGHT, WIDTH), foot.shape
    return foot


def read_labels(labels_path: str | Path) -> np.ndarray:
    """Catalogue mask: labels.tif >= 1 (int8 raster with -1 outside footprint)."""
    with rasterio.open(labels_path) as s:
        a = s.read(1)
    assert a.shape == (HEIGHT, WIDTH), a.shape
    return a >= 1


def read_band_filled(features_path: str | Path, band: int,
                     footprint: np.ndarray | None = None) -> np.ndarray:
    """Read one feature band as float64 with sentinel/non-finite filled by the
    footprint median (required before any FFT-based operator)."""
    with rasterio.open(features_path) as s:
        a = s.read(band).astype(np.float64)
    assert a.shape == (HEIGHT, WIDTH), (a.shape, band)
    ok = is_valid_feature_value(a)
    if footprint is not None:
        med = float(np.median(a[ok & footprint]))
    else:
        med = float(np.median(a[ok]))
    out = np.where(ok, a, med)
    return out


def template_profile(sample_submission_path: str | Path) -> dict:
    with rasterio.open(sample_submission_path) as s:
        prof = s.profile.copy()
    return prof


def assert_grid_matches(path: str | Path) -> dict:
    """Verify a raster matches the competition grid; return its meta summary."""
    with rasterio.open(path) as s:
        meta = dict(crs=str(s.crs), width=s.width, height=s.height,
                    transform=tuple(s.transform)[:6], dtype=s.dtypes[0],
                    count=s.count, nodata=s.nodata)
    assert meta["height"] == HEIGHT and meta["width"] == WIDTH, meta
    assert "32611" in meta["crs"], meta
    t = meta["transform"]
    assert abs(t[0] - 100.0) < 1e-9 and abs(t[4] + 100.0) < 1e-9, meta
    assert abs(t[2] - 243350.0) < 1e-9 and abs(t[5] - 4508550.0) < 1e-9, meta
    return meta
