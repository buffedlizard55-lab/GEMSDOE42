"""Reading the competition rasters with correct nodata handling.

Irregularity found and handled here: ``training_features.tif`` declares
``nodata = -3.4028234663852886e+38``, which is a *finite* float32, so ``np.isfinite`` does NOT
exclude it.  Band statistics in ``data/prepared_manifest.json`` record several thousand such
sentinel pixels *inside* the survey footprint (e.g. 3,061 for ``mag_anom``).  Every loader in this
module therefore masks on ``|a| < 1e30`` as well as on finiteness; leaving them in corrupts any
filter, gradient or percentile computed downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio

SENTINEL_LIMIT = 1e30     # anything with |value| above this is the -3.4e38 nodata sentinel


@dataclass
class Grid:
    """The competition grid, taken from ``sample_submission.tif``."""
    height: int
    width: int
    transform: "rasterio.Affine"
    crs: "rasterio.crs.CRS"
    footprint: np.ndarray          # bool: finite in the sample submission = inside the survey area
    dtype: np.dtype

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


def load_grid(path) -> Grid:
    with rasterio.open(path) as ds:
        a = ds.read(1)
        return Grid(height=ds.height, width=ds.width, transform=ds.transform, crs=ds.crs,
                    footprint=np.isfinite(a), dtype=a.dtype)


def band_names(path) -> list[str]:
    with rasterio.open(path) as ds:
        return [str(d).split(" - ")[0].strip() for d in ds.descriptions]


def read_band(path, name: str, grid: Grid) -> np.ndarray:
    """Read one band by short name; nodata/sentinel pixels become NaN."""
    names = band_names(path)
    if name not in names:
        raise KeyError(f"band {name!r} not in {names}")
    with rasterio.open(path) as ds:
        a = ds.read(names.index(name) + 1).astype(np.float32)
    bad = ~np.isfinite(a) | (np.abs(a) >= SENTINEL_LIMIT)
    a[bad] = np.float32("nan")
    if a.shape != grid.shape:
        raise ValueError(f"band {name} shape {a.shape} != grid {grid.shape}")
    return a


def read_binary(path, grid: Grid) -> np.ndarray:
    with rasterio.open(path) as ds:
        a = ds.read(1)
    b = np.isfinite(a) & (a > 0)
    if b.shape != grid.shape:
        raise ValueError(f"{path} shape {b.shape} != grid {grid.shape}")
    return b


def valid_mask(a: np.ndarray, grid: Grid) -> np.ndarray:
    """Footprint AND finite AND not the nodata sentinel."""
    return grid.footprint & np.isfinite(a) & (np.abs(a) < SENTINEL_LIMIT)


def fill(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid pixels with the footprint median so filters do not see the sentinel."""
    med = float(np.nanmedian(a[valid])) if valid.any() else 0.0
    return np.where(valid, np.nan_to_num(a, nan=med, posinf=med, neginf=med), med).astype(np.float64)
