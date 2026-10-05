"""DrivenData GeoTIFF submission writer + strict audit for GEMSDOE42.

Fixes the portal error "Predicted values must be in range [0, 1]" by
construction (two known mechanisms):
  1. float32 sentinel -3.4e38 written through -> outside [0,1];
  2. NaN cells with a NaN nodata tag on a strict validator.
Every primary here is written all-finite (zeros outside footprint,
nodata=None) AND in the competition's own NaN-outside convention, both
audited by re-reading the written bytes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

from .grid import HEIGHT, WIDTH, TOTAL_CELLS, FOOTPRINT_PX


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_geotiff(tif_path: Path, footprint: np.ndarray, labels: np.ndarray,
                  mode: str = "zeros") -> dict:
    """12-point validator audit by re-reading the written file."""
    tif_path = Path(tif_path)
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        crs = str(src.crs)
        tr = tuple(round(float(v), 4) for v in src.transform)[:6]
        dtype = src.dtypes[0]
        count = src.count
        H, W = src.height, src.width
        nodata = src.nodata
    foot = np.asarray(footprint, bool)
    lab = np.asarray(labels, bool)
    inside = arr[foot]
    outside = arr[~foot]
    n_finite_in = int(np.isfinite(inside).sum())
    in_range = bool(n_finite_in == FOOTPRINT_PX
                    and np.all((inside >= 0.0) & (inside <= 1.0)))
    if mode == "zeros":
        full_ok = bool(int(np.isfinite(arr).sum()) == TOTAL_CELLS
                       and np.all((arr >= 0.0) & (arr <= 1.0)))
        outside_ok = bool(np.all(outside == 0.0))
    else:
        full_ok = in_range
        outside_ok = bool(np.all(np.isnan(outside)))
    emitted = int(((arr > 0.5) & foot).sum())
    on_cat = int(((arr > 0.5) & lab).sum())
    checks = {
        "single_band": count == 1,
        "dtype_float32": dtype == "float32",
        "dimensions_3730x3292": (H == HEIGHT and W == WIDTH),
        "crs_epsg_32611": "32611" in crs,
        "in_footprint_all_finite": n_finite_in == FOOTPRINT_PX,
        "in_footprint_zero_nan": int(np.isnan(inside).sum()) == 0,
        "in_footprint_zero_inf": int(np.isinf(inside).sum()) == 0,
        "in_footprint_zero_sentinel": int((inside < -1e30).sum()) == 0,
        "in_footprint_range_0_1": in_range,
        "outside_footprint_compliant": outside_ok,
        "zero_on_catalogue_leakage": on_cat == 0,
        "validator_range_0_1_guaranteed": full_ok,
    }
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise ValueError(f"audit failed for {tif_path.name}: {failed}")
    return {
        "filename": tif_path.name,
        "size_bytes": tif_path.stat().st_size,
        "sha256": sha256_file(tif_path),
        "mode": mode,
        "crs": crs,
        "transform": list(tr),
        "shape": [H, W],
        "dtype": dtype,
        "nodata": None if nodata is None else (
            None if isinstance(nodata, float) and np.isnan(nodata)
            else float(nodata)),
        "nodata_repr": str(nodata),
        "emitted_positive_pixels": emitted,
        "footprint_fraction": round(emitted / float(FOOTPRINT_PX), 6),
        "on_catalogue_positive_pixels": on_cat,
        "in_footprint_finite_pixels": n_finite_in,
        "in_footprint_min": float(np.min(inside)),
        "in_footprint_max": float(np.max(inside)),
        "full_grid_finite_pixels": int(np.isfinite(arr).sum()),
        "checks": checks,
        "all_checks_passed": True,
    }


def write_pair(mask: np.ndarray, footprint: np.ndarray, labels: np.ndarray,
               template_tif: Path, out_dir: Path, slug: str,
               timestamp_tag: str, description: str,
               submission_note_template: str, extra_meta: dict | None = None):
    """Write + audit zeros/nan pair for a bool/float mask in [0,1].

    Returns the bundle dict (also written as *-audit.json sidecar).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = np.asarray(mask, dtype=np.float64)
    foot = np.asarray(footprint, bool)
    lab = np.asarray(labels, bool)
    if m.shape != foot.shape:
        raise ValueError("mask/footprint shape mismatch")
    clean = np.where(foot & ~lab, np.clip(m, 0.0, 1.0), 0.0).astype(np.float32)
    digest8 = hashlib.sha256(np.packbits(clean > 0.5)).hexdigest()[:8]
    zeros_name = f"gemsdoe42-{slug}-{timestamp_tag}-{digest8}-zeros.tif"
    nan_name = f"gemsdoe42-{slug}-{timestamp_tag}-{digest8}-nan.tif"
    zeros_path = out_dir / zeros_name
    nan_path = out_dir / nan_name
    with rasterio.open(template_tif) as tpl:
        base = tpl.profile.copy()
    prof = base.copy()
    prof.update(driver="GTiff", dtype="float32", count=1, compress="deflate",
                predictor=3, zlevel=9, tiled=True, blockxsize=256,
                blockysize=256, nodata=None)
    arr0 = np.where(foot, clean, 0.0).astype(np.float32)
    with rasterio.open(zeros_path, "w", **prof) as dst:
        dst.write(arr0, 1)
    prof_nan = prof.copy()
    prof_nan.update(nodata=np.nan)
    arrn = np.where(foot, clean, np.nan).astype(np.float32)
    with rasterio.open(nan_path, "w", **prof_nan) as dst:
        dst.write(arrn, 1)
    a0 = audit_geotiff(zeros_path, foot, lab, mode="zeros")
    an = audit_geotiff(nan_path, foot, lab, mode="nan")
    dots = a0["emitted_positive_pixels"]
    bundle = {
        "candidate_id": slug,
        "slug": slug,
        "content_digest8": digest8,
        "description": description,
        "submission_note_zeros": submission_note_template.format(
            filename=zeros_name, sha8=a0["sha256"][:8], dots=dots),
        "submission_note_nan": submission_note_template.format(
            filename=nan_name, sha8=an["sha256"][:8], dots=dots),
        "zeros_tif": a0,
        "nan_tif": an,
        "extra": extra_meta or {},
    }
    sidecar = out_dir / f"gemsdoe42-{slug}-{timestamp_tag}-{digest8}-audit.json"
    sidecar.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle
