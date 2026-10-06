"""Write and verify a competition-conformant submission GeoTIFF.

Format requirements, transcribed from the official problem description (fetched live 2026-10-05,
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format):

  * same projected CRS as the training data - UTM zone 11N, EPSG:32611
  * same resolution as the training data - 100 m
  * same bounds as the training data, with data outside the bounds null or nan
  * a SINGLE layer, datatype 32-bit float, values between 0 and 1

The last line is the one that produces the organiser's form error
"Predicted values must be in range [0, 1]" if it is violated, so ``verify_submission`` checks it
explicitly, along with the CRS, the geotransform, the shape, the band count and the dtype - and it
checks them against ``sample_submission.tif`` itself rather than against values restated in code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from .layers import Grid

OUTSIDE_MODES = ("nan", "zeros")


@dataclass
class WriteResult:
    path: str
    sha256: str
    bytes: int
    positive_px: int
    min_value: float
    max_value: float
    dtype: str
    crs: str
    shape: tuple[int, int]
    transform: list[float]
    outside: str
    checks: dict = field(default_factory=dict)


def write_submission(values: np.ndarray, template_path: Path, out_path: Path,
                     outside: str = "nan") -> WriteResult:
    """Write ``values`` as a single-band float32 GeoTIFF on the template's grid.

    ``values`` must already be in [0, 1].  Pixels outside the template footprint are written as
    NaN (``outside='nan'``, the organiser's own format) or 0 (``outside='zeros'``).
    """
    if outside not in OUTSIDE_MODES:
        raise ValueError(f"outside must be one of {OUTSIDE_MODES}, got {outside!r}")
    with rasterio.open(template_path) as tpl:
        profile = tpl.profile.copy()
        foot = np.isfinite(tpl.read(1))

    v = np.asarray(values, dtype=np.float32)
    if v.shape != foot.shape:
        raise ValueError(f"values shape {v.shape} != template shape {foot.shape}")
    finite = np.isfinite(v)
    if not finite.all():
        raise ValueError("values contain non-finite entries inside the grid")
    lo, hi = float(v.min()), float(v.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(f"values out of [0, 1]: min={lo!r} max={hi!r}")

    out = np.where(foot, v, np.float32("nan") if outside == "nan" else np.float32(0.0))
    out = out.astype(np.float32)

    profile.update(count=1, dtype="float32", compress="deflate", predictor=2,
                   nodata=np.float32("nan") if outside == "nan" else None)
    # Drop any tiling the template carried: libtiff requires tile edges to be multiples of 16,
    # and a 3292-wide grid cannot be tiled on its own width.  Stripped output is both valid and
    # what the organiser's own sample_submission.tif uses.
    for key in ("tiled", "blockxsize", "blockysize", "blocksize"):
        profile.pop(key, None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(out, 1)
        ds.set_band_description(1, "fault_probability")

    import hashlib
    h = hashlib.sha256()
    with out_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return WriteResult(
        path=str(out_path), sha256=h.hexdigest(), bytes=out_path.stat().st_size,
        positive_px=int(((out > 0) & np.isfinite(out)).sum()), min_value=lo, max_value=hi,
        dtype="float32", crs=str(profile["crs"]), shape=(profile["height"], profile["width"]),
        transform=list(Affine(*profile["transform"])[:6]), outside=outside,
    )


def verify_submission(path: Path, template_path: Path) -> dict:
    """Independent pre-flight check of a written file against the organiser's requirements."""
    problems: list[str] = []
    with rasterio.open(template_path) as tpl:
        t_prof = tpl.profile.copy()
        t_foot = np.isfinite(tpl.read(1))
    with rasterio.open(path) as ds:
        prof = ds.profile.copy()
        a = ds.read(1)

    checks: dict[str, object] = {}

    checks["single_band"] = ds.count == 1
    if ds.count != 1:
        problems.append(f"expected 1 band, found {ds.count}")

    checks["dtype_float32"] = a.dtype == np.float32
    if a.dtype != np.float32:
        problems.append(f"expected float32, found {a.dtype}")

    checks["shape_matches"] = (ds.height, ds.width) == (t_prof["height"], t_prof["width"])
    if not checks["shape_matches"]:
        problems.append(f"shape {(ds.height, ds.width)} != template "
                        f"{(t_prof['height'], t_prof['width'])}")

    checks["crs_matches"] = str(ds.crs) == str(t_prof["crs"])
    if not checks["crs_matches"]:
        problems.append(f"CRS {ds.crs} != template {t_prof['crs']}")

    checks["crs_is_epsg32611"] = ds.crs is not None and ds.crs.to_epsg() == 32611
    if not checks["crs_is_epsg32611"]:
        problems.append(f"CRS must be EPSG:32611, got {ds.crs}")

    gt_same = tuple(np.round(ds.transform[:6], 6)) == tuple(np.round(t_prof["transform"][:6], 6))
    checks["geotransform_matches"] = gt_same
    if not gt_same:
        problems.append(f"geotransform {ds.transform[:6]} != template {t_prof['transform'][:6]}")

    res = (abs(ds.transform.a), abs(ds.transform.e))
    checks["resolution_100m"] = res == (100.0, 100.0)
    if not checks["resolution_100m"]:
        problems.append(f"resolution {res} != (100.0, 100.0)")

    inside = a[t_foot]
    checks["footprint_all_finite"] = bool(np.isfinite(inside).all())
    if not checks["footprint_all_finite"]:
        problems.append(f"{int((~np.isfinite(inside)).sum())} non-finite pixels inside the "
                        f"survey footprint")

    finite = np.isfinite(a)
    vmin = float(a[finite].min()) if finite.any() else 0.0
    vmax = float(a[finite].max()) if finite.any() else 0.0
    checks["values_in_0_1"] = bool(vmin >= 0.0 and vmax <= 1.0)
    if not checks["values_in_0_1"]:
        problems.append(f"values out of [0, 1]: min={vmin!r} max={vmax!r}")

    outside_vals = a[~t_foot]
    checks["outside_is_null_or_nan"] = bool(
        outside_vals.size == 0 or not np.isfinite(outside_vals).any()
        or bool((outside_vals == 0).all())
    )
    if not checks["outside_is_null_or_nan"]:
        problems.append("pixels outside the survey footprint must be null or nan")

    checks["float32_roundtrip_exact"] = bool(np.array_equal(a[finite], a[finite].astype(np.float32)))

    return {
        "path": str(path),
        "ok": not problems,
        "problems": problems,
        "checks": checks,
        "min": vmin,
        "max": vmax,
        "positive_px": int(((a > 0) & np.isfinite(a)).sum()),
        "nan_px": int((~np.isfinite(a)).sum()),
        "crs": str(ds.crs),
        "epsg": ds.crs.to_epsg() if ds.crs else None,
        "shape": [ds.height, ds.width],
        "geotransform": list(ds.transform[:6]),
    }


def zip_submission(tif_path: Path, out_zip: Path) -> dict:
    """The organiser also accepts a .zip holding a single GeoTIFF."""
    import hashlib
    import zipfile
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tif_path, arcname=tif_path.name)
    h = hashlib.sha256()
    with out_zip.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(out_zip), "sha256": h.hexdigest(), "bytes": out_zip.stat().st_size}
