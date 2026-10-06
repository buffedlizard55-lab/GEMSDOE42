"""Raster I/O, grid checks, GeoTIFF writing and submission-format validation."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _grid_record(dataset):
    crs = dataset.crs
    return {
        "width": int(dataset.width),
        "height": int(dataset.height),
        "count": int(dataset.count),
        "crs": crs.to_string() if crs else None,
        "epsg": crs.to_epsg() if crs else None,
        "transform": [float(value) for value in dataset.transform[:6]],
        "resolution_m": [abs(float(dataset.transform.a)), abs(float(dataset.transform.e))],
        "bounds": [
            float(dataset.bounds.left), float(dataset.bounds.bottom),
            float(dataset.bounds.right), float(dataset.bounds.top),
        ],
        "dtype": list(dataset.dtypes),
        "nodata": [None if value is None else str(value) for value in dataset.nodatavals],
        "descriptions": list(dataset.descriptions),
    }


def inspect_grid(path: str | Path) -> dict:
    import rasterio

    with rasterio.open(path) as dataset:
        return _grid_record(dataset)


def _assert_aligned(dataset, template, label: str) -> None:
    if (dataset.width, dataset.height) != (template.width, template.height):
        raise ValueError(f"{label} dimensions do not match official sample grid")
    if dataset.crs != template.crs:
        raise ValueError(f"{label} CRS does not match official sample grid")
    if dataset.transform != template.transform:
        raise ValueError(f"{label} affine transform does not match official sample grid")


def _select_band(dataset, role: str, explicit_band: int | None) -> int:
    if explicit_band is not None:
        if not 1 <= explicit_band <= dataset.count:
            raise ValueError(
                f"{role} band {explicit_band} is outside 1..{dataset.count}"
            )
        return explicit_band

    text_by_band = []
    for band in range(1, dataset.count + 1):
        description = dataset.descriptions[band - 1] or ""
        tags = dataset.tags(band)
        text_by_band.append(" ".join([description, *tags.keys(), *tags.values()]).lower())
    aliases = {
        "gravity": ("gravity", "bouguer", "isostatic"),
        "magnetic_rtp": ("magnetic", "magnetics", "reduced to pole", "rtp"),
    }[role]
    matches = [
        index + 1
        for index, text in enumerate(text_by_band)
        if any(alias in text for alias in aliases)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"could not unambiguously infer the {role} band from raster metadata; "
            f"found matches {matches}. Inspect the source and pass the correct band explicitly"
        )
    return matches[0]


def load_aligned_inputs(
    features_path: str | Path,
    template_path: str | Path,
    *,
    gravity_band: int | None = None,
    magnetic_band: int | None = None,
):
    """Read gravity/RTP-magnetic bands and the official one-band footprint template."""
    import numpy as np
    import rasterio

    with rasterio.open(template_path) as template, rasterio.open(features_path) as features:
        if template.count != 1:
            raise ValueError("official sample template must contain exactly one band")
        if template.crs is None or template.crs.to_epsg() != 32611:
            raise ValueError("official sample template must use the documented UTM zone 11N CRS (EPSG:32611)")
        xres, yres = abs(template.transform.a), abs(template.transform.e)
        if not (math.isclose(xres, 100.0, abs_tol=1e-6) and math.isclose(yres, 100.0, abs_tol=1e-6)):
            raise ValueError(f"official sample grid is not 100 m resolution: {(xres, yres)}")
        _assert_aligned(features, template, "training features")

        g_band = _select_band(features, "gravity", gravity_band)
        m_band = _select_band(features, "magnetic_rtp", magnetic_band)
        footprint = template.read_masks(1) > 0
        gravity = features.read(g_band, out_dtype="float32")
        magnetic = features.read(m_band, out_dtype="float32")
        gravity_valid = (features.read_masks(g_band) > 0) & np.isfinite(gravity)
        magnetic_valid = (features.read_masks(m_band) > 0) & np.isfinite(magnetic)

        for name, values, valid in (
            ("gravity", gravity, gravity_valid),
            ("magnetic RTP", magnetic, magnetic_valid),
        ):
            if np.any(valid & (np.abs(values) > 1e30)):
                raise ValueError(
                    f"{name} contains unmasked extreme sentinel-like values; inspect nodata metadata"
                )
            if not np.any(valid & footprint):
                raise ValueError(f"{name} has no valid cells inside the official footprint")

        report = {
            "features": _grid_record(features),
            "template": _grid_record(template),
            "selected_bands": {"gravity": g_band, "magnetic_rtp": m_band},
            "valid_gravity_pixels_in_footprint": int((gravity_valid & footprint).sum()),
            "valid_magnetic_pixels_in_footprint": int((magnetic_valid & footprint).sum()),
            "footprint_pixels": int(footprint.sum()),
        }
    return gravity, magnetic, footprint, report


def load_aligned_binary_labels(path: str | Path, template_path: str | Path):
    """Read an aligned single-band known-fault label raster as a binary mask."""
    import numpy as np
    import rasterio

    with rasterio.open(template_path) as template, rasterio.open(path) as labels:
        _assert_aligned(labels, template, "known-fault labels")
        if labels.count != 1:
            raise ValueError("known-fault label raster must contain one band")
        values = labels.read(1)
        label_valid = (labels.read_masks(1) > 0) & (template.read_masks(1) > 0)
        if not label_valid.any():
            raise ValueError("known-fault raster has no valid cells in sample footprint")
        observed = np.unique(values[label_valid])
        if observed.size > 2 or not np.all(np.isin(observed, [0, 1, 255])):
            raise ValueError(
                "expected binary known-fault values encoded as 0/1 (or 255 for positive); "
                f"observed {observed[:16].tolist()}"
            )
        truth = (values == 1) | (values == 255)
        footprint = template.read_masks(1) > 0
        return truth, label_valid & footprint, {
            "label_path": str(Path(path)),
            "label_sha256": sha256_file(path),
            "label_values": [int(value) for value in observed],
            "known_fault_pixels": int((truth & label_valid).sum()),
            "valid_label_pixels": int(label_valid.sum()),
        }


def load_aligned_score(path: str | Path, template_path: str | Path):
    """Read an aligned one-band incumbent score raster and its valid-data mask."""
    import numpy as np
    import rasterio

    with rasterio.open(template_path) as template, rasterio.open(path) as dataset:
        _assert_aligned(dataset, template, "incumbent score")
        if dataset.count != 1:
            raise ValueError("incumbent score must contain one band")
        values = dataset.read(1, out_dtype="float32")
        valid = (dataset.read_masks(1) > 0) & (template.read_masks(1) > 0)
        if not np.all(np.isfinite(values[valid])):
            raise ValueError("incumbent contains non-finite values in valid cells")
        if np.any((values[valid] < 0.0) | (values[valid] > 1.0)):
            raise ValueError("incumbent values must be in [0,1]")
        return values, valid, {
            "score_path": str(Path(path)),
            "score_sha256": sha256_file(path),
            "positive_pixels": int(np.count_nonzero(values[valid] > 0.0)),
            "valid_pixels": int(valid.sum()),
        }


def write_prediction_tiff(path: str | Path, score, template_path: str | Path) -> dict:
    """Write a single-band float32 TIFF aligned to the sample; outside-footprint is NaN."""
    import numpy as np
    import rasterio

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(score, dtype=np.float32)
    with rasterio.open(template_path) as template:
        if values.shape != (template.height, template.width):
            raise ValueError("score shape does not match official sample grid")
        footprint = template.read_masks(1) > 0
        if not np.all(np.isfinite(values[footprint])):
            raise ValueError("score has non-finite values inside the official footprint")
        if np.any((values[footprint] < 0.0) | (values[footprint] > 1.0)):
            raise ValueError("score values must be in [0,1]")
        output = np.full(values.shape, np.nan, dtype=np.float32)
        output[footprint] = values[footprint]
        profile = template.profile.copy()
        tile_width = min(256, max(16, ((template.width + 15) // 16) * 16))
        tile_height = min(256, max(16, ((template.height + 15) // 16) * 16))
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=float("nan"),
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=tile_width,
            blockysize=tile_height,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(destination, "w", **profile) as target:
            target.write(output, 1)
            target.set_band_description(1, "WPH-01 continuous fault-confidence score")
            target.update_tags(
                1,
                hypothesis="WPH-01",
                score_semantics="continuous ranking/confidence; not calibrated probability",
                values="finite values in [0,1]; NaN outside sample footprint",
            )
    return validate_prediction_tiff(destination, template_path)


def validate_prediction_tiff(path: str | Path, template_path: str | Path) -> dict:
    """Validate shape, mask, CRS, transform, dtype, one-band count and [0,1] range."""
    import numpy as np
    import rasterio

    with rasterio.open(template_path) as template, rasterio.open(path) as prediction:
        _assert_aligned(prediction, template, "prediction")
        if prediction.count != 1:
            raise ValueError("prediction must contain exactly one band")
        if prediction.dtypes[0] != "float32":
            raise ValueError(f"prediction dtype must be float32, got {prediction.dtypes[0]}")
        if prediction.nodata is None or not math.isnan(float(prediction.nodata)):
            raise ValueError("prediction must use NaN nodata outside the official footprint")
        expected_mask = template.read_masks(1) > 0
        actual_mask = prediction.read_masks(1) > 0
        if not np.array_equal(actual_mask, expected_mask):
            raise ValueError("prediction valid-data mask does not match sample footprint")
        values = prediction.read(1)
        valid_values = values[expected_mask]
        if not np.all(np.isfinite(valid_values)):
            raise ValueError("prediction contains non-finite in-footprint values")
        minimum = float(valid_values.min())
        maximum = float(valid_values.max())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(f"prediction range [{minimum}, {maximum}] exceeds [0,1]")
        if np.any(np.isfinite(values[~expected_mask])):
            raise ValueError("prediction contains finite values outside the sample footprint")
        return {
            "path": str(Path(path)),
            "sha256": sha256_file(path),
            "bytes": Path(path).stat().st_size,
            "width": prediction.width,
            "height": prediction.height,
            "crs": prediction.crs.to_string() if prediction.crs else None,
            "epsg": prediction.crs.to_epsg() if prediction.crs else None,
            "transform": [float(value) for value in prediction.transform[:6]],
            "count": prediction.count,
            "dtype": prediction.dtypes[0],
            "nodata": str(prediction.nodata),
            "mask_matches_template": True,
            "min": minimum,
            "max": maximum,
            "nonzero_pixels": int(np.count_nonzero(valid_values)),
        }
