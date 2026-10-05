"""Raw-raster distinctness audit against every hash-pinned prior output."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"submission_id", "artifact_path", "sha256"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"prior manifest must include columns {sorted(required)}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise ValueError("prior manifest has no rows; completeness cannot be established")
    if any(not row.get("submission_id") for row in rows):
        raise ValueError("every prior manifest row must have a submission_id")
    if len({row["submission_id"] for row in rows}) != len(rows):
        raise ValueError("prior manifest contains duplicate submission_id values")
    return rows


def _pearson_from_sums(n, sx, sy, sxx, syy, sxy, identical):
    if n < 2:
        raise ValueError("not enough common finite cells for correlation")
    covariance = sxy - sx * sy / n
    variance_x = sxx - sx * sx / n
    variance_y = syy - sy * sy / n
    if variance_x <= 0.0 or variance_y <= 0.0:
        return 1.0 if identical else 0.0
    return float(covariance / math.sqrt(variance_x * variance_y))


def audit_prior_correlations(
    candidate_path: str | Path,
    template_path: str | Path,
    manifest_path: str | Path,
    *,
    max_absolute_pearson: float = 0.90,
    max_absolute_spearman: float = 0.90,
    max_spearman_sample: int = 250_000,
    base_dir: str | Path | None = None,
) -> dict:
    """Compare one candidate with all registered prior raw rasters; fail closed.

    Pearson is accumulated exactly over all common valid pixels by tile. Spearman is
    estimated from a deterministic systematic sample of at most ``max_spearman_sample``
    cells; top-1%-rank overlap uses the same reproducible sample. Positive-support
    Jaccard is exact over all valid pixels. Every manifest entry must resolve to a
    readable raster with a matching SHA-256 before a passing result is possible.
    """
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    from scipy.stats import rankdata

    from .raster import _assert_aligned, sha256_file

    candidate_path = Path(candidate_path).resolve()
    template_path = Path(template_path).resolve()
    base = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
    rows = _read_manifest(manifest_path)
    if not (0.0 < max_absolute_pearson < 1.0 and 0.0 < max_absolute_spearman < 1.0):
        raise ValueError("correlation thresholds must be between 0 and 1")
    if max_spearman_sample < 2:
        raise ValueError("max_spearman_sample must be at least 2")

    reports = []
    missing = []
    with rasterio.open(template_path) as template, rasterio.open(candidate_path) as candidate:
        _assert_aligned(candidate, template, "candidate")
        if candidate.count != 1:
            raise ValueError("candidate correlation input must be one band")
        candidate_hash = sha256_file(candidate_path)
        width, height = template.width, template.height
        stride = max(1, math.ceil(width * height / max_spearman_sample))
        tile = 512

        for row in rows:
            raw_path = Path(row["artifact_path"]) if row.get("artifact_path") else None
            if raw_path is not None and not raw_path.is_absolute():
                raw_path = (base / raw_path).resolve()
            if raw_path is None or not raw_path.is_file() or not row.get("sha256"):
                missing.append(row["submission_id"])
                continue
            actual_prior_hash = sha256_file(raw_path)
            if actual_prior_hash.lower() != row["sha256"].lower():
                raise ValueError(
                    f"SHA-256 mismatch for prior {row['submission_id']}: "
                    f"manifest={row['sha256']} actual={actual_prior_hash}"
                )

            with rasterio.open(raw_path) as prior:
                _assert_aligned(prior, template, f"prior {row['submission_id']}")
                if prior.count != 1:
                    raise ValueError(f"prior {row['submission_id']} must be single-band")

                count = 0
                sx = sy = sxx = syy = sxy = 0.0
                identical = True
                both_support = either_support = 0
                sample_x = []
                sample_y = []
                for y0 in range(0, height, tile):
                    h = min(tile, height - y0)
                    for x0 in range(0, width, tile):
                        w = min(tile, width - x0)
                        window = Window(x0, y0, w, h)
                        a = candidate.read(1, window=window).astype(np.float64, copy=False)
                        b = prior.read(1, window=window).astype(np.float64, copy=False)
                        expected_mask = template.read_masks(1, window=window) > 0
                        candidate_mask = candidate.read_masks(1, window=window) > 0
                        prior_mask = prior.read_masks(1, window=window) > 0
                        if not np.array_equal(candidate_mask, expected_mask):
                            raise ValueError("candidate mask does not match the official sample footprint")
                        if not np.array_equal(prior_mask, expected_mask):
                            raise ValueError(
                                f"prior {row['submission_id']} mask does not match the official sample footprint"
                            )
                        valid = expected_mask
                        if np.any(~np.isfinite(a[valid])) or np.any(~np.isfinite(b[valid])):
                            raise ValueError(
                                f"candidate/prior {row['submission_id']} has non-finite values inside the footprint"
                            )
                        if np.any((a[valid] < 0.0) | (a[valid] > 1.0)) or np.any(
                            (b[valid] < 0.0) | (b[valid] > 1.0)
                        ):
                            raise ValueError(
                                f"candidate/prior {row['submission_id']} has values outside [0,1]"
                            )
                        if not valid.any():
                            continue
                        av = a[valid]
                        bv = b[valid]
                        count += av.size
                        sx += float(av.sum(dtype=np.float64))
                        sy += float(bv.sum(dtype=np.float64))
                        sxx += float(np.dot(av, av))
                        syy += float(np.dot(bv, bv))
                        sxy += float(np.dot(av, bv))
                        if identical and not np.array_equal(av, bv):
                            identical = False
                        apos = av > 0.0
                        bpos = bv > 0.0
                        both_support += int(np.count_nonzero(apos & bpos))
                        either_support += int(np.count_nonzero(apos | bpos))

                        local_rows, local_cols = np.nonzero(valid)
                        global_flat = (local_rows + y0) * width + (local_cols + x0)
                        take = (global_flat % stride) == 0
                        if take.any():
                            sample_x.append(av[take])
                            sample_y.append(bv[take])

                if count < 2:
                    raise ValueError(f"prior {row['submission_id']} has <2 common valid cells")
                pearson = _pearson_from_sums(count, sx, sy, sxx, syy, sxy, identical)
                if sample_x:
                    sampled_x = np.concatenate(sample_x)
                    sampled_y = np.concatenate(sample_y)
                else:
                    sampled_x = np.array([], dtype=np.float64)
                    sampled_y = np.array([], dtype=np.float64)
                if sampled_x.size < 2:
                    raise ValueError(
                        f"systematic sample too small for prior {row['submission_id']}"
                    )
                rank_x = rankdata(sampled_x, method="average")
                rank_y = rankdata(sampled_y, method="average")
                rank_x -= rank_x.mean()
                rank_y -= rank_y.mean()
                denom = float(np.sqrt(np.dot(rank_x, rank_x) * np.dot(rank_y, rank_y)))
                spearman = (
                    float(np.dot(rank_x, rank_y) / denom)
                    if denom > 0.0
                    else (1.0 if np.array_equal(sampled_x, sampled_y) else 0.0)
                )
                qx = float(np.quantile(sampled_x, 0.99))
                qy = float(np.quantile(sampled_y, 0.99))
                top_x = sampled_x >= qx
                top_y = sampled_y >= qy
                top_union = int(np.count_nonzero(top_x | top_y))
                top_intersection = int(np.count_nonzero(top_x & top_y))
                top_jaccard = top_intersection / top_union if top_union else 1.0
                support_jaccard = both_support / either_support if either_support else 1.0
                fail = (
                    abs(pearson) >= max_absolute_pearson
                    or abs(spearman) >= max_absolute_spearman
                )
                reports.append(
                    {
                        "submission_id": row["submission_id"],
                        "artifact_path": str(raw_path),
                        "sha256": actual_prior_hash,
                        "common_valid_pixels": count,
                        "pearson_all_pixels": pearson,
                        "spearman_systematic_sample": spearman,
                        "spearman_sample_pixels": int(sampled_x.size),
                        "support_jaccard_all_pixels": float(support_jaccard),
                        "top_1_percent_jaccard_sample": float(top_jaccard),
                        "correlation_gate_failed": bool(fail),
                    }
                )

    if missing:
        raise FileNotFoundError(
            "prior-raster audit incomplete; missing path or SHA-256 for: " + ", ".join(missing)
        )
    if len(reports) != len(rows):
        raise RuntimeError("not every prior manifest row produced a correlation report")
    failed = [report["submission_id"] for report in reports if report["correlation_gate_failed"]]
    return {
        "candidate_path": str(candidate_path),
        "candidate_sha256": candidate_hash,
        "candidate_bytes": candidate_path.stat().st_size,
        "prior_count": len(rows),
        "pearson_threshold_abs": max_absolute_pearson,
        "spearman_threshold_abs": max_absolute_spearman,
        "spearman_sampling": f"systematic global-flat-index stride {stride}, maximum target {max_spearman_sample}",
        "reports": reports,
        "failed_prior_ids": failed,
        "all_priors_distinct": not failed,
        "ready_for_distinctness_gate": not failed,
    }
