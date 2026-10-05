#!/usr/bin/env python3
"""Generate a quarantined WPH-01 research raster from local authorized inputs.

This command never writes a public download or marks a candidate submission-ready.
Holdout, format and all-prior distinctness evidence are separate mandatory gates.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gemsdoe42.pipeline import wph01_score
from gemsdoe42.provenance import CANDIDATE_SOURCE_MODULES, hash_source_bundle
from gemsdoe42.raster import (
    load_aligned_inputs,
    sha256_file,
    write_prediction_tiff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--gravity-band", type=int)
    parser.add_argument("--magnetic-band", type=int)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "wph01.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "candidates")
    return parser.parse_args()


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    features = (args.features or data_dir / "training_features.tif").resolve()
    template = (args.template or data_dir / "sample_submission.tif").resolve()
    config_path = args.config.resolve()
    for path in (features, template, config_path):
        if not path.is_file():
            raise SystemExit(f"Required file is absent: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    gravity, magnetic, footprint, input_report = load_aligned_inputs(
        features,
        template,
        gravity_band=args.gravity_band,
        magnetic_band=args.magnetic_band,
    )
    score, inference_report = wph01_score(
        gravity,
        magnetic,
        footprint,
        pixel_size_m=100.0,
        continuation_heights_m=config["continuation_heights_m"],
        smoothing_sigmas_pixels=config["smoothing_sigmas_pixels"],
        worm_match_radius_pixels=config["worm_match_radius_pixels"],
        topology_match_radius_pixels=config["topology_match_radius_pixels"],
    )
    feature_hash = sha256_file(features)
    config_hash = sha256_file(config_path)
    method_hash, module_hashes = hash_source_bundle(CANDIDATE_SOURCE_MODULES)
    builder_hash = sha256_file(Path(__file__).resolve())
    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_nonce = uuid.uuid4().hex[:8]
    candidate_id = f"wph01-{run_stamp}-{feature_hash[:10]}-{config_hash[:8]}-{method_hash[:8]}-{builder_hash[:8]}-{run_nonce}"
    output_dir = args.out_dir.resolve()
    candidate_path = output_dir / f"{candidate_id}.tif"
    format_receipt = write_prediction_tiff(candidate_path, score, template)
    report = {
        "candidate_id": candidate_id,
        "status": "quarantined_candidate_holdout_and_prior_audit_required",
        "submission_ready": False,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "hypothesis_id": config["hypothesis_id"],
        "inputs": {
            "training_features": {"path": str(features), "sha256": feature_hash},
            "sample_template": {"path": str(template), "sha256": sha256_file(template)},
            "config": {"path": str(config_path), "sha256": config_hash},
        },
        "implementation": {
            "source_bundle_sha256": method_hash,
            "module_sha256": module_hashes,
            "builder_script_sha256": builder_hash,
            "package_version": "0.1.0",
            "dependency_versions": {
                name: _installed_version(name)
                for name in ("numpy", "scipy", "rasterio", "numba")
            },
        },
        "input_checks": input_report,
        "inference": inference_report,
        "format_receipt": format_receipt,
        "holdout": {"status": "not_run_by_build_command"},
        "prior_correlation": {"status": "not_run_by_build_command"},
        "message": "Local candidate only. Do not use a competition slot or enable a download until evaluate_candidate.py produces a complete passing receipt.",
    }
    report_path = output_dir / f"{candidate_id}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Quarantined research raster: {candidate_path}")
    print(f"Candidate run receipt: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
