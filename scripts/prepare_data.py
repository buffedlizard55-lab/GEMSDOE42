#!/usr/bin/env python3
"""Validate locally authorized competition rasters and write a provenance manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from gemsdoe42.raster import (
    inspect_grid,
    load_aligned_binary_labels,
    load_aligned_inputs,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--features", type=Path, help="defaults to DATA_DIR/training_features.tif")
    parser.add_argument("--labels", type=Path, help="defaults to DATA_DIR/labels.tif")
    parser.add_argument("--template", type=Path, help="defaults to DATA_DIR/sample_submission.tif")
    parser.add_argument("--gravity-band", type=int, help="1-based band; otherwise inferred only from unambiguous metadata")
    parser.add_argument("--magnetic-band", type=int, help="1-based RTP band; otherwise inferred only from unambiguous metadata")
    parser.add_argument("--manifest", type=Path, help="defaults to DATA_DIR/prepared/input-manifest.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    features = (args.features or data_dir / "training_features.tif").resolve()
    labels = (args.labels or data_dir / "labels.tif").resolve()
    template = (args.template or data_dir / "sample_submission.tif").resolve()
    manifest_path = (args.manifest or data_dir / "prepared" / "input-manifest.json").resolve()

    missing = [str(path) for path in (features, labels, template) if not path.is_file()]
    if missing:
        print("Missing local, authorized input(s):", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        print(
            "Obtain the competition package through the participant's official "
            "DrivenData account; this script does not download or bypass login.",
            file=sys.stderr,
        )
        return 2

    gravity, magnetic, footprint, input_report = load_aligned_inputs(
        features,
        template,
        gravity_band=args.gravity_band,
        magnetic_band=args.magnetic_band,
    )
    del gravity, magnetic, footprint
    _, label_valid, label_report = load_aligned_binary_labels(labels, template)
    report = {
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "source_policy": "locally supplied authorized competition inputs; no automated download",
        "files": {
            "training_features": {"path": str(features), "sha256": sha256_file(features), "grid": inspect_grid(features)},
            "labels": {"path": str(labels), "sha256": sha256_file(labels), "grid": inspect_grid(labels)},
            "sample_template": {"path": str(template), "sha256": sha256_file(template), "grid": inspect_grid(template)},
        },
        "alignment_and_band_checks": input_report,
        "label_check": label_report,
        "valid_label_pixels": int(label_valid.sum()),
        "status": "inputs_aligned_and_manifested; holdout_and_distinctness_not_run",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote local input manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
