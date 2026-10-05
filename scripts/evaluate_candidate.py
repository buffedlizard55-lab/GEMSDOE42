#!/usr/bin/env python3
"""Run format, spatial holdout, concentration and all-prior distinctness audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gemsdoe42.diagnostics import concentration_report, single_scale_gradient_baseline
from gemsdoe42.distinctness import audit_prior_correlations
from gemsdoe42.holdout import evaluate_matched_support_holdout, four_spatial_fold_masks
from gemsdoe42.provenance import (
    CANDIDATE_SOURCE_MODULES,
    EVALUATION_SOURCE_MODULES,
    hash_source_bundle,
)
from gemsdoe42.raster import (
    load_aligned_binary_labels,
    load_aligned_inputs,
    load_aligned_score,
    sha256_file,
    validate_prediction_tiff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True, help="frozen current best holdout raster")
    parser.add_argument("--prior-manifest", type=Path, required=True, help="complete prior-output manifest with paths and SHA-256s")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "wph01.json")
    parser.add_argument("--gravity-band", type=int)
    parser.add_argument("--magnetic-band", type=int)
    parser.add_argument("--guard-pixels", type=int, choices=(3,), default=3, help="frozen 300 m guard on the 100 m grid")
    parser.add_argument("--bootstrap-iterations", type=int, choices=(20000,), default=20000, help="frozen paired block-bootstrap iterations")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "evaluations")
    return parser.parse_args()


def _mask_hash(mask) -> str:
    import numpy as np

    packed = np.packbits(np.asarray(mask, dtype=bool).ravel(), bitorder="little")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    candidate_path = args.candidate.resolve()
    incumbent_path = args.incumbent.resolve()
    prior_manifest = args.prior_manifest.resolve()
    features = (args.features or data_dir / "training_features.tif").resolve()
    labels = (args.labels or data_dir / "labels.tif").resolve()
    template = (args.template or data_dir / "sample_submission.tif").resolve()
    config_path = args.config.resolve()
    candidate_run_receipt_path = candidate_path.with_suffix(".json")
    needed = (
        candidate_path, candidate_run_receipt_path, incumbent_path, prior_manifest,
        features, labels, template, config_path,
    )
    absent = [str(path) for path in needed if not path.is_file()]
    if absent:
        raise SystemExit("Required evaluation input(s) missing:\n  " + "\n  ".join(absent))

    candidate_run_receipt = json.loads(candidate_run_receipt_path.read_text(encoding="utf-8"))
    candidate_hash = sha256_file(candidate_path)
    feature_hash = sha256_file(features)
    template_hash = sha256_file(template)
    config_hash = sha256_file(config_path)
    candidate_source_hash, _ = hash_source_bundle(CANDIDATE_SOURCE_MODULES)
    builder_hash = sha256_file(ROOT / "scripts" / "build_candidate.py")
    provenance_mismatches = []
    if candidate_run_receipt.get("candidate_id") != candidate_path.stem:
        provenance_mismatches.append("candidate_id/file name mismatch")
    if candidate_run_receipt.get("format_receipt", {}).get("sha256") != candidate_hash:
        provenance_mismatches.append("candidate TIFF hash differs from build receipt")
    if candidate_run_receipt.get("inputs", {}).get("training_features", {}).get("sha256") != feature_hash:
        provenance_mismatches.append("feature hash differs from build receipt")
    if candidate_run_receipt.get("inputs", {}).get("sample_template", {}).get("sha256") != template_hash:
        provenance_mismatches.append("template hash differs from build receipt")
    if candidate_run_receipt.get("inputs", {}).get("config", {}).get("sha256") != config_hash:
        provenance_mismatches.append("config hash differs from build receipt")
    implementation = candidate_run_receipt.get("implementation", {})
    if implementation.get("source_bundle_sha256") != candidate_source_hash:
        provenance_mismatches.append("inference source hash differs from current code")
    if implementation.get("builder_script_sha256") != builder_hash:
        provenance_mismatches.append("builder script hash differs from current code")
    if provenance_mismatches:
        raise SystemExit("Candidate provenance validation failed: " + "; ".join(provenance_mismatches))

    evaluation_source_hash, evaluation_module_hashes = hash_source_bundle(EVALUATION_SOURCE_MODULES)
    evaluation_script_hash = sha256_file(Path(__file__).resolve())
    format_receipt = validate_prediction_tiff(candidate_path, template)
    candidate, candidate_valid, candidate_meta = load_aligned_score(candidate_path, template)
    incumbent, incumbent_valid, incumbent_meta = load_aligned_score(incumbent_path, template)
    truth, label_valid, label_meta = load_aligned_binary_labels(labels, template)
    gravity, magnetic, footprint, input_meta = load_aligned_inputs(
        features,
        template,
        gravity_band=args.gravity_band,
        magnetic_band=args.magnetic_band,
    )
    if candidate_run_receipt.get("input_checks", {}).get("selected_bands") != input_meta.get("selected_bands"):
        raise SystemExit("Evaluation band selection differs from candidate inference receipt")
    if candidate_run_receipt.get("hypothesis_id") != "WPH-01":
        raise SystemExit("Candidate receipt is not for preregistered WPH-01")
    evaluation_valid = footprint & candidate_valid & incumbent_valid & label_valid
    if not evaluation_valid.any():
        raise SystemExit("No common labeled, valid cells for holdout evaluation")

    fold_records = []
    fold_masks = four_spatial_fold_masks(footprint, guard_pixels=args.guard_pixels)
    for index, domain_mask in enumerate(fold_masks, start=1):
        evaluation_mask = domain_mask & evaluation_valid
        fold_records.append({
            "fold": index,
            "domain_mask_sha256": _mask_hash(domain_mask),
            "evaluation_mask_sha256": _mask_hash(evaluation_mask),
            "domain_pixels": int(domain_mask.sum()),
            "evaluation_pixels": int(evaluation_mask.sum()),
        })
    split_hash = hashlib.sha256(
        "".join(row["evaluation_mask_sha256"] for row in fold_records).encode("ascii")
    ).hexdigest()

    holdout = {"status": "failed", "error": None}
    comparator_holdout = {"status": "not_run", "error": None}
    try:
        holdout_result = evaluate_matched_support_holdout(
            candidate,
            incumbent,
            truth,
            evaluation_valid,
            pixel_size_m=100.0,
            radius_m=300.0,
            guard_pixels=args.guard_pixels,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=42,
            mass_area_fractions=(0.005, 0.01, 0.02),
            fold_domain=footprint,
        )
        holdout = {"status": "complete", **holdout_result}
    except Exception as error:  # noqa: BLE001 - persist the failed gate and continue independent audits
        holdout = {"status": "failed", "error": f"{type(error).__name__}: {error}"}

    comparator = single_scale_gradient_baseline(gravity, magnetic, footprint, pixel_size_m=100.0)
    candidate_concentration = concentration_report(candidate, footprint)
    comparator_concentration = concentration_report(comparator, footprint)
    try:
        comparator_result = evaluate_matched_support_holdout(
            comparator,
            incumbent,
            truth,
            evaluation_valid,
            pixel_size_m=100.0,
            radius_m=300.0,
            guard_pixels=args.guard_pixels,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=42,
            mass_area_fractions=(0.005, 0.01, 0.02),
            fold_domain=footprint,
        )
        comparator_holdout = {"status": "complete", **comparator_result}
    except Exception as error:  # noqa: BLE001 - capture diagnostic-comparator failure only
        comparator_holdout = {"status": "failed", "error": f"{type(error).__name__}: {error}"}

    try:
        correlation = audit_prior_correlations(
            candidate_path,
            template,
            prior_manifest,
            max_absolute_pearson=0.90,
            max_absolute_spearman=0.90,
            max_spearman_sample=250_000,
            base_dir=ROOT,
        )
        correlation_status = "complete"
    except Exception as error:  # noqa: BLE001 - distinctness failure must be recorded as closed
        correlation = {"error": f"{type(error).__name__}: {error}"}
        correlation_status = "failed_closed"

    holdout_passed = bool(holdout.get("promotion_gate_passed", False))
    distinctness_passed = bool(correlation.get("all_priors_distinct", False))
    eligible = bool(holdout_passed and distinctness_passed and format_receipt["mask_matches_template"])
    report = {
        "candidate_id": candidate_path.stem,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "status": "eligible_for_human_review" if eligible else "not_eligible_gates_not_passed",
        "competition_download_enabled": False,
        "competition_slot_used": False,
        "eligibility": {
            "format_passed": True,
            "holdout_passed": holdout_passed,
            "all_prior_correlations_passed": distinctness_passed,
            "eligible_for_human_review": eligible,
        },
        "hashes": {
            "candidate": format_receipt["sha256"],
            "incumbent": incumbent_meta["score_sha256"],
            "features": sha256_file(features),
            "labels": label_meta["label_sha256"],
            "template": template_hash,
            "config": config_hash,
            "candidate_run_receipt": sha256_file(candidate_run_receipt_path),
            "prior_manifest": sha256_file(prior_manifest),
            "candidate_inference_source_bundle": candidate_source_hash,
            "candidate_builder_script": builder_hash,
            "evaluation_source_bundle": evaluation_source_hash,
            "evaluation_script": evaluation_script_hash,
        },
        "implementation": {
            "candidate_run_receipt": candidate_run_receipt,
            "evaluation_source_module_sha256": evaluation_module_hashes,
        },
        "grid_and_input_receipt": input_meta,
        "candidate_receipt": {**candidate_meta, **format_receipt},
        "incumbent_receipt": incumbent_meta,
        "label_receipt": label_meta,
        "spatial_split": {
            "definition": "four contiguous equal pixel quadrants; 300 m guard from block edges",
            "guard_pixels": args.guard_pixels,
            "footprint_pixels": int(footprint.sum()),
            "fold_mask_hashes": fold_records,
            "combined_split_sha256": split_hash,
        },
        "holdout": holdout,
        "single_scale_gradient_comparator": {
            "concentration": comparator_concentration,
            "candidate_concentration": candidate_concentration,
            "holdout": comparator_holdout,
            "comparison_note": "The comparator is an unsupervised single-scale gradient-magnitude map from the same layers, robustly normalized and averaged. Concentration is reported as a measurement, not assumed.",
        },
        "prior_correlation_audit": {"status": correlation_status, **correlation},
        "submission_materials": {
            "portal_note": (
                "WPH-01: gravity/RTP-magnetic edge survival across upward continuation × 0D ridge "
                "persistence across smoothing scales. Continuous confidence ranking; not a calibrated probability."
                if eligible else None
            ),
            "ai_disclosure_file": "docs/ai-disclosure-template.txt",
            "note": "A passing local audit is not organizer scoring or a guarantee of newly unmapped-fault discovery. A human entrant must review the evidence, arrange an authorized public artifact URL, and decide whether to spend a submission slot.",
        },
    }
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{candidate_path.stem}-evaluation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Evaluation receipt: {report_path}")
    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
