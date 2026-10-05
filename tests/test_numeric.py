from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_RASTERIO = importlib.util.find_spec("rasterio") is not None

def _sample_grid(rasterio, np, path: Path, *, width=80, height=80):
    from affine import Affine

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32611",
        "transform": Affine(100, 0, 500000, 0, -100, 4100000),
        "nodata": -9999.0,
        "compress": "deflate",
    }
    data = np.zeros((height, width), dtype=np.float32)
    data[0, 0] = -9999.0
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(data, 1)
    return profile


@unittest.skipUnless(HAS_NUMPY, "NumPy is an optional runtime dependency")
class NumericRuntimeTests(unittest.TestCase):
    def test_production_persistence_matches_reference(self):
        import numpy as np

        from gemsdoe42.topology import h0_persistence_map
        from gemsdoe42.topology_reference import h0_superlevel_persistence

        values = np.array([[3.0, 1.0, 2.0], [0.0, 0.5, 0.0]], dtype=np.float32)
        valid = np.ones(values.shape, dtype=bool)
        actual = h0_persistence_map(values, valid)
        expected = np.asarray(h0_superlevel_persistence(values.tolist()), dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required")
    def test_production_metric_matches_reference(self):
        import numpy as np

        from gemsdoe42.metric import distance_weighted_tversky
        from gemsdoe42.metric_reference import distance_weighted_tversky_reference

        prediction = np.array(
            [[0.0, 0.0, 0.25, 0.0], [0.0, 0.8, 0.0, 0.1], [0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        truth = np.zeros(prediction.shape, dtype=bool)
        truth[1, 2] = True
        truth[2, 3] = True
        valid = np.ones(prediction.shape, dtype=bool)
        valid[0, 0] = False
        production = distance_weighted_tversky(prediction, truth, valid)
        reference = distance_weighted_tversky_reference(
            prediction.tolist(), truth.tolist(), valid.tolist()
        )
        for key in ("tp_weighted", "fp_weighted", "fn_weighted", "dti"):
            self.assertAlmostEqual(production[key], reference[key], places=6, msg=key)

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required")
    def test_wph01_runs_on_synthetic_rasters_without_label_input(self):
        import numpy as np

        from gemsdoe42.pipeline import wph01_score

        rng = np.random.default_rng(42)
        row, col = np.mgrid[0:48, 0:48]
        gravity = (col > 23).astype(np.float32) * 10.0
        gravity += rng.normal(0.0, 0.15, gravity.shape).astype(np.float32)
        magnetic = (row > 22).astype(np.float32) * 4.0
        magnetic += rng.normal(0.0, 0.12, magnetic.shape).astype(np.float32)
        footprint = np.ones(gravity.shape, dtype=bool)
        footprint[:2, :2] = False
        score, report = wph01_score(
            gravity,
            magnetic,
            footprint,
            continuation_heights_m=(0, 100, 200),
            smoothing_sigmas_pixels=(1, 2, 4),
        )
        self.assertEqual(score.shape, gravity.shape)
        self.assertTrue(np.isfinite(score).all())
        self.assertGreater(float(score[footprint].max()), 0.0)
        self.assertEqual(float(score[~footprint].max()), 0.0)
        self.assertLessEqual(float(score.max()), 1.0)
        self.assertEqual(report["hypothesis_id"], "WPH-01")
        self.assertEqual(report["score_semantics"], "continuous ranking/confidence, not calibrated probability")

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required")
    def test_multiscale_topology_is_bounded_and_detects_nonessential_peak(self):
        import numpy as np

        from gemsdoe42.topology import multiscale_topology_score

        surface = np.zeros((31, 31), dtype=np.float32)
        surface[8, 8] = 5.0
        surface[22, 22] = 4.0
        valid = np.ones(surface.shape, dtype=bool)
        result = multiscale_topology_score(surface, valid, (0.5, 1.0, 2.0, 4.0))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(float(result.max()), 0.0)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required")
    def test_spatial_holdout_prefers_candidate_at_matched_support(self):
        import numpy as np

        from gemsdoe42.holdout import (
            evaluate_matched_support_holdout,
            four_spatial_fold_masks,
        )

        shape = (80, 80)
        truth = np.zeros(shape, dtype=bool)
        candidate = np.zeros(shape, dtype=np.float32)
        incumbent = np.zeros(shape, dtype=np.float32)
        locations = ((20, 20), (20, 60), (60, 20), (60, 60))
        for row, col in locations:
            truth[row, col] = True
            candidate[row, col] = 1.0
            incumbent[row + 5, col] = 1.0
        valid = np.ones(shape, dtype=bool)
        folds = four_spatial_fold_masks(valid, guard_pixels=3)
        self.assertEqual(len(folds), 4)
        result = evaluate_matched_support_holdout(
            candidate,
            incumbent,
            truth,
            valid,
            fold_domain=valid,
            guard_pixels=3,
            bootstrap_iterations=2000,
        )
        self.assertTrue(result["promotion_gate_passed"])
        self.assertEqual(result["positive_folds"], 4)
        self.assertGreater(result["mean_paired_delta"], 0.0)
        self.assertGreater(result["bootstrap_95_percent_interval"][0], 0.0)
        self.assertEqual(result["mass_sweep_area_fractions"], [0.005, 0.01, 0.02])
        self.assertTrue(all(
            entry["status"] != "evaluated"
            for entry in result["folds"][0]["mass_sweep"]
        ))

    @unittest.skipUnless(HAS_NUMPY and HAS_RASTERIO, "NumPy and Rasterio are required")
    def test_official_input_preflight_selects_metadata_bands_and_checks_labels(self):
        import numpy as np
        import rasterio
        from affine import Affine

        from gemsdoe42.raster import load_aligned_binary_labels, load_aligned_inputs

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "template.tif"
            features = root / "features.tif"
            labels = root / "labels.tif"
            _sample_grid(rasterio, np, template, width=32, height=32)
            profile = {
                "driver": "GTiff",
                "width": 32,
                "height": 32,
                "count": 2,
                "dtype": "float32",
                "crs": "EPSG:32611",
                "transform": Affine(100, 0, 500000, 0, -100, 4100000),
                "nodata": -9999.0,
            }
            gravity = np.arange(1024, dtype=np.float32).reshape(32, 32)
            magnetic = gravity * -0.5
            with rasterio.open(features, "w", **profile) as dataset:
                dataset.write(gravity, 1)
                dataset.write(magnetic, 2)
                dataset.set_band_description(1, "Isostatic gravity anomaly (mGal)")
                dataset.set_band_description(2, "RTP magnetic anomaly (nT)")
            label_data = np.zeros((32, 32), dtype=np.uint8)
            label_data[16, 16] = 1
            with rasterio.open(
                labels, "w", driver="GTiff", width=32, height=32, count=1,
                dtype="uint8", crs="EPSG:32611",
                transform=Affine(100, 0, 500000, 0, -100, 4100000), nodata=255,
            ) as dataset:
                dataset.write(label_data, 1)
            _, _, footprint, report = load_aligned_inputs(features, template)
            truth, label_valid, label_report = load_aligned_binary_labels(labels, template)
            self.assertEqual(report["selected_bands"], {"gravity": 1, "magnetic_rtp": 2})
            self.assertTrue(footprint[1:, 1:].all())
            self.assertTrue(truth[16, 16])
            self.assertTrue(label_valid[16, 16])
            self.assertEqual(label_report["known_fault_pixels"], 1)

    @unittest.skipUnless(HAS_NUMPY and HAS_RASTERIO, "NumPy and Rasterio are required")
    def test_tiff_writer_reopens_and_checks_grid_mask_and_range(self):
        import numpy as np
        import rasterio

        from gemsdoe42.raster import validate_prediction_tiff, write_prediction_tiff

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "template.tif"
            output = root / "prediction.tif"
            _sample_grid(rasterio, np, template, width=20, height=18)
            score = np.full((18, 20), 0.4, dtype=np.float32)
            receipt = write_prediction_tiff(output, score, template)
            self.assertTrue(receipt["mask_matches_template"])
            self.assertEqual(receipt["dtype"], "float32")
            self.assertEqual(receipt["count"], 1)
            self.assertAlmostEqual(receipt["min"], 0.4, places=6)
            reopened = validate_prediction_tiff(output, template)
            self.assertEqual(receipt["sha256"], reopened["sha256"])

    @unittest.skipUnless(HAS_NUMPY and HAS_SCIPY, "NumPy and SciPy are required")
    def test_concentration_reports_support_and_top_mass(self):
        import numpy as np

        from gemsdoe42.diagnostics import concentration_report

        score = np.zeros((10, 10), dtype=np.float32)
        score[2, 2] = 1.0
        score[7, 7] = 0.5
        valid = np.ones(score.shape, dtype=bool)
        report = concentration_report(score, valid)
        self.assertEqual(report["nonzero_pixels"], 2)
        self.assertAlmostEqual(report["nonzero_support_fraction"], 0.02)
        self.assertAlmostEqual(report["top_1_percent_mass_fraction"], 2.0 / 3.0)
        self.assertAlmostEqual(report["top_5_percent_mass_fraction"], 1.0)

    @unittest.skipUnless(HAS_NUMPY and HAS_RASTERIO and HAS_SCIPY, "geospatial dependencies are required")
    def test_prior_audit_flags_identical_raster_and_accepts_verified_manifest(self):
        import csv

        import numpy as np
        import rasterio

        from gemsdoe42.distinctness import audit_prior_correlations
        from gemsdoe42.raster import write_prediction_tiff

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "template.tif"
            candidate_path = root / "candidate.tif"
            prior_path = root / "prior.tif"
            manifest = root / "manifest.csv"
            _sample_grid(rasterio, np, template, width=40, height=40)
            score = np.linspace(0.0, 1.0, 1600, dtype=np.float32).reshape(40, 40)
            write_prediction_tiff(candidate_path, score, template)
            write_prediction_tiff(prior_path, score, template)
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["submission_id", "artifact_path", "sha256"])
                writer.writeheader()
                writer.writerow({
                    "submission_id": "prior-identical",
                    "artifact_path": str(prior_path),
                    "sha256": __import__("hashlib").sha256(prior_path.read_bytes()).hexdigest(),
                })
            report = audit_prior_correlations(
                candidate_path,
                template,
                manifest,
                base_dir=root,
                max_spearman_sample=1000,
            )
            self.assertFalse(report["all_priors_distinct"])
            self.assertEqual(report["failed_prior_ids"], ["prior-identical"])
            self.assertAlmostEqual(report["reports"][0]["pearson_all_pixels"], 1.0, places=8)
            self.assertAlmostEqual(report["reports"][0]["spearman_systematic_sample"], 1.0, places=8)
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["submission_id", "artifact_path", "sha256"])
                writer.writeheader()
                writer.writerow({
                    "submission_id": "prior-missing",
                    "artifact_path": "",
                    "sha256": "",
                })
            with self.assertRaises(FileNotFoundError):
                audit_prior_correlations(
                    candidate_path,
                    template,
                    manifest,
                    base_dir=root,
                    max_spearman_sample=1000,
                )


if __name__ == "__main__":
    unittest.main()
