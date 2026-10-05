from __future__ import annotations

import json
import unittest
from pathlib import Path

from gemsdoe42.metric_reference import (
    distance_weighted_tversky_reference,
    dti_from_counts,
)
from gemsdoe42.topology_reference import h0_superlevel_persistence


class PreregistrationConsistencyTests(unittest.TestCase):
    def test_public_site_config_matches_canonical_preregistration(self):
        root = Path(__file__).resolve().parents[1]
        canonical = json.loads((root / "configs" / "wph01.json").read_text(encoding="utf-8"))
        public = json.loads((root / "docs" / "wph01-preregistration.json").read_text(encoding="utf-8"))
        self.assertEqual(canonical, public)
        self.assertEqual(canonical["status"], "not_run_data_blocked")
        self.assertEqual(canonical["distinctness_policy"]["max_absolute_pearson"], 0.9)


class PersistenceReferenceTests(unittest.TestCase):
    def test_bridge_kills_younger_peak_at_merge_level(self):
        result = h0_superlevel_persistence([[3.0, 1.0, 2.0]])
        self.assertEqual(result, [[0.0, 0.0, 1.0]])

    def test_global_essential_component_is_omitted(self):
        self.assertEqual(h0_superlevel_persistence([[4.0]]), [[0.0]])

    def test_equal_height_plateau_has_no_positive_lifetime(self):
        self.assertEqual(h0_superlevel_persistence([[1.0, 1.0]]), [[0.0, 0.0]])

    def test_mask_separates_components_and_nonfinite_cells_are_invalid(self):
        result = h0_superlevel_persistence(
            [[3.0, float("nan"), 2.0]],
            [[True, True, True]],
        )
        self.assertEqual(result, [[0.0, 0.0, 0.0]])
        separated = h0_superlevel_persistence([[3.0, 1.0, 2.0]], [[True, False, True]])
        self.assertEqual(separated, [[0.0, 0.0, 0.0]])

    def test_invalid_shapes_fail_loudly(self):
        with self.assertRaises(ValueError):
            h0_superlevel_persistence([[1.0], [1.0, 2.0]])
        with self.assertRaises(ValueError):
            h0_superlevel_persistence([[1.0, 2.0]], [[True]])


class MetricReferenceTests(unittest.TestCase):
    def test_official_weighted_count_example(self):
        score = dti_from_counts(3.0, 1.89, 2.0, alpha=0.2, beta=0.8)
        self.assertAlmostEqual(score, 3.0 / 4.978, places=12)

    def test_perfect_prediction(self):
        result = distance_weighted_tversky_reference(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[False, False, False], [False, True, False], [False, False, False]],
        )
        self.assertAlmostEqual(result["tp_weighted"], 1.0)
        self.assertAlmostEqual(result["fp_weighted"], 0.0)
        self.assertAlmostEqual(result["fn_weighted"], 0.0)
        self.assertAlmostEqual(result["dti"], 1.0, places=10)

    def test_one_pixel_offset_uses_triangular_kernel(self):
        result = distance_weighted_tversky_reference(
            [[0.0, 1.0, 0.0]],
            [[False, False, True]],
            pixel_size_m=100.0,
            radius_m=300.0,
        )
        self.assertAlmostEqual(result["tp_weighted"], 2.0 / 3.0)
        self.assertAlmostEqual(result["fp_weighted"], 1.0 / 3.0)
        self.assertAlmostEqual(result["fn_weighted"], 1.0 / 3.0)
        self.assertAlmostEqual(result["dti"], 2.0 / 3.0, places=10)

    def test_prediction_outside_radius_is_not_a_true_positive(self):
        result = distance_weighted_tversky_reference(
            [[1.0, 0.0, 0.0, 0.0, 0.0]],
            [[False, False, False, False, True]],
            pixel_size_m=100.0,
            radius_m=300.0,
        )
        self.assertEqual(result["tp_weighted"], 0.0)
        self.assertEqual(result["fp_weighted"], 1.0)
        self.assertEqual(result["fn_weighted"], 1.0)
        self.assertEqual(result["dti"], 0.0)

    def test_mask_ignores_invalid_nan_prediction_and_truth(self):
        result = distance_weighted_tversky_reference(
            [[1.0, float("nan")]],
            [[False, True]],
            [[True, False]],
        )
        self.assertEqual(result["tp_weighted"], 0.0)
        self.assertEqual(result["fp_weighted"], 1.0)
        self.assertEqual(result["fn_weighted"], 0.0)

    def test_valid_prediction_values_must_be_bounded(self):
        for value in (float("nan"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                distance_weighted_tversky_reference([[value]], [[False]])

    def test_invalid_metric_coefficients_fail(self):
        with self.assertRaises(ValueError):
            dti_from_counts(1, 0, 0, alpha=-0.1)
        with self.assertRaises(ValueError):
            distance_weighted_tversky_reference([[0]], [[False]], pixel_size_m=0)


if __name__ == "__main__":
    unittest.main()
