import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_disc_baselines import build_auc_table, save_evaluation_outputs


class DiscBaselineEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.results = pd.DataFrame([
            {"image_path": "a0.png", "domain": "A", "ground_truth": "GON-", "cdr": 0.1, "segmentation_valid": True},
            {"image_path": "a1.png", "domain": "A", "ground_truth": "GON+", "cdr": 0.9, "segmentation_valid": True},
            {"image_path": "b0.png", "domain": "B", "ground_truth": "GON-", "cdr": 0.8, "segmentation_valid": True},
            {"image_path": "b1.png", "domain": "B", "ground_truth": "GON+", "cdr": 0.2, "segmentation_valid": True},
            {"image_path": "c0.png", "domain": "C", "ground_truth": "GON-", "cdr": 0.1, "segmentation_valid": True},
            {"image_path": "c1.png", "domain": "C", "ground_truth": "GON-", "cdr": 0.3, "segmentation_valid": True},
            {"image_path": "bad.png", "domain": "A", "ground_truth": "GON+", "cdr": np.nan, "segmentation_valid": False},
        ])

    def test_cdr_is_used_directly_as_gon_plus_score_per_domain(self):
        table, curves = build_auc_table(self.results, bootstrap_repetitions=20)

        cdr = table[table["model"] == "CDR"].set_index("domain")
        self.assertEqual(cdr.loc["A", "auc"], 1.0)
        self.assertEqual(cdr.loc["B", "auc"], 0.0)
        self.assertTrue(np.isfinite(cdr.loc["A", "auc_ci_low"]))
        self.assertLessEqual(cdr.loc["A", "auc_ci_low"], cdr.loc["A", "auc_ci_high"])
        self.assertNotIn("C", cdr.index)
        self.assertIn(("CDR", "A"), curves)
        self.assertIn(("CDR", "B"), curves)
        self.assertNotIn(("CDR", "C"), curves)

    def test_probability_join_skips_unmatched_nan_and_single_class_rows(self):
        probabilities = pd.DataFrame([
            {"image_path": "a0.png", "domain": "A", "gonet_prob": 0.2},
            {"image_path": "a1.png", "domain": "A", "gonet_prob": 0.8},
            {"image_path": "b0.png", "domain": "B", "gonet_prob": 0.9},
            {"image_path": "b1.png", "domain": "B", "gonet_prob": 0.1},
            {"image_path": "c0.png", "domain": "C", "gonet_prob": 0.2},
            {"image_path": "c1.png", "domain": "C", "gonet_prob": 0.3},
            {"image_path": "bad.png", "domain": "A", "gonet_prob": np.nan},
            {"image_path": "missing.png", "domain": "A", "gonet_prob": 0.99},
        ])

        table, curves = build_auc_table(
            self.results, probabilities, bootstrap_repetitions=20
        )

        model_rows = table[table["model"] == "GONet/DINOv2"].set_index("domain")
        self.assertEqual(model_rows.loc["A", "auc"], 1.0)
        self.assertEqual(model_rows.loc["A", "n"], 2)
        self.assertEqual(model_rows.loc["B", "auc"], 0.0)
        self.assertNotIn("C", model_rows.index)
        self.assertNotIn(("GONet/DINOv2", "C"), curves)

    def test_evaluation_outputs_include_auc_csv_and_per_domain_roc_plots(self):
        table, curves = build_auc_table(self.results, bootstrap_repetitions=20)
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)

            table_path = save_evaluation_outputs(table, curves, output_dir)

            self.assertEqual(table_path.name, "auc_comparison.csv")
            self.assertTrue(table_path.is_file())
            self.assertTrue((output_dir / "roc_A.png").is_file())
            self.assertTrue((output_dir / "roc_B.png").is_file())
            self.assertFalse((output_dir / "roc_C.png").exists())
            saved_table = pd.read_csv(table_path)
            self.assertEqual(list(saved_table.columns), [
                "model", "domain", "auc", "n", "auc_ci_low", "auc_ci_high"
            ])


if __name__ == "__main__":
    unittest.main()
