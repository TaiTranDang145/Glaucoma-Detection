import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

from src.infer_disc_baselines import load_segmenter, run_inference


class StaticSegmenter:
    def __init__(self, disc, cup):
        self.disc = disc
        self.cup = cup

    def predict_masks(self, image_rgb):
        return self.disc, self.cup


class DiscBaselineInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        self.data_root.mkdir()
        self.image_path = self.data_root / "sample.png"
        Image.new("RGB", (10, 12), color=(20, 40, 60)).save(self.image_path)
        self.manifest_path = self.root / "manifest.csv"
        pd.DataFrame([{
            "image_path": "sample.png",
            "domain": "TEST",
            "label": 1,
            "patient_id": "",
        }]).to_csv(self.manifest_path, index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def _masks(self):
        disc = np.zeros((12, 10), dtype=np.uint8)
        cup = np.zeros_like(disc)
        disc[2:10, 1:9] = 1
        cup[4:8, 3:7] = 1
        return disc, cup

    def test_run_inference_saves_features_and_masks(self):
        disc, cup = self._masks()
        output_dir = self.root / "results"

        csv_path = run_inference(
            self.manifest_path, self.data_root, output_dir, StaticSegmenter(disc, cup)
        )

        row = pd.read_csv(csv_path).iloc[0]
        self.assertEqual(len(pd.read_csv(csv_path)), 1)
        self.assertEqual(row["ground_truth"], "GON+")
        self.assertTrue(pd.isna(row["patient_id"]))
        self.assertEqual(row["cdr"], 0.5)
        self.assertTrue(np.isnan(row["rdr"]))
        self.assertEqual(row["rdr_status"], "unverified_mask_level_definition")
        self.assertTrue(row["segmentation_valid"])
        self.assertEqual(row["segmentation_status"], "ok")
        for column, expected in (("od_mask_path", disc), ("oc_mask_path", cup)):
            mask_path = output_dir / row[column]
            self.assertTrue(mask_path.is_file())
            mask = np.asarray(Image.open(mask_path)) > 0
            np.testing.assert_array_equal(mask, expected > 0)

    def test_invalid_segmenter_masks_create_invalid_result_row(self):
        disc, _ = self._masks()
        cup = np.zeros((8, 8), dtype=np.uint8)
        cup[2:6, 2:6] = 1

        csv_path = run_inference(
            self.manifest_path,
            self.data_root,
            self.root / "invalid-results",
            StaticSegmenter(disc, cup),
        )

        row = pd.read_csv(csv_path).iloc[0]
        self.assertFalse(row["segmentation_valid"])
        self.assertTrue(np.isnan(row["cdr"]))
        self.assertEqual(row["segmentation_status"], "shape_mismatch")

    def test_nonfinite_mask_values_are_marked_invalid_and_not_saved(self):
        disc, cup = self._masks()
        disc = disc.astype(np.float32)
        disc[2, 1] = np.nan

        csv_path = run_inference(
            self.manifest_path,
            self.data_root,
            self.root / "nonfinite-results",
            StaticSegmenter(disc, cup),
        )

        row = pd.read_csv(csv_path).iloc[0]
        self.assertFalse(row["segmentation_valid"])
        self.assertTrue(np.isnan(row["cdr"]))
        self.assertEqual(row["segmentation_status"], "invalid_mask_values")
        self.assertTrue(pd.isna(row["od_mask_path"]))

    def test_same_stem_with_different_extensions_gets_distinct_mask_paths(self):
        Image.new("RGB", (10, 12), color=(100, 40, 20)).save(self.data_root / "sample.jpg")
        pd.DataFrame([
            {"image_path": "sample.png", "domain": "TEST", "label": 1, "patient_id": ""},
            {"image_path": "sample.jpg", "domain": "TEST", "label": 0, "patient_id": ""},
        ]).to_csv(self.manifest_path, index=False)
        disc, cup = self._masks()

        csv_path = run_inference(
            self.manifest_path,
            self.data_root,
            self.root / "colliding-results",
            StaticSegmenter(disc, cup),
        )

        rows = pd.read_csv(csv_path)
        self.assertEqual(rows["od_mask_path"].nunique(), 2)
        self.assertEqual(rows["oc_mask_path"].nunique(), 2)

    def test_missing_factory_is_reported(self):
        cfg = OmegaConf.create({"segmenter": {"factory": "", "checkpoint": ""}})

        with self.assertRaisesRegex(ValueError, "segmenter.factory"):
            load_segmenter(cfg)

    def test_missing_checkpoint_is_reported(self):
        cfg = OmegaConf.create({
            "segmenter": {"factory": "some.module:make_segmenter", "checkpoint": ""}
        })

        with self.assertRaisesRegex(ValueError, "segmenter.checkpoint"):
            load_segmenter(cfg)


if __name__ == "__main__":
    unittest.main()
