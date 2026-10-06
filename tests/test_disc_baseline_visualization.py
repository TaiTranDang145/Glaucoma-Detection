import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.visualize_disc_baselines import render_sample


class DiscBaselineVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        self.results_root = self.root / "results"
        self.data_root.mkdir()
        self.results_root.mkdir()
        Image.new("RGB", (10, 12), color=(35, 80, 120)).save(self.data_root / "fundus.png")

        disc = np.zeros((12, 10), dtype=np.uint8)
        cup = np.zeros_like(disc)
        disc[2:10, 1:9] = 255
        cup[4:8, 3:7] = 255
        (self.results_root / "masks").mkdir()
        Image.fromarray(disc).save(self.results_root / "masks" / "od.png")
        Image.fromarray(cup).save(self.results_root / "masks" / "oc.png")
        self.row = pd.Series({
            "image_path": "fundus.png",
            "od_mask_path": "masks/od.png",
            "oc_mask_path": "masks/oc.png",
            "cdr": 0.5,
            "rdr": np.nan,
            "rdr_status": "unverified_mask_level_definition",
        })

    def tearDown(self):
        self.temporary.cleanup()

    def test_render_sample_writes_reopenable_overlay(self):
        output_path = self.root / "visualization.png"

        result_path = render_sample(self.row, self.data_root, self.results_root, output_path)

        self.assertEqual(result_path, output_path)
        self.assertTrue(output_path.is_file())
        with Image.open(output_path) as rendered:
            self.assertEqual(rendered.format, "PNG")
            self.assertGreater(rendered.width, 0)
            self.assertGreater(rendered.height, 0)

    def test_missing_mask_path_fails_with_path_in_message(self):
        row = self.row.copy()
        row["oc_mask_path"] = "masks/missing.png"

        with self.assertRaisesRegex(FileNotFoundError, "masks/missing.png"):
            render_sample(row, self.data_root, self.results_root, self.root / "out.png")

    def test_mask_with_wrong_image_dimensions_fails_clearly(self):
        Image.fromarray(np.zeros((5, 5), dtype=np.uint8)).save(
            self.results_root / "masks" / "od.png"
        )

        with self.assertRaisesRegex(ValueError, "dimensions"):
            render_sample(self.row, self.data_root, self.results_root, self.root / "out.png")


if __name__ == "__main__":
    unittest.main()
