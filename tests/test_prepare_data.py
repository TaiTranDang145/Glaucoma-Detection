import tempfile
import unittest
from pathlib import Path

from src.prepare_data import build_manifest


class BuildManifestTests(unittest.TestCase):
    def test_uses_available_labeled_domains_when_refuge_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hydr = root / "HYDR"
            (hydr / "Images").mkdir(parents=True)
            (hydr / "Images" / "patient-01.jpg").write_bytes(b"image")
            (hydr / "Labels.csv").write_text(
                "Image Name,Label,Patient\npatient-01.jpg,GON+,patient-01\n",
                encoding="utf-8",
            )

            records = build_manifest(root)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["domain"], "HYRD")
        self.assertEqual(records[0]["label"], 1)
        self.assertEqual(records[0]["patient_id"], "HYRD:patient-01")


if __name__ == "__main__":
    unittest.main()
