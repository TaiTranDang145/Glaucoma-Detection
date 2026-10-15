import copy
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit
import torch
from torch import nn

from src.clinical_protocol import (
    PREPROCESS, ClinicalBundle, ClinicalImageDataset, FusionHead, aligned_features,
    assign_roles, calibrated_probability, fit_calibrator, inference_inputs, nested_fold_plan,
    prepare_rgb, save_bundle, validate_manifest,
)
from src.clinical_training import fit_fusion, run_seed, train_nested_oof
from src.evaluate_zero_shot_refuge import find_refuge_gt_mask, run_zero_shot_refuge
from src.models.mobilenetv3 import MobileNetV3Classifier
from src.models.segmenter import MobileNetV3UNet
from src.train_segmenter import compute_predicted_cdr


def cohort(count=20, split="train", prefix="P"):
    return pd.DataFrame([{"image_id": f"{prefix}{patient}_{eye}", "patient_id": f"{prefix}{patient}",
                          "label": patient % 2, "split": split, "role": split,
                          "path": f"{prefix}{patient}_{eye}.png"}
                         for patient in range(count) for eye in ["OD", "OS"]])


def all_roles():
    frame = pd.concat([cohort(20), cohort(6, "val", "V"), cohort(6, "calibration", "C"),
                       cohort(6, "test", "T")], ignore_index=True)
    frame.loc[frame.role == "calibration", "split"] = "train"
    return frame


def config(mask_dir):
    return {"device": "cpu", "batch_size": 8, "workers": 0, "mask_dir": str(mask_dir),
            "segmenter_lr": .001, "global_lr": .0001, "head_lr": .001,
            "weight_decay": .01, "segmenter_epochs": 2, "head_epochs": 2, "pretrained": False}


class ClinicalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_patient_overlap_rejected(self):
        frame = pd.concat([cohort(), cohort(6, "val", "V"), cohort(6, "test", "T")], ignore_index=True)
        validate_manifest(frame)
        frame.loc[frame.index[-1], "patient_id"] = frame.iloc[0].patient_id
        with self.assertRaisesRegex(ValueError, "Patient overlap"):
            validate_manifest(frame)

    def test_calibration_reserved_before_training_and_fold_plan_is_disjoint(self):
        frame = pd.concat([cohort(40), cohort(8, "val", "V"), cohort(8, "test", "T")], ignore_index=True)
        assigned = assign_roles(frame)
        pd.testing.assert_frame_equal(assigned, assign_roles(frame))
        calibration = set(assigned.loc[assigned.role == "calibration", "patient_id"])
        train = assigned[assigned.role == "train"]
        plans = nested_fold_plan(train)
        held_ids = []
        lookup = train.set_index("image_id")
        for plan in plans:
            held = set(lookup.loc[plan["heldout_ids"], "patient_id"])
            selected = set(lookup.loc[plan["selection_val_ids"], "patient_id"])
            fitted = set(lookup.loc[plan["selection_train_ids"], "patient_id"])
            self.assertFalse(held & (selected | fitted | calibration))
            self.assertFalse(selected & fitted)
            self.assertEqual(set(plan["selection_train_ids"]) | set(plan["selection_val_ids"]), set(plan["refit_train_ids"]))
            held_ids += plan["heldout_ids"]
        self.assertCountEqual(held_ids, train.image_id)
        with self.assertRaisesRegex(ValueError, "only model-training"):
            nested_fold_plan(assigned)

    def test_feature_alignment_checks_provenance_labels_and_duplicate_ids(self):
        frame = cohort(2)
        cache = {"image_ids": frame.image_id[::-1].tolist(), "features": torch.arange(12.).reshape(4, 3),
                 "checkpoint_sha256": "expected", "preprocessing": PREPROCESS,
                 "row_metadata": frame.iloc[::-1][["image_id", "patient_id", "label", "role"]].to_dict("records")}
        torch.testing.assert_close(aligned_features(cache, frame, "expected"), cache["features"].flip(0))
        with self.assertRaisesRegex(ValueError, "provenance"):
            aligned_features(cache, frame, "wrong")
        bad = copy.deepcopy(cache)
        bad["image_ids"][0] = bad["image_ids"][1]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            aligned_features(bad, frame, "expected")
        bad = copy.deepcopy(cache)
        bad["row_metadata"][0]["label"] = 1 - bad["row_metadata"][0]["label"]
        with self.assertRaisesRegex(ValueError, "metadata mismatch: label"):
            aligned_features(bad, frame, "expected")

    def test_train_and_inference_use_identical_geometry_and_normalization(self):
        rgb = np.full((19, 35, 3), [20, 90, 160], dtype=np.uint8)
        prepared = prepare_rgb(rgb, 224)
        self.assertTrue((prepared[:45] == 0).all())
        g, s = inference_inputs(rgb)
        self.assertEqual(g.shape, (3, 224, 224))
        self.assertEqual(s.shape, (3, 256, 256))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.fromarray(rgb).save(path)
            frame = pd.DataFrame([{"image_id": "image", "path": str(path), "label": 0}])
            torch.testing.assert_close(ClinicalImageDataset(frame, "global")[0]["image"], g, rtol=0, atol=0)
            torch.testing.assert_close(ClinicalImageDataset(frame, "segmenter")[0]["image"], s, rtol=0, atol=0)

    def test_calibration_rejects_training_patients(self):
        frame = all_roles()
        train, val, cal = [frame[frame.role == role] for role in ["train", "val", "calibration"]]
        calibration = fit_calibrator(np.linspace(-1, 1, len(cal)), cal, train, val)
        self.assertCountEqual(calibration["fit_image_ids"], cal.image_id)
        bad = cal.copy()
        bad.loc[bad.index[0], "patient_id"] = train.iloc[0].patient_id
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_calibrator(np.zeros(len(bad)), bad, train, val)

    def test_test_labels_cannot_affect_head_scaler_calibrator_or_threshold(self):
        frame = all_roles()
        rng = np.random.default_rng(7)
        features = torch.tensor(rng.normal(size=(len(frame), 8)), dtype=torch.float32)
        cdr = pd.DataFrame({"image_id": frame.image_id, "cdr_pred": rng.uniform(.1, .8, len(frame))})
        first = fit_fusion(frame, features, cdr, config("unused"), 42)
        changed = frame.copy()
        changed.loc[changed.role == "test", "label"] = 1 - changed.loc[changed.role == "test", "label"]
        second = fit_fusion(changed, features, cdr, config("unused"), 42)
        for key, value in first[0].state_dict().items():
            torch.testing.assert_close(value, second[0].state_dict()[key], rtol=0, atol=0)
        self.assertEqual(first[1:4], second[1:4])
        self.assertCountEqual(first[2]["fit_image_ids"], frame.loc[frame.role == "calibration", "image_id"])

    def test_nested_training_exports_inner_selection_and_outer_refit_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = root / "masks"
            masks.mkdir()
            train = cohort(20)
            rgb = np.full((32, 48, 3), 90, np.uint8)
            mask = np.zeros((256, 256), np.uint8)
            mask[80:180, 90:170] = 255
            for index, row in train.iterrows():
                path = root / row.path
                Image.fromarray(rgb).save(path)
                train.loc[index, "path"] = str(path)
                for structure in ["disc", "cup"]:
                    cv2.imwrite(str(masks / f"{row.image_id}_{structure}_consensus.png"), mask)
            plans = nested_fold_plan(train, outer_folds=2, inner_folds=2)
            out = root / "oof"
            # Exercise actual gradient updates and checkpoint selection cheaply.
            predictions = train_nested_oof(train, plans, config(masks), out, 42,
                                            factory=lambda: nn.Conv2d(3, 2, kernel_size=1))
            self.assertCountEqual(predictions.image_id, train.image_id)
            for plan in plans:
                selected = torch.load(out / f"fold_{plan['fold']}" / "selection.pt", weights_only=True)
                refit = torch.load(out / f"fold_{plan['fold']}" / "refit.pt", weights_only=True)
                self.assertFalse(set(selected["metadata"]["selection"]["image_ids"]) & set(plan["heldout_ids"]))
                self.assertEqual(refit["metadata"]["train"]["image_ids"], plan["refit_train_ids"])
                self.assertIsNone(refit["metadata"]["selection"])
                self.assertEqual(refit["metadata"]["selected_epoch"], selected["metadata"]["selected_epoch"])
                self.assertEqual(refit["metadata"]["schedule_horizon"], config(masks)["segmenter_epochs"])
            bad = copy.deepcopy(plans)
            bad[0]["selection_val_ids"] += [bad[0]["heldout_ids"][0]]
            with self.assertRaisesRegex(ValueError, "overlap"):
                train_nested_oof(train, bad, config(masks), root / "invalid", 42,
                                 factory=lambda: nn.Conv2d(3, 2, kernel_size=1))

    def test_real_model_bundle_roundtrip_matches_manual_full_pipeline(self):
        torch.manual_seed(8)
        global_model = MobileNetV3Classifier(pretrained=False).eval()
        segmenter = MobileNetV3UNet(pretrained=False).eval()
        head = FusionHead().eval()
        frame = all_roles()
        roles = {role: {"image_ids": part.image_id.tolist(), "patient_ids": part.patient_id.unique().tolist()}
                 for role, part in frame.groupby("role")}
        scaler = {"mean": .4, "scale": .2, "feature": "cdr_pred_oof", "fit_image_ids": roles["train"]["image_ids"]}
        calibrator = {"method": "platt_logistic_regression", "coefficient": .7, "intercept": -.3,
                      "fit_image_ids": roles["calibration"]["image_ids"], "fit_patient_ids": roles["calibration"]["patient_ids"]}
        rgb = np.random.default_rng(4).integers(0, 256, (160, 240, 3), dtype=np.uint8)
        g, s = inference_inputs(rgb)
        with torch.inference_mode():
            probs = torch.sigmoid(segmenter(s[None]))[0].numpy()
            cdr = compute_predicted_cdr(probs[0] >= .5, probs[1] >= .5)[0]
            z = torch.cat([global_model.extract_features(g[None]), torch.tensor([[(cdr - .4) / .2]], dtype=torch.float32)], 1)
            logit = head(z).item()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.pt"
            payload = save_bundle(path, global_model, segmenter, head, scaler, calibrator, .45,
                                  {"seed": 8, "roles": roles, "training_complete": True, "purpose": "unit_test_random_weights"})
            bundle = ClinicalBundle.load(path)
            actual = bundle.predict_rgb(rgb)
            self.assertAlmostEqual(actual["logit"], logit, places=6)
            self.assertAlmostEqual(actual["cdr_pred"], cdr, places=6)
            self.assertAlmostEqual(actual["prob_raw"], expit(logit), places=6)
            self.assertAlmostEqual(actual["prob_calibrated"], calibrated_probability(logit, calibrator), places=6)
            self.assertEqual(actual["prediction_binary"], int(expit(logit) >= .45))
            with self.assertRaisesRegex(ValueError, "224"):
                bundle.predict_tensors(s[None], s[None])
            bad = copy.deepcopy(payload)
            bad["calibrator"]["fit_patient_ids"] = [*bad["calibrator"]["fit_patient_ids"], roles["train"]["patient_ids"][0]]
            with self.assertRaisesRegex(ValueError, "Calibrator provenance"):
                ClinicalBundle(bad)
        with self.assertRaisesRegex(ValueError, "clinical_oof_v2"):
            ClinicalBundle({"model_state_dict": {}})

    def test_training400_mask_lookup_includes_diagnosis_subdirectories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "REFUGE/Refuge/Annotation-Training400/Disc_Cup_Masks/Glaucoma/g0001.bmp"
            path.parent.mkdir(parents=True)
            Image.new("L", (20, 20), 255).save(path)
            row = {"image_path": "REFUGE/Refuge/Training400/Glaucoma/g0001.jpg", "subfolder": "Training400"}
            self.assertEqual(find_refuge_gt_mask(row, root), path)

    def test_full_synthetic_training_export_reload_and_external_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = root / "masks"
            masks.mkdir()
            frame = all_roles()
            for index, row in frame.iterrows():
                rgb = np.full((32, 48, 3), 70 + row.label * 40, np.uint8)
                path = root / row.path
                Image.fromarray(rgb).save(path)
                frame.loc[index, "path"] = str(path)
                disc, cup = np.zeros((256, 256), np.uint8), np.zeros((256, 256), np.uint8)
                disc[80:180, 90:170], cup[110:150, 110:150] = 255, 255
                for structure, mask in [("disc", disc), ("cup", cup)]:
                    cv2.imwrite(str(masks / f"{row.image_id}_{structure}_consensus.png"), mask)
            settings = {**config(masks), "segmenter_epochs": 1, "global_epochs": 1, "head_epochs": 1}
            plans = nested_fold_plan(frame[frame.role == "train"], 2, 2)
            output = root / "run"
            summary = run_seed(frame, plans, settings, output, 42, {"purpose": "synthetic_integration_only"})
            self.assertTrue(summary["reload_check_passed"])
            for name in ["global.pt", "segmenter.pt", "head.pt", "bundle.pt", "bundle.json", "predictions.csv", "summary.json"]:
                self.assertTrue((output / name).is_file(), name)
            payload = torch.load(output / "bundle.pt", weights_only=True)
            self.assertCountEqual(payload["calibrator"]["fit_image_ids"], frame.loc[frame.role == "calibration", "image_id"])
            external_rows = []
            for label in [0, 1]:
                relative = f"REFUGE/Refuge/Test400/T{label}.png"
                path = root / "external_data" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (30, 40), color=(50 + label * 60, 90, 120)).save(path)
                external_rows.append({"image_path": relative, "domain": "REFUGE", "label": label})
            manifest = root / "external.csv"
            pd.DataFrame(external_rows).to_csv(manifest, index=False)
            external = root / "external_results"
            result = run_zero_shot_refuge(output / "bundle.pt", external, manifest, root / "external_data")
            self.assertEqual(result["partitions"]["all"]["n"], 2)
            self.assertEqual(result["bundle_sha256"], summary["bundle_sha256"])
            predicted = pd.read_csv(external / "predictions.csv")
            np.testing.assert_allclose(predicted.prob_calibrated,
                calibrated_probability(predicted.logit, payload["calibrator"]), atol=1e-7)
            np.testing.assert_array_equal(predicted.prediction_binary,
                (predicted.prob_raw >= payload["threshold"]["value"]).astype(int))
            with self.assertRaises(FileExistsError):
                run_zero_shot_refuge(output / "bundle.pt", external, manifest, root / "external_data")


if __name__ == "__main__":
    unittest.main()
