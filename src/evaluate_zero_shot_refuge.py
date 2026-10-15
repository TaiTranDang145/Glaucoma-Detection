"""Evaluate REFUGE using an explicitly supplied clinical_oof_v2 bundle.

The evaluator never fits a scaler, calibrator, threshold, or model on REFUGE.
Historical exploratory results remain in results/refuge_zero_shot/.
"""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from src.clinical_protocol import (
    ClinicalBundle, ClinicalImageDataset, PREPROCESS, environment_info, sha256_file, write_json,
)
from src.clinical_training import metrics


def get_eval_transforms(target_size=256):
    """Legacy direct-resize helper retained only for replaying the historical audit."""
    return A.Compose([A.Resize(target_size, target_size, interpolation=cv2.INTER_LINEAR),
                      A.Normalize(mean=PREPROCESS["mean"], std=PREPROCESS["std"]), ToTensorV2()])


def find_refuge_gt_mask(row, data_root=Path("data")):
    root = Path(data_root) / "REFUGE/Refuge"
    name = Path(row["image_path"]).stem + ".bmp"
    folder = row["subfolder"]
    if folder == "REFUGE-Validation400":
        candidates = [root / "REFUGE-Validation400-GT/REFUGE-Validation400-GT/Disc_Cup_Masks" / name]
    elif folder == "Test400":
        candidates = [root / "REFUGE-Test-GT/Disc_Cup_Masks" / group / name for group in ["G", "N"]]
    else:
        candidates = [root / directory / group / name
                      for directory in ["Annotation-Training400/Disc_Cup_Masks", "Disc_Cup_Masks"]
                      for group in ["Glaucoma", "Non-Glaucoma", ""]]
    return next((p for p in candidates if p.is_file()), None)


def native_cdr(mask):
    def height(binary):
        rows = np.flatnonzero(binary.any(axis=1))
        return int(rows[-1] - rows[0] + 1) if len(rows) else 0
    disc_height, cup_height = height(mask <= 128), height(mask == 0)
    if not disc_height:
        raise ValueError("Empty REFUGE ground-truth disc")
    return cup_height / disc_height


def run_zero_shot_refuge(bundle_path, output_dir, manifest_path=Path("data/manifests/all_public.csv"),
                         data_root=Path("data"), device="cpu", batch_size=16):
    bundle_path, output_dir = Path(bundle_path), Path(output_dir)
    bundle = ClinicalBundle.load(bundle_path, device)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose an empty output directory; historical results are preserved")
    frame = pd.read_csv(manifest_path)
    frame = frame[frame.domain == "REFUGE"].copy().reset_index(drop=True)
    if frame.empty or frame.image_path.duplicated().any():
        raise ValueError("REFUGE manifest is empty or contains duplicate image paths")
    frame["subfolder"] = frame.image_path.map(lambda p: Path(p).parts[2])
    frame["image_id"] = frame.image_path.map(lambda p: Path(p).stem)
    if frame.image_id.duplicated().any():
        raise ValueError("Duplicate REFUGE image IDs")
    frame["path"] = frame.image_path.map(lambda p: str(Path(data_root) / p))
    global_loader = DataLoader(ClinicalImageDataset(frame, "global"), batch_size=batch_size, shuffle=False)
    seg_loader = DataLoader(ClinicalImageDataset(frame, "segmenter"), batch_size=batch_size, shuffle=False)
    batches = []
    for global_batch, seg_batch in zip(global_loader, seg_loader):
        if global_batch["image_id"] != seg_batch["image_id"]:
            raise ValueError("Branch image IDs do not match")
        predicted = bundle.predict_tensors(global_batch["image"], seg_batch["image"])
        batches.append(pd.DataFrame({"image_id": global_batch["image_id"], **predicted}))
    predictions = frame.merge(pd.concat(batches, ignore_index=True), on="image_id", validate="one_to_one")
    predictions["threshold_applied"] = bundle.payload["threshold"]["value"]
    # Ground truth is read only after all predictions have been finalized.
    gt_records = []
    for _, row in predictions.iterrows():
        mask_path = find_refuge_gt_mask(row, data_root)
        if mask_path is None:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or not set(np.unique(mask)).issubset({0, 128, 255}):
            raise ValueError(f"Invalid REFUGE mask: {mask_path}")
        gt_records.append({"image_id": row.image_id, "gt_cdr_native": native_cdr(mask)})
    if gt_records:
        predictions = predictions.merge(pd.DataFrame(gt_records), on="image_id", how="left", validate="one_to_one")
    summary = {"bundle_sha256": sha256_file(bundle_path), "manifest_sha256": sha256_file(manifest_path),
               "bundle_seed": bundle.payload["metadata"]["seed"], "preprocessing": PREPROCESS,
               "threshold": bundle.payload["threshold"], "environment": environment_info(), "partitions": {}}
    parts = [("all", predictions), *predictions.groupby("subfolder")]
    for name, part in parts:
        values = metrics(part)
        if "gt_cdr_native" in part:
            valid = part.dropna(subset=["gt_cdr_native"])
            values["cdr_gt_n"] = len(valid)
            if len(valid):
                values["cdr_mae_native"] = float(np.abs(valid.cdr_pred - valid.gt_cdr_native).mean())
                values["cdr_pearson_r_native"] = (float(pearsonr(valid.cdr_pred, valid.gt_cdr_native).statistic)
                    if len(valid) > 2 and valid.cdr_pred.nunique() > 1 and valid.gt_cdr_native.nunique() > 1 else None)
        summary["partitions"][name] = values
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    write_json(output_dir / "metrics.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/refuge_oof_v2"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/all_public.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    summary = run_zero_shot_refuge(args.bundle, args.output_dir, args.manifest, args.data_root, args.device, args.batch_size)
    print(summary["partitions"])


if __name__ == "__main__":
    main()
