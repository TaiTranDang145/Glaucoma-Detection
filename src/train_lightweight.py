"""Unified training and evaluation pipeline for Lightweight MobileNetV3 GON models.

Strictly adheres to:
1. Patient-stratified split from papila_morphology.csv (no patient leakage).
2. Exactly identical augmentations, optimizer, scheduler, and selection criterion for Global (M2) and Local (M3).
3. Checkpoint selection strictly on validation set.
4. Classification threshold tuned strictly on validation set via Youden's J.
5. Calibrated Brier score fitted on train set via Platt scaling.
6. Both eye-level and patient-clustered bootstrap 95% CIs reported on the frozen test set.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.mobilenetv3 import MobileNetV3Classifier


# ── Reproducibility ──────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ── Dataset & Transforms ─────────────────────────────────────────────────────

def get_transforms(split: str, image_size: int = 224) -> A.Compose:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_CONSTANT,
                interpolation=cv2.INTER_LINEAR,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.1,
                p=0.5,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


class FundusDataset(Dataset):
    """Square-padded fundus image dataset."""

    def __init__(self, df: pd.DataFrame, image_col: str, transform: A.Compose):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row[self.image_col]

        with Image.open(img_path) as source:
            image = source.convert("RGB")

        # Square padding to preserve 1:1 circular anatomy without distortion
        w, h = image.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(image, ((side - w) // 2, (side - h) // 2))
        np_img = np.array(canvas)

        if self.transform:
            np_img = self.transform(image=np_img)["image"]

        return {
            "image": np_img,
            "label": torch.tensor(row["label"], dtype=torch.float32),
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "eye": row["eye"],
        }


# ── Bootstrap Confidence Interval Calculations ───────────────────────────────

def compute_bootstrap_cis(
    df_eval: pd.DataFrame,
    score_col: str,
    n_iterations: int = 2000,
    seed: int = 42,
) -> Tuple[List[float], List[float]]:
    """Computes both eye-level and patient-clustered bootstrap 95% CIs for ROC-AUC."""
    # 1. Eye-level bootstrap
    np.random.seed(seed)
    n_eyes = len(df_eval)
    eye_aucs = []
    for _ in range(n_iterations):
        idx = np.random.choice(n_eyes, size=n_eyes, replace=True)
        sample = df_eval.iloc[idx]
        if len(sample["label"].unique()) > 1:
            eye_aucs.append(roc_auc_score(sample["label"], sample[score_col]))
    ci_eye = np.percentile(eye_aucs, [2.5, 97.5]).round(4).tolist()

    # 2. Patient-clustered bootstrap
    np.random.seed(seed)
    patients = df_eval["patient_id"].unique()
    n_patients = len(patients)
    cluster_aucs = []
    for _ in range(n_iterations):
        sampled_patients = np.random.choice(patients, size=n_patients, replace=True)
        sampled_rows = pd.concat(
            [df_eval[df_eval["patient_id"] == p] for p in sampled_patients],
            ignore_index=True,
        )
        if len(sampled_rows["label"].unique()) > 1:
            cluster_aucs.append(roc_auc_score(sampled_rows["label"], sampled_rows[score_col]))
    ci_cluster = np.percentile(cluster_aucs, [2.5, 97.5]).round(4).tolist()

    return ci_eye, ci_cluster


# ── Inference Helper ─────────────────────────────────────────────────────────

def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """Runs deterministic inference and gathers logits, probs, and metadata."""
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            image_ids = batch["image_id"]
            patient_ids = batch["patient_id"]
            eyes = batch["eye"]

            logits = model(imgs).squeeze(-1).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))

            for i in range(len(labels)):
                records.append({
                    "image_id": image_ids[i],
                    "patient_id": patient_ids[i],
                    "eye": eyes[i],
                    "label": int(labels[i]),
                    "logit": float(logits[i]),
                    "prob_raw": float(probs[i]),
                })
    return pd.DataFrame(records)


# ── Main Training Routine ────────────────────────────────────────────────────

def train_and_evaluate(args):
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"=== Running Lightweight GON Experiment: {args.branch.upper()} ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Determine column
    image_col = "path" if args.branch == "global" else "crop_path"
    branch_name = "M2_global" if args.branch == "global" else "M3_local"

    # Load manifest
    df_manifest = pd.read_csv(args.manifest)
    train_df = df_manifest[df_manifest["split"] == "train"].reset_index(drop=True)
    val_df = df_manifest[df_manifest["split"] == "val"].reset_index(drop=True)
    test_df = df_manifest[df_manifest["split"] == "test"].reset_index(drop=True)

    print(f"Dataset summary from {args.manifest}:")
    print(f"  Train: {len(train_df)} images ({sum(train_df['label'] == 1)} GON+, {sum(train_df['label'] == 0)} GON-)")
    print(f"  Val:   {len(val_df)} images ({sum(val_df['label'] == 1)} GON+, {sum(val_df['label'] == 0)} GON-)")
    print(f"  Test:  {len(test_df)} images ({sum(test_df['label'] == 1)} GON+, {sum(test_df['label'] == 0)} GON-)")

    # DataLoaders
    train_tf = get_transforms("train", args.image_size)
    val_tf = get_transforms("val", args.image_size)

    train_ds = FundusDataset(train_df, image_col, train_tf)
    val_ds = FundusDataset(val_df, image_col, val_tf)
    test_ds = FundusDataset(test_df, image_col, val_tf)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # Initialize model
    model = MobileNetV3Classifier(
        model_name="mobilenetv3_small_100",
        pretrained=True,
        dropout_rate=args.dropout,
        num_classes=1,
    ).to(device)

    params_info = model.count_parameters()
    print(f"Model: mobilenetv3_small_100 | Total params: {params_info['total']:,} | Trainable: {params_info['trainable']:,}")

    # Loss with positive class weight
    n_neg = (train_df["label"] == 0).sum()
    n_pos = (train_df["label"] == 1).sum()
    pos_weight_val = n_neg / n_pos
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"BCEWithLogitsLoss pos_weight: {pos_weight_val:.4f} (neg: {n_neg}, pos: {n_pos})")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Create output directories
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best_model.pt"

    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    history = []

    print(f"\nStarting training (Selection criterion: {args.criterion})...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        scheduler.step()
        train_loss /= n_train_batches

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        val_logits_list = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)
                logits = model(imgs)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()
                n_val_batches += 1
                val_logits_list.extend(logits.squeeze(-1).cpu().numpy())
                val_labels_list.extend(labels.squeeze(-1).cpu().numpy())

        val_loss /= n_val_batches
        val_probs = 1.0 / (1.0 + np.exp(-np.array(val_logits_list)))
        val_auc = float(roc_auc_score(val_labels_list, val_probs))
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_auc": round(val_auc, 4),
            "lr": current_lr,
        })

        if args.criterion == "val_auc":
            is_best = (val_auc > best_val_auc) or (np.isclose(val_auc, best_val_auc) and val_loss < best_val_loss)
        else:
            is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_auc": val_auc,
                "args": vars(args),
            }, best_ckpt_path)

        if epoch % 5 == 0 or epoch == args.epochs or is_best:
            mark = " [*BEST*]" if is_best else ""
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | LR: {current_lr:.6f}{mark}")

    # Save history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed! Best checkpoint at epoch {best_epoch:02d} with Val Loss: {best_val_loss:.4f}, Val AUC: {best_val_auc:.4f}")

    # ── Validation-Guided Threshold Tuning ───────────────────────────────────
    print("\n--- Phase 2: Validation-Guided Threshold Optimization ---")
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_preds_df = run_inference(model, val_loader, device)
    fpr, tpr, thresholds = roc_curve(val_preds_df["label"], val_preds_df["prob_raw"])
    youdens_j = tpr - fpr
    best_j_idx = int(np.argmax(youdens_j))
    optimal_thresh = float(thresholds[best_j_idx])
    val_sens = float(tpr[best_j_idx])
    val_spec = float(1.0 - fpr[best_j_idx])

    print(f"Optimal Threshold (Youden's J on Val): {optimal_thresh:.4f}")
    print(f"Val Sensitivity: {val_sens:.4f} | Val Specificity: {val_spec:.4f} | Val Youden's J: {youdens_j[best_j_idx]:.4f}")

    # ── Platt Scaling Calibration Fitting (on Train) ─────────────────────────
    print("\n--- Phase 3: Platt Scaling Calibration Fitting (on Train) ---")
    train_eval_loader = DataLoader(
        FundusDataset(train_df, image_col, val_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    train_preds_df = run_inference(model, train_eval_loader, device)
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(train_preds_df[["logit"]], train_preds_df["label"])
    print(f"Platt Scaling fitted: slope={calibrator.coef_[0][0]:.4f}, intercept={calibrator.intercept_[0]:.4f}")

    # ── Frozen Test Set Evaluation ───────────────────────────────────────────
    print("\n--- Phase 4: Frozen Test Set Evaluation ---")
    test_preds_df = run_inference(model, test_loader, device)

    # Apply calibrator
    test_cal_probs = calibrator.predict_proba(test_preds_df[["logit"]])[:, 1]
    test_preds_df["prob_calibrated"] = test_cal_probs

    # Apply locked validation threshold
    test_preds_df["threshold_applied"] = optimal_thresh
    test_preds_df["prediction_binary"] = (test_preds_df["prob_raw"] >= optimal_thresh).astype(int)

    # Compute Test Metrics
    test_auc = float(roc_auc_score(test_preds_df["label"], test_preds_df["prob_raw"]))
    ci_eye, ci_cluster = compute_bootstrap_cis(test_preds_df, score_col="prob_raw", n_iterations=2000, seed=args.seed)

    tn, fp, fn, tp = confusion_matrix(test_preds_df["label"], test_preds_df["prediction_binary"]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc = float((tp + tn) / len(test_preds_df))
    brier_raw = float(brier_score_loss(test_preds_df["label"], test_preds_df["prob_raw"]))
    brier_cal = float(brier_score_loss(test_preds_df["label"], test_preds_df["prob_calibrated"]))

    print(f"Test ROC-AUC:               {test_auc:.4f}")
    print(f"Eye-level 95% CI:           [{ci_eye[0]:.4f}, {ci_eye[1]:.4f}]")
    print(f"Patient-clustered 95% CI:   [{ci_cluster[0]:.4f}, {ci_cluster[1]:.4f}]")
    print(f"Sensitivity:                {sens:.4f} ({tp}/{tp + fn})")
    print(f"Specificity:                {spec:.4f} ({tn}/{tn + fp})")
    print(f"Accuracy:                   {acc:.4f} ({tp + tn}/{len(test_preds_df)})")
    print(f"Brier Score (Calibrated):   {brier_cal:.4f}")
    print(f"Brier Score (Raw):          {brier_raw:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Save test predictions CSV
    preds_out_path = Path(args.predictions_out)
    preds_out_path.parent.mkdir(parents=True, exist_ok=True)
    test_preds_df.to_csv(preds_out_path, index=False)
    print(f"Saved test predictions to: {preds_out_path}")

    # Save metrics JSON
    metrics_data = {
        "experiment": branch_name,
        "input_type": "full_fundus" if args.branch == "global" else "oracle_od_crop",
        "model_architecture": "mobilenetv3_small_100",
        "parameters": params_info,
        "training_setup": {
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pos_weight": float(pos_weight_val),
            "selection_criterion": f"best_{args.criterion}",
            "best_val_loss": round(best_val_loss, 4),
            "best_val_auc": round(best_val_auc, 4),
        },
        "threshold_selection": {
            "selected_on": "validation_set",
            "criterion": "maximum_youdens_j",
            "threshold": round(optimal_thresh, 4),
            "validation_sensitivity": round(val_sens, 4),
            "validation_specificity": round(val_spec, 4),
        },
        "test_results": {
            "n_test_images": len(test_preds_df),
            "n_test_patients": int(test_preds_df["patient_id"].nunique()),
            "gon_plus_count": int(test_preds_df["label"].sum()),
            "gon_minus_count": int((test_preds_df["label"] == 0).sum()),
            "roc_auc": round(test_auc, 4),
            "bootstrap_95ci_eye_level": ci_eye,
            "bootstrap_95ci_patient_clustered": ci_cluster,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "accuracy": round(acc, 4),
            "brier_score_calibrated": round(brier_cal, 4),
            "brier_score_raw": round(brier_raw, 4),
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            },
        },
        "platt_calibrator": {
            "slope": round(float(calibrator.coef_[0][0]), 4),
            "intercept": round(float(calibrator.intercept_[0]), 4),
        },
    }

    metrics_out_path = Path(args.metrics_out)
    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics JSON to: {metrics_out_path}")

    return metrics_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lightweight MobileNetV3 for GON Classification")
    parser.add_argument("--branch", choices=["global", "local"], required=True, help="'global' (M2: Full Fundus) or 'local' (M3: OD Crop)")
    parser.add_argument("--manifest", default="data/manifests/papila_morphology.csv", help="Path to manifest CSV")
    parser.add_argument("--image-size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--criterion", choices=["val_auc", "val_loss"], default="val_auc", help="Model checkpoint selection criterion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--output-dir", required=True, help="Directory to save model checkpoints and logs")
    parser.add_argument("--predictions-out", required=True, help="Path to save test predictions CSV")
    parser.add_argument("--metrics-out", required=True, help="Path to save test metrics JSON")

    args = parser.parse_args()
    train_and_evaluate(args)
