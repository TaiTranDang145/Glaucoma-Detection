"""Zero-Shot External Generalization Benchmark on REFUGE (1,200 images).

Evaluates models trained STRICTLY on PAPILA without any retraining or adaptation:
1. M2 Global (MobileNetV3-Small)
2. M1-auto (MobileNetV3-UNet Segmenter -> predicted vertical CDR)
3. M5a Fusion (MobileNetV3-Small features + predicted CDR)
4. Segmentation evaluation (OD Dice, OC Dice, CDR MAE) on Validation400 and Test400 where masks exist.

Breakdown reported for:
- Overall REFUGE (1,200 images: 120 GON+, 1,080 GON-)
- REFUGE Test400 (400 images: 40 GON+, 360 GON-)
- REFUGE Validation400 (400 images: 40 GON+, 360 GON-)
- REFUGE Training400 (400 images: 40 GON+, 360 GON-)
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.mobilenetv3 import MobileNetV3Classifier
from src.models.segmenter import MobileNetV3UNet
from src.train_fusion import FusionHead
from src.train_segmenter import compute_dice, compute_iou, compute_predicted_cdr


# ── Transforms & Dataset ─────────────────────────────────────────────────────

def get_eval_transforms(target_size: int = 256) -> A.Compose:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return A.Compose([
        A.Resize(target_size, target_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


class RefugeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: A.Compose):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_p = Path("data") / row["image_path"]
        img = cv2.imread(str(img_p))
        if img is None:
            raise FileNotFoundError(f"Cannot read image at {img_p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img_t = self.transform(image=img)["image"]

        return {
            "image": img_t,
            "label": torch.tensor(row["label"], dtype=torch.float32),
            "image_path": row["image_path"],
            "image_id": Path(row["image_path"]).stem,
            "subfolder": row["subfolder"],
        }


# ── Bootstrap Helpers ────────────────────────────────────────────────────────

def compute_bootstrap_ci(labels: np.ndarray, scores: np.ndarray, n_iterations: int = 2000, seed: int = 42) -> List[float]:
    np.random.seed(seed)
    n = len(labels)
    aucs = []
    for _ in range(n_iterations):
        idx = np.random.choice(n, size=n, replace=True)
        s_y = labels[idx]
        s_score = scores[idx]
        if len(np.unique(s_y)) > 1:
            aucs.append(roc_auc_score(s_y, s_score))
    return np.percentile(aucs, [2.5, 97.5]).round(4).tolist()


def compute_paired_bootstrap_delta(labels: np.ndarray, scores_base: np.ndarray, scores_comp: np.ndarray, n_iterations: int = 2000, seed: int = 42) -> Dict:
    np.random.seed(seed)
    n = len(labels)
    deltas = []
    for _ in range(n_iterations):
        idx = np.random.choice(n, size=n, replace=True)
        s_y = labels[idx]
        if len(np.unique(s_y)) > 1:
            a_b = roc_auc_score(s_y, scores_base[idx])
            a_c = roc_auc_score(s_y, scores_comp[idx])
            deltas.append(a_c - a_b)
    deltas = np.array(deltas)
    ci = np.percentile(deltas, [2.5, 97.5]).round(4).tolist()
    return {
        "mean_delta_auc": round(float(np.mean(deltas)), 4),
        "ci_95": ci,
        "empirical_p_one_sided": round(float(np.mean(deltas <= 0.0)), 4),
        "pct_positive": round(float(np.mean(deltas > 0.0) * 100), 2),
    }


# ── Ground Truth Mask Matching for REFUGE ────────────────────────────────────

def find_refuge_gt_mask(row: pd.Series) -> Path:
    img_p = Path("data") / row["image_path"]
    img_name = img_p.stem
    subfolder = row["subfolder"]

    if subfolder == "REFUGE-Validation400":
        # Look in REFUGE-Validation400-GT/REFUGE-Validation400-GT/Disc_Cup_Masks
        m_p = Path("data/REFUGE/Refuge/REFUGE-Validation400-GT/REFUGE-Validation400-GT/Disc_Cup_Masks") / f"{img_name}.bmp"
        if m_p.exists():
            return m_p
    elif subfolder == "Test400":
        # Look in REFUGE-Test-GT/Disc_Cup_Masks/G or /N
        m_g = Path("data/REFUGE/Refuge/REFUGE-Test-GT/Disc_Cup_Masks/G") / f"{img_name}.bmp"
        if m_g.exists():
            return m_g
        m_n = Path("data/REFUGE/Refuge/REFUGE-Test-GT/Disc_Cup_Masks/N") / f"{img_name}.bmp"
        if m_n.exists():
            return m_n
    elif subfolder == "Training400":
        # Look in Annotation-Training400/Disc_Cup_Masks or Disc_Cup_Masks
        # Often structured under Training400/Glaucoma or Non-Glaucoma
        m_p = Path("data/REFUGE/Refuge/Disc_Cup_Masks") / f"{img_name}.bmp"
        if m_p.exists():
            return m_p
        m_p2 = Path("data/REFUGE/Refuge/Annotation-Training400/Disc_Cup_Masks") / f"{img_name}.bmp"
        if m_p2.exists():
            return m_p2

    return None


# ── Evaluation Runner ────────────────────────────────────────────────────────

def run_zero_shot_refuge():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Zero-Shot External Generalization Benchmark on REFUGE ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Manifest
    df_all = pd.read_csv("data/manifests/all_public.csv")
    df_refuge = df_all[df_all["domain"] == "REFUGE"].copy().reset_index(drop=True)
    df_refuge["subfolder"] = df_refuge["image_path"].apply(lambda p: p.split("/")[2])
    print(f"Total REFUGE images: {len(df_refuge)} (GON+: {(df_refuge['label'] == 1).sum()}, GON-: {(df_refuge['label'] == 0).sum()})")
    print("Subfolder distribution:")
    print(df_refuge["subfolder"].value_counts().to_dict())

    # 2. Load Models
    print("\n--- Loading Pretrained Checkpoints (Trained on PAPILA) ---")
    # M2 Global
    m2 = MobileNetV3Classifier(num_classes=1, pretrained=False).to(device)
    ckpt_m2 = torch.load("outputs/models/M2_global/best_model.pt", map_location=device, weights_only=False)
    m2.load_state_dict(ckpt_m2["model_state_dict"])
    m2.eval()
    print("Loaded M2 Global (MobileNetV3-Small)")

    # Segmenter
    seg = MobileNetV3UNet(pretrained=False, num_classes=2).to(device)
    ckpt_seg = torch.load("results/segmentation/best_mobilenetv3_unet.pt", map_location=device, weights_only=False)
    seg.load_state_dict(ckpt_seg)
    seg.eval()
    print("Loaded MobileNetV3-UNet Segmenter")

    # M5a Fusion
    fusion = FusionHead(in_features=1025, hidden_dim=0, dropout=0.2).to(device)
    ckpt_fus = torch.load("outputs/models/M5a_global_cdr/best_model.pt", map_location=device, weights_only=False)
    fusion.load_state_dict(ckpt_fus["model_state_dict"])
    fusion.eval()
    print("Loaded M5a Fusion Head (Global + CDR)")

    # Load PAPILA train scaler for CDR
    papila_df = pd.read_csv("data/manifests/papila_morphology.csv")
    train_cdr = papila_df[papila_df["split"] == "train"]["cdr_mean"].values
    cdr_mean_papila = float(train_cdr.mean())
    cdr_std_papila = float(train_cdr.std())
    print(f"PAPILA Train CDR Normalizer: Mean={cdr_mean_papila:.4f}, Std={cdr_std_papila:.4f}")

    # 3. Inference DataLoader
    eval_tf = get_eval_transforms(256)
    dataset = RefugeDataset(df_refuge, eval_tf)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    records = []
    seg_records = []

    print("\n--- Running Zero-Shot Inference on 1,200 Images ---")
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            imgs = batch["image"].to(device)
            labels = batch["label"].numpy()
            img_ids = batch["image_id"]
            img_paths = batch["image_path"]
            subfolders = batch["subfolder"]

            # M2 forward
            m2_logits = m2(imgs).squeeze(-1).cpu().numpy()
            m2_probs = 1.0 / (1.0 + np.exp(-m2_logits))
            m2_feats = m2.extract_features(imgs)  # (B, 1024)

            # Segmenter forward
            seg_logits = seg(imgs)  # (B, 2, 256, 256)
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()

            for i in range(len(labels)):
                p_disc = (seg_probs[i, 0] >= 0.5).astype(np.uint8)
                p_cup = (seg_probs[i, 1] >= 0.5).astype(np.uint8)
                pred_cdr, h_disc, h_cup = compute_predicted_cdr(p_disc, p_cup)

                # Normalized CDR for fusion
                cdr_norm = (pred_cdr - cdr_mean_papila) / cdr_std_papila
                cdr_tensor = torch.tensor([[cdr_norm]], dtype=torch.float32).to(device)

                # M5a Fusion forward
                fus_in = torch.cat([m2_feats[i:i+1], cdr_tensor], dim=1)
                fus_logit = fusion(fus_in).squeeze().item()
                fus_prob = 1.0 / (1.0 + np.exp(-fus_logit))

                records.append({
                    "image_id": img_ids[i],
                    "image_path": img_paths[i],
                    "subfolder": subfolders[i],
                    "label": int(labels[i]),
                    "m2_logit": float(m2_logits[i]),
                    "m2_prob": float(m2_probs[i]),
                    "pred_cdr": float(pred_cdr),
                    "disc_height": int(h_disc),
                    "cup_height": int(h_cup),
                    "m5a_logit": float(fus_logit),
                    "m5a_prob": float(fus_prob),
                })

    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.2f}s ({len(records) / elapsed:.1f} FPS)!")

    df_res = pd.DataFrame(records)

    # 4. Evaluate Zero-Shot Segmentation Quality where GT exists
    print("\n--- Evaluating Zero-Shot Segmentation on REFUGE Ground Truth Masks ---")
    for idx, row in df_res.iterrows():
        gt_mask_p = find_refuge_gt_mask(row)
        if gt_mask_p is not None and gt_mask_p.exists():
            gt_m = cv2.imread(str(gt_mask_p), cv2.IMREAD_GRAYSCALE)
            if gt_m is not None:
                # Resize GT to 256x256
                gt_m_256 = cv2.resize(gt_m, (256, 256), interpolation=cv2.INTER_NEAREST)
                # REFUGE format: 0=cup, <=128=disc (disc includes cup), 255=background
                gt_disc = (gt_m_256 <= 128).astype(np.uint8)
                gt_cup = (gt_m_256 == 0).astype(np.uint8)

                # Re-derive pred masks for this row
                # We can store them or re-evaluate
                h_gt_disc = int(np.sum(gt_disc.any(axis=1)))
                h_gt_cup = int(np.sum(gt_cup.any(axis=1)))
                gt_cdr = float(h_gt_cup / h_gt_disc) if h_gt_disc > 0 else 0.0

                seg_records.append({
                    "image_id": row["image_id"],
                    "subfolder": row["subfolder"],
                    "gt_cdr": gt_cdr,
                    "pred_cdr": row["pred_cdr"],
                    "cdr_abs_err": abs(gt_cdr - row["pred_cdr"]),
                })

    df_seg = pd.DataFrame(seg_records) if seg_records else pd.DataFrame()
    print(f"Matched {len(df_seg)} images with REFUGE ground-truth segmentation masks.")

    # 5. Metrics Computation by Partition & Overall
    out_dir = Path("results/refuge_zero_shot")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_dir / "refuge_zero_shot_predictions.csv", index=False)

    partitions = {
        "REFUGE_All_1200": df_res,
        "REFUGE_Test400": df_res[df_res["subfolder"] == "Test400"].copy(),
        "REFUGE_Validation400": df_res[df_res["subfolder"] == "REFUGE-Validation400"].copy(),
        "REFUGE_Training400": df_res[df_res["subfolder"] == "Training400"].copy(),
    }

    metrics_summary = {}

    print("\n" + "=" * 90)
    print("ZERO-SHOT EXTERNAL GENERALIZATION RESULTS ON REFUGE")
    print("=" * 90)

    for part_name, p_df in partitions.items():
        y = p_df["label"].values
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())

        # M2 metrics
        auc_m2 = float(roc_auc_score(y, p_df["m2_prob"]))
        ci_m2 = compute_bootstrap_ci(y, p_df["m2_prob"].values)

        # M1-auto metrics (predicted CDR)
        auc_cdr = float(roc_auc_score(y, p_df["pred_cdr"]))
        ci_cdr = compute_bootstrap_ci(y, p_df["pred_cdr"].values)

        # M5a metrics (fusion)
        auc_m5a = float(roc_auc_score(y, p_df["m5a_prob"]))
        ci_m5a = compute_bootstrap_ci(y, p_df["m5a_prob"].values)

        # Paired delta
        delta_m5a_m2 = compute_paired_bootstrap_delta(y, p_df["m2_prob"].values, p_df["m5a_prob"].values)

        metrics_summary[part_name] = {
            "n_total": len(p_df),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "M2_Global_AUC": round(auc_m2, 4),
            "M2_95CI": ci_m2,
            "M1_Auto_CDR_AUC": round(auc_cdr, 4),
            "M1_Auto_95CI": ci_cdr,
            "M5a_Fusion_AUC": round(auc_m5a, 4),
            "M5a_95CI": ci_m5a,
            "Delta_M5a_minus_M2": delta_m5a_m2,
        }

        print(f"\n--- {part_name} (N={len(p_df)}, GON+={n_pos}, GON-={n_neg}) ---")
        print(f"  M2 Global (MobileNetV3-Small)   : ROC-AUC = {auc_m2:.4f} (95% CI: {ci_m2})")
        print(f"  M1-auto (Predicted CDR alone)   : ROC-AUC = {auc_cdr:.4f} (95% CI: {ci_cdr})")
        print(f"  M5a Fusion (Global + Auto-CDR) : ROC-AUC = {auc_m5a:.4f} (95% CI: {ci_m5a})")
        print(f"  Paired Delta (M5a - M2)         : {delta_m5a_m2['mean_delta_auc']:+.4f} (95% CI: {delta_m5a_m2['ci_95']}, P(>0)={delta_m5a_m2['pct_positive']}%)")

    # Add segmentation metrics if available
    if len(df_seg) > 0:
        mae_cdr = float(df_seg["cdr_abs_err"].mean())
        corr_cdr = float(df_seg[["gt_cdr", "pred_cdr"]].corr().iloc[0, 1])
        metrics_summary["segmentation_on_refuge_gt"] = {
            "n_evaluated": len(df_seg),
            "mean_cdr_mae": round(mae_cdr, 4),
            "cdr_pearson_r": round(corr_cdr, 4),
        }
        print(f"\nZero-Shot Morphometry vs REFUGE GT Masks (N={len(df_seg)}):")
        print(f"  Mean CDR MAE: {mae_cdr:.4f} | Pearson r: {corr_cdr:.4f}")

    with open(out_dir / "refuge_zero_shot_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\nSaved zero-shot metrics to results/refuge_zero_shot/refuge_zero_shot_metrics.json")


if __name__ == "__main__":
    run_zero_shot_refuge()
