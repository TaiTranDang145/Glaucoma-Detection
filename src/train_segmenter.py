"""Train lightweight MobileNetV3-UNet for Optic Disc & Cup segmentation.

Strict protocol:
- Trained strictly on PAPILA Train partition (294 images, 147 patients).
- Checkpoint selection strictly based on Validation mean Dice (62 images, 31 patients).
- Frozen Test partition (64 images, 32 patients) strictly for final evaluation.
- Outputs predicted CDR for all 420 images to data/manifests/papila_predicted_cdr.csv.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.segmenter import MobileNetV3UNet


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


# ── Post-processing & Morphometry ───────────────────────────────────────────

def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component to filter spurious noise using OpenCV."""
    mask_u8 = binary_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return binary_mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    return (labels == largest_label).astype(np.uint8)



def get_vertical_height(mask: np.ndarray) -> int:
    """Computes vertical height in pixels."""
    rows = np.flatnonzero(mask.any(axis=1))
    if len(rows) == 0:
        return 0
    return int(rows[-1] - rows[0] + 1)


def compute_predicted_cdr(pred_disc: np.ndarray, pred_cup: np.ndarray) -> Tuple[float, int, int]:
    """Computes vertical CDR from predicted binary masks with morphological constraints."""
    # 1. Clean disc (largest component)
    disc_clean = keep_largest_component(pred_disc)
    # 2. Constrain cup to lie strictly within disc
    cup_inside = (pred_cup > 0) & (disc_clean > 0)
    cup_clean = keep_largest_component(cup_inside.astype(np.uint8))

    h_disc = get_vertical_height(disc_clean)
    h_cup = get_vertical_height(cup_clean)

    if h_disc == 0:
        return 0.0, 0, 0
    cdr = min(float(h_cup) / float(h_disc), 1.0)
    return cdr, h_disc, h_cup


def compute_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    intersection = np.sum((pred > 0) & (target > 0))
    total = np.sum(pred > 0) + np.sum(target > 0)
    if total == 0:
        return 1.0
    return float((2.0 * intersection) / (total + eps))


def compute_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    intersection = np.sum((pred > 0) & (target > 0))
    union = np.sum((pred > 0) | (target > 0))
    if union == 0:
        return 1.0
    return float(intersection / (union + eps))


def compute_icc_2_1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes two-way random, single measure ICC(2,1)."""
    Y = np.column_stack([y_true, y_pred])
    n, k = Y.shape
    grand_mean = np.mean(Y)
    ss_total = np.sum((Y - grand_mean) ** 2)
    ss_rows = k * np.sum((np.mean(Y, axis=1) - grand_mean) ** 2)
    ss_cols = n * np.sum((np.mean(Y, axis=0) - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error)
    if denom == 0:
        return 0.0
    icc = (ms_rows - ms_error) / denom
    return float(icc)


# ── Loss Functions ──────────────────────────────────────────────────────────

class SoftDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        intersection = torch.sum(probs * targets, dim=dims)
        cardinality = torch.sum(probs * probs + targets * targets, dim=dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return torch.mean(1.0 - dice)


class CombinedBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.w_bce = bce_weight
        self.w_dice = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Separate channels: 0 = OD, 1 = OC
        loss_od = self.w_bce * self.bce(logits[:, 0:1], targets[:, 0:1]) + self.w_dice * self.dice(logits[:, 0:1], targets[:, 0:1])
        loss_oc = self.w_bce * self.bce(logits[:, 1:2], targets[:, 1:2]) + self.w_dice * self.dice(logits[:, 1:2], targets[:, 1:2])
        return loss_od + 1.2 * loss_oc  # Slightly upweight OC because it is smaller


# ── Dataset ──────────────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, mask_dir: Path, transform: A.Compose = None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_id = row["image_id"]

        img_p = self.img_dir / f"{img_id}.png"
        img = cv2.imread(str(img_p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        d_p = self.mask_dir / f"{img_id}_disc_consensus.png"
        c_p = self.mask_dir / f"{img_id}_cup_consensus.png"
        disc = cv2.imread(str(d_p), cv2.IMREAD_GRAYSCALE)
        cup = cv2.imread(str(c_p), cv2.IMREAD_GRAYSCALE)

        mask = np.stack([disc, cup], axis=-1)  # (256, 256, 2)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            img = ToTensorV2()(image=img)["image"]
            mask = torch.from_numpy(mask)

        # Ensure mask is (2, H, W) and normalized to [0, 1]
        if mask.ndim == 3 and mask.shape[-1] == 2:
            mask = mask.permute(2, 0, 1)

        mask = mask.float() / 255.0  # Normalized to [0, 1]

        return {
            "image": img,
            "mask": mask,
            "image_id": img_id,
            "patient_id": row["patient_id"],
            "eye": row["eye"],
            "label": row["label"],
            "cdr_oracle": float(row["cdr_mean"]),
        }


def get_seg_transforms(split: str) -> A.Compose:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


# ── Training & Evaluation Loop ───────────────────────────────────────────────

def evaluate_segmenter(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_dir: Path,
) -> Tuple[Dict[str, float], List[Dict]]:
    model.eval()
    disc_dices_cons = []
    cup_dices_cons = []
    disc_dices_e1 = []
    cup_dices_e1 = []
    disc_dices_e2 = []
    cup_dices_e2 = []
    disc_ious = []
    cup_ious = []

    pred_cdrs = []
    oracle_cdrs = []
    detailed_records = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(len(imgs)):
                img_id = batch["image_id"][i]
                p_disc = (probs[i, 0] >= 0.5).astype(np.uint8)
                p_cup = (probs[i, 1] >= 0.5).astype(np.uint8)

                # Ground truth masks
                gt_d_cons = cv2.imread(str(mask_dir / f"{img_id}_disc_consensus.png"), cv2.IMREAD_GRAYSCALE) >= 128
                gt_c_cons = cv2.imread(str(mask_dir / f"{img_id}_cup_consensus.png"), cv2.IMREAD_GRAYSCALE) >= 128
                gt_d_e1 = cv2.imread(str(mask_dir / f"{img_id}_disc_exp1.png"), cv2.IMREAD_GRAYSCALE) > 0
                gt_c_e1 = cv2.imread(str(mask_dir / f"{img_id}_cup_exp1.png"), cv2.IMREAD_GRAYSCALE) > 0
                gt_d_e2 = cv2.imread(str(mask_dir / f"{img_id}_disc_exp2.png"), cv2.IMREAD_GRAYSCALE) > 0
                gt_c_e2 = cv2.imread(str(mask_dir / f"{img_id}_cup_exp2.png"), cv2.IMREAD_GRAYSCALE) > 0

                # Compute Dice & IoU
                d_dice_c = compute_dice(p_disc, gt_d_cons)
                c_dice_c = compute_dice(p_cup, gt_c_cons)
                disc_dices_cons.append(d_dice_c)
                cup_dices_cons.append(c_dice_c)

                disc_dices_e1.append(compute_dice(p_disc, gt_d_e1))
                cup_dices_e1.append(compute_dice(p_cup, gt_c_e1))
                disc_dices_e2.append(compute_dice(p_disc, gt_d_e2))
                cup_dices_e2.append(compute_dice(p_cup, gt_c_e2))

                disc_ious.append(compute_iou(p_disc, gt_d_cons))
                cup_ious.append(compute_iou(p_cup, gt_c_cons))

                # Morphometry: CDR
                pred_cdr, h_disc, h_cup = compute_predicted_cdr(p_disc, p_cup)
                orc_cdr = float(batch["cdr_oracle"][i])

                pred_cdrs.append(pred_cdr)
                oracle_cdrs.append(orc_cdr)

                detailed_records.append({
                    "image_id": img_id,
                    "patient_id": batch["patient_id"][i],
                    "eye": batch["eye"][i],
                    "label": int(batch["label"][i]),
                    "cdr_oracle": round(orc_cdr, 4),
                    "cdr_pred": round(pred_cdr, 4),
                    "disc_height_pred": int(h_disc),
                    "cup_height_pred": int(h_cup),
                    "disc_dice_consensus": round(d_dice_c, 4),
                    "cup_dice_consensus": round(c_dice_c, 4),
                })

    pred_cdrs = np.array(pred_cdrs)
    oracle_cdrs = np.array(oracle_cdrs)

    mae_cdr = float(np.mean(np.abs(pred_cdrs - oracle_cdrs)))
    pearson_r = float(pearsonr(pred_cdrs, oracle_cdrs)[0]) if len(pred_cdrs) > 2 else 0.0
    spearman_rho = float(spearmanr(pred_cdrs, oracle_cdrs)[0]) if len(pred_cdrs) > 2 else 0.0
    icc = compute_icc_2_1(oracle_cdrs, pred_cdrs) if len(pred_cdrs) > 2 else 0.0

    metrics = {
        "disc_dice_consensus": round(float(np.mean(disc_dices_cons)), 4),
        "cup_dice_consensus": round(float(np.mean(cup_dices_cons)), 4),
        "mean_dice_consensus": round(float((np.mean(disc_dices_cons) + np.mean(cup_dices_cons)) / 2.0), 4),
        "disc_dice_exp1": round(float(np.mean(disc_dices_e1)), 4),
        "cup_dice_exp1": round(float(np.mean(cup_dices_e1)), 4),
        "disc_dice_exp2": round(float(np.mean(disc_dices_e2)), 4),
        "cup_dice_exp2": round(float(np.mean(cup_dices_e2)), 4),
        "disc_iou_consensus": round(float(np.mean(disc_ious)), 4),
        "cup_iou_consensus": round(float(np.mean(cup_ious)), 4),
        "cdr_mae": round(mae_cdr, 4),
        "cdr_pearson_r": round(pearson_r, 4),
        "cdr_spearman_rho": round(spearman_rho, 4),
        "cdr_icc_2_1": round(icc, 4),
    }

    return metrics, detailed_records


def train_segmenter_pipeline(epochs: int = 50, batch_size: int = 16, lr: float = 1e-3):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training MobileNetV3-UNet Segmenter ===")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_p = Path("data/manifests/papila_morphology.csv")
    df = pd.read_csv(manifest_p)
    img_dir = Path("data/cache/fundus_256")
    mask_dir = Path("data/cache/masks_256")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Partitions: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)} (FROZEN)")

    train_ds = SegmentationDataset(train_df, img_dir, mask_dir, get_seg_transforms("train"))
    val_ds = SegmentationDataset(val_df, img_dir, mask_dir, get_seg_transforms("val"))
    test_ds = SegmentationDataset(test_df, img_dir, mask_dir, get_seg_transforms("val"))

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
        worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = MobileNetV3UNet(pretrained=True, num_classes=2).to(device)
    loss_fn = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_score = 0.0
    best_epoch = -1
    best_weights = None

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_tr = 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_tr += 1
        scheduler.step()
        train_loss /= n_tr

        # Validation on Val partition
        val_metrics, _ = evaluate_segmenter(model, val_loader, device, mask_dir)
        val_score = val_metrics["mean_dice_consensus"]

        is_best = val_score > best_val_score
        if is_best:
            best_val_score = val_score
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs or is_best:
            mark = " [*BEST]" if is_best else ""
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Mean Dice: {val_score:.4f} (OD: {val_metrics['disc_dice_consensus']:.4f}, OC: {val_metrics['cup_dice_consensus']:.4f}) | Val CDR MAE: {val_metrics['cdr_mae']:.4f}{mark}")

    print(f"\nTraining completed in {time.time() - t0:.1f}s. Best Epoch: {best_epoch} with Val Mean Dice: {best_val_score:.4f}")

    # Load best checkpoint
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Save checkpoint
    out_seg = Path("results/segmentation")
    out_seg.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_seg / "best_mobilenetv3_unet.pt"
    torch.save(best_weights, ckpt_path)
    print(f"Saved best model weights to {ckpt_path}")

    # ── Final Evaluation on Val and Frozen Test ──────────────────────────────
    val_final_metrics, _ = evaluate_segmenter(model, val_loader, device, mask_dir)
    test_final_metrics, test_records = evaluate_segmenter(model, test_loader, device, mask_dir)

    print("\n" + "=" * 80)
    print("FROZEN TEST PARTITION SEGMENTATION & CDR METRICS (64 images, 32 patients)")
    print("=" * 80)
    print(f"Disc Dice (Consensus): {test_final_metrics['disc_dice_consensus']:.4f} | Cup Dice (Consensus): {test_final_metrics['cup_dice_consensus']:.4f} | Mean Dice: {test_final_metrics['mean_dice_consensus']:.4f}")
    print(f"Disc Dice (Exp1):      {test_final_metrics['disc_dice_exp1']:.4f} | Cup Dice (Exp1):      {test_final_metrics['cup_dice_exp1']:.4f}")
    print(f"Disc Dice (Exp2):      {test_final_metrics['disc_dice_exp2']:.4f} | Cup Dice (Exp2):      {test_final_metrics['cup_dice_exp2']:.4f}")
    print(f"Disc IoU:              {test_final_metrics['disc_iou_consensus']:.4f} | Cup IoU:              {test_final_metrics['cup_iou_consensus']:.4f}")
    print("-" * 80)
    print(f"CDR MAE (vs Oracle):   {test_final_metrics['cdr_mae']:.4f}")
    print(f"CDR Pearson r:         {test_final_metrics['cdr_pearson_r']:.4f}")
    print(f"CDR Spearman rho:      {test_final_metrics['cdr_spearman_rho']:.4f}")
    print(f"CDR ICC(2,1):          {test_final_metrics['cdr_icc_2_1']:.4f}")
    print("=" * 80)

    # ── Generate Predicted CDR for ALL 420 Images ────────────────────────────
    print("\nGenerating predicted CDR for all 420 images across Train, Val, Test...")
    all_ds = SegmentationDataset(df, img_dir, mask_dir, get_seg_transforms("val"))
    all_loader = DataLoader(all_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    _, all_records = evaluate_segmenter(model, all_loader, device, mask_dir)

    df_pred_cdr = pd.DataFrame(all_records)
    # Merge split and label from original manifest
    df_pred_cdr = df_pred_cdr.merge(df[["image_id", "split"]], on="image_id", how="left")
    pred_cdr_path = Path("data/manifests/papila_predicted_cdr.csv")
    df_pred_cdr.to_csv(pred_cdr_path, index=False)
    print(f"Saved complete predicted CDR manifest to {pred_cdr_path}")

    # Save summary metrics
    summary_seg = {
        "best_epoch": best_epoch,
        "validation_metrics": val_final_metrics,
        "test_metrics": test_final_metrics,
    }
    with open(out_seg / "segmenter_metrics.json", "w") as f:
        json.dump(summary_seg, f, indent=2)
    print(f"Saved metrics summary to {out_seg / 'segmenter_metrics.json'}")


if __name__ == "__main__":
    train_segmenter_pipeline(epochs=50, batch_size=16, lr=1e-3)
