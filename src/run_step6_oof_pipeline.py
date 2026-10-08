"""Step 6 Final: 5-Fold Patient-Stratified Out-Of-Fold (OOF) Predicted CDR & Final Evaluation.

Protocol:
1. PAPILA Train (294 images, 147 patients) is split into 5 patient-stratified folds using StratifiedGroupKFold.
2. For each fold k (0..4):
   - Train MobileNetV3-UNet on the other 4 folds (35 epochs).
   - Predict masks on fold k (completely unseen patients).
   - Compute vertical CDR -> cdr_pred_oof for fold k.
3. Combine all 5 folds to create unbiased cdr_pred_oof for the entire Train partition.
4. For Val (62 images) and Test (64 images), use the segmenter trained on the full Train partition.
5. Save combined manifest to data/manifests/papila_oof_predicted_cdr.csv.
6. Train M5a-auto-OOF across 5 seeds: [42, 123, 2024, 3407, 777].
7. Compute paired patient-clustered bootstrap for:
   - Delta AUC(M5a-auto-OOF - M2)
   - Delta AUC(M1-auto - M1-oracle)
8. Save results to results/automated/step6_oof_summary.json.
"""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.segmenter import MobileNetV3UNet
from src.train_segmenter import (
    CombinedBCEDiceLoss,
    SegmentationDataset,
    compute_dice,
    compute_icc_2_1,
    compute_iou,
    compute_predicted_cdr,
    get_seg_transforms,
    seed_everything,
    seed_worker,
)


# ── Bootstrap Helpers ────────────────────────────────────────────────────────

def compute_paired_patient_bootstrap_delta_auc(
    df_base: pd.DataFrame,
    df_comp: pd.DataFrame,
    score_col_base: str,
    score_col_comp: str,
    n_iterations: int = 2000,
    seed: int = 42,
) -> Dict:
    """Computes paired patient-clustered bootstrap confidence interval for Delta AUC = AUC(comp) - AUC(base)."""
    assert (df_base["image_id"] == df_comp["image_id"]).all()
    assert (df_base["label"] == df_comp["label"]).all()

    np.random.seed(seed)
    patients = df_base["patient_id"].unique()
    n_patients = len(patients)

    delta_aucs = []
    for _ in range(n_iterations):
        sampled_patients = np.random.choice(patients, size=n_patients, replace=True)
        sampled_base = pd.concat([df_base[df_base["patient_id"] == p] for p in sampled_patients], ignore_index=True)
        sampled_comp = pd.concat([df_comp[df_comp["patient_id"] == p] for p in sampled_patients], ignore_index=True)

        if len(sampled_base["label"].unique()) > 1:
            auc_b = roc_auc_score(sampled_base["label"], sampled_base[score_col_base])
            auc_c = roc_auc_score(sampled_comp["label"], sampled_comp[score_col_comp])
            delta_aucs.append(auc_c - auc_b)

    delta_aucs = np.array(delta_aucs)
    ci = np.percentile(delta_aucs, [2.5, 97.5]).round(4).tolist()
    mean_delta = float(np.mean(delta_aucs))
    p_value = float(np.mean(delta_aucs <= 0.0))

    return {
        "mean_delta_auc": round(mean_delta, 4),
        "bootstrap_95ci_patient_clustered": ci,
        "empirical_p_value_one_sided": round(p_value, 4),
        "pct_bootstrap_positive": round(float(np.mean(delta_aucs > 0.0) * 100), 2),
    }


def compute_bootstrap_cis(df_eval: pd.DataFrame, score_col: str, n_iterations: int = 2000, seed: int = 42):
    np.random.seed(seed)
    n_eyes = len(df_eval)
    eye_aucs = []
    for _ in range(n_iterations):
        idx = np.random.choice(n_eyes, size=n_eyes, replace=True)
        sample = df_eval.iloc[idx]
        if len(sample["label"].unique()) > 1:
            eye_aucs.append(roc_auc_score(sample["label"], sample[score_col]))
    ci_eye = np.percentile(eye_aucs, [2.5, 97.5]).round(4).tolist()

    patients = df_eval["patient_id"].unique()
    n_patients = len(patients)
    cluster_aucs = []
    for _ in range(n_iterations):
        sampled_patients = np.random.choice(patients, size=n_patients, replace=True)
        sampled_rows = pd.concat([df_eval[df_eval["patient_id"] == p] for p in sampled_patients], ignore_index=True)
        if len(sampled_rows["label"].unique()) > 1:
            cluster_aucs.append(roc_auc_score(sampled_rows["label"], sampled_rows[score_col]))
    ci_cluster = np.percentile(cluster_aucs, [2.5, 97.5]).round(4).tolist()
    return ci_eye, ci_cluster


# ── Step 1: 5-Fold OOF Segmentation Training ────────────────────────────────

def train_and_predict_oof(
    train_df: pd.DataFrame,
    img_dir: Path,
    mask_dir: Path,
    device: torch.device,
    epochs: int = 35,
    batch_size: int = 16,
) -> Tuple[pd.DataFrame, Dict]:
    print("\n--- Running 5-Fold Patient-Stratified OOF Training on PAPILA Train ---")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_records = []
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(train_df, train_df["label"], train_df["patient_id"])):
        print(f"\n>>> Fold {fold}/4: Train {len(tr_idx)} images, OOF Val {len(val_idx)} images...")
        fold_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        seed_everything(42 + fold)
        train_ds = SegmentationDataset(fold_train_df, img_dir, mask_dir, get_seg_transforms("train"))
        val_ds = SegmentationDataset(fold_val_df, img_dir, mask_dir, get_seg_transforms("val"))

        g = torch.Generator()
        g.manual_seed(42 + fold)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True,
            worker_init_fn=seed_worker, generator=g,
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = MobileNetV3UNet(pretrained=True, num_classes=2).to(device)
        loss_fn = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_dice = 0.0
        best_weights = None

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = loss_fn(logits, masks)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Evaluate on OOF Val
            model.eval()
            d_dices = []
            c_dices = []
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(device)
                    probs = torch.sigmoid(model(imgs)).cpu().numpy()
                    for i in range(len(imgs)):
                        img_id = batch["image_id"][i]
                        p_disc = (probs[i, 0] >= 0.5).astype(np.uint8)
                        p_cup = (probs[i, 1] >= 0.5).astype(np.uint8)
                        gt_d = cv2.imread(str(mask_dir / f"{img_id}_disc_consensus.png"), cv2.IMREAD_GRAYSCALE) >= 128
                        gt_c = cv2.imread(str(mask_dir / f"{img_id}_cup_consensus.png"), cv2.IMREAD_GRAYSCALE) >= 128
                        d_dices.append(compute_dice(p_disc, gt_d))
                        c_dices.append(compute_dice(p_cup, gt_c))
            m_dice = (np.mean(d_dices) + np.mean(c_dices)) / 2.0
            if m_dice > best_dice:
                best_dice = m_dice
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Fold {fold} finished in {time.time() - t0:.1f}s | Best OOF Mean Dice: {best_dice:.4f}")

        # Final prediction on unseen OOF Val with best weights
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                probs = torch.sigmoid(model(imgs)).cpu().numpy()
                for i in range(len(imgs)):
                    img_id = batch["image_id"][i]
                    p_disc = (probs[i, 0] >= 0.5).astype(np.uint8)
                    p_cup = (probs[i, 1] >= 0.5).astype(np.uint8)
                    cdr_pred, h_disc, h_cup = compute_predicted_cdr(p_disc, p_cup)
                    orc_cdr = float(batch["cdr_oracle"][i])
                    gt_d = cv2.imread(str(mask_dir / f"{img_id}_disc_consensus.png"), cv2.IMREAD_GRAYSCALE) >= 128
                    gt_c = cv2.imread(str(mask_dir / f"{img_id}_cup_consensus.png"), cv2.IMREAD_GRAYSCALE) >= 128

                    oof_records.append({
                        "image_id": img_id,
                        "patient_id": batch["patient_id"][i],
                        "eye": batch["eye"][i],
                        "label": int(batch["label"][i]),
                        "cdr_oracle": round(orc_cdr, 4),
                        "cdr_pred_oof": round(cdr_pred, 4),
                        "disc_height_pred": int(h_disc),
                        "cup_height_pred": int(h_cup),
                        "disc_dice": round(compute_dice(p_disc, gt_d), 4),
                        "cup_dice": round(compute_dice(p_cup, gt_c), 4),
                        "fold": fold,
                    })

    df_oof = pd.DataFrame(oof_records)
    mae_oof = float(np.mean(np.abs(df_oof["cdr_pred_oof"] - df_oof["cdr_oracle"])))
    r_oof = float(pearsonr(df_oof["cdr_pred_oof"], df_oof["cdr_oracle"])[0])
    rho_oof = float(spearmanr(df_oof["cdr_pred_oof"], df_oof["cdr_oracle"])[0])
    icc_oof = compute_icc_2_1(df_oof["cdr_oracle"].values, df_oof["cdr_pred_oof"].values)

    oof_summary = {
        "train_oof_cdr_mae": round(mae_oof, 4),
        "train_oof_cdr_pearson_r": round(r_oof, 4),
        "train_oof_cdr_spearman_rho": round(rho_oof, 4),
        "train_oof_cdr_icc": round(icc_oof, 4),
        "train_oof_mean_disc_dice": round(float(df_oof["disc_dice"].mean()), 4),
        "train_oof_mean_cup_dice": round(float(df_oof["cup_dice"].mean()), 4),
    }
    print(f"\nComplete Train OOF Metrics (294 images):")
    print(f"  CDR MAE: {mae_oof:.4f} | Pearson r: {r_oof:.4f} | ICC: {icc_oof:.4f} | Mean Disc Dice: {oof_summary['train_oof_mean_disc_dice']:.4f} | Mean Cup Dice: {oof_summary['train_oof_mean_cup_dice']:.4f}")

    return df_oof, oof_summary


# ── Step 2: Assemble Full OOF Manifest ──────────────────────────────────────

def build_oof_manifest(df_oof: pd.DataFrame, existing_pred_p: Path, out_path: Path) -> pd.DataFrame:
    df_existing = pd.read_csv(existing_pred_p)
    # df_existing has cdr_pred for train, val, test from the full train model
    val_test_df = df_existing[df_existing["split"].isin(["val", "test"])].copy()
    val_test_df["cdr_pred_oof"] = val_test_df["cdr_pred"]
    val_test_df["fold"] = -1

    # Train uses pure OOF predictions
    train_oof_merged = df_oof.merge(
        df_existing[["image_id", "split"]],
        on="image_id",
        how="left",
    )

    combined = pd.concat([train_oof_merged, val_test_df], ignore_index=True)
    # Ensure identical row ordering as original morphology manifest
    morph_df = pd.read_csv("data/manifests/papila_morphology.csv")
    order_map = {img_id: i for i, img_id in enumerate(morph_df["image_id"])}
    combined["orig_order"] = combined["image_id"].map(order_map)
    combined = combined.sort_values("orig_order").drop(columns=["orig_order"]).reset_index(drop=True)

    combined.to_csv(out_path, index=False)
    print(f"Saved complete OOF manifest (420 rows) to {out_path}")
    return combined


# ── Step 3: Train M5a-auto-OOF across 5 Seeds ────────────────────────────────

def train_m5a_auto_oof_seed(
    seed: int,
    df_manifest: pd.DataFrame,
    f_global: torch.Tensor,
    cdr_tensor: torch.Tensor,
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
) -> Tuple[pd.DataFrame, Dict]:
    seed_everything(seed)
    features = torch.cat([f_global, cdr_tensor], dim=1)
    in_dim = features.shape[1]

    train_idx = (df_manifest["split"] == "train").values
    val_idx = (df_manifest["split"] == "val").values
    test_idx = (df_manifest["split"] == "test").values

    labels = torch.tensor(df_manifest["label"].values, dtype=torch.float32).unsqueeze(1)

    X_train, y_train = features[train_idx].to(device), labels[train_idx].to(device)
    X_val, y_val = features[val_idx].to(device), labels[val_idx].to(device)
    X_test, y_test = features[test_idx].to(device), labels[test_idx].to(device)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Dropout(0.2),
        nn.Linear(in_dim, 1),
    ).to(device)

    pos_weight = torch.tensor([233.0 / 61.0], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(X_val)
            v_loss = loss_fn(v_logits, y_val).item()
            v_probs = torch.sigmoid(v_logits).squeeze(1).cpu().numpy()
            v_auc = float(roc_auc_score(y_val.cpu().numpy(), v_probs))

        is_best = (v_auc > best_val_auc) or (np.isclose(v_auc, best_val_auc) and v_loss < best_val_loss)
        if is_best:
            best_val_auc = v_auc
            best_val_loss = v_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()

    # Optimal Threshold on Val
    with torch.no_grad():
        v_logits = model(X_val).squeeze(1).cpu().numpy()
        v_probs = 1.0 / (1.0 + np.exp(-v_logits))
    val_y = y_val.squeeze(1).cpu().numpy().astype(int)
    fpr, tpr, thresholds = roc_curve(val_y, v_probs)
    j = tpr - fpr
    best_j_idx = int(np.argmax(j))
    optimal_thresh = float(thresholds[best_j_idx])

    # Platt calibrator on Train
    with torch.no_grad():
        train_logits = model(X_train).squeeze(1).cpu().numpy()
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(train_logits.reshape(-1, 1), y_train.squeeze(1).cpu().numpy().astype(int))

    # Test Evaluation
    with torch.no_grad():
        test_logits = model(X_test).squeeze(1).cpu().numpy()
        test_probs_raw = 1.0 / (1.0 + np.exp(-test_logits))

    test_cal_probs = calibrator.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    test_y = y_test.squeeze(1).cpu().numpy().astype(int)

    test_preds_df = df_manifest[test_idx].copy().reset_index(drop=True)
    test_preds_df["logit"] = test_logits
    test_preds_df["prob_raw"] = test_probs_raw
    test_preds_df["prob_calibrated"] = test_cal_probs
    test_preds_df["threshold_applied"] = optimal_thresh
    test_preds_df["prediction_binary"] = (test_probs_raw >= optimal_thresh).astype(int)

    test_auc = float(roc_auc_score(test_y, test_probs_raw))
    ci_eye, ci_cluster = compute_bootstrap_cis(test_preds_df, "prob_raw", n_iterations=2000, seed=seed)
    tn, fp, fn, tp = confusion_matrix(test_y, test_preds_df["prediction_binary"]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc = float((tp + tn) / len(test_preds_df))
    brier_cal = float(brier_score_loss(test_y, test_cal_probs))

    metrics = {
        "seed": seed,
        "val_auc": round(best_val_auc, 4),
        "test_auc": round(test_auc, 4),
        "ci_eye": ci_eye,
        "ci_cluster": ci_cluster,
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "accuracy": round(acc, 4),
        "brier_calibrated": round(brier_cal, 4),
        "threshold": round(optimal_thresh, 4),
    }

    return test_preds_df, metrics


# ── Step 4: Main Execution Suite ─────────────────────────────────────────────

def run_step6_oof_suite():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Step 6 Final: Out-Of-Fold Automated Pipeline Evaluation ===")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_p = Path("data/manifests/papila_morphology.csv")
    df_manifest = pd.read_csv(manifest_p)
    img_dir = Path("data/cache/fundus_256")
    mask_dir = Path("data/cache/masks_256")

    train_df = df_manifest[df_manifest["split"] == "train"].reset_index(drop=True)

    # 1. Train 5-fold OOF segmenters and predict on unseen folds
    df_oof, oof_summary = train_and_predict_oof(train_df, img_dir, mask_dir, device, epochs=35, batch_size=16)

    # 2. Build full OOF manifest
    out_manifest = Path("data/manifests/papila_oof_predicted_cdr.csv")
    df_full_oof = build_oof_manifest(df_oof, Path("data/manifests/papila_predicted_cdr.csv"), out_manifest)

    # 3. Load Global Features & Standardize on Train OOF CDR
    feat_dict = torch.load("data/features/papila_m2_global_features.pt", weights_only=True)
    f_global = feat_dict["features"]

    train_idx = (df_full_oof["split"] == "train").values
    val_idx = (df_full_oof["split"] == "val").values
    test_idx = (df_full_oof["split"] == "test").values

    scaler = StandardScaler()
    cdr_train = scaler.fit_transform(df_full_oof.loc[train_idx, ["cdr_pred_oof"]].values)
    cdr_val = scaler.transform(df_full_oof.loc[val_idx, ["cdr_pred_oof"]].values)
    cdr_test = scaler.transform(df_full_oof.loc[test_idx, ["cdr_pred_oof"]].values)

    cdr_all = np.zeros((len(df_full_oof), 1), dtype=np.float32)
    cdr_all[train_idx] = cdr_train
    cdr_all[val_idx] = cdr_val
    cdr_all[test_idx] = cdr_test
    cdr_tensor = torch.tensor(cdr_all, dtype=torch.float32)

    # 4. Train M5a-auto-OOF across 5 seeds
    print("\n--- Training M5a-auto-OOF across 5 Seeds ---")
    seeds = [42, 123, 2024, 3407, 777]
    out_dir = Path("results/automated")
    out_dir.mkdir(parents=True, exist_ok=True)

    m5a_oof_runs = []
    seed_preds = {}

    for s in seeds:
        df_pred, met = train_m5a_auto_oof_seed(
            s, df_full_oof, f_global, cdr_tensor, device, epochs=40, batch_size=16, lr=1e-3
        )
        m5a_oof_runs.append(met)
        seed_preds[s] = df_pred
        df_pred.to_csv(out_dir / f"M5a_auto_oof_seed_{s}_test_predictions.csv", index=False)
        print(f"  Seed {s} -> Test AUC: {met['test_auc']:.4f} | Sens: {met['sensitivity']:.4f} | Spec: {met['specificity']:.4f} | Brier: {met['brier_calibrated']:.4f}")

    aucs = [m["test_auc"] for m in m5a_oof_runs]
    briers = [m["brier_calibrated"] for m in m5a_oof_runs]
    sens_list = [m["sensitivity"] for m in m5a_oof_runs]
    spec_list = [m["specificity"] for m in m5a_oof_runs]

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    ens_probs = np.mean([seed_preds[s]["prob_raw"].values for s in seeds], axis=0)
    test_y = seed_preds[seeds[0]]["label"].values
    ens_auc = float(roc_auc_score(test_y, ens_probs))

    # 5. Paired bootstrap comparisons
    print("\n--- Running Final Paired Patient-Clustered Bootstrap Tests ---")
    # A. M5a-auto-OOF vs M2 (using Seed 42 and Ensemble)
    df_m2 = pd.read_csv("results/multiseed/M2_seed_42_test_predictions.csv")
    df_m5a_s42 = seed_preds[42]
    paired_m5a_m2 = compute_paired_patient_bootstrap_delta_auc(df_m2, df_m5a_s42, "prob_raw", "prob_raw")

    # B. M1-auto vs M1-oracle
    df_m1_orc = pd.read_csv("results/M1_cdr_test_predictions.csv")
    df_m1_auto = df_full_oof[df_full_oof["split"] == "test"].copy().reset_index(drop=True)
    paired_m1_auto_orc = compute_paired_patient_bootstrap_delta_auc(df_m1_orc, df_m1_auto, "cdr_mean", "cdr_pred_oof")

    # Load Step 5 summary
    with open("results/multiseed/multiseed_summary.json") as f:
        step5_summary = json.load(f)

    summary_final = {
        "oof_segmentation_train_summary": oof_summary,
        "M5a_auto_oof": {
            "mean_auc": round(mean_auc, 4),
            "std_auc": round(std_auc, 4),
            "min_auc": round(float(np.min(aucs)), 4),
            "max_auc": round(float(np.max(aucs)), 4),
            "mean_calibrated_brier": round(float(np.mean(briers)), 4),
            "mean_sensitivity": round(float(np.mean(sens_list)), 4),
            "mean_specificity": round(float(np.mean(spec_list)), 4),
            "ensemble_auc": round(ens_auc, 4),
            "runs": m5a_oof_runs,
        },
        "comparison_3_levels": {
            "M2_Global_mean_auc": step5_summary["M2_global"]["mean_auc"],
            "M5a_Oracle_mean_auc": step5_summary["M5a_global_cdr"]["mean_auc"],
            "M5a_Auto_OOF_mean_auc": round(mean_auc, 4),
            "delta_auto_vs_m2": round(mean_auc - step5_summary["M2_global"]["mean_auc"], 4),
        },
        "paired_bootstraps": {
            "M5a_auto_OOF_vs_M2_seed42": paired_m5a_m2,
            "M1_auto_vs_M1_oracle": paired_m1_auto_orc,
        },
    }

    out_json = out_dir / "step6_oof_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary_final, f, indent=2)
    print(f"\nSaved complete final summary to {out_json}")

    print("\n" + "=" * 80)
    print("STEP 6 FINAL OOF PIPELINE RESULTS")
    print("=" * 80)
    print(f"M2 Global Alone:               Mean AUC = {step5_summary['M2_global']['mean_auc']:.4f}")
    print(f"M5a-Oracle (Expert CDR):        Mean AUC = {step5_summary['M5a_global_cdr']['mean_auc']:.4f}")
    print(f"M5a-Auto-OOF (Predicted CDR):  Mean AUC = {mean_auc:.4f} ± {std_auc:.4f} (range: {np.min(aucs):.4f} - {np.max(aucs):.4f}) | Ensemble: {ens_auc:.4f}")
    print(f"Sensitivity (Catching Glaucoma): {np.mean(sens_list)*100:.1f}% across runs")
    print("-" * 80)
    print(f"Paired Delta AUC (M5a-auto vs M2, Seed 42): {paired_m5a_m2['mean_delta_auc']:+.4f} (95% CI: {paired_m5a_m2['bootstrap_95ci_patient_clustered']}, P(>0)={paired_m5a_m2['pct_bootstrap_positive']}%)")
    print(f"Paired Delta AUC (M1-auto vs M1-oracle):     {paired_m1_auto_orc['mean_delta_auc']:+.4f} (95% CI: {paired_m1_auto_orc['bootstrap_95ci_patient_clustered']}, P(>0)={paired_m1_auto_orc['pct_bootstrap_positive']}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_step6_oof_suite()
