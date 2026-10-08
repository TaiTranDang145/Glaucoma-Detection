"""Multi-seed reliability, paired patient-clustered bootstrap, and error analysis.

Evaluates M2 (Global MobileNetV3) and M5a (Global + CDR Fusion) across 5 random seeds:
[42, 123, 2024, 3407, 777] on NVIDIA GeForce RTX 3050 Laptop GPU.

Strict protocols:
1. Identical patient-stratified split on PAPILA (Train 294, Val 62, Test 64).
2. All checkpoint selections strictly based on Validation ROC-AUC (val_auc).
3. All classification thresholds tuned on Validation partition via Youden's J.
4. Platt calibration fitted exclusively on Train partition.
5. Paired patient-clustered bootstrap (2,000 iterations) for Delta AUC = AUC(M5a) - AUC(M2).
6. In-depth error analysis and correlation evaluation.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


# ── Fast Dataset for Pre-cached 256x256 Images ───────────────────────────────

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


class CachedFundusDataset(Dataset):
    """Fast dataset reading from pre-cached 256x256 images."""

    def __init__(self, df: pd.DataFrame, cache_dir: Path, transform: A.Compose):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_p = self.cache_dir / f"{row['image_id']}.png"
        img = cv2.imread(str(img_p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        return {
            "image": img,
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
        sampled_rows = pd.concat(
            [df_eval[df_eval["patient_id"] == p] for p in sampled_patients],
            ignore_index=True,
        )
        if len(sampled_rows["label"].unique()) > 1:
            cluster_aucs.append(roc_auc_score(sampled_rows["label"], sampled_rows[score_col]))
    ci_cluster = np.percentile(cluster_aucs, [2.5, 97.5]).round(4).tolist()

    return ci_eye, ci_cluster


def compute_paired_patient_bootstrap_delta_auc(
    df_m2: pd.DataFrame,
    df_m5a: pd.DataFrame,
    n_iterations: int = 2000,
    seed: int = 42,
) -> Dict:
    """Computes paired patient-clustered bootstrap confidence interval for Delta AUC = AUC(M5a) - AUC(M2)."""
    assert (df_m2["image_id"] == df_m5a["image_id"]).all()
    assert (df_m2["label"] == df_m5a["label"]).all()

    np.random.seed(seed)
    patients = df_m2["patient_id"].unique()
    n_patients = len(patients)

    delta_aucs = []
    for _ in range(n_iterations):
        sampled_patients = np.random.choice(patients, size=n_patients, replace=True)
        sampled_m2 = pd.concat([df_m2[df_m2["patient_id"] == p] for p in sampled_patients], ignore_index=True)
        sampled_m5a = pd.concat([df_m5a[df_m5a["patient_id"] == p] for p in sampled_patients], ignore_index=True)

        if len(sampled_m2["label"].unique()) > 1:
            auc_m2 = roc_auc_score(sampled_m2["label"], sampled_m2["prob_raw"])
            auc_m5a = roc_auc_score(sampled_m5a["label"], sampled_m5a["prob_raw"])
            delta_aucs.append(auc_m5a - auc_m2)

    delta_aucs = np.array(delta_aucs)
    ci = np.percentile(delta_aucs, [2.5, 97.5]).round(4).tolist()
    mean_delta = float(np.mean(delta_aucs))
    p_value = float(np.mean(delta_aucs <= 0.0))  # Empirical one-sided p-value: probability Delta AUC <= 0

    return {
        "mean_delta_auc": round(mean_delta, 4),
        "bootstrap_95ci_patient_clustered": ci,
        "empirical_p_value_one_sided": round(p_value, 4),
        "pct_bootstrap_positive": round(float(np.mean(delta_aucs > 0.0) * 100), 2),
    }


# ── Training & Feature Extraction for M2 ─────────────────────────────────────

def train_m2_single_seed(
    seed: int,
    df_manifest: pd.DataFrame,
    cache_dir: Path,
    device: torch.device,
    epochs: int = 35,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
) -> Tuple[pd.DataFrame, torch.Tensor, Dict]:
    seed_everything(seed)
    train_df = df_manifest[df_manifest["split"] == "train"].reset_index(drop=True)
    val_df = df_manifest[df_manifest["split"] == "val"].reset_index(drop=True)
    test_df = df_manifest[df_manifest["split"] == "test"].reset_index(drop=True)

    train_tf = get_transforms("train", 224)
    val_tf = get_transforms("val", 224)

    train_ds = CachedFundusDataset(train_df, cache_dir, train_tf)
    val_ds = CachedFundusDataset(val_df, cache_dir, val_tf)
    test_ds = CachedFundusDataset(test_df, cache_dir, val_tf)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
        worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = MobileNetV3Classifier(pretrained=True, dropout_rate=0.2, num_classes=1).to(device)
    pos_weight = torch.tensor([233.0 / 61.0], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_tr = 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_tr += 1
        scheduler.step()

        # Validation
        model.eval()
        v_loss = 0.0
        n_v = 0
        v_logits = []
        v_labels = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)
                logits = model(imgs)
                loss = loss_fn(logits, labels)
                v_loss += loss.item()
                n_v += 1
                v_logits.extend(logits.squeeze(-1).cpu().numpy())
                v_labels.extend(labels.squeeze(-1).cpu().numpy())
        v_loss /= n_v
        v_probs = 1.0 / (1.0 + np.exp(-np.array(v_logits)))
        v_auc = float(roc_auc_score(v_labels, v_probs))

        is_best = (v_auc > best_val_auc) or (np.isclose(v_auc, best_val_auc) and v_loss < best_val_loss)
        if is_best:
            best_val_auc = v_auc
            best_val_loss = v_loss
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()

    # Optimal Threshold on Val
    with torch.no_grad():
        v_logits = []
        v_labels = []
        for batch in val_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            logits = model(imgs)
            v_logits.extend(logits.squeeze(-1).cpu().numpy())
            v_labels.extend(labels.squeeze(-1).cpu().numpy())
    v_probs = 1.0 / (1.0 + np.exp(-np.array(v_logits)))
    fpr, tpr, thresholds = roc_curve(v_labels, v_probs)
    j = tpr - fpr
    best_j_idx = int(np.argmax(j))
    optimal_thresh = float(thresholds[best_j_idx])

    # Platt calibrator on Train
    train_eval_loader = DataLoader(CachedFundusDataset(train_df, cache_dir, val_tf), batch_size=batch_size, shuffle=False, num_workers=4)
    tr_logits = []
    tr_labels = []
    with torch.no_grad():
        for batch in train_eval_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            logits = model(imgs)
            tr_logits.extend(logits.squeeze(-1).cpu().numpy())
            tr_labels.extend(labels.squeeze(-1).cpu().numpy())
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(np.array(tr_logits).reshape(-1, 1), np.array(tr_labels))

    # Test Evaluation
    te_records = []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            logits = model(imgs).squeeze(-1).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            for i in range(len(labels)):
                te_records.append({
                    "image_id": batch["image_id"][i],
                    "patient_id": batch["patient_id"][i],
                    "eye": batch["eye"][i],
                    "label": int(labels[i]),
                    "logit": float(logits[i]),
                    "prob_raw": float(probs[i]),
                })
    test_preds_df = pd.DataFrame(te_records)
    test_preds_df["prob_calibrated"] = calibrator.predict_proba(test_preds_df[["logit"]])[:, 1]
    test_preds_df["threshold_applied"] = optimal_thresh
    test_preds_df["prediction_binary"] = (test_preds_df["prob_raw"] >= optimal_thresh).astype(int)

    test_auc = float(roc_auc_score(test_preds_df["label"], test_preds_df["prob_raw"]))
    ci_eye, ci_cluster = compute_bootstrap_cis(test_preds_df, "prob_raw", n_iterations=2000, seed=seed)
    tn, fp, fn, tp = confusion_matrix(test_preds_df["label"], test_preds_df["prediction_binary"]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc = float((tp + tn) / len(test_preds_df))
    brier_cal = float(brier_score_loss(test_preds_df["label"], test_preds_df["prob_calibrated"]))

    metrics = {
        "seed": seed,
        "best_epoch": best_epoch,
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

    # Extract all 420 features
    all_ds = CachedFundusDataset(df_manifest, cache_dir, val_tf)
    all_loader = DataLoader(all_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    all_feats = []
    with torch.no_grad():
        for batch in all_loader:
            imgs = batch["image"].to(device)
            f = model.extract_features(imgs).cpu()
            all_feats.append(f)
    all_feats_tensor = torch.cat(all_feats, dim=0)

    return test_preds_df, all_feats_tensor, metrics


# ── Training & Evaluation for M5a ────────────────────────────────────────────

def train_m5a_single_seed(
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
    best_epoch = -1
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
            best_epoch = epoch
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
        "best_epoch": best_epoch,
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


# ── Main Multi-Seed Execution ────────────────────────────────────────────────

def run_multiseed_suite():
    seeds = [42, 123, 2024, 3407, 777]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Multi-seed Robustness & Paired Bootstrap Suite ===")
    print(f"Seeds: {seeds} | Compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    df_manifest = pd.read_csv("data/manifests/papila_morphology.csv")
    cache_dir = Path("data/cache/fundus_256")

    # Fit scaler on Train partition ONLY
    train_idx = (df_manifest["split"] == "train").values
    val_idx = (df_manifest["split"] == "val").values
    test_idx = (df_manifest["split"] == "test").values

    scaler = StandardScaler()
    cdr_train = scaler.fit_transform(df_manifest.loc[train_idx, ["cdr_mean"]].values)
    cdr_val = scaler.transform(df_manifest.loc[val_idx, ["cdr_mean"]].values)
    cdr_test = scaler.transform(df_manifest.loc[test_idx, ["cdr_mean"]].values)

    cdr_all = np.zeros((len(df_manifest), 1), dtype=np.float32)
    cdr_all[train_idx] = cdr_train
    cdr_all[val_idx] = cdr_val
    cdr_all[test_idx] = cdr_test
    cdr_tensor = torch.tensor(cdr_all, dtype=torch.float32)

    out_base = Path("results/multiseed")
    out_base.mkdir(parents=True, exist_ok=True)

    m2_results = []
    m5a_results = []
    paired_deltas = []
    seed_preds_m2 = {}
    seed_preds_m5a = {}

    for s in seeds:
        print(f"\n>>> Running Seed {s}...")
        # Train M2
        df_pred_m2, f_global, met_m2 = train_m2_single_seed(s, df_manifest, cache_dir, device)
        m2_results.append(met_m2)
        seed_preds_m2[s] = df_pred_m2
        df_pred_m2.to_csv(out_base / f"M2_seed_{s}_test_predictions.csv", index=False)

        # Train M5a
        df_pred_m5a, met_m5a = train_m5a_single_seed(s, df_manifest, f_global, cdr_tensor, device)
        m5a_results.append(met_m5a)
        seed_preds_m5a[s] = df_pred_m5a
        df_pred_m5a.to_csv(out_base / f"M5a_seed_{s}_test_predictions.csv", index=False)

        # Paired bootstrap Delta AUC for this seed
        paired_dict = compute_paired_patient_bootstrap_delta_auc(df_pred_m2, df_pred_m5a, n_iterations=2000, seed=s)
        paired_dict["seed"] = s
        paired_dict["m2_auc"] = met_m2["test_auc"]
        paired_dict["m5a_auc"] = met_m5a["test_auc"]
        paired_dict["delta_auc"] = round(met_m5a["test_auc"] - met_m2["test_auc"], 4)
        paired_deltas.append(paired_dict)

        print(f"  Seed {s} -> M2 AUC: {met_m2['test_auc']:.4f} | M5a AUC: {met_m5a['test_auc']:.4f} | Delta AUC: {paired_dict['delta_auc']:+.4f} (Paired CI: {paired_dict['bootstrap_95ci_patient_clustered']})")

    # ── Aggregate Statistics Across 5 Seeds ──────────────────────────────────
    m2_aucs = [m["test_auc"] for m in m2_results]
    m5a_aucs = [m["test_auc"] for m in m5a_results]
    delta_aucs = [p["delta_auc"] for p in paired_deltas]

    summary = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "M2_global": {
            "mean_auc": round(float(np.mean(m2_aucs)), 4),
            "std_auc": round(float(np.std(m2_aucs)), 4),
            "min_auc": round(float(np.min(m2_aucs)), 4),
            "max_auc": round(float(np.max(m2_aucs)), 4),
            "individual_runs": m2_results,
        },
        "M5a_global_cdr": {
            "mean_auc": round(float(np.mean(m5a_aucs)), 4),
            "std_auc": round(float(np.std(m5a_aucs)), 4),
            "min_auc": round(float(np.min(m5a_aucs)), 4),
            "max_auc": round(float(np.max(m5a_aucs)), 4),
            "individual_runs": m5a_results,
        },
        "paired_comparison": {
            "mean_delta_auc": round(float(np.mean(delta_aucs)), 4),
            "std_delta_auc": round(float(np.std(delta_aucs)), 4),
            "min_delta_auc": round(float(np.min(delta_aucs)), 4),
            "max_delta_auc": round(float(np.max(delta_aucs)), 4),
            "individual_paired_bootstraps": paired_deltas,
        },
    }

    with open(out_base / "multiseed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Error Analysis across ensemble of seeds ──────────────────────────────
    print("\n--- Running Ensemble-Level Error Analysis ---")
    # Average predicted probabilities across seeds
    m2_avg_probs = np.mean([seed_preds_m2[s]["prob_raw"].values for s in seeds], axis=0)
    m5a_avg_probs = np.mean([seed_preds_m5a[s]["prob_raw"].values for s in seeds], axis=0)
    test_y = seed_preds_m2[seeds[0]]["label"].values
    image_ids = seed_preds_m2[seeds[0]]["image_id"].values
    patient_ids = seed_preds_m2[seeds[0]]["patient_id"].values
    cdrs = df_manifest[test_idx]["cdr_mean"].values

    # Overall AUC of ensembled predictions
    ens_auc_m2 = float(roc_auc_score(test_y, m2_avg_probs))
    ens_auc_m5a = float(roc_auc_score(test_y, m5a_avg_probs))

    # Threshold for ensembled predictions
    # Best threshold on test set or median threshold
    med_thresh_m2 = float(np.median([m["threshold"] for m in m2_results]))
    med_thresh_m5a = float(np.median([m["threshold"] for m in m5a_results]))

    bin_m2 = (m2_avg_probs >= med_thresh_m2).astype(int)
    bin_m5a = (m5a_avg_probs >= med_thresh_m5a).astype(int)

    rescue_cases = []
    harmed_cases = []
    both_wrong_cases = []

    for i in range(len(test_y)):
        corr_m2 = (bin_m2[i] == test_y[i])
        corr_m5a = (bin_m5a[i] == test_y[i])
        record = {
            "image_id": image_ids[i],
            "patient_id": patient_ids[i],
            "label": "GON+" if test_y[i] == 1 else "GON-",
            "cdr": round(float(cdrs[i]), 4),
            "m2_prob": round(float(m2_avg_probs[i]), 4),
            "m5a_prob": round(float(m5a_avg_probs[i]), 4),
            "m2_pred": int(bin_m2[i]),
            "m5a_pred": int(bin_m5a[i]),
        }
        if (not corr_m2) and corr_m5a:
            rescue_cases.append(record)
        elif corr_m2 and (not corr_m5a):
            harmed_cases.append(record)
        elif (not corr_m2) and (not corr_m5a):
            both_wrong_cases.append(record)

    # Correlations
    m3_probs = pd.read_csv("results/M3_local_test_predictions.csv")["prob_raw"].values
    p_corr_m2_m3 = float(pearsonr(m2_avg_probs, m3_probs)[0])
    p_corr_m2_cdr = float(pearsonr(m2_avg_probs, cdrs)[0])
    p_corr_m3_cdr = float(pearsonr(m3_probs, cdrs)[0])
    s_corr_m2_cdr = float(spearmanr(m2_avg_probs, cdrs)[0])

    error_analysis_data = {
        "ensemble_m2_auc": round(ens_auc_m2, 4),
        "ensemble_m5a_auc": round(ens_auc_m5a, 4),
        "ensemble_delta_auc": round(ens_auc_m5a - ens_auc_m2, 4),
        "correlations": {
            "pearson_m2_m3": round(p_corr_m2_m3, 4),
            "pearson_m2_cdr": round(p_corr_m2_cdr, 4),
            "spearman_m2_cdr": round(s_corr_m2_cdr, 4),
            "pearson_m3_cdr": round(p_corr_m3_cdr, 4),
        },
        "rescue_cases_count": len(rescue_cases),
        "harmed_cases_count": len(harmed_cases),
        "both_wrong_count": len(both_wrong_cases),
        "rescue_cases": rescue_cases,
        "harmed_cases": harmed_cases,
        "both_wrong_cases": both_wrong_cases,
    }

    with open(out_base / "error_analysis.json", "w") as f:
        json.dump(error_analysis_data, f, indent=2)

    print("\n" + "=" * 80)
    print("STEP 5 MULTI-SEED SUMMARY (5 SEEDS: 42, 123, 2024, 3407, 777)")
    print("=" * 80)
    print(f"M2 Global MobileNet:   Mean AUC = {summary['M2_global']['mean_auc']:.4f} ± {summary['M2_global']['std_auc']:.4f} (range: {summary['M2_global']['min_auc']:.4f} - {summary['M2_global']['max_auc']:.4f})")
    print(f"M5a Global + CDR:      Mean AUC = {summary['M5a_global_cdr']['mean_auc']:.4f} ± {summary['M5a_global_cdr']['std_auc']:.4f} (range: {summary['M5a_global_cdr']['min_auc']:.4f} - {summary['M5a_global_cdr']['max_auc']:.4f})")
    print(f"Paired Delta AUC:      Mean Delta = {summary['paired_comparison']['mean_delta_auc']:+.4f} ± {summary['paired_comparison']['std_delta_auc']:.4f}")
    print(f"Ensemble AUC:          M2 = {ens_auc_m2:.4f} | M5a = {ens_auc_m5a:.4f} | Delta = {ens_auc_m5a - ens_auc_m2:+.4f}")
    print(f"Rescue Cases (CDR fixed M2): {len(rescue_cases)} cases")
    print(f"Harmed Cases (M5a broke M2): {len(harmed_cases)} cases")
    print(f"Correlations: Pearson r(M2, CDR) = {p_corr_m2_cdr:.4f}, Spearman rho(M2, CDR) = {s_corr_m2_cdr:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run_multiseed_suite()
