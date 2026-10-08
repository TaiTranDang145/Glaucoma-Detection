"""DINOv2 Foundation Model Reference Evaluation on PAPILA Frozen Split.

Compares:
- DINOv2 ViT-B/14 (86.6M params, 43.89 GFLOPs) vs Proposed Lightweight Pipeline (2.78M params, 1.98 GFLOPs).
- Protocol: 5 seeds [42, 123, 2024, 3407, 777], Youden's J on Val, Platt calibrator on Train, frozen Test (64 images).
- Measures: Test ROC-AUC, Sensitivity, Specificity, Brier, params, FLOPs, GPU/CPU latency.
"""

import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class SimpleFundusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_p = self.img_dir / f"{row['image_id']}.png"
        img = cv2.imread(str(img_p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor = torch.from_numpy(img).permute(2, 0, 1)

        return {
            "image": tensor,
            "label": torch.tensor(row["label"], dtype=torch.float32),
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "eye": row["eye"],
        }


def extract_dinov2_features(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            f = model(imgs)
            feats.append(f.cpu())
            labels.append(batch["label"])
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def train_dinov2_head_single_seed(
    seed: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    test_df: pd.DataFrame,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
) -> Tuple[pd.DataFrame, Dict]:
    seed_everything(seed)
    in_dim = X_train.shape[1]

    train_ds = TensorDataset(X_train.to(device), y_train.unsqueeze(1).to(device))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    head = nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Dropout(0.3),
        nn.Linear(in_dim, 1),
    ).to(device)

    pos_weight = torch.tensor([233.0 / 61.0], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        head.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = head(bx)
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        head.eval()
        with torch.no_grad():
            v_logits = head(X_val.to(device))
            v_loss = loss_fn(v_logits, y_val.unsqueeze(1).to(device)).item()
            v_probs = torch.sigmoid(v_logits).squeeze(1).cpu().numpy()
            v_auc = float(roc_auc_score(y_val.numpy(), v_probs))

        is_best = (v_auc > best_val_auc) or (np.isclose(v_auc, best_val_auc) and v_loss < best_val_loss)
        if is_best:
            best_val_auc = v_auc
            best_val_loss = v_loss
            best_weights = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    head.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    head.eval()

    # Val threshold
    with torch.no_grad():
        v_logits = head(X_val.to(device)).squeeze(1).cpu().numpy()
        v_probs = 1.0 / (1.0 + np.exp(-v_logits))
    val_y = y_val.numpy().astype(int)
    fpr, tpr, thresholds = roc_curve(val_y, v_probs)
    j = tpr - fpr
    best_j_idx = int(np.argmax(j))
    optimal_thresh = float(thresholds[best_j_idx])

    # Platt calibrator on Train
    with torch.no_grad():
        train_logits = head(X_train.to(device)).squeeze(1).cpu().numpy()
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(train_logits.reshape(-1, 1), y_train.numpy().astype(int))

    # Test Evaluation
    with torch.no_grad():
        test_logits = head(X_test.to(device)).squeeze(1).cpu().numpy()
        test_probs_raw = 1.0 / (1.0 + np.exp(-test_logits))

    test_cal_probs = calibrator.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    test_y = y_test.numpy().astype(int)

    test_preds_df = test_df.copy().reset_index(drop=True)
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


def main():
    print("=== DINOv2 Foundation Model Heavy Reference Evaluation ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_p = Path("data/manifests/papila_morphology.csv")
    df = pd.read_csv(manifest_p)
    img_dir = Path("data/cache/fundus_256")
    out_dir = Path("results/dinov2_reference")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # 1. Instantiate DINOv2 ViT-B/14
    print("\n--- 1. Loading Pre-trained DINOv2 ViT-B/14 ---")
    dino_model = timm.create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=0, dynamic_img_size=True).to(device)
    total_params = sum(p.numel() for p in dino_model.parameters())
    print(f"DINOv2 Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # 2. Extract features
    print("\n--- 2. Extracting DINOv2 Frozen Features for PAPILA ---")
    tr_loader = DataLoader(SimpleFundusDataset(train_df, img_dir), batch_size=16, shuffle=False, num_workers=4)
    v_loader = DataLoader(SimpleFundusDataset(val_df, img_dir), batch_size=16, shuffle=False, num_workers=4)
    te_loader = DataLoader(SimpleFundusDataset(test_df, img_dir), batch_size=16, shuffle=False, num_workers=4)

    X_train, y_train = extract_dinov2_features(dino_model, tr_loader, device)
    X_val, y_val = extract_dinov2_features(dino_model, v_loader, device)
    X_test, y_test = extract_dinov2_features(dino_model, te_loader, device)
    print(f"Extracted features shape: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    # 3. Train probe head across 5 seeds
    print("\n--- 3. Training Linear Probe across 5 Seeds ---")
    seeds = [42, 123, 2024, 3407, 777]
    dino_runs = []
    seed_preds = {}

    for s in seeds:
        df_p, met = train_dinov2_head_single_seed(
            s, X_train, y_train, X_val, y_val, X_test, y_test, test_df, device, epochs=50, batch_size=16, lr=1e-3
        )
        dino_runs.append(met)
        seed_preds[s] = df_p
        df_p.to_csv(out_dir / f"dinov2_seed_{s}_test_predictions.csv", index=False)
        print(f"  Seed {s} -> Test AUC: {met['test_auc']:.4f} | Sens: {met['sensitivity']:.4f} | Spec: {met['specificity']:.4f} | Brier: {met['brier_calibrated']:.4f}")

    aucs = [m["test_auc"] for m in dino_runs]
    briers = [m["brier_calibrated"] for m in dino_runs]
    sens = [m["sensitivity"] for m in dino_runs]
    spec = [m["specificity"] for m in dino_runs]

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    ens_probs = np.mean([seed_preds[s]["prob_raw"].values for s in seeds], axis=0)
    ens_auc = float(roc_auc_score(y_test.numpy(), ens_probs))

    # 4. Measure FLOPs & Latency
    print("\n--- 4. Benchmarking DINOv2 Compute & Latency ---")
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with FlopCounterMode(display=False) as fcm:
        with torch.no_grad():
            _ = dino_model(dummy)
    flops = fcm.get_total_flops()

    # GPU Latency
    gpu_timings = []
    with torch.no_grad():
        for _ in range(30):
            _ = dino_model(dummy)
        torch.cuda.synchronize()
        for _ in range(100):
            t0 = time.perf_counter()
            _ = dino_model(dummy)
            torch.cuda.synchronize()
            gpu_timings.append((time.perf_counter() - t0) * 1000.0)

    # CPU Latency
    dino_cpu = dino_model.cpu()
    dino_cpu.eval()
    dummy_cpu = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(5):
            _ = dino_cpu(dummy_cpu)
    cpu_timings = []
    for _ in range(20):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = dino_cpu(dummy_cpu)
        cpu_timings.append((time.perf_counter() - t0) * 1000.0)

    bench_res = {
        "params": total_params,
        "params_m": round(total_params / 1e6, 2),
        "checkpoint_size_mb": round(total_params * 4 / (1024 * 1024), 2),
        "total_flops": flops,
        "gflops": round(flops / 1e9, 2),
        "mmacs": round((flops / 2.0) / 1e6, 2),
        "gpu_latency_ms": round(float(np.mean(gpu_timings)), 3),
        "gpu_std_ms": round(float(np.std(gpu_timings)), 3),
        "gpu_fps": round(1000.0 / float(np.mean(gpu_timings)), 1),
        "cpu_latency_ms": round(float(np.mean(cpu_timings)), 3),
        "cpu_std_ms": round(float(np.std(cpu_timings)), 3),
        "cpu_fps": round(1000.0 / float(np.mean(cpu_timings)), 1),
    }

    results = {
        "model": "DINOv2_ViT_B_14",
        "mean_test_auc": round(mean_auc, 4),
        "std_test_auc": round(std_auc, 4),
        "min_test_auc": round(float(np.min(aucs)), 4),
        "max_test_auc": round(float(np.max(aucs)), 4),
        "ensemble_test_auc": round(ens_auc, 4),
        "mean_sensitivity": round(float(np.mean(sens)), 4),
        "mean_specificity": round(float(np.mean(spec)), 4),
        "mean_brier": round(float(np.mean(briers)), 4),
        "efficiency": bench_res,
        "runs": dino_runs,
    }

    out_file = out_dir / "dinov2_reference_metrics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("DINOV2 FOUNDATION MODEL REFERENCE RESULTS")
    print("=" * 80)
    print(f"DINOv2 ViT-B/14 Mean AUC:      {mean_auc:.4f} ± {std_auc:.4f} (range: {np.min(aucs):.4f} - {np.max(aucs):.4f}) | Ensemble: {ens_auc:.4f}")
    print(f"DINOv2 Mean Sensitivity:       {np.mean(sens)*100:.1f}%")
    print(f"DINOv2 Mean Specificity:       {np.mean(spec)*100:.1f}%")
    print(f"DINOv2 Calibrated Brier:       {np.mean(briers):.4f}")
    print("-" * 80)
    print(f"Parameters:                    {bench_res['params_m']}M params (~{bench_res['checkpoint_size_mb']} MB)")
    print(f"FLOPs:                         {bench_res['gflops']} GFLOPs ({bench_res['mmacs']} MMACs)")
    print(f"GPU Latency RTX 3050:          {bench_res['gpu_latency_ms']} ± {bench_res['gpu_std_ms']} ms ({bench_res['gpu_fps']} FPS)")
    print(f"CPU Latency Intel Core:        {bench_res['cpu_latency_ms']} ± {bench_res['cpu_std_ms']} ms ({bench_res['cpu_fps']} FPS)")
    print("=" * 80)


if __name__ == "__main__":
    main()
