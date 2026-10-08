"""Evaluation of Automated CDR Pipeline: M1-auto and M5a-auto.

Strict Protocol:
1. M1-auto: Classifies directly from predicted CDR (threshold on Val, test on Test).
2. M5a-auto: MobileNetV3-Small Global features + predicted CDR (Scaler fit on Train only).
   Trained across 5 seeds: [42, 123, 2024, 3407, 777].
3. Comparison table:
   - M2 (Global alone)
   - M5a-oracle (Global + Expert CDR)
   - M5a-auto (Global + Predicted CDR)
4. End-to-end efficiency benchmark:
   - Parameters (Segmenter + Classifier)
   - Checkpoint size (MB)
   - FLOPs/MACs (combined)
   - End-to-end latency on GPU and CPU (batch size 1)
"""

import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.mobilenetv3 import MobileNetV3Classifier
from src.models.segmenter import MobileNetV3UNet


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


def evaluate_m1_auto(df_pred: pd.DataFrame) -> Dict:
    """Evaluates M1-auto (predicted CDR baseline)."""
    train_df = df_pred[df_pred["split"] == "train"].copy()
    val_df = df_pred[df_pred["split"] == "val"].copy()
    test_df = df_pred[df_pred["split"] == "test"].copy()

    # Optimal threshold on Val via Youden's J
    fpr, tpr, thresholds = roc_curve(val_df["label"], val_df["cdr_pred"])
    j = tpr - fpr
    best_j_idx = int(np.argmax(j))
    optimal_thresh = float(thresholds[best_j_idx])

    # Platt calibration fitted on Train
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(train_df[["cdr_pred"]].values, train_df["label"].values)

    test_probs_cal = calibrator.predict_proba(test_df[["cdr_pred"]].values)[:, 1]
    test_preds_bin = (test_df["cdr_pred"].values >= optimal_thresh).astype(int)

    test_auc = float(roc_auc_score(test_df["label"], test_df["cdr_pred"]))
    ci_eye, ci_cluster = compute_bootstrap_cis(test_df, "cdr_pred")
    tn, fp, fn, tp = confusion_matrix(test_df["label"], test_preds_bin).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc = float((tp + tn) / len(test_df))
    brier = float(brier_score_loss(test_df["label"], test_probs_cal))

    return {
        "model": "M1_auto_predicted_cdr",
        "threshold": round(optimal_thresh, 4),
        "test_auc": round(test_auc, 4),
        "ci_eye": ci_eye,
        "ci_cluster": ci_cluster,
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "accuracy": round(acc, 4),
        "brier_calibrated": round(brier, 4),
    }


def train_m5a_auto_single_seed(
    seed: int,
    df_pred: pd.DataFrame,
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

    train_idx = (df_pred["split"] == "train").values
    val_idx = (df_pred["split"] == "val").values
    test_idx = (df_pred["split"] == "test").values

    labels = torch.tensor(df_pred["label"].values, dtype=torch.float32).unsqueeze(1)

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

    test_preds_df = df_pred[test_idx].copy().reset_index(drop=True)
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


# ── End-to-End Latency & Efficiency Benchmark ───────────────────────────────

class EndToEndAutomatedSystem(nn.Module):
    """Full deployed pipeline: Fundus -> Segmenter -> Morphometry -> MobileNetV3 Global + CDR -> Output."""

    def __init__(self, segmenter: MobileNetV3UNet, backbone: MobileNetV3Classifier, scaler_mean: float, scaler_std: float):
        super().__init__()
        self.segmenter = segmenter
        self.backbone = backbone
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        in_dim = 1024 + 1
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(0.2),
            nn.Linear(in_dim, 1),
        )

    def forward(self, img_256: torch.Tensor) -> Tuple[torch.Tensor, float]:
        # Step 1: Segment OD and OC
        seg_logits = self.segmenter(img_256)
        probs = torch.sigmoid(seg_logits)[0].detach().cpu().numpy()
        p_disc = (probs[0] >= 0.5).astype(np.uint8)
        p_cup = (probs[1] >= 0.5).astype(np.uint8)

        # Step 2: Extract CDR
        from src.train_segmenter import compute_predicted_cdr
        cdr_pred, _, _ = compute_predicted_cdr(p_disc, p_cup)

        # Step 3: Standardize CDR
        cdr_norm = (cdr_pred - self.scaler_mean) / self.scaler_std
        cdr_tensor = torch.tensor([[cdr_norm]], dtype=torch.float32, device=img_256.device)

        # Step 4: Extract visual features & fuse
        img_224 = F.interpolate(img_256, size=(224, 224), mode="bilinear", align_corners=False)
        feat = self.backbone.extract_features(img_224)
        fused = torch.cat([feat, cdr_tensor], dim=1)
        logit = self.head(fused)
        prob = torch.sigmoid(logit)
        return prob, cdr_pred


def benchmark_end_to_end(device: torch.device, scaler_mean: float, scaler_std: float) -> Dict:
    print("\n--- Benchmarking End-to-End Automated System ---")
    seg = MobileNetV3UNet(pretrained=False).to(device)
    bb = MobileNetV3Classifier(pretrained=False, dropout_rate=0.2, num_classes=1).to(device)
    e2e = EndToEndAutomatedSystem(seg, bb, scaler_mean, scaler_std).to(device)
    e2e.eval()

    dummy_input = torch.randn(1, 3, 256, 256).to(device)

    # 1. Parameter counts
    seg_params = sum(p.numel() for p in seg.parameters())
    bb_params = sum(p.numel() for p in bb.parameters())
    head_params = sum(p.numel() for p in e2e.head.parameters())
    total_params = sum(p.numel() for p in e2e.parameters())

    # 2. FLOPs calculation
    with FlopCounterMode(display=False) as fcm_seg:
        with torch.no_grad():
            _ = seg(dummy_input)
    flops_seg = fcm_seg.get_total_flops()

    dummy_224 = torch.randn(1, 3, 224, 224).to(device)
    with FlopCounterMode(display=False) as fcm_bb:
        with torch.no_grad():
            _ = bb(dummy_224)
    flops_bb = fcm_bb.get_total_flops()

    total_flops = flops_seg + flops_bb + 2

    # 3. GPU Latency
    gpu_timings = []
    if device.type == "cuda":
        # Warmup
        with torch.no_grad():
            for _ in range(30):
                _ = e2e(dummy_input)
        torch.cuda.synchronize()

        for _ in range(100):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = e2e(dummy_input)
            torch.cuda.synchronize()
            gpu_timings.append((time.perf_counter() - t0) * 1000.0)

    # 4. CPU Latency
    e2e_cpu = EndToEndAutomatedSystem(seg.cpu(), bb.cpu(), scaler_mean, scaler_std).cpu()
    e2e_cpu.eval()
    dummy_cpu = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        for _ in range(10):
            _ = e2e_cpu(dummy_cpu)
    cpu_timings = []
    for _ in range(30):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = e2e_cpu(dummy_cpu)
        cpu_timings.append((time.perf_counter() - t0) * 1000.0)

    res = {
        "segmenter_params": seg_params,
        "classifier_params": bb_params,
        "head_params": head_params,
        "total_params": total_params,
        "total_params_m": round(total_params / 1e6, 3),
        "total_flops": total_flops,
        "total_mflops": round(total_flops / 1e6, 2),
        "total_gflops": round(total_flops / 1e9, 3),
        "total_mmacs": round((total_flops / 2.0) / 1e6, 2),
        "gpu_latency_ms": round(float(np.mean(gpu_timings)), 3) if gpu_timings else None,
        "gpu_std_ms": round(float(np.std(gpu_timings)), 3) if gpu_timings else None,
        "gpu_fps": round(1000.0 / float(np.mean(gpu_timings)), 1) if gpu_timings else None,
        "cpu_latency_ms": round(float(np.mean(cpu_timings)), 3),
        "cpu_std_ms": round(float(np.std(cpu_timings)), 3),
        "cpu_fps": round(1000.0 / float(np.mean(cpu_timings)), 1),
    }
    return res


# ── Main Suite ───────────────────────────────────────────────────────────────

def main():
    print("=== Step 6: Automated CDR Pipeline Evaluation (M1-auto & M5a-auto) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    pred_cdr_p = Path("data/manifests/papila_predicted_cdr.csv")
    df_pred = pd.read_csv(pred_cdr_p)

    out_dir = Path("results/automated")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Evaluate M1-auto (Standalone Predicted CDR) ───────────────────────
    print("\n--- 1. Evaluating M1-auto (Standalone Predicted CDR) ---")
    m1_auto_res = evaluate_m1_auto(df_pred)
    print(f"M1-auto Test AUC: {m1_auto_res['test_auc']:.4f} (Eye CI: {m1_auto_res['ci_eye']}, Clustered CI: {m1_auto_res['ci_cluster']})")
    print(f"Sensitivity: {m1_auto_res['sensitivity']:.4f} | Specificity: {m1_auto_res['specificity']:.4f} | Accuracy: {m1_auto_res['accuracy']:.4f} | Calibrated Brier: {m1_auto_res['brier_calibrated']:.4f}")

    # ── 2. Load Global Features & Standardize Predicted CDR ──────────────────
    print("\n--- 2. Loading Global Features & Standardizing Predicted CDR on Train ---")
    feat_dict = torch.load("data/features/papila_m2_global_features.pt", weights_only=True)
    f_global = feat_dict["features"]  # (420, 1024)

    train_idx = (df_pred["split"] == "train").values
    val_idx = (df_pred["split"] == "val").values
    test_idx = (df_pred["split"] == "test").values

    # Standardize CDR strictly on Train
    scaler = StandardScaler()
    cdr_train = scaler.fit_transform(df_pred.loc[train_idx, ["cdr_pred"]].values)
    cdr_val = scaler.transform(df_pred.loc[val_idx, ["cdr_pred"]].values)
    cdr_test = scaler.transform(df_pred.loc[test_idx, ["cdr_pred"]].values)

    cdr_all = np.zeros((len(df_pred), 1), dtype=np.float32)
    cdr_all[train_idx] = cdr_train
    cdr_all[val_idx] = cdr_val
    cdr_all[test_idx] = cdr_test
    cdr_tensor = torch.tensor(cdr_all, dtype=torch.float32)

    scaler_mean = float(scaler.mean_[0])
    scaler_std = float(scaler.scale_[0])

    # ── 3. Train & Evaluate M5a-auto across 5 Seeds ───────────────────────────
    print("\n--- 3. Training & Evaluating M5a-auto (Global + Predicted CDR) across 5 Seeds ---")
    seeds = [42, 123, 2024, 3407, 777]
    m5a_auto_runs = []
    seed_preds = {}

    for s in seeds:
        df_seed_pred, met = train_m5a_auto_single_seed(
            s, df_pred, f_global, cdr_tensor, device, epochs=40, batch_size=16, lr=1e-3
        )
        m5a_auto_runs.append(met)
        seed_preds[s] = df_seed_pred
        df_seed_pred.to_csv(out_dir / f"M5a_auto_seed_{s}_test_predictions.csv", index=False)
        print(f"  Seed {s} -> Test AUC: {met['test_auc']:.4f} | Sens: {met['sensitivity']:.4f} | Spec: {met['specificity']:.4f} | Brier: {met['brier_calibrated']:.4f}")

    # Aggregate M5a-auto metrics
    aucs = [m["test_auc"] for m in m5a_auto_runs]
    briers = [m["brier_calibrated"] for m in m5a_auto_runs]
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))

    # Ensemble of 5 seeds
    ens_probs = np.mean([seed_preds[s]["prob_raw"].values for s in seeds], axis=0)
    test_y = seed_preds[seeds[0]]["label"].values
    ens_auc = float(roc_auc_score(test_y, ens_probs))

    # Paired comparison with Oracle M5a and M2
    multiseed_summary_p = Path("results/multiseed/multiseed_summary.json")
    with open(multiseed_summary_p) as f:
        step5_summary = json.load(f)

    m2_mean_auc = step5_summary["M2_global"]["mean_auc"]
    m5a_orc_mean_auc = step5_summary["M5a_global_cdr"]["mean_auc"]

    # ── 4. End-to-End Benchmark ──────────────────────────────────────────────
    e2e_bench = benchmark_end_to_end(device, scaler_mean, scaler_std)

    # ── 5. Save Summary JSON ─────────────────────────────────────────────────
    final_summary = {
        "M1_auto": m1_auto_res,
        "M5a_auto": {
            "mean_auc": round(mean_auc, 4),
            "std_auc": round(std_auc, 4),
            "min_auc": round(float(np.min(aucs)), 4),
            "max_auc": round(float(np.max(aucs)), 4),
            "mean_calibrated_brier": round(float(np.mean(briers)), 4),
            "ensemble_auc": round(ens_auc, 4),
            "runs": m5a_auto_runs,
        },
        "comparison_3_levels": {
            "M2_Global_mean_auc": m2_mean_auc,
            "M5a_Oracle_mean_auc": m5a_orc_mean_auc,
            "M5a_Auto_mean_auc": round(mean_auc, 4),
            "delta_auto_vs_m2": round(mean_auc - m2_mean_auc, 4),
            "retention_of_oracle_gain_pct": round(float((mean_auc - m2_mean_auc) / (m5a_orc_mean_auc - m2_mean_auc) * 100), 2) if (m5a_orc_mean_auc != m2_mean_auc) else 100.0,
        },
        "end_to_end_efficiency": e2e_bench,
    }

    out_json = out_dir / "step6_automated_evaluation.json"
    with open(out_json, "w") as f:
        json.dump(final_summary, f, indent=2)
    print(f"\nSaved complete results to {out_json}")

    # ── 6. Print Formatted Table ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 6 SUMMARY: 3-TIER COMPARISON & AUTOMATION RETENTION")
    print("=" * 80)
    print(f"M2 (Global Alone):            Mean AUC = {m2_mean_auc:.4f}")
    print(f"M5a-Oracle (Global + Expert):  Mean AUC = {m5a_orc_mean_auc:.4f}")
    print(f"M5a-Auto (Global + Pred CDR):  Mean AUC = {mean_auc:.4f} ± {std_auc:.4f} (range: {np.min(aucs):.4f} - {np.max(aucs):.4f}) | Ensemble AUC: {ens_auc:.4f}")
    print(f"Delta Auto vs M2:              {mean_auc - m2_mean_auc:+.4f}")
    print(f"Retention of Oracle Gain:      {final_summary['comparison_3_levels']['retention_of_oracle_gain_pct']:.1f}%")
    print("-" * 80)
    print(f"End-to-End Parameters:         {e2e_bench['total_params_m']}M params")
    print(f"End-to-End FLOPs:              {e2e_bench['total_gflops']} GFLOPs ({e2e_bench['total_mmacs']} MMACs)")
    print(f"End-to-End Latency RTX 3050:   {e2e_bench['gpu_latency_ms']} ± {e2e_bench['gpu_std_ms']} ms ({e2e_bench['gpu_fps']} FPS)")
    print(f"End-to-End Latency CPU:        {e2e_bench['cpu_latency_ms']} ± {e2e_bench['cpu_std_ms']} ms ({e2e_bench['cpu_fps']} FPS)")
    print("=" * 80)


if __name__ == "__main__":
    main()

