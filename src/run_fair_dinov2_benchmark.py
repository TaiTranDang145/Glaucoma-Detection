"""Step 8: Fair DINOv2 Full Fine-Tuning Comparison at 224x224 and 392x392.

Methodology:
1. Architecture: timm `vit_base_patch14_dinov2` (86.58M params).
2. Protocol: Full fine-tuning (backbone + head) on PAPILA Train (294 images, 147 patients).
   - Input resolutions: 224x224 and 392x392 (native GONet patch grid: 28x28 patches).
   - Hyperparameters: AdamW (lr=1e-5, weight_decay=0.01), CosineAnnealingLR, BCEWithLogitsLoss with pos_weight=[233/61].
   - Mixed precision (torch.amp.autocast('cuda'), GradScaler) for memory and compute efficiency on RTX 3050 Laptop GPU.
   - Checkpoint selection: Strictly based on Validation ROC-AUC (62 images, 31 patients).
   - Operating threshold: Tuned on Validation via Youden's J.
   - Platt scaling: Fitted strictly on Train partition.
   - Test evaluation: Permanently frozen Test partition (64 images, 32 patients).
3. Evaluated across seeds: [42, 123, 2024].
4. Paired patient-clustered bootstrap (2,000 resamples) for Delta AUC = AUC(Proposed M5a-auto-OOF) - AUC(DINOv2).
5. Comprehensive computational efficiency benchmarking (FLOPs, latency GPU/CPU).
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
import timm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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


# ── Transforms & Dataset ─────────────────────────────────────────────────────

def get_transforms(split: str, target_size: int) -> A.Compose:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if split == "train":
        return A.Compose([
            A.Resize(target_size, target_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.4),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(target_size, target_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


class FundusDataset(Dataset):
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


# ── Bootstrap Helpers ────────────────────────────────────────────────────────

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


def compute_paired_patient_bootstrap_delta_auc(
    df_base: pd.DataFrame,
    df_comp: pd.DataFrame,
    score_col_base: str,
    score_col_comp: str,
    n_iterations: int = 2000,
    seed: int = 42,
) -> Dict:
    """Computes paired patient-clustered bootstrap for Delta AUC = AUC(comp) - AUC(base)."""
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


# ── Full Fine-Tuning Function ────────────────────────────────────────────────

def train_dinov2_finetune_single_seed(
    seed: int,
    resolution: int,
    cache_dir: Path,
    df_manifest: pd.DataFrame,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 8,
    accum_steps: int = 1,
    lr: float = 1e-5,
    weight_decay: float = 0.01,
) -> Tuple[pd.DataFrame, Dict]:
    seed_everything(seed)
    train_df = df_manifest[df_manifest["split"] == "train"].reset_index(drop=True)
    val_df = df_manifest[df_manifest["split"] == "val"].reset_index(drop=True)
    test_df = df_manifest[df_manifest["split"] == "test"].reset_index(drop=True)

    train_tf = get_transforms("train", resolution)
    eval_tf = get_transforms("val", resolution)

    train_ds = FundusDataset(train_df, cache_dir, train_tf)
    val_ds = FundusDataset(val_df, cache_dir, eval_tf)
    test_ds = FundusDataset(test_df, cache_dir, eval_tf)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
        worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Instantiate pre-trained DINOv2 ViT-B/14
    model = timm.create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=1, dynamic_img_size=True).to(device)

    pos_weight = torch.tensor([233.0 / 61.0], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    scaler = GradScaler("cuda")

    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = -1

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_tr = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            with autocast("cuda"):
                logits = model(imgs)
                loss = loss_fn(logits, labels) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps
            n_tr += 1

        scheduler.step()
        train_loss /= n_tr

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
                with autocast("cuda"):
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

        mark = " [*BEST]" if is_best else ""
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {v_loss:.4f} | Val AUC: {v_auc:.4f}{mark}")

    print(f"  Finished in {time.time() - t0:.1f}s. Best Epoch: {best_epoch} with Val AUC: {best_val_auc:.4f}")

    # Load best checkpoint
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()

    # Optimal Threshold on Val via Youden's J
    v_logits = []
    v_labels = []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            with autocast("cuda"):
                logits = model(imgs)
            v_logits.extend(logits.squeeze(-1).cpu().numpy())
            v_labels.extend(labels.squeeze(-1).cpu().numpy())
    v_probs = 1.0 / (1.0 + np.exp(-np.array(v_logits)))
    fpr, tpr, thresholds = roc_curve(v_labels, v_probs)
    j = tpr - fpr
    best_j_idx = int(np.argmax(j))
    optimal_thresh = float(thresholds[best_j_idx])

    # Platt calibrator fitted strictly on Train
    train_eval_loader = DataLoader(FundusDataset(train_df, cache_dir, eval_tf), batch_size=batch_size, shuffle=False, num_workers=4)
    tr_logits = []
    tr_labels = []
    with torch.no_grad():
        for batch in train_eval_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            with autocast("cuda"):
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
            with autocast("cuda"):
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
    test_preds_df["prob_calibrated"] = calibrator.predict_proba(test_preds_df[["logit"]].values)[:, 1]
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
        "resolution": resolution,
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


# ── Latency Benchmarking ─────────────────────────────────────────────────────

def benchmark_dinov2_resolution(resolution: int, device: torch.device) -> Dict:
    print(f"\n--- Measuring DINOv2 Compute & Latency at {resolution}x{resolution} ---")
    model = timm.create_model("vit_base_patch14_dinov2", pretrained=False, num_classes=1, dynamic_img_size=True).to(device)
    model.eval()

    dummy = torch.randn(1, 3, resolution, resolution).to(device)
    with FlopCounterMode(display=False) as fcm:
        with torch.no_grad():
            _ = model(dummy)
    flops = fcm.get_total_flops()

    # GPU Latency
    gpu_timings = []
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy)
        torch.cuda.synchronize()
        for _ in range(50):
            t0 = time.perf_counter()
            _ = model(dummy)
            torch.cuda.synchronize()
            gpu_timings.append((time.perf_counter() - t0) * 1000.0)

    # CPU Latency
    model_cpu = model.cpu()
    model_cpu.eval()
    dummy_cpu = torch.randn(1, 3, resolution, resolution)
    with torch.no_grad():
        for _ in range(5):
            _ = model_cpu(dummy_cpu)
    cpu_timings = []
    for _ in range(15):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_cpu(dummy_cpu)
        cpu_timings.append((time.perf_counter() - t0) * 1000.0)

    total_params = sum(p.numel() for p in model.parameters())
    return {
        "resolution": resolution,
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


# ── Main Suite ───────────────────────────────────────────────────────────────

def run_suite():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Step 8: Fair DINOv2 Full Fine-Tuning Comparison Suite ===")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_p = Path("data/manifests/papila_morphology.csv")
    df_manifest = pd.read_csv(manifest_p)

    out_dir = Path("results/dinov2_fair")
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 2024]
    results_224 = []
    results_392 = []
    preds_224 = {}
    preds_392 = {}

    # 1. Full Fine-Tuning DINOv2 @ 224x224
    print("\n" + "=" * 80)
    print("PHASE 1: Full Fine-Tuning DINOv2 ViT-B/14 @ 224x224 (Batch size 8, lr=1e-5)")
    print("=" * 80)
    cache_224 = Path("data/cache/fundus_256")
    for s in seeds:
        print(f"\n>>> Running DINOv2@224 Seed {s}...")
        df_p, met = train_dinov2_finetune_single_seed(
            seed=s, resolution=224, cache_dir=cache_224, df_manifest=df_manifest,
            device=device, epochs=15, batch_size=8, accum_steps=1, lr=1e-5
        )
        results_224.append(met)
        preds_224[s] = df_p
        df_p.to_csv(out_dir / f"dinov2_224_seed_{s}_test_predictions.csv", index=False)
        print(f"  Result Seed {s} -> Test AUC: {met['test_auc']:.4f} | Sens: {met['sensitivity']:.4f} | Spec: {met['specificity']:.4f} | Brier: {met['brier_calibrated']:.4f}")

    # 2. Full Fine-Tuning DINOv2 @ 392x392 (GONet Native Resolution)
    print("\n" + "=" * 80)
    print("PHASE 2: Full Fine-Tuning DINOv2 ViT-B/14 @ 392x392 (Batch size 4, accum 2 = eff bs 8, lr=1e-5)")
    print("=" * 80)
    cache_392 = Path("data/cache/fundus_392")
    for s in seeds:
        print(f"\n>>> Running DINOv2@392 Seed {s}...")
        df_p, met = train_dinov2_finetune_single_seed(
            seed=s, resolution=392, cache_dir=cache_392, df_manifest=df_manifest,
            device=device, epochs=15, batch_size=4, accum_steps=2, lr=1e-5
        )
        results_392.append(met)
        preds_392[s] = df_p
        df_p.to_csv(out_dir / f"dinov2_392_seed_{s}_test_predictions.csv", index=False)
        print(f"  Result Seed {s} -> Test AUC: {met['test_auc']:.4f} | Sens: {met['sensitivity']:.4f} | Spec: {met['specificity']:.4f} | Brier: {met['brier_calibrated']:.4f}")

    # 3. Efficiency Benchmarks
    bench_224 = benchmark_dinov2_resolution(224, device)
    bench_392 = benchmark_dinov2_resolution(392, device)

    # 4. Aggregates & Ensemble
    aucs_224 = [m["test_auc"] for m in results_224]
    aucs_392 = [m["test_auc"] for m in results_392]

    ens_probs_224 = np.mean([preds_224[s]["prob_raw"].values for s in seeds], axis=0)
    ens_probs_392 = np.mean([preds_392[s]["prob_raw"].values for s in seeds], axis=0)
    test_y = preds_224[seeds[0]]["label"].values

    ens_auc_224 = float(roc_auc_score(test_y, ens_probs_224))
    ens_auc_392 = float(roc_auc_score(test_y, ens_probs_392))

    # Save ensemble predictions
    df_ens_224 = preds_224[seeds[0]].copy()
    df_ens_224["prob_raw"] = ens_probs_224
    df_ens_224.to_csv(out_dir / "dinov2_224_ensemble_test_predictions.csv", index=False)

    df_ens_392 = preds_392[seeds[0]].copy()
    df_ens_392["prob_raw"] = ens_probs_392
    df_ens_392.to_csv(out_dir / "dinov2_392_ensemble_test_predictions.csv", index=False)

    # 5. Paired Bootstrap vs Proposed Autonomous Model (M5a-auto-OOF)
    print("\n--- Running Paired Patient-Clustered Bootstrap Tests vs Proposed System ---")
    df_proposed_s42 = pd.read_csv("results/automated/M5a_auto_oof_seed_42_test_predictions.csv")

    # Load all 3 seeds of proposed model to construct proposed ensemble
    df_prop_list = [pd.read_csv(f"results/automated/M5a_auto_oof_seed_{s}_test_predictions.csv") for s in seeds]
    df_proposed_ens = df_proposed_s42.copy()
    df_proposed_ens["prob_raw"] = np.mean([df_p["prob_raw"].values for df_p in df_prop_list], axis=0)

    paired_vs_224_s42 = compute_paired_patient_bootstrap_delta_auc(preds_224[42], df_proposed_s42, "prob_raw", "prob_raw")
    paired_vs_392_s42 = compute_paired_patient_bootstrap_delta_auc(preds_392[42], df_proposed_s42, "prob_raw", "prob_raw")
    paired_vs_224_ens = compute_paired_patient_bootstrap_delta_auc(df_ens_224, df_proposed_ens, "prob_raw", "prob_raw")
    paired_vs_392_ens = compute_paired_patient_bootstrap_delta_auc(df_ens_392, df_proposed_ens, "prob_raw", "prob_raw")

    # Load Proposed model metrics
    with open("results/automated/step6_oof_summary.json") as f:
        step6_summary = json.load(f)

    prop_mean_auc = step6_summary["M5a_auto_oof"]["mean_auc"]
    prop_ens_auc = step6_summary["M5a_auto_oof"]["ensemble_auc"]

    final_payload = {
        "DINOv2_224_finetuned": {
            "mean_auc": round(float(np.mean(aucs_224)), 4),
            "std_auc": round(float(np.std(aucs_224)), 4),
            "ensemble_auc": round(ens_auc_224, 4),
            "mean_sensitivity": round(float(np.mean([m["sensitivity"] for m in results_224])), 4),
            "mean_specificity": round(float(np.mean([m["specificity"] for m in results_224])), 4),
            "mean_accuracy": round(float(np.mean([m["accuracy"] for m in results_224])), 4),
            "mean_calibrated_brier": round(float(np.mean([m["brier_calibrated"] for m in results_224])), 4),
            "runs": results_224,
            "efficiency": bench_224,
        },
        "DINOv2_392_finetuned": {
            "mean_auc": round(float(np.mean(aucs_392)), 4),
            "std_auc": round(float(np.std(aucs_392)), 4),
            "ensemble_auc": round(ens_auc_392, 4),
            "mean_sensitivity": round(float(np.mean([m["sensitivity"] for m in results_392])), 4),
            "mean_specificity": round(float(np.mean([m["specificity"] for m in results_392])), 4),
            "mean_accuracy": round(float(np.mean([m["accuracy"] for m in results_392])), 4),
            "mean_calibrated_brier": round(float(np.mean([m["brier_calibrated"] for m in results_392])), 4),
            "runs": results_392,
            "efficiency": bench_392,
        },
        "paired_comparisons": {
            "delta_proposed_vs_dinov2_224_seed42": paired_vs_224_s42,
            "delta_proposed_vs_dinov2_392_seed42": paired_vs_392_s42,
            "delta_proposed_vs_dinov2_224_ensemble": paired_vs_224_ens,
            "delta_proposed_vs_dinov2_392_ensemble": paired_vs_392_ens,
        },
        "summary_comparison": {
            "Proposed_M5a_auto_OOF_mean_auc": prop_mean_auc,
            "Proposed_M5a_auto_OOF_ensemble_auc": prop_ens_auc,
            "DINOv2_224_mean_auc": round(float(np.mean(aucs_224)), 4),
            "DINOv2_224_ensemble_auc": round(ens_auc_224, 4),
            "DINOv2_392_mean_auc": round(float(np.mean(aucs_392)), 4),
            "DINOv2_392_ensemble_auc": round(ens_auc_392, 4),
            "delta_proposed_minus_dinov2_224_mean": round(prop_mean_auc - float(np.mean(aucs_224)), 4),
            "delta_proposed_minus_dinov2_392_mean": round(prop_mean_auc - float(np.mean(aucs_392)), 4),
            "delta_proposed_minus_dinov2_224_ensemble": round(prop_ens_auc - ens_auc_224, 4),
            "delta_proposed_minus_dinov2_392_ensemble": round(prop_ens_auc - ens_auc_392, 4),
        },
    }

    out_file = out_dir / "fair_dinov2_comparison.json"
    with open(out_file, "w") as f:
        json.dump(final_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("STEP 8 FAIR DINOv2 FULL FINE-TUNING SUMMARY")
    print("=" * 80)
    print(f"Proposed Model (M5a-auto-OOF): Mean AUC = {prop_mean_auc:.4f} (2.78M params, 1.98 GFLOPs)")
    print(f"DINOv2 ViT-B/14 @ 224 (Fine-tuned): Mean AUC = {final_payload['DINOv2_224_finetuned']['mean_auc']:.4f} ± {final_payload['DINOv2_224_finetuned']['std_auc']:.4f} | Ensemble: {ens_auc_224:.4f}")
    print(f"DINOv2 ViT-B/14 @ 392 (Fine-tuned): Mean AUC = {final_payload['DINOv2_392_finetuned']['mean_auc']:.4f} ± {final_payload['DINOv2_392_finetuned']['std_auc']:.4f} | Ensemble: {ens_auc_392:.4f}")
    print("-" * 80)
    print(f"Advantage vs DINOv2@224: {final_payload['summary_comparison']['delta_proposed_minus_dinov2_224_mean']:+.4f} AUC (Paired CI: {paired_vs_224_s42['bootstrap_95ci_patient_clustered']}, P(>0)={paired_vs_224_s42['pct_bootstrap_positive']}%)")
    print(f"Advantage vs DINOv2@392: {final_payload['summary_comparison']['delta_proposed_minus_dinov2_392_mean']:+.4f} AUC (Paired CI: {paired_vs_392_s42['bootstrap_95ci_patient_clustered']}, P(>0)={paired_vs_392_s42['pct_bootstrap_positive']}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_suite()
