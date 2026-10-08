"""Training and evaluation pipeline for Multimodal and Multi-branch GON Fusion Models.

Implements the complete Step 4 ablation matrix:
- M4:  Global Full Fundus + Local Optic Disc (z_global + z_local)
- M5a: Global Full Fundus + Clinical CDR (z_global + CDR)
- M5b: Global Full Fundus + Local Optic Disc + Clinical CDR (z_global + z_local + CDR)

Strict methodological rigor:
1. All fusion heads trained on GPU (NVIDIA GeForce RTX 3050 Laptop GPU).
2. CDR standardized via StandardScaler fitted ONLY on the Train partition.
3. Checkpoint selection strictly on Validation set ROC-AUC.
4. Decision threshold selected strictly on Validation set via Youden's J.
5. Calibration fitted on Train partition via Platt scaling.
6. Both eye-level and patient-clustered bootstrap 95% CIs reported on the frozen Test partition.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Reproducibility ──────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Fusion Head Architecture ─────────────────────────────────────────────────

class FusionHead(nn.Module):
    """LayerNorm-stabilized classification head for multi-modal feature fusion."""

    def __init__(self, in_features: int, hidden_dim: int = 0, dropout: float = 0.2):
        super().__init__()
        self.in_features = in_features
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(dropout),
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(dropout),
                nn.Linear(in_features, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


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


# ── Main Fusion Pipeline ─────────────────────────────────────────────────────

def train_and_evaluate_fusion(
    variant: str,
    df_manifest: pd.DataFrame,
    f_global: torch.Tensor,
    f_local: torch.Tensor,
    cdr_tensor: torch.Tensor,
    args,
    device: torch.device,
) -> Dict:
    seed_everything(args.seed)
    print(f"\n============================================================")
    print(f"EXPERIMENT {variant.upper()}: Training on GPU ({device})")
    print(f"============================================================")

    # Construct input feature tensor based on variant
    if variant == "M4_global_local":
        features = torch.cat([f_global, f_local], dim=1)
        input_desc = "Global Full Fundus (1024-d) + Local OD Crop (1024-d)"
    elif variant == "M5a_global_cdr":
        features = torch.cat([f_global, cdr_tensor], dim=1)
        input_desc = "Global Full Fundus (1024-d) + Clinical Consensus CDR (1-d)"
    elif variant == "M5b_global_local_cdr":
        features = torch.cat([f_global, f_local, cdr_tensor], dim=1)
        input_desc = "Global Full Fundus (1024-d) + Local OD Crop (1024-d) + Clinical CDR (1-d)"
    else:
        raise ValueError(f"Unknown fusion variant: {variant}")

    in_dim = features.shape[1]
    print(f"Feature matrix dimension: {features.shape} (Input features: {in_dim})")
    print(f"Modalities: {input_desc}")

    # Splits
    train_idx = (df_manifest["split"] == "train").values
    val_idx = (df_manifest["split"] == "val").values
    test_idx = (df_manifest["split"] == "test").values

    labels = torch.tensor(df_manifest["label"].values, dtype=torch.float32).unsqueeze(1)

    X_train, y_train = features[train_idx].to(device), labels[train_idx].to(device)
    X_val, y_val = features[val_idx].to(device), labels[val_idx].to(device)
    X_test, y_test = features[test_idx].to(device), labels[test_idx].to(device)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Initialize Fusion Head on GPU
    model = FusionHead(in_features=in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    params_info = model.count_parameters()
    print(f"Fusion Head parameters: {params_info['total']:,} (Trainable: {params_info['trainable']:,})")

    # Pos weight for class imbalance
    n_neg = (y_train == 0).sum().item()
    n_pos = (y_train == 1).sum().item()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    out_dir = Path(f"outputs/models/{variant}")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best_model.pt"

    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = model(X_val)
            v_loss = loss_fn(v_logits, y_val).item()
            v_probs = torch.sigmoid(v_logits).squeeze(1).cpu().numpy()
            v_auc = float(roc_auc_score(y_val.cpu().numpy(), v_probs))

        current_lr = scheduler.get_last_lr()[0]
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(v_loss, 4),
            "val_auc": round(v_auc, 4),
            "lr": current_lr,
        })

        is_best = (v_auc > best_val_auc) or (np.isclose(v_auc, best_val_auc) and v_loss < best_val_loss)
        if is_best:
            best_val_auc = v_auc
            best_val_loss = v_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": v_auc,
                "val_loss": v_loss,
                "variant": variant,
                "in_features": in_dim,
                "args": vars(args),
            }, best_ckpt_path)

        if epoch % 10 == 0 or epoch == args.epochs or is_best:
            mark = " [*BEST*]" if is_best else ""
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {v_loss:.4f} | Val AUC: {v_auc:.4f}{mark}")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training completed! Best checkpoint at epoch {best_epoch:02d} (Val AUC: {best_val_auc:.4f}, Val Loss: {best_val_loss:.4f})")

    # ── Phase 2: Validation-Guided Threshold Optimization ────────────────────
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        val_logits = model(X_val).squeeze(1).cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

    val_y = y_val.squeeze(1).cpu().numpy().astype(int)
    fpr, tpr, thresholds = roc_curve(val_y, val_probs)
    youdens_j = tpr - fpr
    best_j_idx = int(np.argmax(youdens_j))
    optimal_thresh = float(thresholds[best_j_idx])
    val_sens = float(tpr[best_j_idx])
    val_spec = float(1.0 - fpr[best_j_idx])

    print(f"Optimal Threshold (Youden's J on Val): {optimal_thresh:.4f} (Sens: {val_sens:.4f}, Spec: {val_spec:.4f})")

    # ── Phase 3: Platt Scaling Calibration Fitting (on Train) ────────────────
    with torch.no_grad():
        train_logits = model(X_train).squeeze(1).cpu().numpy()
    train_y = y_train.squeeze(1).cpu().numpy().astype(int)

    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(train_logits.reshape(-1, 1), train_y)
    print(f"Platt Scaling fitted: slope={calibrator.coef_[0][0]:.4f}, intercept={calibrator.intercept_[0]:.4f}")

    # ── Phase 4: Frozen Test Set Evaluation ──────────────────────────────────
    with torch.no_grad():
        test_logits = model(X_test).squeeze(1).cpu().numpy()
        test_probs_raw = 1.0 / (1.0 + np.exp(-test_logits))

    test_cal_probs = calibrator.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    test_y = y_test.squeeze(1).cpu().numpy().astype(int)

    test_df = df_manifest[test_idx].copy().reset_index(drop=True)
    test_df["logit"] = test_logits
    test_df["prob_raw"] = test_probs_raw
    test_df["prob_calibrated"] = test_cal_probs
    test_df["threshold_applied"] = optimal_thresh
    test_df["prediction_binary"] = (test_probs_raw >= optimal_thresh).astype(int)

    test_auc = float(roc_auc_score(test_y, test_probs_raw))
    ci_eye, ci_cluster = compute_bootstrap_cis(test_df, score_col="prob_raw", n_iterations=2000, seed=args.seed)

    tn, fp, fn, tp = confusion_matrix(test_y, test_df["prediction_binary"]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc = float((tp + tn) / len(test_df))
    brier_raw = float(brier_score_loss(test_y, test_probs_raw))
    brier_cal = float(brier_score_loss(test_y, test_cal_probs))

    print(f"\n--- Frozen Test Results for {variant.upper()} ---")
    print(f"Test ROC-AUC:               {test_auc:.4f}")
    print(f"Eye-level 95% CI:           [{ci_eye[0]:.4f}, {ci_eye[1]:.4f}]")
    print(f"Patient-clustered 95% CI:   [{ci_cluster[0]:.4f}, {ci_cluster[1]:.4f}]")
    print(f"Sensitivity:                {sens:.4f} ({tp}/{tp + fn})")
    print(f"Specificity:                {spec:.4f} ({tn}/{tn + fp})")
    print(f"Accuracy:                   {acc:.4f} ({tp + tn}/{len(test_df)})")
    print(f"Brier Score (Calibrated):   {brier_cal:.4f}")
    print(f"Brier Score (Raw):          {brier_raw:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Save test predictions CSV
    preds_out_path = Path(f"results/{variant}_test_predictions.csv")
    preds_out_path.parent.mkdir(parents=True, exist_ok=True)
    save_cols = [
        "image_id", "patient_id", "eye", "label", "logit",
        "prob_raw", "prob_calibrated", "prediction_binary", "threshold_applied"
    ]
    test_df[save_cols].to_csv(preds_out_path, index=False)
    print(f"Saved test predictions to: {preds_out_path}")

    # Save metrics JSON
    metrics_data = {
        "experiment": variant,
        "input_modalities": input_desc,
        "feature_dim": in_dim,
        "fusion_head_architecture": f"LayerNorm({in_dim}) -> Dropout({args.dropout}) -> Linear({in_dim}, {1 if args.hidden_dim == 0 else args.hidden_dim})",
        "parameters": params_info,
        "training_setup": {
            "device": str(device),
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pos_weight": float(n_neg / n_pos),
            "selection_criterion": "best_val_auc",
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
            "n_test_images": len(test_df),
            "n_test_patients": int(test_df["patient_id"].nunique()),
            "gon_plus_count": int(test_df["label"].sum()),
            "gon_minus_count": int((test_df["label"] == 0).sum()),
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

    metrics_out_path = Path(f"results/{variant}_metrics.json")
    with open(metrics_out_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics JSON to: {metrics_out_path}")

    return metrics_data


def main():
    parser = argparse.ArgumentParser(description="Train Step 4 Fusion Models on GPU")
    parser.add_argument("--manifest", default="data/manifests/papila_morphology.csv", help="Path to manifest CSV")
    parser.add_argument("--global-features", default="data/features/papila_m2_global_features.pt", help="Path to M2 features")
    parser.add_argument("--local-features", default="data/features/papila_m3_local_features.pt", help="Path to M3 features")
    parser.add_argument("--hidden-dim", type=int, default=0, help="Hidden dim for fusion head (0 = linear probe)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device ('cuda' or 'cpu')")

    args = parser.parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Active compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load data
    df_manifest = pd.read_csv(args.manifest)
    f_global = torch.load(args.global_features, weights_only=False)["features"]
    f_local = torch.load(args.local_features, weights_only=False)["features"]

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

    variants = [
        "M4_global_local",
        "M5a_global_cdr",
        "M5b_global_local_cdr",
    ]

    summary_results = {}
    for var in variants:
        res = train_and_evaluate_fusion(var, df_manifest, f_global, f_local, cdr_tensor, args, device)
        summary_results[var] = res

    print("\n" + "=" * 80)
    print("STEP 4 FUSION ABLATION SUMMARY (FROZEN TEST PARTITION)")
    print("=" * 80)
    print(f"{'Model':<22} | {'AUC':<7} | {'Eye 95% CI':<18} | {'Clustered 95% CI':<18} | {'Sens':<6} | {'Spec':<6} | {'Acc':<6} | {'Brier':<6}")
    print("-" * 80)
    for var, res in summary_results.items():
        tr = res["test_results"]
        ci_e = f"[{tr['bootstrap_95ci_eye_level'][0]:.4f}, {tr['bootstrap_95ci_eye_level'][1]:.4f}]"
        ci_c = f"[{tr['bootstrap_95ci_patient_clustered'][0]:.4f}, {tr['bootstrap_95ci_patient_clustered'][1]:.4f}]"
        print(f"{var:<22} | {tr['roc_auc']:<7.4f} | {ci_e:<18} | {ci_c:<18} | {tr['sensitivity']:<6.4f} | {tr['specificity']:<6.4f} | {tr['accuracy']:<6.4f} | {tr['brier_score_calibrated']:<6.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
