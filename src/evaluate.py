"""
evaluate.py
-----------
Standalone evaluation script. Loads a saved checkpoint and evaluates
on the test set, printing a full classification report and saving figures.

Usage:
    python src/evaluate.py \
        --config configs/efficientnet_b3.yaml \
        --checkpoint outputs/checkpoints/best_model.pth
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
from omegaconf import OmegaConf
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc
)

from dataset import build_dataloaders
from models import build_model
from utils import set_seed, get_logger, compute_metrics, load_checkpoint


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Glaucoma Classifier")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--split",      type=str, default="test",
                        choices=["val", "test"])
    return parser.parse_args()


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, save_path: str) -> float:
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2563EB")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontsize=13)
    ax.set_title("ROC Curve — GON Detection",  fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Evaluate] ROC curve saved → {save_path}")
    return roc_auc


def plot_confusion_matrix(
    labels: np.ndarray, preds: np.ndarray, save_path: str
) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["GON-", "GON+"],
        yticklabels=["GON-", "GON+"],
        ax=ax, linewidths=0.5, annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Evaluate] Confusion matrix saved → {save_path}")


def plot_probability_distribution(
    labels: np.ndarray, probs: np.ndarray, save_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probs[labels == 0], bins=30, alpha=0.6, color="#10B981", label="GON−", edgecolor="white")
    ax.hist(probs[labels == 1], bins=30, alpha=0.6, color="#EF4444", label="GON+", edgecolor="white")
    ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold = 0.5")
    ax.set_xlabel("Predicted Probability (GON+)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Prediction Probability Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Evaluate] Probability distribution saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    set_seed(cfg.experiment.seed)
    device = torch.device(
        cfg.experiment.device if torch.cuda.is_available() else "cpu"
    )
    logger = get_logger("evaluate")

    Path(cfg.paths.figures_dir).mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    _, val_loader, test_loader, _ = build_dataloaders(
        labels_csv        = cfg.paths.labels_csv,
        images_dir        = cfg.paths.images_dir,
        image_size        = cfg.dataset.image_size,
        batch_size        = cfg.training.batch_size,
        num_workers       = cfg.training.num_workers,
        min_quality_score = cfg.dataset.min_quality_score,
        val_size          = cfg.dataset.val_size,
        test_size         = cfg.dataset.test_size,
        seed              = cfg.experiment.seed,
    )
    loader = test_loader if args.split == "test" else val_loader

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        model_name   = cfg.model.architecture,
        pretrained   = False,
        dropout_rate = cfg.model.dropout_rate,
        device       = str(device),
    )
    load_checkpoint(args.checkpoint, model, device=str(device))
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    all_labels, all_probs, all_names = [], [], []
    with torch.no_grad():
        for images, labels, names in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            all_names.extend(names)

    labels = np.array(all_labels)
    probs  = np.array(all_probs)
    preds  = (probs >= args.threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(labels, probs, threshold=args.threshold)
    logger.info(f"\n[{args.split.upper()} SET RESULTS]")
    for k, v in metrics.items():
        logger.info(f"  {k:15s}: {v}")

    logger.info("\n[Classification Report]")
    logger.info("\n" + classification_report(labels, preds, target_names=["GON-", "GON+"]))

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_roc_curve(
        labels, probs,
        save_path=f"{cfg.paths.figures_dir}/roc_curve.png"
    )
    plot_confusion_matrix(
        labels, preds,
        save_path=f"{cfg.paths.figures_dir}/confusion_matrix.png"
    )
    plot_probability_distribution(
        labels, probs,
        save_path=f"{cfg.paths.figures_dir}/prob_distribution.png"
    )

    # ── Save predictions ──────────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "Image Name": all_names,
        "True Label": ["GON+" if l == 1 else "GON-" for l in labels],
        "Pred Label": ["GON+" if p == 1 else "GON-" for p in preds],
        "GON+ Probability": probs.round(4),
    })
    out_csv = f"{cfg.paths.log_dir}/{args.split}_predictions.csv"
    pred_df.to_csv(out_csv, index=False)
    logger.info(f"\nPredictions saved → {out_csv}")


if __name__ == "__main__":
    main()
