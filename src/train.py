"""
train.py
--------
Main training script for the Glaucoma Detection model.

Usage:
    python src/train.py --config configs/efficientnet_b3.yaml
    python src/train.py --config configs/efficientnet_b3.yaml --epochs 50 --lr 1e-4
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model, build_criterion
from utils import (
    set_seed, get_logger, compute_metrics,
    log_metrics, save_checkpoint, EarlyStopping,
)


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Glaucoma Classifier")
    parser.add_argument("--config",     type=str, required=True,  help="Path to YAML config")
    parser.add_argument("--epochs",     type=int, default=None,   help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None,   help="Override batch size")
    parser.add_argument("--lr",         type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume",     type=str, default=None,   help="Resume from checkpoint path")
    return parser.parse_args()


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger,
) -> dict:
    """Runs one training epoch. Returns averaged loss and metrics."""
    model.train()

    all_labels, all_probs, total_loss = [], [], 0.0
    pbar = tqdm(loader, desc=f"Train Epoch {epoch:03d}", leave=False)

    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad()
        logits = model(images).squeeze(1)   # (B,)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(np.array(all_labels), np.array(all_probs))
    metrics["loss"] = round(avg_loss, 4)
    log_metrics(metrics, "train", epoch, logger)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    split: str,
    logger,
) -> dict:
    """Runs evaluation on val or test set. Returns metrics."""
    model.eval()

    all_labels, all_probs, total_loss = [], [], 0.0

    for images, labels, _ in tqdm(loader, desc=f"{split.capitalize()} Epoch {epoch:03d}", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        logits = model(images).squeeze(1)
        loss   = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(np.array(all_labels), np.array(all_probs))
    metrics["loss"] = round(avg_loss, 4)
    log_metrics(metrics, split, epoch, logger)
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)

    # CLI overrides
    if args.epochs:     cfg.training.epochs     = args.epochs
    if args.batch_size: cfg.training.batch_size = args.batch_size
    if args.lr:         cfg.training.lr         = args.lr

    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(cfg.experiment.seed)
    device = torch.device(
        cfg.experiment.device if torch.cuda.is_available() else "cpu"
    )

    Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.figures_dir).mkdir(parents=True, exist_ok=True)

    logger  = get_logger("train", log_file=f"{cfg.paths.log_dir}/train.log")
    writer  = SummaryWriter(log_dir=cfg.paths.log_dir)

    logger.info(f"Experiment : {cfg.experiment.name}")
    logger.info(f"Device     : {device}")
    logger.info(OmegaConf.to_yaml(cfg))

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, split_info = build_dataloaders(
        labels_csv           = cfg.paths.labels_csv,
        images_dir           = cfg.paths.images_dir,
        image_size           = cfg.dataset.image_size,
        batch_size           = cfg.training.batch_size,
        num_workers          = cfg.training.num_workers,
        min_quality_score    = cfg.dataset.min_quality_score,
        val_size             = cfg.dataset.val_size,
        test_size            = cfg.dataset.test_size,
        use_weighted_sampler = cfg.training.use_weighted_loss,
        seed                 = cfg.experiment.seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        model_name   = cfg.model.architecture,
        pretrained   = cfg.model.pretrained,
        dropout_rate = cfg.model.dropout_rate,
        device       = str(device),
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    if cfg.training.use_weighted_loss:
        n_pos = split_info["train"]["GON+"]
        n_neg = split_info["train"]["GON-"]
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float).to(device)
        logger.info(f"[Loss] pos_weight = {pos_weight.item():.4f}")
    else:
        pos_weight = None

    criterion = build_criterion(
        pos_weight       = pos_weight,
        label_smoothing  = cfg.training.label_smoothing,
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.training.lr,
        weight_decay = cfg.training.weight_decay,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = cfg.training.epochs,
        eta_min = cfg.training.min_lr,
    )

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stopper = EarlyStopping(
        patience = cfg.training.early_stopping.patience,
        mode     = cfg.training.early_stopping.mode,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch  = 1
    best_val_auc = 0.0

    if args.resume:
        from utils import load_checkpoint
        start_epoch, prev_metrics = load_checkpoint(
            args.resume, model, optimizer, device=str(device)
        )
        start_epoch  += 1
        best_val_auc  = prev_metrics.get("auc", 0.0)
        logger.info(f"Resumed from epoch {start_epoch - 1}, best AUC = {best_val_auc:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting training ...")
    logger.info("=" * 60)

    history = {"train": [], "val": []}
    t0 = time.time()

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, epoch, "val", logger
        )

        scheduler.step()

        # ── TensorBoard ───────────────────────────────────────────────────────
        for key, val in train_metrics.items():
            writer.add_scalar(f"Train/{key}", val, epoch)
        for key, val in val_metrics.items():
            writer.add_scalar(f"Val/{key}", val, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # ── Save best checkpoint ──────────────────────────────────────────────
        val_auc = val_metrics["auc"]
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_checkpoint(
                model     = model,
                optimizer = optimizer,
                epoch     = epoch,
                metrics   = val_metrics,
                path      = f"{cfg.paths.checkpoint_dir}/best_model.pth",
                config    = OmegaConf.to_container(cfg),
            )
            logger.info(f"  ✓ Best model saved  (val_auc = {best_val_auc:.4f})")

        # ── Early stopping ────────────────────────────────────────────────────
        if early_stopper(val_auc):
            logger.info(f"Early stopping at epoch {epoch} (patience exhausted).")
            break

    elapsed = time.time() - t0
    logger.info(f"\nTraining complete in {elapsed / 60:.1f} min | Best val AUC = {best_val_auc:.4f}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on TEST set with best checkpoint ...")
    logger.info("=" * 60)

    from utils import load_checkpoint
    load_checkpoint(
        f"{cfg.paths.checkpoint_dir}/best_model.pth",
        model, device=str(device)
    )
    test_metrics = evaluate(
        model, test_loader, criterion, device, epoch=0, split="test", logger=logger
    )

    logger.info("\n[TEST RESULTS]")
    for k, v in test_metrics.items():
        logger.info(f"  {k:15s}: {v}")

    # Save history and test results
    pd.DataFrame(history["train"]).to_csv(f"{cfg.paths.log_dir}/train_history.csv", index=False)
    pd.DataFrame(history["val"]).to_csv(f"{cfg.paths.log_dir}/val_history.csv",   index=False)
    pd.DataFrame([test_metrics]).to_csv(f"{cfg.paths.log_dir}/test_results.csv",   index=False)
    logger.info(f"\nLogs saved to: {cfg.paths.log_dir}")

    writer.close()


if __name__ == "__main__":
    main()
