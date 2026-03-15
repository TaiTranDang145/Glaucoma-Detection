"""
utils.py
--------
Utility functions: seed setting, checkpoint I/O, metric computation,
early stopping, and logging helpers.
"""

import os
import random
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, classification_report,
    precision_score, recall_score,
)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Sets all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Utils] Random seed set to {seed}")


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Creates a logger with console and optional file handler.

    Args:
        name (str): Logger name.
        log_file (str, optional): Path to log file.

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes classification metrics for binary GON detection.

    Args:
        labels (np.ndarray): Ground truth binary labels (0 or 1).
        probs (np.ndarray):  Predicted probabilities (0.0–1.0).
        threshold (float):   Decision threshold. Default: 0.5.

    Returns:
        Dict with keys: auc, accuracy, f1, sensitivity, specificity, precision.
    """
    preds = (probs >= threshold).astype(int)

    auc       = roc_auc_score(labels, probs)
    accuracy  = accuracy_score(labels, preds)
    f1        = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)   # sensitivity

    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-8)

    return {
        "auc":         round(float(auc),         4),
        "accuracy":    round(float(accuracy),     4),
        "f1":          round(float(f1),           4),
        "sensitivity": round(float(recall),       4),
        "specificity": round(float(specificity),  4),
        "precision":   round(float(precision),    4),
    }


def log_metrics(metrics: Dict[str, float], split: str, epoch: int, logger: logging.Logger) -> None:
    """Pretty-prints a metrics dict."""
    logger.info(
        f"[{split.upper()} | Epoch {epoch:03d}] "
        f"AUC={metrics['auc']:.4f}  "
        f"Acc={metrics['accuracy']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"Sens={metrics['sensitivity']:.4f}  "
        f"Spec={metrics['specificity']:.4f}"
    )


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    config: Optional[dict] = None,
) -> None:
    """
    Saves model checkpoint.

    Args:
        model: PyTorch model.
        optimizer: Optimizer state.
        epoch: Current epoch number.
        metrics: Evaluation metrics dict.
        path: File path to save (.pth).
        config: Optional config dict to embed in checkpoint.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "metrics":     metrics,
        "config":      config,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Tuple[int, Dict[str, float]]:
    """
    Loads a checkpoint into model (and optionally optimizer).

    Args:
        path: Path to .pth file.
        model: Model to load weights into.
        optimizer: Optimizer to restore state (optional).
        device: Target device.

    Returns:
        Tuple of (epoch, metrics).
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optim_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optim_state"])
    epoch   = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    print(f"[Utils] Checkpoint loaded from '{path}' (epoch {epoch})")
    return epoch, metrics


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when the monitored metric stops improving.

    Args:
        patience (int): Epochs to wait before stopping. Default: 10.
        mode (str): 'max' (higher is better) or 'min'.
        delta (float): Minimum change to qualify as improvement.
    """

    def __init__(self, patience: int = 10, mode: str = "max", delta: float = 1e-4):
        self.patience  = patience
        self.mode      = mode
        self.delta     = delta
        self.counter   = 0
        self.best      = None
        self.stopped   = False

    def __call__(self, value: float) -> bool:
        """
        Returns True if training should stop.

        Args:
            value (float): Current monitored metric value.
        """
        if self.best is None:
            self.best = value
            return False

        improved = (
            value > self.best + self.delta
            if self.mode == "max"
            else value < self.best - self.delta
        )

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                return True
        return False
