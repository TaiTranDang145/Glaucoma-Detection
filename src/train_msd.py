"""Train GONet with multi-source leave-one-domain-out evaluation."""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dataset import build_manifest_loader
from losses import build_criterion
from models import build_model
from train import train_one_epoch, evaluate
from utils import EarlyStopping, compute_metrics, get_logger, set_seed


REPO_ROOT = Path(__file__).absolute().parents[1]


def split_sources(source_df, val_fraction, seed):
    groups = source_df["patient_id"].replace("", np.nan).fillna(source_df["image_path"]).astype(str)
    split = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(split.split(source_df, groups=groups))
    return source_df.iloc[train_idx].reset_index(drop=True), source_df.iloc[val_idx].reset_index(drop=True)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, probabilities, image_paths = [], [], []
    for images, batch_labels, batch_image_paths in tqdm(loader, desc="Target evaluation", leave=False):
        logits = model(images.to(device, non_blocking=True)).squeeze(1)
        probabilities.extend(torch.sigmoid(logits).cpu().numpy())
        labels.extend(batch_labels.numpy())
        image_paths.extend(batch_image_paths)
    return np.asarray(labels, dtype=int), np.asarray(probabilities), image_paths


def bootstrap_auc_ci(labels, probabilities, repetitions=1000, fraction=0.95, seed=42):
    rng = np.random.default_rng(seed)
    sample_size = max(2, round(len(labels) * fraction))
    scores = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(labels), size=sample_size)
        if np.unique(labels[indices]).size == 2:
            scores.append(roc_auc_score(labels[indices], probabilities[indices]))
    if not scores:
        return float("nan"), float("nan")
    return tuple(np.percentile(scores, [2.5, 97.5]).tolist())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gonet_msd.yaml")
    parser.add_argument("--data-root", help="Root containing labeled dataset folders such as HYDR/, DRISHTI-GS/, and PAPILA/.")
    parser.add_argument("--manifest", help="CSV manifest path; overrides the config.")
    parser.add_argument("--output-dir", help="Writable output directory; overrides the config.")
    parser.add_argument("--target-domain", help="Run just one leave-one-domain-out fold.")
    parser.add_argument("--epochs", type=int, help="Override configured epoch count.")
    parser.add_argument("--batch-size", type=int, help="Override configured batch size.")
    parser.add_argument("--num-workers", type=int, help="Override configured DataLoader workers.")
    parser.add_argument(
        "--domain-balanced-sampling",
        action="store_true",
        help="Balance source-domain contribution in training batches (ablation).",
    )
    return parser.parse_args()


def _repo_relative(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def main():
    args = parse_args()
    cfg = OmegaConf.load(_repo_relative(args.config))
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers

    data_root = _repo_relative(args.data_root or cfg.paths.data_root)
    manifest_path = _repo_relative(args.manifest or cfg.paths.manifest)
    manifest = pd.read_csv(manifest_path)
    required = {"image_path", "domain", "label", "patient_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    domains = sorted(manifest["domain"].dropna().unique().tolist())
    if len(domains) < 2:
        raise ValueError("MSD training needs at least two labeled domains.")
    targets = [args.target_domain] if args.target_domain else domains
    unknown = set(targets) - set(domains)
    if unknown:
        raise ValueError(f"Unknown target domain(s): {sorted(unknown)}. Available: {domains}")

    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    output_dir = _repo_relative(args.output_dir or cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("gonet_msd", log_file=str(output_dir / "train.log"))
    logger.info("Domains: %s | device: %s", domains, device)

    summary = []
    prediction_frames = []
    for target in targets:
        set_seed(cfg.experiment.seed)
        source = manifest[manifest["domain"] != target].reset_index(drop=True)
        target_df = manifest[manifest["domain"] == target].reset_index(drop=True)
        train_df, val_df = split_sources(source, cfg.dataset.validation_fraction, cfg.experiment.seed)
        logger.info(
            "Target %s: train=%d val=%d target=%d",
            target, len(train_df), len(val_df), len(target_df),
        )
        if args.domain_balanced_sampling:
            logger.info("Domain-balanced sampling enabled for training only.")

        train_loader = build_manifest_loader(
            train_df, data_root, cfg.dataset.image_size, cfg.training.batch_size,
            cfg.training.num_workers, "train",
            domain_balanced_sampling=args.domain_balanced_sampling,
        )
        val_loader = build_manifest_loader(
            val_df, data_root, cfg.dataset.image_size, cfg.training.batch_size,
            cfg.training.num_workers, "val",
        )
        target_loader = build_manifest_loader(
            target_df, data_root, cfg.dataset.image_size, cfg.training.batch_size,
            cfg.training.num_workers, "test",
        )

        model = build_model(
            model_name=cfg.model.architecture,
            pretrained=cfg.model.pretrained,
            dropout_rate=cfg.model.dropout_rate,
            device=str(device),
            image_size=cfg.dataset.image_size,
        )
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        logger.info("Full fine-tuning: %d/%d parameters trainable", trainable, total)

        criterion = build_criterion().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=cfg.training.scheduler_patience,
            factor=cfg.training.scheduler_factor,
        )
        stopper = EarlyStopping(
            patience=cfg.training.patience,
            mode="min",
            delta=cfg.training.early_stopping_delta,
        )
        best_loss, best_auc_at_best_loss, best_state, best_epoch = float("inf"), float("nan"), None, 0

        for epoch in range(1, cfg.training.epochs + 1):
            train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger)
            val_metrics = evaluate(model, val_loader, criterion, device, epoch, "val", logger)
            scheduler.step(val_metrics["loss"])
            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                best_auc_at_best_loss = val_metrics["auc"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if stopper(val_metrics["loss"]):
                logger.info("Early stop %s at epoch %d.", target, epoch)
                break

        if best_state is None:
            raise RuntimeError(f"No validation checkpoint was produced for target {target}.")
        model.load_state_dict(best_state)
        labels, probabilities, image_paths = predict(model, target_loader, device)
        metrics = compute_metrics(labels, probabilities)
        low, high = bootstrap_auc_ci(
            labels, probabilities,
            repetitions=cfg.evaluation.bootstrap_repetitions,
            fraction=cfg.evaluation.bootstrap_fraction,
            seed=cfg.experiment.seed,
        )
        metrics.update({"auc_ci_low": round(low, 4), "auc_ci_high": round(high, 4)})
        metrics.update({
            "target_domain": target,
            "n": len(labels),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
        })

        fold_dir = output_dir / target
        fold_dir.mkdir(parents=True, exist_ok=True)
        predictions = pd.DataFrame({
            "image_path": image_paths,
            "domain": target,
            "gonet_prob": probabilities,
        })
        predictions.to_csv(fold_dir / "test_predictions.csv", index=False)
        prediction_frames.append(predictions)
        torch.save({
            "model_state": best_state,
            "target_domain": target,
            "best_epoch": best_epoch,
            "validation_loss": best_loss,
            "validation_auc": best_auc_at_best_loss,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, fold_dir / "best_model.pth")
        logger.info(
            "OOD %s: AUC %.4f (95%% CI %.4f-%.4f), Brier %.4f | best val loss %.4f at epoch %d",
            target, metrics["auc"], low, high, metrics["brier"], best_loss, best_epoch,
        )
        summary.append(metrics)

    pd.DataFrame(summary).to_csv(output_dir / "ood_metrics.csv", index=False)
    logger.info("Saved OOD metrics to %s", output_dir / "ood_metrics.csv")
    predictions_path = output_dir / "gonet_predictions.csv"
    pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["domain", "image_path"]
    ).to_csv(predictions_path, index=False)
    logger.info("Saved per-image target probabilities to %s", predictions_path)


if __name__ == "__main__":
    main()
