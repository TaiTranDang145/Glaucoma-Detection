"""Shared, leakage-aware training and evaluation engine for DINOv2 SSD/MSD."""

import argparse
import copy
import json
import logging
import math
import os
import platform
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from dataset import build_manifest_loader
from models.gonet import DINO_MODEL
from utils import get_logger, set_seed


REPO_ROOT = Path(__file__).resolve().parents[1]
DOMAIN_ALIASES = {"HYDR": "HYRD", "DRISHTI-GS": "DRISHTI_GS"}


def canonical_domain(value):
    value = str(value).strip()
    return DOMAIN_ALIASES.get(value, value)


def _resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _seed_everything(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _domain_split_score(frame, validation, fraction):
    total = len(frame)
    desired_n = max(1, round(total * fraction))
    score = abs(len(validation) - desired_n) / desired_n
    labels = frame["label"].astype(int)
    validation_labels = validation["label"].astype(int)
    for label in (0, 1):
        full_count = int((labels == label).sum())
        if not full_count:
            continue
        desired_count = max(1, round(full_count * fraction))
        actual_count = int((validation_labels == label).sum())
        score += abs(actual_count - desired_count) / desired_count
        if full_count > 1 and actual_count in (0, full_count):
            score += 100.0
    return score


def split_source_domains(source_frame, source_domains, validation_fraction, seed):
    """Make reproducible per-domain, group-aware train/validation splits."""
    training_parts, validation_parts, audit = [], [], {}
    for domain_index, domain in enumerate(source_domains):
        frame = source_frame[source_frame["domain"] == domain].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"Source domain {domain!r} has no rows.")
        group_column = "split_group" if "split_group" in frame else "patient_id"
        groups = frame[group_column].fillna("").astype(str).str.strip()
        groups = groups.where(groups.ne(""), domain + "::" + frame["image_path"].astype(str))
        if groups.nunique() < 2:
            raise ValueError(f"Source domain {domain!r} has fewer than two independent groups.")

        splitter = GroupShuffleSplit(
            n_splits=64,
            test_size=validation_fraction,
            random_state=seed + domain_index,
        )
        best = None
        best_score = math.inf
        for train_indices, validation_indices in splitter.split(frame, groups=groups):
            train = frame.iloc[train_indices].reset_index(drop=True)
            validation = frame.iloc[validation_indices].reset_index(drop=True)
            score = _domain_split_score(frame, validation, validation_fraction)
            if score < best_score and len(train) and len(validation):
                best, best_score = (train, validation), score
        if best is None:
            raise RuntimeError(f"Could not create a valid train/validation split for {domain}.")

        train, validation = best
        train_groups = set(train["patient_id"].fillna("").astype(str).str.strip())
        val_groups = set(validation["patient_id"].fillna("").astype(str).str.strip())
        train_groups.discard("")
        val_groups.discard("")
        overlap = train_groups & val_groups
        if overlap:
            raise RuntimeError(f"Patient leakage in {domain}: {sorted(overlap)[:5]}")
        if "split_group" in frame:
            group_overlap = set(train["split_group"]) & set(validation["split_group"])
            if group_overlap:
                raise RuntimeError(f"Related-sample split-group leakage in {domain}: {sorted(group_overlap)[:5]}")
        training_parts.append(train)
        validation_parts.append(validation)
        audit[domain] = {
            "train_n": len(train),
            "val_n": len(validation),
            "train_gon_minus": int((train["label"] == 0).sum()),
            "train_gon_plus": int((train["label"] == 1).sum()),
            "val_gon_minus": int((validation["label"] == 0).sum()),
            "val_gon_plus": int((validation["label"] == 1).sum()),
            "train_patient_groups": len(train_groups),
            "val_patient_groups": len(val_groups),
            "patient_id_available": bool(frame["patient_id"].fillna("").astype(str).str.strip().ne("").any()),
            "split_score": round(float(best_score), 6),
        }
    return (
        pd.concat(training_parts, ignore_index=True),
        pd.concat(validation_parts, ignore_index=True),
        audit,
    )


def _prepared_source_split(source_frame, source_domains):
    """Use a prepared train/val split when every requested source has one."""
    split_column = "source_split" if "source_split" in source_frame else "split"
    if split_column not in source_frame:
        return None
    training_parts, validation_parts, audit = [], [], {}
    for domain in source_domains:
        frame = source_frame[source_frame["domain"] == domain].reset_index(drop=True)
        splits = set(frame[split_column].astype(str))
        if not {"train", "val"}.issubset(splits) or not splits.issubset({"train", "val"}):
            return None
        train = frame[frame[split_column] == "train"].drop(columns=[split_column]).reset_index(drop=True)
        validation = frame[frame[split_column] == "val"].drop(columns=[split_column]).reset_index(drop=True)
        if train.empty or validation.empty:
            return None
        known_train = set(train.loc[train["patient_id"].astype(str).str.strip().ne(""), "patient_id"])
        known_val = set(validation.loc[validation["patient_id"].astype(str).str.strip().ne(""), "patient_id"])
        if known_train & known_val:
            raise RuntimeError(f"Patient leakage in prepared split for {domain}.")
        if "split_group" in frame and set(train["split_group"]) & set(validation["split_group"]):
            raise RuntimeError(f"Related-sample split-group leakage in prepared split for {domain}.")
        if "sha256" in frame and set(train["sha256"]) & set(validation["sha256"]):
            raise RuntimeError(f"Exact image-hash leakage in prepared split for {domain}.")
        training_parts.append(train)
        validation_parts.append(validation)
        audit[domain] = {
            "train_n": len(train),
            "val_n": len(validation),
            "train_gon_minus": int((train["label"] == 0).sum()),
            "train_gon_plus": int((train["label"] == 1).sum()),
            "val_gon_minus": int((validation["label"] == 0).sum()),
            "val_gon_plus": int((validation["label"] == 1).sum()),
            "train_patient_groups": len(known_train),
            "val_patient_groups": len(known_val),
            "patient_id_available": bool(known_train or known_val),
            "split_score": None,
            "split_source": "prepared_manifest",
        }
    return pd.concat(training_parts, ignore_index=True), pd.concat(validation_parts, ignore_index=True), audit


def _stratified_cap(frame, maximum, seed):
    if maximum is None or maximum <= 0 or len(frame) <= maximum:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    labels = sorted(frame["label"].astype(int).unique())
    per_class = max(1, maximum // max(len(labels), 1))
    selected_indices = []
    remaining = []
    for label in labels:
        indices = frame.index[frame["label"].astype(int) == label].to_numpy()
        rng.shuffle(indices)
        take = min(per_class, len(indices))
        selected_indices.extend(indices[:take].tolist())
        remaining.extend(indices[take:].tolist())
    if len(selected_indices) < maximum and remaining:
        rng.shuffle(remaining)
        selected_indices.extend(remaining[:maximum - len(selected_indices)].tolist())
    return frame.loc[selected_indices].reset_index(drop=True)


def _autocast(device, enabled):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)


def _make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _is_cuda_oom(error):
    oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(error, oom_type) or "out of memory" in str(error).casefold()


def _train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, epoch, logger):
    model.train()
    total_loss = 0.0
    labels_all, probabilities_all = [], []
    oom_retries = 0
    for cpu_images, cpu_labels, _ in loader:
        count = len(cpu_labels)
        chunk_size = count
        while True:
            optimizer.zero_grad(set_to_none=True)
            chunk_probabilities = []
            batch_loss = 0.0
            images = labels = logits = loss = weighted_loss = None
            try:
                for start in range(0, count, chunk_size):
                    end = min(start + chunk_size, count)
                    images = cpu_images[start:end].to(device, non_blocking=True)
                    labels = cpu_labels[start:end].to(device, non_blocking=True).float()
                    with _autocast(device, use_amp):
                        logits = model(images).squeeze(1)
                        loss = criterion(logits, labels)
                        weighted_loss = loss * ((end - start) / count)
                    scaler.scale(weighted_loss).backward()
                    batch_loss += float(weighted_loss.detach().float().item())
                    chunk_probabilities.extend(torch.sigmoid(logits.detach().float()).cpu().numpy())

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                break
            except RuntimeError as error:
                if device.type != "cuda" or not _is_cuda_oom(error) or chunk_size <= 1:
                    raise
                optimizer.zero_grad(set_to_none=True)
                chunk_size = max(1, chunk_size // 2)
                oom_retries += 1
                images = labels = logits = loss = weighted_loss = None
                torch.cuda.empty_cache()

        total_loss += batch_loss
        labels_all.extend(cpu_labels.numpy().astype(int).tolist())
        probabilities_all.extend(float(value) for value in chunk_probabilities)

    if not len(loader):
        raise ValueError("Training loader is empty.")
    labels_np = np.asarray(labels_all, dtype=int)
    probabilities_np = np.asarray(probabilities_all, dtype=float)
    auc = float(roc_auc_score(labels_np, probabilities_np)) if np.unique(labels_np).size == 2 else None
    if oom_retries:
        logger.warning("CUDA OOM fallback split batches into smaller microbatches (%d retries).", oom_retries)
    return {"loss": total_loss / len(loader), "auc": auc, "oom_retries": oom_retries}


@torch.no_grad()
def _evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    labels_all, probabilities_all = [], []
    total_loss, total_rows = 0.0, 0
    oom_retries = 0
    for cpu_images, cpu_labels, _ in loader:
        count = len(cpu_labels)
        chunk_size = count
        while True:
            labels_batch, probabilities_batch = [], []
            batch_loss = 0.0
            images = labels = logits = loss = None
            try:
                for start in range(0, count, chunk_size):
                    end = min(start + chunk_size, count)
                    images = cpu_images[start:end].to(device, non_blocking=True)
                    labels = cpu_labels[start:end].to(device, non_blocking=True).float()
                    with _autocast(device, use_amp):
                        logits = model(images).squeeze(1)
                        loss = criterion(logits, labels)
                    batch_loss += float(loss.float().item()) * (end - start)
                    probabilities_batch.extend(torch.sigmoid(logits.float()).cpu().numpy())
                    labels_batch.extend(cpu_labels[start:end].numpy().astype(int).tolist())
                break
            except RuntimeError as error:
                if device.type != "cuda" or not _is_cuda_oom(error) or chunk_size <= 1:
                    raise
                chunk_size = max(1, chunk_size // 2)
                oom_retries += 1
                images = labels = logits = loss = None
                torch.cuda.empty_cache()
        total_loss += batch_loss
        total_rows += count
        labels_all.extend(labels_batch)
        probabilities_all.extend(float(value) for value in probabilities_batch)

    if not total_rows:
        raise ValueError("Evaluation loader is empty.")
    labels_np = np.asarray(labels_all, dtype=int)
    probabilities_np = np.asarray(probabilities_all, dtype=float)
    auc = float(roc_auc_score(labels_np, probabilities_np)) if np.unique(labels_np).size == 2 else None
    return {
        "loss": total_loss / total_rows,
        "auc": auc,
        "n": total_rows,
        "oom_retries": oom_retries,
    }


@torch.no_grad()
def _predict(model, loader, device, use_amp):
    model.eval()
    labels_all, probabilities_all, paths_all = [], [], []
    oom_retries = 0
    for cpu_images, cpu_labels, paths in loader:
        count = len(cpu_labels)
        chunk_size = count
        while True:
            labels_batch, probabilities_batch = [], []
            images = logits = None
            try:
                for start in range(0, count, chunk_size):
                    end = min(start + chunk_size, count)
                    images = cpu_images[start:end].to(device, non_blocking=True)
                    with _autocast(device, use_amp):
                        logits = model(images).squeeze(1)
                    probs = torch.sigmoid(logits.float()).cpu().numpy()
                    probabilities_batch.extend(float(value) for value in probs)
                    labels_batch.extend(cpu_labels[start:end].numpy().astype(int).tolist())
                break
            except RuntimeError as error:
                if device.type != "cuda" or not _is_cuda_oom(error) or chunk_size <= 1:
                    raise
                chunk_size = max(1, chunk_size // 2)
                oom_retries += 1
                images = logits = None
                torch.cuda.empty_cache()
        labels_all.extend(labels_batch)
        probabilities_all.extend(probabilities_batch)
        paths_all.extend(paths)
    if not labels_all:
        raise ValueError("Target evaluation loader is empty.")
    probabilities = np.asarray(probabilities_all, dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise RuntimeError("Inference produced probabilities outside [0, 1].")
    return np.asarray(labels_all, dtype=int), probabilities, paths_all, oom_retries


def _bootstrap_auc(labels, probabilities, repetitions, fraction, seed):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) == 0 or np.unique(labels).size != 2:
        return {"low": None, "high": None, "valid_resamples": 0, "requested_resamples": repetitions}
    rng = np.random.default_rng(seed)
    sample_size = max(2, round(len(labels) * fraction))
    scores = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(labels), size=sample_size)
        if np.unique(labels[indices]).size < 2:
            continue
        scores.append(float(roc_auc_score(labels[indices], probabilities[indices])))
    if not scores:
        return {"low": None, "high": None, "valid_resamples": 0, "requested_resamples": repetitions}
    low, high = np.percentile(scores, [2.5, 97.5]).tolist()
    return {
        "low": float(low), "high": float(high),
        "valid_resamples": len(scores), "requested_resamples": repetitions,
    }


def _load_manifest(path):
    frame = pd.read_csv(path, keep_default_na=False)
    if "domain" not in frame and "dataset" in frame:
        frame = frame.rename(columns={"dataset": "domain"})
    required = {"image_path", "domain", "label", "patient_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    frame["domain"] = frame["domain"].map(canonical_domain)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Manifest labels must use GON+=1 and GON-=0.")
    if frame["image_path"].duplicated().any():
        examples = frame.loc[frame["image_path"].duplicated(False), "image_path"].head().tolist()
        raise ValueError(f"Duplicate image paths in manifest: {examples}")
    if "sha256" in frame:
        repeated = frame[frame["sha256"].astype(str).ne("")].duplicated("sha256", keep=False)
        if repeated.any():
            examples = frame.loc[repeated, ["image_path", "domain", "sha256"]].head().to_dict("records")
            raise ValueError(f"Exact duplicate image contents in manifest: {examples}")
    return frame


def _versions(device):
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda,
    }


def _parse_overrides(config, args):
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.data_root:
        config.paths.data_root = args.data_root
    if args.manifest:
        config.paths.manifest = args.manifest
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    if args.no_pretrained:
        config.model.pretrained = False
    return config


def add_common_arguments(parser):
    parser.add_argument("--config", default="configs/dinov2_baseline.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-samples-per-split", type=int,
                        help="Deterministic small subset for a debug run.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Use random DINOv2 initialization for offline smoke tests only.")


def run_experiment(
    mode, source_domains, target_domain, args, model_instance=None,
    defer_target_evaluation=False,
):
    from omegaconf import OmegaConf

    config_path = _resolve_path(args.config)
    cfg = _parse_overrides(OmegaConf.load(config_path), args)
    source_domains = [canonical_domain(domain) for domain in source_domains]
    target_domain = canonical_domain(target_domain)
    if not source_domains:
        raise ValueError("At least one source domain is required.")
    if len(set(source_domains)) != len(source_domains):
        raise ValueError(f"Duplicate source domain names: {source_domains}")
    if target_domain in source_domains:
        raise ValueError("The target domain must not appear in source_domains.")

    seed = int(cfg.experiment.seed)
    _seed_everything(seed)
    device = torch.device(str(cfg.experiment.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "This configuration requests CUDA, but no CUDA device is available. "
            "Select a GPU runtime or explicitly set experiment.device=cpu for a deliberate smoke test."
        )
    data_root = _resolve_path(cfg.paths.data_root)
    manifest_path = _resolve_path(cfg.paths.manifest)
    output_dir = _resolve_path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"dinov2_{mode}", str(output_dir / "train.log"))
    frame = _load_manifest(manifest_path)
    available = sorted(frame["domain"].unique().tolist())
    if target_domain not in available:
        raise ValueError(f"Target {target_domain!r} absent from manifest; available: {available}")
    unknown_sources = set(source_domains) - set(available)
    if unknown_sources:
        raise ValueError(f"Unknown source domain(s): {sorted(unknown_sources)}; available: {available}")
    if set(source_domains) & {target_domain}:
        raise ValueError("Source and target domains overlap.")

    # Only source rows enter splitting, data-loader construction, validation,
    # checkpoint selection, and scheduler/early-stopping decisions.
    source_frame = frame[frame["domain"].isin(source_domains)].reset_index(drop=True)
    prepared = _prepared_source_split(source_frame, source_domains)
    if prepared is None:
        train_frame, val_frame, split_audit = split_source_domains(
            source_frame, source_domains, float(cfg.dataset.validation_fraction), seed,
        )
    else:
        train_frame, val_frame, split_audit = prepared
    if args.max_samples_per_split:
        train_frame = pd.concat([
            _stratified_cap(part, args.max_samples_per_split, seed + index)
            for index, (_, part) in enumerate(train_frame.groupby("domain", sort=True))
        ], ignore_index=True)
        val_frame = pd.concat([
            _stratified_cap(part, args.max_samples_per_split, seed + 100 + index)
            for index, (_, part) in enumerate(val_frame.groupby("domain", sort=True))
        ], ignore_index=True)
    if train_frame.empty or val_frame.empty:
        raise ValueError("Source train/validation split is empty.")
    if set(train_frame["domain"]) - set(source_domains) or set(val_frame["domain"]) - set(source_domains):
        raise RuntimeError("A non-source row entered the source training or validation split.")
    if set(train_frame["image_path"]) & set(val_frame["image_path"]):
        raise RuntimeError("Image path leakage between source train and validation.")
    if "sha256" in frame:
        train_hashes = set(train_frame["sha256"].astype(str))
        val_hashes = set(val_frame["sha256"].astype(str))
        if train_hashes & val_hashes:
            raise RuntimeError("Exact image hash leakage between source train and validation.")

    batch_size = int(cfg.training.batch_size)
    workers = int(cfg.training.num_workers)
    image_size = int(cfg.dataset.image_size)
    train_loader = build_manifest_loader(
        train_frame, data_root, image_size, batch_size, workers, "train", seed,
    )
    val_loader = build_manifest_loader(
        val_frame, data_root, image_size, batch_size, workers, "val", seed + 1,
    )

    logger.info("Mode=%s | source=%s | target=%s | device=%s", mode, source_domains, target_domain, device)
    logger.info("Source-only split audit: %s", json.dumps(split_audit, sort_keys=True))
    logger.info("Train rows=%d; val rows=%d; target evaluation is deferred until checkpoint selection.",
                len(train_frame), len(val_frame))
    logger.info("Pretrained DINOv2 weights=%s", bool(cfg.model.pretrained))

    if model_instance is None:
        model = __import__("models", fromlist=["build_model"]).build_model(
            model_name=cfg.model.architecture,
            pretrained=bool(cfg.model.pretrained),
            dropout_rate=float(cfg.model.dropout_rate),
            device=str(device),
            image_size=image_size,
        )
    else:
        model = model_instance.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    if trainable_count != total_count:
        raise RuntimeError("The DINOv2 pilot requires full-backbone fine-tuning.")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=int(cfg.training.scheduler_patience),
        factor=float(cfg.training.scheduler_factor),
    )
    scaler = _make_scaler(bool(cfg.training.mixed_precision and device.type == "cuda"))
    use_amp = bool(cfg.training.mixed_precision and device.type == "cuda")
    patience = int(cfg.training.early_stopping_patience)
    best_loss, best_state, best_epoch, bad_epochs = math.inf, None, 0, 0
    history = []
    start_time = time.time()
    checkpoint_path = output_dir / (
        f"dinov2_{mode}_{'_'.join(source_domains)}_target_{target_domain}_seed{seed}_best.pt"
    )

    for epoch in range(1, int(cfg.training.epochs) + 1):
        train_stats = _train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, epoch, logger,
        )
        val_stats = _evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step(val_stats["loss"])
        improved = val_stats["loss"] < best_loss
        if improved:
            best_loss = val_stats["loss"]
            best_epoch = epoch
            bad_epochs = 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            torch.save({
                "model_state": best_state,
                "epoch": best_epoch,
                "validation_loss": best_loss,
                "source_domains": source_domains,
                "target_domain": target_domain,
                "seed": seed,
                "model_name": DINO_MODEL,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "trainable_parameters": trainable_count,
            }, checkpoint_path)
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_auc": train_stats["auc"],
            "val_loss": val_stats["loss"],
            "val_auc": val_stats["auc"],
            "lr": optimizer.param_groups[0]["lr"],
            "train_oom_retries": train_stats["oom_retries"],
            "val_oom_retries": val_stats["oom_retries"],
        }
        history.append(row)
        logger.info(
            "Epoch %d/%d train_loss=%.5f val_loss=%.5f val_auc=%s lr=%.2g%s",
            epoch, int(cfg.training.epochs), train_stats["loss"], val_stats["loss"],
            "NA" if val_stats["auc"] is None else f"{val_stats['auc']:.4f}",
            optimizer.param_groups[0]["lr"], " checkpoint" if improved else "",
        )
        if bad_epochs >= patience:
            logger.info("Early stopping after %d epochs without lower source-validation loss.", bad_epochs)
            break

    if best_state is None:
        raise RuntimeError("No source-validation checkpoint was selected.")
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    context = {
        "cfg": cfg,
        "frame": frame,
        "data_root": data_root,
        "device": device,
        "output_dir": output_dir,
        "seed": seed,
        "mode": mode,
        "source_domains": source_domains,
        "target_domain": target_domain,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "trainable_count": trainable_count,
        "total_count": total_count,
        "use_amp": use_amp,
        "split_audit": split_audit,
        "start_time": start_time,
        "checkpoint_path": checkpoint_path,
        "max_samples_per_split": args.max_samples_per_split,
        "logger": logger,
    }
    if defer_target_evaluation:
        return {
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(output_dir),
            "best_epoch": best_epoch,
            "best_source_validation_loss": float(best_loss),
            "history": history,
            "_evaluation_context": context,
        }

    model.load_state_dict(best_state)
    return evaluate_target_checkpoint(model, context)


@torch.no_grad()
def evaluate_target_checkpoint(model, context):
    """Evaluate the held-out target after source-only checkpoint selection."""
    from omegaconf import OmegaConf

    cfg = context["cfg"]
    frame = context["frame"]
    data_root = context["data_root"]
    device = context["device"]
    output_dir = context["output_dir"]
    seed = context["seed"]
    mode = context["mode"]
    source_domains = context["source_domains"]
    target_domain = context["target_domain"]
    best_epoch = context["best_epoch"]
    best_loss = context["best_loss"]
    trainable_count = context["trainable_count"]
    total_count = context["total_count"]
    use_amp = context["use_amp"]
    split_audit = context["split_audit"]
    start_time = context["start_time"]
    checkpoint_path = context["checkpoint_path"]
    logger = context["logger"]

    # Target rows are fetched only after source validation selected the
    # checkpoint. Their labels are used only for this final evaluation.
    target_frame = frame[frame["domain"] == target_domain].reset_index(drop=True)
    if context["max_samples_per_split"]:
        target_frame = _stratified_cap(
            target_frame, context["max_samples_per_split"], seed + 200,
        )
    target_loader = build_manifest_loader(
        target_frame, data_root, image_size, batch_size, workers, "test", seed + 2,
    )
    labels, probabilities, image_paths, prediction_oom_retries = _predict(
        model, target_loader, device, use_amp,
    )
    if np.unique(labels).size == 2:
        auc = float(roc_auc_score(labels, probabilities))
    else:
        auc = None
    brier = float(brier_score_loss(labels, probabilities))
    bootstrap = _bootstrap_auc(
        labels, probabilities,
        int(cfg.evaluation.bootstrap_repetitions),
        float(cfg.evaluation.bootstrap_fraction),
        seed,
    )
    target_metadata = target_frame.set_index("image_path")
    predictions = pd.DataFrame({
        "image_id": [Path(path).stem for path in image_paths],
        "image_path": image_paths,
        "patient_id": [target_metadata.loc[path, "patient_id"] for path in image_paths],
        "dataset": target_domain,
        "label": labels,
        "prob_gon": probabilities,
        "split": "test",
        "model": DINO_MODEL,
        "source_domains": "+".join(source_domains),
        "target_domain": target_domain,
        "seed": seed,
    })
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    metrics = {
        "protocol": "Public-data method reproduction of DINOv2 single-source and multi-source domain training for GON detection.",
        "mode": mode,
        "model": DINO_MODEL,
        "architecture": "ViT-B/14 with a one-logit linear classification head",
        "source_domains": source_domains,
        "target_domain": target_domain,
        "seed": seed,
        "n_target": int(len(labels)),
        "gon_minus": int((labels == 0).sum()),
        "gon_plus": int((labels == 1).sum()),
        "roc_auc": auc,
        "roc_auc_bootstrap_95ci": bootstrap,
        "brier_score": brier,
        "selection_metric": "minimum source-validation BCE loss",
        "best_epoch": best_epoch,
        "best_source_validation_loss": float(best_loss),
        "trainable_parameters": int(trainable_count),
        "total_parameters": int(total_count),
        "mixed_precision": use_amp,
        "target_prediction_oom_retries": prediction_oom_retries,
        "source_split": split_audit,
        "training_seconds": round(time.time() - start_time, 2),
        "versions": _versions(device),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "outputs": {
            "checkpoint": checkpoint_path.name,
            "predictions_csv": "predictions.csv",
            "history_csv": "history.csv",
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Target results: AUC=%s (bootstrap valid=%d/%d), Brier=%.4f; outputs=%s",
        "NA" if auc is None else f"{auc:.4f}", bootstrap["valid_resamples"],
        bootstrap["requested_resamples"], brier, output_dir,
    )
    return metrics


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    add_common_arguments(parser)
    parser.add_argument("--target-domain", "--target", dest="target_domain", required=True)
    return parser
