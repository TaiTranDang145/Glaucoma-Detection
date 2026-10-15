"""Train the clinical OOF v2 protocol; legacy result files are never reused."""

import argparse
import hashlib
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, get_worker_info

from src.clinical_protocol import (
    PREPROCESS, ClinicalBundle, ClinicalImageDataset, FusionHead, aligned_features,
    assign_roles, calibrated_probability, cpu_state, environment_info, fit_calibrator,
    nested_fold_plan, require_disjoint, save_bundle, save_checkpoint, select_threshold,
    sha256_file, validate_roles, write_json,
)
from src.models.mobilenetv3 import MobileNetV3Classifier
from src.models.segmenter import MobileNetV3UNet
from src.train_segmenter import CombinedBCEDiceLoss, compute_predicted_cdr


def seed_run(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_loader_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    info = get_worker_info()
    if hasattr(info.dataset, "transform"):
        info.dataset.transform.set_random_seed(seed)


def image_loader(frame, branch, config, training=False, masks=False, seed=42):
    dataset = ClinicalImageDataset(frame, branch, training,
                                   config["mask_dir"] if masks else None, seed)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=training,
                      num_workers=config["workers"], worker_init_fn=seed_loader_worker,
                      persistent_workers=config["workers"] > 0,
                      generator=torch.Generator().manual_seed(seed),
                      pin_memory=str(config["device"]).startswith("cuda"))


def partition_record(frame):
    return {"image_ids": frame.image_id.tolist(),
            "patient_ids": sorted(frame.patient_id.unique().tolist())}


def select_rows(frame, image_ids):
    if len(set(image_ids)) != len(image_ids) or not set(image_ids).issubset(set(frame.image_id)):
        raise ValueError("Invalid or duplicated fold IDs")
    return frame.set_index("image_id").loc[image_ids].reset_index()


def fit_image_model(factory, branch, train, selection, config, seed, epochs, schedule_epochs=None):
    """Selection=None performs a fixed-epoch refit without inspecting held-out data."""
    if selection is not None:
        require_disjoint(train, selection)
    seed_run(seed)
    model = factory().to(config["device"])
    initial_digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        initial_digest.update(key.encode())
        initial_digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    train_loader = image_loader(train, branch, config, True, branch == "segmenter", seed)
    val_loader = (image_loader(selection, branch, config, masks=branch == "segmenter")
                  if selection is not None else None)
    if branch == "segmenter":
        loss_fn, lr = CombinedBCEDiceLoss(), config["segmenter_lr"]
    else:
        count = train.label.value_counts()
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([count[0] / count[1]],
                                          dtype=torch.float32, device=config["device"]))
        lr = config["global_lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    horizon = schedule_epochs or epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=horizon, eta_min=1e-6)
    best_score, best_loss, best_epoch, best_state = -float("inf"), float("inf"), None, None
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        train_loss = 0.
        for batch in train_loader:
            x = batch["image"].to(config["device"])
            target = batch["mask"] if branch == "segmenter" else batch["label"][:, None]
            target = target.to(config["device"])
            optimizer.zero_grad()
            loss = loss_fn(model(x), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss / len(train)}
        if val_loader is not None:
            model.eval()
            losses, scores, targets = [], [], []
            with torch.inference_mode():
                for batch in val_loader:
                    target = batch["mask"] if branch == "segmenter" else batch["label"][:, None]
                    target = target.to(config["device"])
                    logits = model(batch["image"].to(config["device"]))
                    losses.append((loss_fn(logits, target).item(), len(target)))
                    if branch == "segmenter":
                        p, t = torch.sigmoid(logits) >= .5, target >= .5
                        intersection = (p & t).sum((2, 3)).float()
                        dice = (2 * intersection + 1e-6) / (p.sum((2, 3)) + t.sum((2, 3)) + 1e-6)
                        scores.extend(dice.mean(1).cpu().tolist())
                    else:
                        scores.extend(logits.flatten().cpu().tolist())
                        targets.extend(target.flatten().cpu().tolist())
            score = float(np.mean(scores)) if branch == "segmenter" else float(roc_auc_score(targets, scores))
            val_loss = sum(loss * n for loss, n in losses) / len(selection)
            record.update(selection_score=score, selection_loss=val_loss)
            if score > best_score or (score == best_score and val_loss < best_loss):
                best_score, best_loss, best_epoch, best_state = score, val_loss, epoch, cpu_state(model)
        history.append(record)
        selection_text = (f" selection={record['selection_score']:.5f}" if selection is not None else " refit")
        print(f"{branch} seed={seed} epoch={epoch}/{epochs}"
              f" train_loss={record['train_loss']:.5f}{selection_text}"
              f" elapsed={time.perf_counter() - started:.1f}s", flush=True)
    if selection is None:
        best_epoch = epochs
    else:
        model.load_state_dict(best_state)
    model.eval()
    metadata = {"branch": branch, "seed": seed, "selected_epoch": best_epoch,
                "initial_state_sha256": initial_digest.hexdigest(),
                "schedule_horizon": horizon, "train": partition_record(train),
                "selection": partition_record(selection) if selection is not None else None,
                "history": history, "config": config, "preprocessing": PREPROCESS}
    return model, metadata


@torch.inference_mode()
def predict_cdr(model, frame, config):
    model.eval()
    records = []
    for batch in image_loader(frame, "segmenter", config):
        probabilities = torch.sigmoid(model(batch["image"].to(config["device"]))).cpu().numpy()
        for image_id, masks in zip(batch["image_id"], probabilities):
            cdr, disc_height, cup_height = compute_predicted_cdr(masks[0] >= .5, masks[1] >= .5)
            records.append({"image_id": image_id, "cdr_pred": cdr,
                            "disc_height_pred": disc_height, "cup_height_pred": cup_height})
    return pd.DataFrame(records)


def train_nested_oof(train, plans, config, output_dir, seed, factory=None):
    factory = factory or (lambda: MobileNetV3UNet(pretrained=config["pretrained"]))
    output_dir = Path(output_dir)
    all_predictions, evidence = [], []
    expected_ids = sorted(train.image_id)
    actual_ids = sorted(i for plan in plans for i in plan["heldout_ids"])
    if expected_ids != actual_ids:
        raise ValueError("OOF plan does not cover the training cohort exactly once")
    for plan in plans:
        fold = plan["fold"]
        selection_train = select_rows(train, plan["selection_train_ids"])
        selection_val = select_rows(train, plan["selection_val_ids"])
        outer_train = select_rows(train, plan["refit_train_ids"])
        held = select_rows(train, plan["heldout_ids"])
        require_disjoint(selection_train, selection_val, held)
        require_disjoint(outer_train, held)
        if set(selection_train.image_id) | set(selection_val.image_id) != set(outer_train.image_id):
            raise ValueError("Inner partitions must cover outer training patients")
        print(f"OOF fold {fold}: inner selection, then outer-train refit", flush=True)
        selected, selection_info = fit_image_model(factory, "segmenter", selection_train, selection_val,
                                                  config, seed + fold, config["segmenter_epochs"])
        fold_dir = output_dir / f"fold_{fold}"
        save_checkpoint(fold_dir / "selection.pt", selected, selection_info)
        selected_epoch = selection_info["selected_epoch"]
        del selected
        refit, refit_info = fit_image_model(factory, "segmenter", outer_train, None, config,
                                          seed + fold, selected_epoch, config["segmenter_epochs"])
        refit_info.update(fold=fold, heldout=partition_record(held), selection_epoch_source="inner_validation")
        save_checkpoint(fold_dir / "refit.pt", refit, refit_info)
        # Held-out pixels/GT are never read during epoch selection or refit.
        predicted = predict_cdr(refit, held, config)
        predicted["fold"] = fold
        predicted.to_csv(fold_dir / "heldout_predictions.csv", index=False)
        all_predictions.append(predicted)
        evidence.append({"fold": fold, "selected_epoch": selected_epoch,
                         "selection": selection_info, "refit": refit_info})
        del refit
    predictions = pd.concat(all_predictions, ignore_index=True)
    if predictions.image_id.duplicated().any() or set(predictions.image_id) != set(train.image_id):
        raise ValueError("OOF output IDs do not match training manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    write_json(output_dir / "fold_evidence.json", evidence)
    return predictions


@torch.inference_mode()
def extract_features(model, frame, config, checkpoint):
    values, ids = [], []
    model.eval()
    for batch in image_loader(frame, "global", config):
        values.append(model.extract_features(batch["image"].to(config["device"])).cpu())
        ids.extend(batch["image_id"])
    return {"image_ids": ids, "features": torch.cat(values), "checkpoint_sha256": sha256_file(checkpoint),
            "preprocessing": PREPROCESS,
            "row_metadata": frame[["image_id", "patient_id", "label", "role"]].to_dict("records")}


def fit_fusion(frame, features, cdr, config, seed):
    validate_roles(frame)
    seed_run(seed)
    if cdr.image_id.duplicated().any() or set(cdr.image_id) != set(frame.image_id):
        raise ValueError("CDR IDs do not match the feature manifest")
    predicted_cdr = cdr.set_index("image_id").loc[frame.image_id, "cdr_pred"].to_numpy()
    if not np.isfinite(predicted_cdr).all() or np.any((predicted_cdr < 0) | (predicted_cdr > 1)):
        raise ValueError("Invalid predicted CDR")
    train = frame.role == "train"
    val = frame.role == "val"
    cal = frame.role == "calibration"
    train_cdr = predicted_cdr[train]
    scaler = {"mean": float(train_cdr.mean()), "scale": float(train_cdr.std()) or 1.,
              "fit_image_ids": frame.loc[train, "image_id"].tolist(), "feature": "cdr_pred_oof"}
    normalized = torch.tensor((predicted_cdr - scaler["mean"]) / scaler["scale"], dtype=torch.float32)[:, None]
    x = torch.cat([features, normalized], dim=1).to(config["device"])
    y = torch.tensor(frame.label.to_numpy(), dtype=torch.float32, device=config["device"])[:, None]
    train_indices, val_indices, cal_indices = [np.flatnonzero(mask) for mask in [train, val, cal]]
    head = FusionHead(x.shape[1]).to(config["device"])
    loader = DataLoader(TensorDataset(x[train_indices], y[train_indices]), batch_size=config["batch_size"],
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    target = frame.loc[train, "label"].value_counts()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([target[0] / target[1]], dtype=torch.float32, device=config["device"]))
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["head_lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["head_epochs"], eta_min=1e-6)
    best_score, best_loss, history = -float("inf"), float("inf"), []
    for epoch in range(1, config["head_epochs"] + 1):
        head.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = loss_fn(head(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()
        head.eval()
        with torch.inference_mode():
            logits = head(x[val_indices]).flatten().cpu().numpy()
            val_loss = loss_fn(head(x[val_indices]), y[val_indices]).item()
        score = roc_auc_score(frame.loc[val, "label"], logits)
        history.append({"epoch": epoch, "val_auc": float(score), "val_loss": val_loss})
        if score > best_score or (score == best_score and val_loss < best_loss):
            best_score, best_loss, best_epoch, weights = score, val_loss, epoch, cpu_state(head)
        print(f"head seed={seed} epoch={epoch}/{config['head_epochs']} val_auc={score:.5f}", flush=True)
    head.load_state_dict(weights)
    head.eval()
    with torch.inference_mode():
        # Calibration/test predictions are computed only after head selection is finished.
        logits = head(x).flatten().cpu().numpy()
    calibrator = fit_calibrator(logits[cal_indices], frame[cal], frame[train], frame[val])
    threshold = select_threshold(frame.loc[val, "label"], logits[val_indices])
    predictions = frame[["image_id", "patient_id", "label", "split", "role"]].copy()
    predictions["cdr_pred"] = predicted_cdr
    predictions["logit"] = logits
    predictions["prob_raw"] = expit(logits)
    predictions["prob_calibrated"] = calibrated_probability(logits, calibrator)
    predictions["threshold_applied"] = threshold
    predictions["prediction_binary"] = (predictions.prob_raw >= threshold).astype(int)
    evidence = {"seed": seed, "selected_epoch": best_epoch, "history": history,
                "train": partition_record(frame[train]), "selection": partition_record(frame[val]),
                "calibration": partition_record(frame[cal]), "config": config}
    return head, scaler, calibrator, threshold, predictions, evidence


def metrics(frame):
    tn, fp, fn, tp = confusion_matrix(frame.label, frame.prediction_binary, labels=[0, 1]).ravel()
    return {"n": len(frame), "auc": float(roc_auc_score(frame.label, frame.prob_raw)),
            "brier_calibrated": float(brier_score_loss(frame.label, frame.prob_calibrated)),
            "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp))}


def run_seed(frame, plans, config, output_dir, seed, provenance):
    validate_roles(frame)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    train, val = frame[frame.role == "train"], frame[frame.role == "val"]
    oof = train_nested_oof(train, plans, config, output_dir / "oof", seed)
    print("Selecting final segmenter and global backbone on source validation", flush=True)
    segmenter, seg_info = fit_image_model(lambda: MobileNetV3UNet(pretrained=config["pretrained"]),
        "segmenter", train, val, config, seed, config["segmenter_epochs"])
    global_model, global_info = fit_image_model(lambda: MobileNetV3Classifier(pretrained=config["pretrained"]),
        "global", train, val, config, seed, config["global_epochs"])
    seg_path, global_path = output_dir / "segmenter.pt", output_dir / "global.pt"
    save_checkpoint(seg_path, segmenter, seg_info)
    save_checkpoint(global_path, global_model, global_info)
    final_cdr = predict_cdr(segmenter, frame[frame.role != "train"], config)
    final_cdr["fold"] = -1
    cdr = pd.concat([oof, final_cdr], ignore_index=True)
    cdr.to_csv(output_dir / "cdr_predictions.csv", index=False)
    cache = extract_features(global_model, frame, config, global_path)
    torch.save(cache, output_dir / "global_features.pt")
    features = aligned_features(torch.load(output_dir / "global_features.pt", weights_only=True), frame, sha256_file(global_path))
    head, scaler, calibrator, threshold, predictions, head_info = fit_fusion(frame, features, cdr, config, seed)
    save_checkpoint(output_dir / "head.pt", head, head_info)
    roles = {role: partition_record(part) for role, part in frame.groupby("role")}
    metadata = {"seed": seed, "roles": roles, "config": config, "provenance": provenance,
                "oof_fold_plan": plans,
                "oof_checkpoints": [{"fold": plan["fold"],
                    "selection_sha256": sha256_file(output_dir / "oof" / f"fold_{plan['fold']}" / "selection.pt"),
                    "refit_sha256": sha256_file(output_dir / "oof" / f"fold_{plan['fold']}" / "refit.pt")}
                    for plan in plans],
                "oof_epoch_selection": "inner_validation_then_outer_train_refit",
                "calibration": "patient_holdout_excluded_from_all_training_and_selection",
                "training_complete": True,
                "component_checkpoints": {name: {"file": path.name, "sha256": sha256_file(path)}
                    for name, path in [("global", global_path), ("segmenter", seg_path), ("head", output_dir / "head.pt")]}}
    bundle_path = output_dir / "bundle.pt"
    payload = save_bundle(bundle_path, global_model, segmenter, head, scaler, calibrator, threshold, metadata)
    reloaded = ClinicalBundle.load(bundle_path, config["device"])
    before = ClinicalBundle(payload, config["device"])
    with Image.open(val.iloc[0].path) as image:
        rgb = np.asarray(image.convert("RGB"))
    expected, actual = before.predict_rgb(rgb), reloaded.predict_rgb(rgb)
    np.testing.assert_allclose(list(expected.values()), list(actual.values()), atol=1e-6, rtol=1e-6)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary = {"seed": seed, "bundle_sha256": sha256_file(bundle_path), "reload_check_passed": True,
               "metrics": {role: metrics(part) for role, part in predictions.groupby("role") if role != "train"},
               "calibration_note": "Calibration-role metrics are fitted-sample diagnostics, not independent validation."}
    write_json(output_dir / "summary.json", summary)
    return summary


def prepare_run(args):
    config = vars(args).copy()
    config.pop("plan_only")
    for key in ["output_dir", "manifest", "split_manifest", "mask_dir"]:
        config[key] = str(config[key])
    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if min(config[k] for k in ["segmenter_epochs", "global_epochs", "head_epochs", "batch_size", "threads"]) < 1 or config["workers"] < 0:
        raise ValueError("Epochs, batch size and threads must be positive; workers nonnegative")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Duplicate training seeds")
    frame = pd.read_csv(args.manifest)
    canonical = pd.read_csv(args.split_manifest)
    if canonical.image_id.duplicated().any() or set(frame.image_id) != set(canonical.image_id):
        raise ValueError("Morphology and frozen split manifests have different IDs")
    canonical = canonical.set_index("image_id").loc[frame.image_id].reset_index()
    for key in ["patient_id", "label", "split", "path"]:
        if not np.array_equal(frame[key], canonical[key]):
            raise ValueError(f"Frozen split metadata mismatch: {key}")
    frame["sha256"] = canonical.sha256.to_numpy()
    frame = assign_roles(frame, args.split_seed, args.calibration_folds)
    plans = nested_fold_plan(frame[frame.role == "train"], args.outer_folds, args.inner_folds, args.split_seed)
    mask_fingerprints = {}
    for row in frame.itertuples(index=False):
        if sha256_file(row.path) != row.sha256:
            raise ValueError(f"Raw image hash mismatch: {row.image_id}")
        for structure in ["disc", "cup"]:
            path = args.mask_dir / f"{row.image_id}_{structure}_consensus.png"
            mask_fingerprints[str(path)] = sha256_file(path)
    provenance = {"manifest_sha256": sha256_file(args.manifest),
                  "split_manifest_sha256": sha256_file(args.split_manifest),
                  "mask_sha256": mask_fingerprints,
                  "source_sha256": {p: sha256_file(p) for p in ["src/clinical_protocol.py", "src/clinical_training.py",
                      "src/models/mobilenetv3.py", "src/models/segmenter.py", "src/train_segmenter.py"]}}
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "roles.csv", index=False)
    write_json(output / "fold_plan.json", plans)
    write_json(output / "protocol.json", {"config": config, "preprocessing": PREPROCESS,
        "provenance": provenance, "environment": environment_info(),
        "roles": {role: {**partition_record(part), "images": len(part), "positive_images": int(part.label.sum())}
                  for role, part in frame.groupby("role")}})
    return frame, plans, config, provenance


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=Path("data/manifests/papila_morphology.csv"))
    result.add_argument("--split-manifest", type=Path, default=Path("data/manifests/papila_manifest.csv"))
    result.add_argument("--mask-dir", type=Path, default=Path("data/cache/masks_256"))
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--plan-only", action="store_true")
    result.add_argument("--seeds", type=int, nargs="+", default=[42])
    result.add_argument("--split-seed", type=int, default=42)
    result.add_argument("--calibration-folds", type=int, default=5)
    result.add_argument("--outer-folds", type=int, default=5)
    result.add_argument("--inner-folds", type=int, default=3)
    result.add_argument("--segmenter-epochs", type=int, default=35)
    result.add_argument("--global-epochs", type=int, default=35)
    result.add_argument("--head-epochs", type=int, default=40)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--device", default="auto")
    result.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--segmenter-lr", type=float, default=1e-3)
    result.add_argument("--global-lr", type=float, default=1e-4)
    result.add_argument("--head-lr", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=1e-2)
    return result


def main():
    args = parser().parse_args()
    torch.set_num_threads(args.threads)
    seed_run(args.split_seed)
    frame, plans, config, provenance = prepare_run(args)
    print(frame.groupby("role").agg(images=("image_id", "size"), patients=("patient_id", "nunique"), positives=("label", "sum")).to_string(), flush=True)
    if args.plan_only:
        print(f"Validated protocol saved to {args.output_dir}; no models trained.", flush=True)
        return
    runs = [run_seed(frame, plans, config, args.output_dir / f"seed_{seed}", seed, provenance) for seed in args.seeds]
    write_json(args.output_dir / "runs.json", runs)
