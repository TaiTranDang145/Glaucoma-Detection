"""Recompute existing research evidence without training or overwriting results.

Run from the repository root with the project virtual environment. All outputs
are diagnostic artifacts; they do not replace the original benchmark results.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score


SEEDS = [42, 123, 2024, 3407, 777]


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def classification_metrics(frame, score="prob_raw"):
    result = {"n": len(frame), "auc": float(roc_auc_score(frame.label, frame[score]))}
    if "prediction_binary" in frame:
        tn, fp, fn, tp = confusion_matrix(frame.label, frame.prediction_binary, labels=[0, 1]).ravel()
        result.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                      sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                      false_negative_ids=frame.loc[(frame.label == 1) & (frame.prediction_binary == 0), "image_id"].tolist())
    if "threshold_applied" in frame:
        result["thresholds"] = frame.threshold_applied.unique().tolist()
        result["binary_threshold_consistent"] = bool(np.array_equal(
            (frame[score] >= frame.threshold_applied).astype(int), frame.prediction_binary))
    if "prob_calibrated" in frame:
        result["brier_calibrated"] = float(brier_score_loss(frame.label, frame.prob_calibrated))
    return result


def cluster_bootstrap(frame, base, comp, iterations=2000):
    """Paired patient bootstrap using multiplicities, including bilateral eyes.

    The sign frequency is descriptive bootstrap support, not a null-based p-value.
    AUC is calculated from weighted positive-negative pairs, with half credit for ties.
    """
    y = frame.label.to_numpy()
    codes, patients = pd.factorize(frame.patient_id, sort=False)
    rng = np.random.RandomState(42)
    weights = []
    for _ in range(iterations):
        counts = np.bincount(rng.choice(len(patients), len(patients), replace=True), minlength=len(patients))
        weights.append(counts[codes])
    weights = np.asarray(weights)
    wp, wn = weights[:, y == 1], weights[:, y == 0]
    denom = wp.sum(axis=1) * wn.sum(axis=1)
    valid = denom > 0

    def aucs(scores):
        pos, neg = scores[y == 1], scores[y == 0]
        comparison = (pos[:, None] > neg).astype(float) + .5 * (pos[:, None] == neg)
        return np.einsum("bi,ij,bj->b", wp[valid], comparison, wn[valid]) / denom[valid]

    delta = aucs(np.asarray(comp)) - aucs(np.asarray(base))
    return {
        "point_delta_auc": float(roc_auc_score(y, comp) - roc_auc_score(y, base)),
        "bootstrap_mean_delta_auc": float(delta.mean()),
        "patient_clustered_ci95": np.percentile(delta, [2.5, 97.5]).tolist(),
        "bootstrap_fraction_delta_le_zero": float(np.mean(delta <= 0)),
        "valid_resamples": int(valid.sum()),
    }


def height(mask):
    rows = np.flatnonzero(mask.any(axis=1))
    return int(rows[-1] - rows[0] + 1) if rows.size else 0


def find_original_mask(row):
    root = Path("data/REFUGE/Refuge")
    if row.subfolder == "REFUGE-Validation400":
        candidates = [root / "REFUGE-Validation400-GT/REFUGE-Validation400-GT/Disc_Cup_Masks" / f"{row.image_id}.bmp"]
    elif row.subfolder == "Test400":
        candidates = [root / f"REFUGE-Test-GT/Disc_Cup_Masks/{group}/{row.image_id}.bmp" for group in ["G", "N"]]
    else:
        # The original evaluator omits these diagnosis subdirectories.
        candidates = [root / folder / group / f"{row.image_id}.bmp"
                      for folder in ["Annotation-Training400/Disc_Cup_Masks", "Disc_Cup_Masks"]
                      for group in ["Glaucoma", "Non-Glaucoma"]]
    return next((p for p in candidates if p.exists()), None)


def morphometry_summary(frame):
    return {
        "n": len(frame),
        "mae_256": float(np.mean(np.abs(frame.pred_cdr - frame.gt_cdr_256))),
        "pearson_r_256": float(pearsonr(frame.pred_cdr, frame.gt_cdr_256).statistic),
        "mae_native": float(np.mean(np.abs(frame.pred_cdr - frame.gt_cdr_native))),
        "pearson_r_native": float(pearsonr(frame.pred_cdr, frame.gt_cdr_native).statistic),
        "mean_signed_error_256": float((frame.pred_cdr - frame.gt_cdr_256).mean()),
        "pred_cdr_median": float(frame.pred_cdr.median()),
        "gt_cdr_256_median": float(frame.gt_cdr_256.median()),
        "empty_predicted_disc": int((frame.disc_height == 0).sum()),
        "zero_predicted_cup": int((frame.cup_height == 0).sum()),
        "max_256_height_definition_difference": float(frame.gt_height_definition_difference.max()),
    }


def plot_refuge_cdr(masks, summaries, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for axis, (name, d) in zip(axes, masks.groupby("subfolder")):
        for label, color, title in [(0, "#31688e", "GON-"), (1, "#b63629", "GON+")]:
            selected = d[d.label == label]
            axis.scatter(selected.gt_cdr_256, selected.pred_cdr, s=10, alpha=.45, color=color, label=title)
        axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        values = summaries[name]
        axis.set(title=f"{name}\nMAE={values['mae_256']:.3f}; r={values['pearson_r_256']:.3f}",
                 xlabel="Ground-truth vertical CDR", xlim=(0, 1), ylim=(0, 1))
        axis.set_aspect("equal")
    axes[0].set_ylabel("Predicted vertical CDR")
    axes[-1].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(output_dir / f"refuge_cdr_scatter.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/research_claims_audit"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv("data/manifests/papila_manifest.csv")
    morphology = pd.read_csv("data/manifests/papila_morphology.csv")
    oof = pd.read_csv("data/manifests/papila_oof_predicted_cdr.csv")
    test = manifest.loc[manifest.split == "test"].reset_index(drop=True)
    report = {"scope": "Recomputation from saved predictions and original masks; no training or model selection."}
    report["papila_split"] = {
        "splits": {s: {"images": len(d), "patients": int(d.patient_id.nunique()),
                        "positive_images": int(d.label.sum()),
                        "positive_patients": int(d.loc[d.label == 1, "patient_id"].nunique())}
                   for s, d in manifest.groupby("split")},
        "patients_in_multiple_splits": int((manifest.groupby("patient_id").split.nunique() > 1).sum()),
        "duplicate_image_ids": int(manifest.image_id.duplicated().sum()),
        "duplicate_manifest_hashes": int(manifest.sha256.duplicated().sum()),
        "all_patients_have_two_eyes": bool((manifest.groupby("patient_id").eye.nunique() == 2).all()),
        "morphology_order_matches": bool(manifest.image_id.equals(morphology.image_id)),
        "oof_order_matches": bool(manifest.image_id.equals(oof.image_id)),
        "train_patient_in_multiple_oof_folds": int((oof[oof.split == "train"].groupby("patient_id").fold.nunique() > 1).sum()),
    }
    raw_hash_mismatches = []
    for row in manifest.itertuples(index=False):
        if fingerprint(Path(row.path))["sha256"] != row.sha256:
            raw_hash_mismatches.append(row.image_id)
    report["papila_split"]["raw_image_hash_mismatches"] = raw_hash_mismatches
    base = test.set_index("image_id")
    frames, runs, input_paths, mismatches = {}, [], [], []
    patterns = {
        "M2": "results/multiseed/M2_seed_{seed}_test_predictions.csv",
        "M5a_oracle": "results/multiseed/M5a_seed_{seed}_test_predictions.csv",
        "M5a_auto_oof": "results/automated/M5a_auto_oof_seed_{seed}_test_predictions.csv",
        "DINOv2_224": "results/dinov2_fair/dinov2_224_seed_{seed}_test_predictions.csv",
        "DINOv2_392": "results/dinov2_fair/dinov2_392_seed_{seed}_test_predictions.csv",
    }
    step5 = json.loads(Path("results/multiseed/multiseed_summary.json").read_text())
    step6 = json.loads(Path("results/automated/step6_oof_summary.json").read_text())
    step8 = json.loads(Path("results/dinov2_fair/fair_dinov2_comparison.json").read_text())
    stored = {"M2": step5["M2_global"]["individual_runs"],
              "M5a_oracle": step5["M5a_global_cdr"]["individual_runs"],
              "M5a_auto_oof": step6["M5a_auto_oof"]["runs"],
              "DINOv2_224": step8["DINOv2_224_finetuned"]["runs"],
              "DINOv2_392": step8["DINOv2_392_finetuned"]["runs"]}
    for model, pattern in patterns.items():
        frames[model] = {}
        for seed in (SEEDS if not model.startswith("DINOv2") else SEEDS[:3]):
            path = Path(pattern.format(seed=seed))
            frame = pd.read_csv(path).set_index("image_id")
            if frame.index.duplicated().any() or set(frame.index) != set(base.index):
                raise ValueError(f"Test image coverage mismatch: {path}")
            frame = frame.loc[base.index].reset_index()
            if not np.array_equal(frame.label, test.label) or not np.array_equal(frame.patient_id, test.patient_id):
                raise ValueError(f"Test metadata mismatch: {path}")
            metrics = classification_metrics(frame)
            runs.append({"model": model, "seed": seed, **metrics})
            frames[model][seed] = frame
            input_paths.append(path)
            expected = next(r for r in stored[model] if r["seed"] == seed)
            for actual_key, stored_key in [("auc", "test_auc"), ("sensitivity", "sensitivity"), ("specificity", "specificity"), ("brier_calibrated", "brier_calibrated")]:
                if abs(metrics[actual_key] - expected[stored_key]) > 0.000051:
                    mismatches.append({"model": model, "seed": seed, "metric": actual_key,
                                       "recomputed": metrics[actual_key], "stored": expected[stored_key]})
    report["classification_runs"] = runs
    report["stored_per_run_metric_mismatches"] = mismatches
    report["classification_summary"] = {}
    for model, by_seed in frames.items():
        aucs = [r["auc"] for r in runs if r["model"] == model]
        mean_scores = np.mean([d.prob_raw.to_numpy() for d in by_seed.values()], axis=0)
        report["classification_summary"][model] = {
            "mean_auc": float(np.mean(aucs)), "std_auc_ddof0": float(np.std(aucs)),
            "ensemble_auc": float(roc_auc_score(test.label, mean_scores)), "seeds": list(by_seed)}
    auto = frames["M5a_auto_oof"]
    report["papila_paired_comparisons"] = {
        "auto_oof_vs_multiseed_M2_seed42": cluster_bootstrap(test, frames["M2"][42].prob_raw, auto[42].prob_raw),
    }
    original_m2 = pd.read_csv("results/M2_global_test_predictions.csv").set_index("image_id").loc[base.index].reset_index()
    report["papila_paired_comparisons"]["auto_oof_vs_cached_backbone_M2"] = {
        str(seed): cluster_bootstrap(test, original_m2.prob_raw, d.prob_raw)
        for seed, d in auto.items()}
    auto_three = np.mean([auto[s].prob_raw.to_numpy() for s in SEEDS[:3]], axis=0)
    report["papila_paired_comparisons"]["auto_oof_3seed_ensemble_auc"] = float(roc_auc_score(test.label, auto_three))
    for model in ["DINOv2_224", "DINOv2_392"]:
        dino_three = np.mean([frames[model][s].prob_raw.to_numpy() for s in SEEDS[:3]], axis=0)
        report["papila_paired_comparisons"][f"auto_oof_vs_{model}_seed42"] = cluster_bootstrap(test, frames[model][42].prob_raw, auto[42].prob_raw)
        report["papila_paired_comparisons"][f"auto_oof_vs_{model}_3seed_ensemble"] = cluster_bootstrap(test, dino_three, auto_three)
    report["original_M2"] = classification_metrics(original_m2)
    correct = (auto[42].prediction_binary.to_numpy() == 1) & (test.label.to_numpy() == 1)
    codes, patients = pd.factorize(test.patient_id, sort=False)
    rng = np.random.RandomState(42)
    sensitivities = []
    for _ in range(2000):
        w = np.bincount(rng.choice(len(patients), len(patients), replace=True), minlength=len(patients))[codes]
        denominator = w[test.label.to_numpy() == 1].sum()
        if denominator:
            sensitivities.append(float(w[correct].sum() / denominator))
    report["sensitivity_patient_clustered_ci95_seed42"] = np.percentile(sensitivities, [2.5, 97.5]).tolist()
    report["papila_cdr"] = {}
    for split, d in oof.groupby("split"):
        report["papila_cdr"][split] = {"n": len(d),
            "mae": float(np.abs(d.cdr_pred_oof - d.cdr_oracle).mean()),
            "pearson_r": float(pearsonr(d.cdr_pred_oof, d.cdr_oracle).statistic)}
    print("PAPILA predictions checked", flush=True)

    refuge_path = Path("results/refuge_zero_shot/refuge_zero_shot_predictions.csv")
    refuge = pd.read_csv(refuge_path)
    refuge_stored = json.loads(Path("results/refuge_zero_shot/refuge_zero_shot_metrics.json").read_text())
    report["refuge_prediction_integrity"] = {
        "duplicate_image_ids": int(refuge.image_id.duplicated().sum()),
        "duplicate_image_paths": int(refuge.image_path.duplicated().sum()),
        "dinov2_score_columns": [c for c in refuge if "dino" in c.lower()],
        "dinov2_json_metrics_have_no_per_image_scores_in_this_artifact": not any("dino" in c.lower() for c in refuge),
        "cdr_equals_stored_height_ratio": bool(np.allclose(refuge.pred_cdr,
            np.divide(refuge.cup_height, refuge.disc_height, out=np.zeros(len(refuge)), where=refuge.disc_height != 0))),
    }
    report["refuge_classification"] = {}
    partitions = {"REFUGE_All_1200": refuge, "REFUGE_Test400": refuge[refuge.subfolder == "Test400"],
                  "REFUGE_Validation400": refuge[refuge.subfolder == "REFUGE-Validation400"],
                  "REFUGE_Training400": refuge[refuge.subfolder == "Training400"]}
    for name, d in partitions.items():
        calculated = {"n": len(d)}
        for metric, score in [("M2_Global_AUC", "m2_prob"), ("M1_Auto_CDR_AUC", "pred_cdr"), ("M5a_Fusion_AUC", "m5a_prob")]:
            calculated[metric] = float(roc_auc_score(d.label, d[score]))
            if abs(calculated[metric] - refuge_stored[name][metric]) > 0.000051:
                mismatches.append({"partition": name, "metric": metric, "recomputed": calculated[metric], "stored": refuge_stored[name][metric]})
        calculated["point_delta_M5a_minus_M2"] = calculated["M5a_Fusion_AUC"] - calculated["M2_Global_AUC"]
        report["refuge_classification"][name] = calculated

    records, mask_issues = [], []
    for index, row in enumerate(refuge.itertuples(index=False)):
        path = find_original_mask(row)
        if path is None:
            mask_issues.append({"image_id": row.image_id, "issue": "missing_mask"})
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unreadable mask: {path}")
        levels = np.flatnonzero(np.bincount(mask.ravel(), minlength=256))
        if not set(levels).issubset({0, 128, 255}):
            mask_issues.append({"image_id": row.image_id, "issue": "unexpected_encoding", "levels": levels.tolist()})
        with Image.open(Path("data") / row.image_path) as image:
            dimensions_match = image.size == (mask.shape[1], mask.shape[0])
        if not dimensions_match:
            mask_issues.append({"image_id": row.image_id, "issue": "image_mask_dimensions"})
        small = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        dh, ch = height(small <= 128), height(small == 0)
        original_dh, original_ch = height(mask <= 128), height(mask == 0)
        legacy_dh, legacy_ch = int((small <= 128).any(axis=1).sum()), int((small == 0).any(axis=1).sum())
        records.append({"image_id": row.image_id, "subfolder": row.subfolder, "label": row.label,
                        "mask_path": str(path), "gt_cdr_native": original_ch / original_dh,
                        "gt_cdr_256": ch / dh, "legacy_gt_cdr_256": legacy_ch / legacy_dh,
                        "gt_height_definition_difference": abs(ch / dh - legacy_ch / legacy_dh),
                        "pred_cdr": row.pred_cdr, "disc_height": row.disc_height, "cup_height": row.cup_height})
        if (index + 1) % 400 == 0:
            print(f"REFUGE masks checked: {index + 1}/{len(refuge)}", flush=True)
    masks = pd.DataFrame(records)
    masks.to_csv(args.output_dir / "refuge_mask_audit.csv", index=False)
    legacy = masks[masks.subfolder != "Training400"]
    report["refuge_mask_issues"] = mask_issues
    report["refuge_morphometry"] = {name: morphometry_summary(d) for name, d in masks.groupby("subfolder")}
    report["refuge_morphometry"]["legacy_800"] = morphometry_summary(legacy)
    report["refuge_morphometry"]["all_1200"] = morphometry_summary(masks)
    plot_refuge_cdr(masks, report["refuge_morphometry"], args.output_dir)
    input_paths += [Path(p) for p in ["data/manifests/papila_manifest.csv", "data/manifests/papila_morphology.csv", "data/manifests/papila_oof_predicted_cdr.csv"]]
    input_paths.append(refuge_path)
    input_paths += [Path(p) for p in ["results/multiseed/multiseed_summary.json",
        "results/automated/step6_oof_summary.json", "results/dinov2_fair/fair_dinov2_comparison.json",
        "results/refuge_zero_shot/refuge_zero_shot_metrics.json", "src/run_step6_oof_pipeline.py",
        "src/evaluate_zero_shot_refuge.py", "src/train_fusion.py", "src/run_multiseed_evaluation.py"] if Path(p).exists()]
    report["input_fingerprints"] = [fingerprint(p) for p in input_paths]
    (args.output_dir / "audit_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in runs]).to_csv(args.output_dir / "classification_runs.csv", index=False)
    print(json.dumps({"classification_summary": report["classification_summary"], "metric_mismatches": mismatches,
                      "refuge_morphometry": report["refuge_morphometry"], "mask_issues": mask_issues,
                      "sensitivity_ci": report["sensitivity_patient_clustered_ci95_seed42"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
