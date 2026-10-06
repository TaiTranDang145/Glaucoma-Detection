"""Evaluate CDR baseline scores and optionally compare classifier probabilities."""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL.PngImagePlugin import PngInfo
from sklearn.metrics import roc_auc_score, roc_curve

from src.metrics.auc import bootstrap_auc_ci


REPO_ROOT = Path(__file__).absolute().parents[1]
AUC_COLUMNS = ["model", "domain", "auc", "n", "auc_ci_low", "auc_ci_high"]


def build_auc_table(
    results: pd.DataFrame,
    gonet_predictions: Optional[pd.DataFrame] = None,
    bootstrap_repetitions: int = 1000,
    bootstrap_fraction: float = 0.95,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]:
    """Build per-domain AUC rows and ROC coordinates for CDR and optional scores."""
    required = {"image_path", "domain", "ground_truth", "cdr", "segmentation_valid"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError("Disc baseline results are missing columns: {}".format(sorted(missing)))

    roc_data: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    cdr_rows = results.copy()
    cdr_rows["_label"] = cdr_rows["ground_truth"].map(_label_value)
    cdr_rows["_score"] = pd.to_numeric(cdr_rows["cdr"], errors="coerce")
    cdr_rows = cdr_rows[
        cdr_rows["segmentation_valid"].map(_is_valid)
        & cdr_rows["_label"].notna()
        & np.isfinite(cdr_rows["_score"])
    ]
    _append_model_results(
        "CDR", cdr_rows, bootstrap_repetitions, bootstrap_fraction, seed, rows, roc_data
    )

    if gonet_predictions is not None:
        probability_columns = {"image_path", "domain", "gonet_prob"}
        missing = probability_columns - set(gonet_predictions.columns)
        if missing:
            raise ValueError("GONet probabilities are missing columns: {}".format(sorted(missing)))
        keys = ["image_path", "domain"]
        if gonet_predictions.duplicated(keys).any():
            raise ValueError("GONet probabilities must have one row per image_path and domain.")
        probability_rows = results[keys + ["ground_truth"]].merge(
            gonet_predictions[keys + ["gonet_prob"]],
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        probability_rows["_label"] = probability_rows["ground_truth"].map(_label_value)
        probability_rows["_score"] = pd.to_numeric(probability_rows["gonet_prob"], errors="coerce")
        probability_rows = probability_rows[
            probability_rows["_label"].notna() & np.isfinite(probability_rows["_score"])
        ]
        _append_model_results(
            "GONet/DINOv2", probability_rows, bootstrap_repetitions,
            bootstrap_fraction, seed, rows, roc_data,
        )

    return pd.DataFrame(rows, columns=AUC_COLUMNS), roc_data


def _append_model_results(
    model: str,
    data: pd.DataFrame,
    repetitions: int,
    fraction: float,
    seed: int,
    rows: List[Dict[str, object]],
    roc_data: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
) -> None:
    for domain in sorted(data["domain"].dropna().unique().tolist()):
        domain_data = data[data["domain"] == domain]
        labels = domain_data["_label"].to_numpy(dtype=int)
        scores = domain_data["_score"].to_numpy(dtype=float)
        if np.unique(labels).size != 2:
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = float(roc_auc_score(labels, scores))
        ci_low, ci_high = bootstrap_auc_ci(
            labels,
            scores,
            repetitions=repetitions,
            fraction=fraction,
            seed=seed,
        )
        rows.append({
            "model": model,
            "domain": domain,
            "auc": auc,
            "n": int(len(domain_data)),
            "auc_ci_low": ci_low,
            "auc_ci_high": ci_high,
        })
        roc_data[(model, domain)] = (fpr, tpr)


def _label_value(value: object) -> Union[int, float]:
    if not isinstance(value, str):
        return np.nan
    if value == "GON+":
        return 1
    if value == "GON-":
        return 0
    return np.nan


def _is_valid(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "yes"}


def save_evaluation_outputs(
    auc_table: pd.DataFrame,
    roc_data: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> Path:
    """Save the comparison table and one ROC plot for each evaluated domain."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "auc_comparison.csv"
    auc_table.to_csv(table_path, index=False)

    for domain in sorted(auc_table["domain"].unique().tolist()):
        figure, axis = plt.subplots(figsize=(6, 5))
        domain_rows = auc_table[auc_table["domain"] == domain]
        for row in domain_rows.to_dict(orient="records"):
            fpr, tpr = roc_data[(row["model"], domain)]
            axis.plot(fpr, tpr, label="{} (AUC={:.3f})".format(row["model"], row["auc"]))
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC — {}".format(domain))
        axis.legend(loc="lower right")
        axis.grid(alpha=0.2)
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(domain)).strip(" .") or "domain"
        figure.tight_layout()
        figure.savefig(
            output_dir / "roc_{}.png".format(filename),
            dpi=150,
            pil_kwargs={"pnginfo": PngInfo()},
        )
        plt.close(figure)
    return table_path


def _repo_relative(path: Union[str, Path]) -> Path:
    path = Path(str(path))
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/disc_baselines.yaml")
    parser.add_argument("--results", help="Inference result CSV; overrides config output location.")
    parser.add_argument("--probabilities", help="Optional CSV with image_path, domain, gonet_prob.")
    parser.add_argument("--output-dir", help="Directory for AUC CSV and ROC plots.")
    args = parser.parse_args(argv)

    cfg = OmegaConf.load(_repo_relative(args.config))
    paths = cfg.get("paths", {})
    evaluation = cfg.get("evaluation", {})
    default_output = _repo_relative(paths.get("output_dir", "outputs/disc_baselines"))
    results_path = _repo_relative(args.results) if args.results else default_output / "disc_baselines.csv"
    output_dir = _repo_relative(args.output_dir) if args.output_dir else default_output
    probability_value = args.probabilities or evaluation.get("gonet_predictions")
    predictions = pd.read_csv(_repo_relative(probability_value)) if probability_value else None
    results = pd.read_csv(results_path)
    auc_table, roc_data = build_auc_table(
        results,
        predictions,
        bootstrap_repetitions=int(evaluation.get("bootstrap_repetitions", 1000)),
        bootstrap_fraction=float(evaluation.get("bootstrap_fraction", 0.95)),
        seed=int(evaluation.get("seed", 42)),
    )
    table_path = save_evaluation_outputs(auc_table, roc_data, output_dir)
    print("AUC comparison: {}".format(table_path))


if __name__ == "__main__":
    main()
