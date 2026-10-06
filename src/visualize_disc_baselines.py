"""Render fundus images with saved optic-disc and optic-cup contours."""

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from omegaconf import OmegaConf
from PIL import Image
from PIL.PngImagePlugin import PngInfo


REPO_ROOT = Path(__file__).absolute().parents[1]


def render_sample(
    row: pd.Series,
    data_root: Path,
    results_root: Path,
    output_path: Path,
) -> Path:
    """Save one fundus image with OD/OC contours and CDR/RDR annotations."""
    image_path = _resolve_input(data_root, row.get("image_path", ""), "fundus image")
    od_path = _resolve_input(results_root, row.get("od_mask_path", ""), "OD mask")
    oc_path = _resolve_input(results_root, row.get("oc_mask_path", ""), "OC mask")

    with Image.open(image_path) as source:
        image_rgb = np.asarray(source.convert("RGB"))
    disc = _read_binary_mask(od_path)
    cup = _read_binary_mask(oc_path)
    if disc.shape != image_rgb.shape[:2] or cup.shape != image_rgb.shape[:2]:
        raise ValueError(
            "Segmentation mask dimensions must match fundus image {}: image={}, OD={}, OC={}".format(
                image_path, image_rgb.shape[:2], disc.shape, cup.shape
            )
        )

    cdr_text = _format_value("CDR", row.get("cdr"))
    rdr_value = _finite_value(row.get("rdr"))
    rdr_status = str(row.get("rdr_status", ""))
    rdr_text = (
        "RDR unavailable"
        if rdr_value is None or rdr_status == "unverified_mask_level_definition"
        else "RDR: {:.3f}".format(rdr_value)
    )

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.imshow(image_rgb)
    if disc.any():
        axis.contour(disc.astype(float), levels=[0.5], colors=["lime"], linewidths=1.6)
    if cup.any():
        axis.contour(cup.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.6)
    axis.set_title("{} | {}".format(cdr_text, rdr_text))
    axis.set_axis_off()
    axis.legend(
        handles=[
            Line2D([0], [0], color="lime", linewidth=1.6, label="Optic disc (OD)"),
            Line2D([0], [0], color="cyan", linewidth=1.6, label="Optic cup (OC)"),
        ],
        loc="lower right",
        framealpha=0.85,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.tight_layout()
        figure.savefig(output_path, dpi=150, pil_kwargs={"pnginfo": PngInfo()})
    finally:
        plt.close(figure)
    return output_path


def _read_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        mask = np.asarray(source)
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional grayscale mask at {}".format(path))
    return mask != 0


def _resolve_input(root: Path, relative_path, description: str) -> Path:
    if relative_path is None or not str(relative_path).strip():
        raise FileNotFoundError("Missing {} path in result row.".format(description))
    root = _absolute_path(root)
    path = Path(str(relative_path).replace("\\", "/"))
    candidate = path if path.is_absolute() else root / path
    candidate = _absolute_path(candidate)
    try:
        common = os.path.commonpath((str(root), str(candidate)))
    except ValueError as exc:
        raise ValueError("{} path is outside its configured root: {}".format(description, relative_path)) from exc
    if os.path.normcase(common) != os.path.normcase(str(root)):
        raise ValueError("{} path is outside its configured root: {}".format(description, relative_path))
    if not candidate.is_file():
        raise FileNotFoundError("Missing {}: {}".format(description, relative_path))
    return candidate


def _absolute_path(path: Union[str, Path]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _finite_value(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _format_value(name: str, value) -> str:
    numeric_value = _finite_value(value)
    return "{} unavailable".format(name) if numeric_value is None else "{}: {:.3f}".format(name, numeric_value)


def _repo_relative(path: Union[str, Path]) -> Path:
    path = Path(str(path))
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/disc_baselines.yaml")
    parser.add_argument("--results", help="Inference result CSV.")
    parser.add_argument("--data-root", help="Root directory containing manifest images.")
    parser.add_argument("--results-root", help="Root directory containing saved masks.")
    parser.add_argument("--output-dir", help="Directory for sample overlay PNG files.")
    parser.add_argument("--count", type=int, help="Maximum number of samples to render.")
    parser.add_argument("--seed", type=int, help="Random seed for deterministic selection.")
    args = parser.parse_args(argv)

    cfg = OmegaConf.load(_repo_relative(args.config))
    paths = cfg.get("paths", {})
    visualization = cfg.get("visualization", {})
    default_results_root = _repo_relative(paths.get("output_dir", "outputs/disc_baselines"))
    results_path = _repo_relative(args.results) if args.results else default_results_root / "disc_baselines.csv"
    results_root = _repo_relative(args.results_root) if args.results_root else results_path.parent
    data_root = _repo_relative(args.data_root or paths.get("data_root", "data"))
    output_dir = _repo_relative(args.output_dir) if args.output_dir else results_root / "visualizations"
    count = args.count if args.count is not None else int(visualization.get("sample_count", 16))
    seed = args.seed if args.seed is not None else int(visualization.get("seed", 42))
    if count < 1:
        parser.error("--count must be a positive integer.")

    results = pd.read_csv(results_path)
    renderable = results[
        results["od_mask_path"].fillna("").astype(str).str.len().gt(0)
        & results["oc_mask_path"].fillna("").astype(str).str.len().gt(0)
    ]
    if renderable.empty:
        parser.error("No rows have saved OD and OC masks to visualize.")
    selected = renderable.sample(n=min(count, len(renderable)), random_state=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (_, row) in enumerate(selected.iterrows(), start=1):
        output_path = output_dir / "sample_{:03d}.png".format(index)
        render_sample(row, data_root, results_root, output_path)
    print("Rendered {} sample(s) under {}".format(len(selected), output_dir))


if __name__ == "__main__":
    main()
