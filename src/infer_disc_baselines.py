"""Run an external optic-disc/cup segmenter and save CDR baseline results."""

import argparse
import importlib
import os
import re
from pathlib import Path
from typing import Protocol, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from src.metrics.optic_disc import CDRResult, calculate_vertical_cdr


REPO_ROOT = Path(__file__).absolute().parents[1]
OUTPUT_COLUMNS = [
    "image_path",
    "patient_id",
    "domain",
    "ground_truth",
    "cdr",
    "rdr",
    "segmentation_valid",
    "segmentation_status",
    "rdr_status",
    "od_mask_path",
    "oc_mask_path",
]


class Segmenter(Protocol):
    """Adapter implemented by a user-supplied OD/OC segmentation model."""

    def predict_masks(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return binary optic-disc and optic-cup masks in input image size."""
        ...


def run_inference(
    manifest_path: Path,
    data_root: Path,
    output_dir: Path,
    segmenter: Segmenter,
) -> Path:
    """Infer OD/OC masks for a manifest and write masks plus one CSV row/image."""
    manifest_path = Path(manifest_path)
    data_root = _absolute_path(data_root)
    output_dir = Path(output_dir)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    required = {"image_path", "patient_id", "domain", "label"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError("Manifest is missing columns: {}".format(sorted(missing)))

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in manifest.to_dict(orient="records"):
        relative_image = Path(str(row["image_path"]).replace("\\", "/"))
        image_path = relative_image if relative_image.is_absolute() else data_root / relative_image
        image_path = _absolute_path(image_path)
        try:
            common = os.path.commonpath((str(data_root), str(image_path)))
        except ValueError as exc:
            raise ValueError("Manifest image_path must resolve under data_root: {}".format(row["image_path"])) from exc
        if os.path.normcase(common) != os.path.normcase(str(data_root)):
            raise ValueError("Manifest image_path must resolve under data_root: {}".format(row["image_path"]))
        relative_image = Path(os.path.relpath(str(image_path), str(data_root)))
        if not image_path.is_file():
            raise FileNotFoundError("Fundus image not found: {}".format(image_path))

        with Image.open(image_path) as source:
            image_rgb = np.asarray(source.convert("RGB"))
        predictions = segmenter.predict_masks(image_rgb)
        if not isinstance(predictions, (tuple, list)) or len(predictions) != 2:
            raise ValueError("Segmenter.predict_masks must return (optic_disc_mask, optic_cup_mask).")
        disc = np.asarray(predictions[0])
        cup = np.asarray(predictions[1])

        if disc.ndim != 2 or cup.ndim != 2:
            result = CDRResult(float("nan"), False, "invalid_dimensions")
        elif disc.shape != cup.shape:
            result = CDRResult(float("nan"), False, "shape_mismatch")
        elif disc.shape != image_rgb.shape[:2]:
            result = CDRResult(float("nan"), False, "invalid_dimensions")
        else:
            result = calculate_vertical_cdr(disc, cup)

        od_mask_path = ""
        oc_mask_path = ""
        if disc.ndim == 2 and cup.ndim == 2 and disc.shape == cup.shape == image_rgb.shape[:2]:
            safe_domain = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["domain"]))
            if safe_domain in {"", ".", ".."}:
                safe_domain = "unknown_domain"
            mask_relative = Path("masks") / safe_domain / relative_image.with_suffix(".png")
            od_output = output_dir / mask_relative
            oc_output = output_dir / mask_relative.parent / (mask_relative.stem + "_cup.png")
            od_output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((disc != 0).astype(np.uint8) * 255, mode="L").save(od_output)
            Image.fromarray((cup != 0).astype(np.uint8) * 255, mode="L").save(oc_output)
            od_mask_path = mask_relative.as_posix()
            oc_mask_path = oc_output.relative_to(output_dir).as_posix()

        records.append({
            "image_path": relative_image.as_posix(),
            "patient_id": row["patient_id"],
            "domain": row["domain"],
            "ground_truth": _ground_truth(row["label"], image_path),
            "cdr": result.cdr,
            "rdr": float("nan"),
            "segmentation_valid": result.segmentation_valid,
            "segmentation_status": result.status,
            "rdr_status": "unverified_mask_level_definition",
            "od_mask_path": od_mask_path,
            "oc_mask_path": oc_mask_path,
        })

    result_path = output_dir / "disc_baselines.csv"
    pd.DataFrame(records, columns=OUTPUT_COLUMNS).to_csv(result_path, index=False)
    return result_path


def load_segmenter(cfg: DictConfig) -> Segmenter:
    """Load the configured external factory and its local model checkpoint."""
    segmenter_cfg = cfg.get("segmenter") if cfg is not None else None
    factory_ref = segmenter_cfg.get("factory") if segmenter_cfg else None
    checkpoint_value = segmenter_cfg.get("checkpoint") if segmenter_cfg else None
    if not factory_ref:
        raise ValueError("Configure segmenter.factory as 'module.path:factory_name'.")
    if not checkpoint_value:
        raise ValueError("Configure segmenter.checkpoint with a compatible OD/OC checkpoint path.")

    checkpoint = _repo_relative(checkpoint_value)
    if not checkpoint.is_file():
        raise FileNotFoundError("Segmenter checkpoint not found: {}".format(checkpoint))

    module_name, separator, factory_name = str(factory_ref).partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("segmenter.factory must use 'module.path:factory_name' syntax.")
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ValueError("Segmenter factory is not callable: {}".format(factory_ref))
    device = str(segmenter_cfg.get("device", "cpu"))
    segmenter = factory(str(checkpoint), device)
    if not callable(getattr(segmenter, "predict_masks", None)):
        raise TypeError("Configured segmenter factory must return an object with predict_masks().")
    return segmenter


def _ground_truth(label: object, image_path: Path) -> str:
    if label in (1, "1", "GON+"):
        return "GON+"
    if label in (0, "0", "GON-"):
        return "GON-"
    raise ValueError("Unsupported glaucoma label {!r} for {}".format(label, image_path))


def _repo_relative(path) -> Path:
    path = Path(str(path))
    return path if path.is_absolute() else REPO_ROOT / path


def _absolute_path(path) -> Path:
    """Normalize a path without filesystem-dependent symlink resolution."""
    return Path(os.path.abspath(os.fspath(path)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/disc_baselines.yaml")
    parser.add_argument("--data-root", help="Root directory containing manifest images.")
    parser.add_argument("--manifest", help="Manifest CSV; overrides the config.")
    parser.add_argument("--output-dir", help="Output directory; overrides the config.")
    args = parser.parse_args(argv)
    config_path = _repo_relative(args.config)
    try:
        cfg = OmegaConf.load(config_path)
        segmenter = load_segmenter(cfg)
        paths = cfg.get("paths", {})
        data_root = _repo_relative(args.data_root or paths.get("data_root", "data"))
        manifest_path = _repo_relative(args.manifest or paths.get("manifest", "data/manifests/gonet_msd.csv"))
        output_dir = _repo_relative(args.output_dir or paths.get("output_dir", "outputs/disc_baselines"))
        result_path = run_inference(manifest_path, data_root, output_dir, segmenter)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.error(str(exc))
    print("Disc baseline results: {}".format(result_path))


if __name__ == "__main__":
    main()
