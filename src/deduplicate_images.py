"""Create a copy of the dataset with exact SHA-256 duplicates excluded."""

import argparse
import csv
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="data/Images")
    parser.add_argument("--labels", default="data/Labels.csv")
    parser.add_argument("--config", default="configs/efficientnet_b3.yaml")
    parser.add_argument("--output", default="outputs/deduplicated")
    args = parser.parse_args()

    source_images = Path(args.images).resolve()
    output = Path(args.output)
    images_out = (output / "Images").resolve()
    if images_out == source_images or images_out in source_images.parents or source_images in images_out.parents:
        raise ValueError("Output Images/ must not overlap the source Images/ directory")

    with Path(args.labels).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or not rows:
        raise ValueError("Labels CSV is empty or has no header")

    groups = defaultdict(list)
    for row in rows:
        name = row["Image Name"]
        if Path(name).name != name:
            raise ValueError(f"Invalid image filename in CSV: {name}")
        path = source_images / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups[digest].append(row)

    kept = set()
    report = []
    for digest, group in groups.items():
        if len({row["Label"] for row in group}) != 1:
            raise ValueError(f"Duplicate images have conflicting labels: {[r['Image Name'] for r in group]}")
        winner = min(group, key=lambda row: (-float(row["Quality Score"]), row["Image Name"]))
        kept.add(winner["Image Name"])
        report.extend({
            "Dropped Image": row["Image Name"],
            "Kept Image": winner["Image Name"],
            "SHA-256": digest,
            "Dropped Patient": row["Patient"],
            "Kept Patient": winner["Patient"],
            "Label": winner["Label"],
        } for row in group if row["Image Name"] != winner["Image Name"])

    output.mkdir(parents=True, exist_ok=True)
    if images_out.exists():
        shutil.rmtree(images_out)
    images_out.mkdir(parents=True)
    for name in sorted(kept):
        shutil.copy2(source_images / name, images_out / name)

    with (output / "Labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for row in rows if row["Image Name"] in kept)

    report_fields = ["Dropped Image", "Kept Image", "SHA-256", "Dropped Patient", "Kept Patient", "Label"]
    with (output / "duplicate_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report_fields)
        writer.writeheader()
        writer.writerows(sorted(report, key=lambda row: row["Dropped Image"]))

    cfg = OmegaConf.load(args.config)
    output_path = output.as_posix()
    cfg.paths.data_dir = f"{output_path}/"
    cfg.paths.images_dir = f"{output_path}/Images/"
    cfg.paths.labels_csv = f"{output_path}/Labels.csv"
    cfg.paths.output_dir = f"{output_path}/training/"
    cfg.paths.checkpoint_dir = f"{output_path}/training/checkpoints/"
    cfg.paths.log_dir = f"{output_path}/training/logs/"
    cfg.paths.figures_dir = f"{output_path}/training/figures/"
    OmegaConf.save(cfg, output / "efficientnet_b3.yaml")

    output_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in images_out.iterdir() if path.is_file()
    }
    assert len(output_hashes) == len(kept) == len(list(images_out.iterdir()))
    print(f"Images kept: {len(kept)} | duplicates excluded: {len(report)} | output: {output}")


if __name__ == "__main__":
    main()
