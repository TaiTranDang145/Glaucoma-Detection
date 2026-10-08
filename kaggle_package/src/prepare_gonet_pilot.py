"""Build a fixed, audited public-data SSD/MSD pilot manifest for GONet."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from prepare_data import build_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DOMAINS = ("HYRD", "PAPILA", "DRISHTI_GS")
TARGET_DOMAIN = "REFUGE"
MANIFEST_FIELDS = (
    "image_path", "domain", "label", "patient_id", "split_group",
    "source_label", "sha256", "source_split", "split",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_and_hash_records(records, data_root):
    """Check paths and image decoding, and attach hashes used for leakage checks."""
    failures = []
    hashed_records = []
    for original in records:
        row = dict(original)
        path = data_root / row["image_path"]
        if not path.is_file():
            failures.append({"image_path": row["image_path"], "error": "missing file"})
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:  # Pillow exposes several decoder-specific exceptions.
            failures.append({"image_path": row["image_path"], "error": str(error)})
            continue
        row["sha256"] = row.get("sha256") or sha256_file(path)
        row["label"] = int(row["label"])
        hashed_records.append(row)
    if failures:
        sample = failures[:10]
        raise ValueError(f"{len(failures)} missing or unreadable images; examples: {sample}")
    return hashed_records


def deduplicate_exact_images(records):
    """Drop same-domain byte-identical images, preserving the first sorted path."""
    groups = defaultdict(list)
    for row in records:
        groups[(row["domain"], row["sha256"])].append(row)

    kept = []
    duplicate_groups = []
    for (domain, digest), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda item: item["image_path"])
        labels = {row["label"] for row in rows}
        if len(labels) != 1:
            raise ValueError(
                f"Identical image bytes have conflicting labels in {domain}: "
                f"{[row['image_path'] for row in rows]}"
            )
        kept.append(rows[0])
        if len(rows) > 1:
            duplicate_groups.append({
                "domain": domain,
                "sha256": digest,
                "kept_image_path": rows[0]["image_path"],
                "removed_image_paths": [row["image_path"] for row in rows[1:]],
                "label": rows[0]["label"],
            })

    cross_domain = defaultdict(set)
    for row in kept:
        cross_domain[row["sha256"]].add(row["domain"])
    collisions = [
        {"sha256": digest, "domains": sorted(domains)}
        for digest, domains in cross_domain.items() if len(domains) > 1
    ]
    if collisions:
        raise ValueError(f"Exact image duplicates occur across domains: {collisions[:10]}")
    return sorted(kept, key=lambda row: (row["domain"], row["image_path"])), duplicate_groups


def _split_score(rows, validation_rows, fraction):
    total = len(rows)
    target_size = round(total * fraction)
    counts = Counter(int(row["label"]) for row in rows)
    val_counts = Counter(int(row["label"]) for row in validation_rows)
    size_loss = abs(len(validation_rows) - target_size) / max(target_size, 1)
    class_losses = [
        abs(val_counts[label] - round(counts[label] * fraction))
        / max(round(counts[label] * fraction), 1)
        for label in (0, 1)
        if counts[label]
    ]
    class_loss = sum(class_losses) / len(class_losses) if class_losses else 0.0
    if not validation_rows or len(validation_rows) == total:
        return float("inf")
    if counts[0] and counts[1] and (not val_counts[0] or not val_counts[1]):
        class_loss += 10.0
    return size_loss + class_loss


def grouped_validation_paths(rows, fraction, seed, candidates=12000):
    """Choose a deterministic patient-group holdout near 10%, stratified by label."""
    by_patient = defaultdict(list)
    for row in rows:
        patient = str(row.get("split_group", "")).strip()
        if not patient:
            patient = str(row.get("patient_id", "")).strip()
        if not patient:
            # DRISHTI-GS has one image per case but no patient metadata table.
            patient = str(row["image_path"])
        by_patient[patient].append(row)

    groups = list(by_patient.values())
    if len(groups) < 2:
        raise ValueError(f"Need at least two patient groups; found {len(groups)}")

    rng = random.Random(seed)
    best_score = float("inf")
    best_validation = None
    for _ in range(candidates):
        validation = [row for group in groups if rng.random() < fraction for row in group]
        score = _split_score(rows, validation, fraction)
        if score < best_score:
            best_score = score
            best_validation = validation

    if best_validation is None:
        raise RuntimeError("Could not create a non-empty patient-level validation split")
    return {row["image_path"] for row in best_validation}, {
        "groups_total": len(groups),
        "groups_validation": len({
            str(row.get("split_group", "")).strip()
            or str(row.get("patient_id", "")).strip() or row["image_path"]
            for row in best_validation
        }),
        "score": round(best_score, 6),
    }


def make_pilot_manifest(records, validation_fraction, seed):
    rows = [dict(row) for row in records]
    split_audit = {}
    for domain_index, domain in enumerate(SOURCE_DOMAINS):
        domain_rows = [row for row in rows if row["domain"] == domain]
        if not domain_rows:
            raise ValueError(f"No records found for source domain {domain}")
        if all(row.get("source_split") in {"train", "val"} for row in domain_rows):
            details = {
                "groups_total": len({
                    row.get("split_group") or row.get("patient_id") or row["image_path"]
                    for row in domain_rows
                }),
                "groups_validation": len({
                    row.get("split_group") or row.get("patient_id") or row["image_path"]
                    for row in domain_rows if row["source_split"] == "val"
                }),
                "score": None,
                "split_source": "all_public.source_split",
            }
            for row in domain_rows:
                row["split"] = row["source_split"]
        else:
            validation_paths, details = grouped_validation_paths(
                domain_rows,
                validation_fraction,
                seed + domain_index,
            )
            for row in domain_rows:
                row["split"] = "val" if row["image_path"] in validation_paths else "train"
        split_audit[domain] = details

    target_rows = [row for row in rows if row["domain"] == TARGET_DOMAIN]
    if not target_rows:
        raise ValueError(f"No records found for target domain {TARGET_DOMAIN}")
    for row in target_rows:
        row["split"] = "test"

    return sorted(rows, key=lambda row: (row["domain"], row["split"], row["image_path"])), split_audit


def summarize(rows):
    summary = {}
    for domain in sorted({row["domain"] for row in rows}):
        summary[domain] = {}
        for split in ("train", "val", "test"):
            subset = [row for row in rows if row["domain"] == domain and row["split"] == split]
            if subset:
                counts = Counter(int(row["label"]) for row in subset)
                summary[domain][split] = {
                    "n": len(subset),
                    "gon_minus": counts[0],
                    "gon_plus": counts[1],
                    "patients": len({row["patient_id"] for row in subset if row["patient_id"]}),
                }
    return summary


def audit_splits(rows):
    problems = []
    source_rows = [row for row in rows if row["domain"] in SOURCE_DOMAINS]
    target_rows = [row for row in rows if row["domain"] == TARGET_DOMAIN]
    for row in rows:
        if row["domain"] == TARGET_DOMAIN and row["split"] != "test":
            problems.append(f"REFUGE row is not test: {row['image_path']}")
        if row["domain"] in SOURCE_DOMAINS and row["split"] not in {"train", "val"}:
            problems.append(f"Source row has unexpected split {row['split']}: {row['image_path']}")
        if row["domain"] not in SOURCE_DOMAINS and row["domain"] != TARGET_DOMAIN:
            problems.append(f"Unexpected domain: {row['domain']}")

    patient_splits = defaultdict(set)
    group_splits = defaultdict(set)
    hash_splits = defaultdict(set)
    for row in source_rows:
        if str(row.get("patient_id", "")).strip():
            patient_splits[(row["domain"], row["patient_id"])].add(row["split"])
        group = str(row.get("split_group", "")).strip()
        if group:
            group_splits[group].add(row["split"])
        hash_splits[row["sha256"]].add(row["split"])
    leaked_patients = [key for key, splits in patient_splits.items() if len(splits) > 1]
    leaked_groups = [key for key, splits in group_splits.items() if len(splits) > 1]
    leaked_hashes = [digest for digest, splits in hash_splits.items() if len(splits) > 1]
    if leaked_patients:
        problems.append(f"Patient leakage between train/val: {leaked_patients[:10]}")
    if leaked_groups:
        problems.append(f"Split-group leakage between train/val: {leaked_groups[:10]}")
    if leaked_hashes:
        problems.append(f"Image hash leakage between train/val: {leaked_hashes[:10]}")
    source_hashes = {row["sha256"] for row in source_rows}
    target_hashes = {row["sha256"] for row in target_rows}
    overlapping_hashes = source_hashes & target_hashes
    if overlapping_hashes:
        problems.append(f"Exact image hashes shared by source and REFUGE: {len(overlapping_hashes)}")
    return {
        "passed": not problems,
        "problems": problems,
        "target_rows_all_test": all(row["split"] == "test" for row in target_rows),
        "source_target_hash_overlap_count": len(overlapping_hashes),
        "source_patient_train_val_leakage_count": len(leaked_patients),
        "source_split_group_train_val_leakage_count": len(leaked_groups),
        "source_hash_train_val_leakage_count": len(leaked_hashes),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--source-manifest", default="data/manifests/all_public.csv")
    parser.add_argument("--output", default="data/manifests/gonet_pilot.csv")
    parser.add_argument("--audit", default="data/manifests/gonet_pilot_audit.json")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def main():
    args = parse_args()
    data_root = resolve_path(args.data_root)
    source_manifest_path = resolve_path(args.source_manifest)
    output_path = resolve_path(args.output)
    audit_path = resolve_path(args.audit)
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")

    with source_manifest_path.open(newline="", encoding="utf-8") as source:
        source_rows = list(csv.DictReader(source))
    selected = [
        row for row in source_rows
        if row["domain"] in (*SOURCE_DOMAINS, TARGET_DOMAIN)
    ]
    if not selected:
        raise ValueError(f"No pilot domains found in {source_manifest_path}")

    hashed = validate_and_hash_records(selected, data_root)
    deduplicated, duplicates = deduplicate_exact_images(hashed)
    manifest_rows, split_details = make_pilot_manifest(
        deduplicated, args.validation_fraction, args.seed
    )
    split_audit = audit_splits(manifest_rows)
    if not split_audit["passed"]:
        raise ValueError(f"Pilot split audit failed: {split_audit['problems']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "protocol": {
            "name": "Preliminary public-only GONet method reproduction",
            "single_source_training": "HYDR train; HYDR validation; REFUGE test",
            "multi_source_training": "HYDR + PAPILA + DRISHTI_GS train; source validation; REFUGE test",
            "target_domain": TARGET_DOMAIN,
            "validation_fraction_per_source_domain": args.validation_fraction,
            "seed": args.seed,
            "exact_duplicate_policy": "deduplicate byte-identical images within domain, retaining lexically first image_path",
            "split_unit": "linked split_group when supplied, otherwise patient_id, otherwise one image/case per group",
        },
        "input_manifest": str(source_manifest_path.relative_to(REPO_ROOT)),
        "input_manifest_sha256": sha256_file(source_manifest_path),
        "input_rows_selected": len(selected),
        "image_integrity_audit": {
            "files_checked": len(selected),
            "missing_or_unreadable": 0,
        },
        "rows_after_deduplication": len(deduplicated),
        "duplicates_removed": len(selected) - len(deduplicated),
        "duplicate_groups": duplicates,
        "source_validation_split_details": split_details,
        "counts": summarize(manifest_rows),
        "split_audit": split_audit,
        "outputs": {
            "manifest": str(output_path.relative_to(REPO_ROOT)),
            "audit": str(audit_path.relative_to(REPO_ROOT)),
        },
        "limitations": [
            "KULRD is unavailable, so this is not an exact numerical reproduction of GONet.",
            "REFUGE patient identifiers are not available in the local manifest; its test bootstrap will be image-level.",
            "The SSD versus MSD comparison changes both source-domain count and source-data volume.",
        ],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Pilot manifest: {output_path}")
    print(f"Audit report: {audit_path}")
    print(f"Rows: {len(selected)} input -> {len(deduplicated)} after deduplication")
    for domain, splits in report["counts"].items():
        parts = []
        for split, counts in splits.items():
            parts.append(
                f"{split} n={counts['n']} GON-={counts['gon_minus']} "
                f"GON+={counts['gon_plus']} patients={counts['patients']}"
            )
        print(f"{domain}: " + " | ".join(parts))
    print(f"Split audit: {'PASS' if split_audit['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
