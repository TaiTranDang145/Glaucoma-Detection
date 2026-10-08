"""Build the public-data manifest and a reproducible audit report."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from prepare_data import BUILD_DIAGNOSTICS, build_manifest
from prepare_gonet_pilot import grouped_validation_paths


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FIELDS = (
    "image_path", "domain", "label", "patient_id", "split_group", "source_label",
    "sha256", "source_split",
)
LABEL_RULES = {
    "HYRD": "HYRD/Labels.csv: GON+=1; GON-=0. Local folder name is HYDR.",
    "PAPILA": "Clinical diagnosis: 1 (glaucoma)=1; 0 (healthy)=0; 2 (suspect) excluded.",
    "DRISHTI_GS": "Image class directory: Glaucoma=1; Normal=0.",
    "REFUGE": "Training class directory or REFUGE validation/test glaucoma label: 1=GON+; 0=GON-.",
    "GAMMA": "Training GT one-hot grade: non=0; early or mid_advanced=1.",
}


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _count_rows(rows):
    counts = defaultdict(Counter)
    patients = defaultdict(set)
    for row in rows:
        counts[row["domain"]][int(row["label"])] += 1
        if str(row.get("patient_id", "")).strip():
            patients[row["domain"]].add(row["patient_id"])
    return {
        domain: {
            "n": int(sum(labels.values())),
            "gon_minus": int(labels[0]),
            "gon_plus": int(labels[1]),
            "patient_or_case_ids": len(patients[domain]),
        }
        for domain, labels in sorted(counts.items())
    }


def audit(data_root, manifest_path, report_path, json_path):
    data_root = _resolve(data_root)
    manifest_path = _resolve(manifest_path)
    report_path = _resolve(report_path)
    json_path = _resolve(json_path)
    records = build_manifest(data_root)

    # Precompute deterministic per-domain source candidates. The engine only
    # reads these rows for configured source domains; the configured target is
    # filtered out before it can affect optimization or model selection.
    split_details = {}
    domains = sorted({row["domain"] for row in records})
    for domain_index, domain in enumerate(domains):
        domain_rows = [row for row in records if row["domain"] == domain]
        validation_paths, details = grouped_validation_paths(
            domain_rows, 0.10, 42 + domain_index,
        )
        for row in domain_rows:
            row["source_split"] = "val" if row["image_path"] in validation_paths else "train"
        split_details[domain] = details

    failures = []
    for row in records:
        image_path = data_root / row["image_path"]
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                image.load()
        except Exception as error:
            failures.append({"image_path": row["image_path"], "error": str(error)})

    counts = _count_rows(records)
    duplicate_rows = BUILD_DIAGNOSTICS.get("duplicates_removed", [])
    duplicate_counts = Counter(row["domain"] for row in duplicate_rows)
    missing_by_domain = {}
    for domain, details in BUILD_DIAGNOSTICS.items():
        if not isinstance(details, dict) or "missing_labeled_images" not in details:
            continue
        missing = details["missing_labeled_images"]
        missing_by_domain[domain] = len(missing) if isinstance(missing, (list, tuple)) else int(missing)
    raw_label_details = {
        domain: details for domain, details in BUILD_DIAGNOSTICS.items()
        if isinstance(details, dict) and "raw_labels" in details
    }
    refuge2 = BUILD_DIAGNOSTICS.get("REFUGE2", {})
    pilot_audit_path = data_root / "manifests" / "gonet_pilot_audit.json"
    pilot_audit = json.loads(pilot_audit_path.read_text(encoding="utf-8")) if pilot_audit_path.is_file() else None
    patient_metadata = {
        "HYRD": "Available in Labels.csv; 288 patients in local source.",
        "PAPILA": "Available in clinical workbooks; patient-level grouping used.",
        "DRISHTI_GS": "No separate patient table; the source has one image per case, so image-level grouping is used.",
        "REFUGE": "No patient ID included in the local label tables; OOD test bootstrap is image-level.",
        "GAMMA": "Case ID is in the training GT workbook; not independently reconciled to the 276-person table.",
    }
    metadata = {
        "manifest": str(manifest_path.relative_to(REPO_ROOT)) if manifest_path.is_relative_to(REPO_ROOT) else str(manifest_path),
        "records_after_exact_deduplication": len(records),
        "records_before_exact_deduplication": len(records) + len(duplicate_rows),
        "counts": counts,
        "source_candidate_splits": split_details,
        "duplicates_removed_by_domain": dict(sorted(duplicate_counts.items())),
        "duplicate_records": duplicate_rows,
        "duplicate_patient_links": BUILD_DIAGNOSTICS.get("duplicate_patient_links", []),
        "missing_labeled_images_by_domain": missing_by_domain,
        "corrupt_images": failures,
        "raw_label_details": raw_label_details,
        "refuge2": refuge2,
        "pilot_split_audit": pilot_audit.get("split_audit") if pilot_audit else None,
        "label_rules": LABEL_RULES,
        "patient_metadata": patient_metadata,
        "limitations": [
            "KULRD is unavailable, so this is a public-data method reproduction and cannot reproduce the paper's KULRD numerical results.",
            "Local HYRD source contains 747 labeled images; the GONet paper's selected HYRD analysis set is 647 after quality/optic-disc exclusions. This pilot uses the local public labels without reapplying those unavailable filters.",
            "PAPILA suspect cases are excluded, not assigned to GON-.",
            "Exact byte hashes detect identical files, not re-encoded or geometrically transformed duplicates.",
            "Some byte-identical HYRD/GAMMA images carry different patient/case IDs. Duplicate rows are removed and linked IDs share one split group; the source data does not clarify whether these are copied images or the same underlying patient.",
            "REFUGE has no patient IDs in local label files; target confidence intervals use image-level resampling.",
        ],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    table = ["| Domain | Images | GON+ | GON- | Patient/case IDs | Exact duplicates removed |",
             "|---|---:|---:|---:|---:|---:|"]
    for domain, row in sorted(counts.items()):
        table.append(
            f"| {domain} | {row['n']} | {row['gon_plus']} | {row['gon_minus']} | "
            f"{row['patient_or_case_ids']} | {duplicate_counts[domain]} |"
        )
    report = [
        "# Public fundus data audit",
        "",
        "## Datasets found and usable",
        "",
        *table,
        "",
        "Included labeled fundus domains: **HYRD, PAPILA, DRISHTI_GS, REFUGE, GAMMA**.",
        "`source_split` in the all-domain manifest is a deterministic, patient/case-grouped candidate split for a domain only when it is configured as a source. The training engine removes the configured target rows before using any source split, so target data does not enter training, validation, early stopping, or threshold selection.",
        "The local image directory `HYDR/` is normalized to the official domain name `HYRD` in manifests; image paths retain the physical directory name.",
        "REFUGE is retained as a held-out target and never enters source training or validation when selected as target.",
        "GAMMA contributes 100 labeled training cases, of which one byte-identical image was deduplicated; 99 unique fundus images remain. It is not part of the default SSD/MSD source list.",
        "",
        "## Datasets excluded",
        "",
        f"- **REFUGE2:** {refuge2.get('image_count', 0)} files are present; {refuge2.get('exact_refuge_duplicates', 0)} are byte-identical to REFUGE and {refuge2.get('image_count', 0) - refuge2.get('exact_refuge_duplicates', 0)} are unique. No glaucoma diagnosis/label table is present locally, so no REFUGE2 row is usable. REFUGE/REFUGE2 are kept as separate dataset names.",
        "- **KULRD:** not present locally and not included.",
        "",
        "## Label mapping",
        "",
        *[f"- **{domain}:** {rule}" for domain, rule in LABEL_RULES.items()],
        "- Unknown labels and suspect/OHT labels are not coerced to GON-. PAPILA suspect labels are excluded explicitly.",
        "",
        "## Integrity and duplicates",
        "",
        f"- Missing labeled images recorded by source adapters: {sum(missing_by_domain.values())}.",
        f"- Missing or corrupt images after full decode check: {len(failures)}.",
        f"- Exact byte-identical duplicates removed within datasets: {len(duplicate_rows)}.",
        "- Exact cross-domain collisions within the usable manifest: 0; the manifest builder fails if any are found.",
        "- Hashes are SHA-256 over original file bytes. Re-encoded near-duplicates are not detectable by this check.",
        f"- Distinct patient/case IDs linked by exact duplicate images: {len(BUILD_DIAGNOSTICS.get('duplicate_patient_links', []))} groups. Linked IDs share one `split_group` so their rows cannot cross train/validation.",
        "",
        "## Patient-level metadata",
        "",
        *[f"- **{domain}:** {detail}" for domain, detail in patient_metadata.items()],
        "",
        "## Pilot leakage checks",
        "",
        (
            f"- Fixed pilot split: target rows all held out = {pilot_audit['split_audit']['target_rows_all_test']}; "
            f"source/target exact-hash overlap = {pilot_audit['split_audit']['source_target_hash_overlap_count']}; "
            f"source train/validation patient leakage = {pilot_audit['split_audit']['source_patient_train_val_leakage_count']}; "
            f"source train/validation split-group leakage = {pilot_audit['split_audit'].get('source_split_group_train_val_leakage_count', 'not audited')}; "
            f"source train/validation hash leakage = {pilot_audit['split_audit']['source_hash_train_val_leakage_count']}."
            if pilot_audit and "split_audit" in pilot_audit else "- The fixed pilot split audit has not been generated yet."
        ),
        "",
        "## Differences from the paper and limits",
        "",
        *[f"- {item}" for item in metadata["limitations"]],
        "- Local inclusion is by valid image and binary label. The public pilot does not apply the paper's FundusQ-Net quality threshold or optic-disc-presence exclusion; this is recorded as a protocol difference.",
        "- GAMMA mapping uses its provided non/early/mid_advanced clinical grade groups. The local labeled subset is only the 100-case training partition; one exact duplicate is removed from the usable manifest.",
        "- GAMMA files are distributed under the license recorded in the supplied dataset README (CC BY-NC-ND); retain attribution and license when using the data package.",
        "",
        f"Manifest: `{manifest_path.relative_to(REPO_ROOT) if manifest_path.is_relative_to(REPO_ROOT) else manifest_path}`",
        f"Machine-readable audit: `{json_path.relative_to(REPO_ROOT) if json_path.is_relative_to(REPO_ROOT) else json_path}`",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    if failures:
        raise RuntimeError(f"Data audit found {len(failures)} corrupt images; see {report_path}.")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest", default="data/manifests/all_public.csv")
    parser.add_argument("--report", default="reports/data_audit.md")
    parser.add_argument("--json", default="data/manifests/all_public_audit.json")
    args = parser.parse_args()
    metadata = audit(args.data_root, args.manifest, args.report, args.json)
    print(f"Manifest rows: {metadata['records_after_exact_deduplication']}")
    for domain, counts in metadata["counts"].items():
        print(f"{domain}: n={counts['n']} GON-={counts['gon_minus']} GON+={counts['gon_plus']}")
    print(f"Wrote {args.report} and {args.json}")


if __name__ == "__main__":
    main()
