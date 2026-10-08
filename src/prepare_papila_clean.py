"""Prepare clean, fixed PAPILA manifest with stratified patient-level 70/15/15 split.

Protocol:
1. Exclude suspect cases (Diagnosis = 2).
2. Binary classification: Diagnosis 1 -> GON+ (1), Diagnosis 0 -> GON- (0).
3. Patient-level grouping: Both eyes (OD and OS) of any patient must be in the same split.
4. Stratified allocation across patient disease profiles (both healthy, both glaucoma, asymmetric).
5. Output manifest with required fields:
   image_id, patient_id, path, label, split
   plus auxiliary metadata (eye, image_path, contour paths, sha256).
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_papila_root(data_root: Path) -> Path:
    candidates = list(data_root.glob("PAPILA/PapilaDB-PAPILA-*"))
    if candidates and candidates[0].is_dir():
        return candidates[0]
    candidate = data_root / "PAPILA"
    if (candidate / "FundusImages").is_dir():
        return candidate
    raise FileNotFoundError(f"Could not locate PAPILA dataset under {data_root}")


def prepare_papila_dataset(
    data_root: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, dict]:
    papila_root = find_papila_root(data_root)
    fundus_dir = papila_root / "FundusImages"
    clinical_dir = papila_root / "ClinicalData"
    contours_dir = papila_root / "ExpertsSegmentations/Contours"

    if not fundus_dir.is_dir() or not clinical_dir.is_dir() or not contours_dir.is_dir():
        raise FileNotFoundError(f"Missing PAPILA directory structure under {papila_root}")

    # Load clinical Excel files
    od_excel = clinical_dir / "patient_data_od.xlsx"
    os_excel = clinical_dir / "patient_data_os.xlsx"

    df_od = pd.read_excel(od_excel, header=None).iloc[3:].dropna(subset=[0])
    df_os = pd.read_excel(os_excel, header=None).iloc[3:].dropna(subset=[0])

    patients_raw = {}
    for _, row in df_od.iterrows():
        pid = int(str(row[0]).strip().replace("#", ""))
        diag = int(row[3])
        patients_raw.setdefault(pid, {})["OD"] = diag

    for _, row in df_os.iterrows():
        pid = int(str(row[0]).strip().replace("#", ""))
        diag = int(row[3])
        patients_raw.setdefault(pid, {})["OS"] = diag

    total_patients_in_source = len(patients_raw)
    suspect_patients = {}
    valid_patients = {}

    for pid, eyes in sorted(patients_raw.items()):
        if eyes.get("OD") == 2 or eyes.get("OS") == 2:
            suspect_patients[pid] = eyes
        elif "OD" in eyes and "OS" in eyes:
            valid_patients[pid] = eyes
        else:
            raise ValueError(f"Patient {pid} is missing one eye: {eyes}")

    # Stratify the 210 non-suspect patients into 3 strata:
    # 1. both_healthy: OD=0, OS=0 (163 patients)
    # 2. both_glaucoma: OD=1, OS=1 (40 patients)
    # 3. asymmetric: (0, 1) or (1, 0) (7 patients)
    strata = {"both_healthy": [], "both_glaucoma": [], "asymmetric": []}
    for pid, eyes in sorted(valid_patients.items()):
        diag_pair = (eyes["OD"], eyes["OS"])
        if diag_pair == (0, 0):
            strata["both_healthy"].append(pid)
        elif diag_pair == (1, 1):
            strata["both_glaucoma"].append(pid)
        else:
            strata["asymmetric"].append(pid)

    rng = random.Random(seed)
    split_assignment = {}

    # Exact deterministic allocation matching ~70 / 15 / 15
    for s_name, pids in strata.items():
        shuffled = list(pids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if s_name == "both_healthy":
            # 163 patients: 114 train, 24 val, 25 test
            n_tr, n_va, n_te = 114, 24, 25
        elif s_name == "both_glaucoma":
            # 40 patients: 28 train, 6 val, 6 test
            n_tr, n_va, n_te = 28, 6, 6
        elif s_name == "asymmetric":
            # 7 patients: 5 train, 1 val, 1 test
            n_tr, n_va, n_te = 5, 1, 1
        else:
            raise ValueError(f"Unknown stratum {s_name}")

        assert n_tr + n_va + n_te == n, f"Strata allocation mismatch: {n_tr}+{n_va}+{n_te} != {n}"
        for p in shuffled[:n_tr]:
            split_assignment[p] = "train"
        for p in shuffled[n_tr : n_tr + n_va]:
            split_assignment[p] = "val"
        for p in shuffled[n_tr + n_va :]:
            split_assignment[p] = "test"

    # Assemble records
    records = []
    repo_root = Path(__file__).resolve().parents[1]

    for pid in sorted(valid_patients.keys()):
        split = split_assignment[pid]
        patient_str = f"RET{pid:03d}"
        for eye in ["OD", "OS"]:
            diag = valid_patients[pid][eye]
            image_id = f"RET{pid:03d}{eye}"
            filename = f"{image_id}.jpg"
            image_abs = fundus_dir / filename
            if not image_abs.is_file():
                raise FileNotFoundError(f"Fundus image not found: {image_abs}")

            # Verify Pillow can open and decode
            with Image.open(image_abs) as im:
                im.verify()

            # Hash for duplicate verification
            file_hash = sha256_file(image_abs)

            # Contour paths
            disc_exp1 = contours_dir / f"{image_id}_disc_exp1.txt"
            cup_exp1 = contours_dir / f"{image_id}_cup_exp1.txt"
            disc_exp2 = contours_dir / f"{image_id}_disc_exp2.txt"
            cup_exp2 = contours_dir / f"{image_id}_cup_exp2.txt"

            for c in [disc_exp1, cup_exp1, disc_exp2, cup_exp2]:
                if not c.is_file():
                    raise FileNotFoundError(f"Contour file missing: {c}")

            # Construct paths
            rel_repo_path = image_abs.relative_to(repo_root).as_posix()
            rel_data_path = image_abs.relative_to(data_root).as_posix()

            records.append({
                "image_id": image_id,
                "patient_id": patient_str,
                "path": rel_repo_path,
                "label": int(diag),
                "split": split,
                "eye": eye,
                "image_path": rel_data_path,
                "disc_exp1": disc_exp1.relative_to(repo_root).as_posix(),
                "cup_exp1": cup_exp1.relative_to(repo_root).as_posix(),
                "disc_exp2": disc_exp2.relative_to(repo_root).as_posix(),
                "cup_exp2": cup_exp2.relative_to(repo_root).as_posix(),
                "sha256": file_hash,
            })

    df_manifest = pd.DataFrame(records)

    # Sanity checks
    assert len(df_manifest) == 420, f"Expected 420 images, got {len(df_manifest)}"
    train_pts = set(df_manifest[df_manifest["split"] == "train"]["patient_id"])
    val_pts = set(df_manifest[df_manifest["split"] == "val"]["patient_id"])
    test_pts = set(df_manifest[df_manifest["split"] == "test"]["patient_id"])

    assert len(train_pts & val_pts) == 0, "Patient leakage between train and val!"
    assert len(train_pts & test_pts) == 0, "Patient leakage between train and test!"
    assert len(val_pts & test_pts) == 0, "Patient leakage between val and test!"

    # Summary audit dictionary
    audit_info = {
        "dataset_name": "PAPILA",
        "total_source_patients": total_patients_in_source,
        "suspect_patients_excluded": len(suspect_patients),
        "suspect_eyes_excluded": len(suspect_patients) * 2,
        "usable_patients": len(valid_patients),
        "usable_images": len(df_manifest),
        "split_patient_counts": {
            "train": len(train_pts),
            "val": len(val_pts),
            "test": len(test_pts),
        },
        "split_image_counts": df_manifest["split"].value_counts().to_dict(),
        "split_label_counts": {
            s: {
                "total": int(len(sub)),
                "gon_plus": int((sub["label"] == 1).sum()),
                "gon_minus": int((sub["label"] == 0).sum()),
                "gon_plus_ratio": round(float((sub["label"] == 1).mean()), 4),
            }
            for s, sub in df_manifest.groupby("split")
        },
        "patient_overlap_check": {
            "train_val_overlap": len(train_pts & val_pts),
            "train_test_overlap": len(train_pts & test_pts),
            "val_test_overlap": len(val_pts & test_pts),
        },
        "hash_duplicates_within_manifest": int(df_manifest["sha256"].duplicated().sum()),
        "random_seed": seed,
    }

    return df_manifest, audit_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="Root folder containing dataset subdirectories.")
    parser.add_argument(
        "--output-manifest",
        default="data/manifests/papila_manifest.csv",
        help="Target path for clean PAPILA manifest CSV.",
    )
    parser.add_argument(
        "--output-audit",
        default="data/manifests/papila_manifest_audit.json",
        help="Target path for machine-readable audit report JSON.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for patient stratification.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / args.data_root
    out_manifest = repo_root / args.output_manifest
    out_audit = repo_root / args.output_audit

    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    df_manifest, audit_info = prepare_papila_dataset(data_root=data_root, seed=args.seed)

    # Save manifest CSV
    df_manifest.to_csv(out_manifest, index=False)
    print(f"[Done] Clean manifest written to: {out_manifest} ({len(df_manifest)} rows)")

    # Save audit JSON
    with open(out_audit, "w", encoding="utf-8") as f:
        json.dump(audit_info, f, indent=2)
    print(f"[Done] Audit report written to: {out_audit}")

    # Print summary
    print("\n" + "=" * 50)
    print("PAPILA CLEAN PROTOCOL FREEZE SUMMARY")
    print("=" * 50)
    print(f"Total non-suspect patients: {audit_info['usable_patients']}")
    print(f"Total images in manifest:   {audit_info['usable_images']}")
    print(f"Train : {audit_info['split_patient_counts']['train']} patients, {audit_info['split_image_counts']['train']} images "
          f"({audit_info['split_label_counts']['train']['gon_plus']} GON+, {audit_info['split_label_counts']['train']['gon_minus']} GON- | {audit_info['split_label_counts']['train']['gon_plus_ratio']*100:.2f}%)")
    print(f"Val   : {audit_info['split_patient_counts']['val']} patients, {audit_info['split_image_counts']['val']} images "
          f"({audit_info['split_label_counts']['val']['gon_plus']} GON+, {audit_info['split_label_counts']['val']['gon_minus']} GON- | {audit_info['split_label_counts']['val']['gon_plus_ratio']*100:.2f}%)")
    print(f"Test  : {audit_info['split_patient_counts']['test']} patients, {audit_info['split_image_counts']['test']} images "
          f"({audit_info['split_label_counts']['test']['gon_plus']} GON+, {audit_info['split_label_counts']['test']['gon_minus']} GON- | {audit_info['split_label_counts']['test']['gon_plus_ratio']*100:.2f}%)")
    print(f"Patient Leakage (Train/Val/Test): {sum(audit_info['patient_overlap_check'].values())}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
