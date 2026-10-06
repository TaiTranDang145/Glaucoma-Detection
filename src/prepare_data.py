"""Build a label-verified CSV manifest for the locally available GONet data."""

import argparse
import csv
import hashlib
import posixpath
import re
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


REPO_ROOT = Path(__file__).absolute().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _column_index(cell_ref):
    index = 0
    for char in cell_ref:
        if not char.isalpha():
            break
        index = index * 26 + ord(char.upper()) - ord("A") + 1
    return index - 1


def _xlsx_rows(path):
    """Read the first worksheet with the stdlib, including shared strings."""
    with zipfile.ZipFile(path) as workbook:
        root = ET.fromstring(workbook.read("xl/workbook.xml"))
        rels = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in rels}
        sheet = root.find("main:sheets/main:sheet", NS)
        target = targets[sheet.attrib[f"{{{NS['rel']}}}id"]].lstrip("/")
        sheet_path = target if target.startswith("xl/") else posixpath.join("xl", target)

        strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            strings = [
                "".join(part.text or "" for part in item.findall(".//main:t", NS))
                for item in shared.findall("main:si", NS)
            ]

        sheet_xml = ET.fromstring(workbook.read(sheet_path))
        rows = []
        for row in sheet_xml.findall(".//main:sheetData/main:row", NS):
            values = {}
            for cell in row.findall("main:c", NS):
                value = cell.find("main:v", NS)
                text = value.text if value is not None else ""
                if cell.attrib.get("t") == "s" and text:
                    text = strings[int(text)]
                elif cell.attrib.get("t") == "inlineStr":
                    text = "".join(part.text or "" for part in cell.findall(".//main:t", NS))
                values[_column_index(cell.attrib.get("r", "A1"))] = text
            rows.append(values)
        return rows


def _data_path(path, data_root):
    return path.absolute().relative_to(data_root).as_posix()


def _record(path, data_root, domain, label, patient_id):
    return {
        "image_path": _data_path(path, data_root),
        "domain": domain,
        "label": int(label),
        "patient_id": f"{domain}:{patient_id}" if patient_id else "",
    }


def _hydr_records(data_root):
    labels = data_root / "HYDR" / "Labels.csv"
    images = data_root / "HYDR" / "Images"
    records = []
    with labels.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            name = row.get("Image Name", "").strip()
            label = row.get("Label", "").strip().upper()
            image = images / name
            if not image.is_file() or label not in {"GON+", "GON-"}:
                continue
            patient = row.get("Patient", "").strip() or image.stem
            records.append(_record(image, data_root, "HYDR", label == "GON+", patient))
    return records


def _drishti_records(data_root):
    records = []
    for images in data_root.joinpath("DRISHTI-GS").rglob("Images"):
        if "__MACOSX" in images.parts:
            continue
        for class_dir in images.iterdir():
            class_name = class_dir.name.casefold()
            if class_name == "glaucoma":
                label = 1
            elif class_name == "normal":
                label = 0
            else:
                continue
            for image in sorted(class_dir.iterdir()):
                if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append(_record(image, data_root, "DRISHTI_GS", label, image.stem))
    return records


def _papila_records(data_root):
    dataset = data_root / "PAPILA"
    fundus_dir = next((p for p in dataset.rglob("FundusImages") if p.is_dir()), None)
    clinical_dir = next((p for p in dataset.rglob("ClinicalData") if p.is_dir()), None)
    if fundus_dir is None or clinical_dir is None:
        raise FileNotFoundError("PAPILA needs FundusImages/ and ClinicalData/.")

    image_by_eye = {}
    for image in fundus_dir.rglob("*"):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = re.fullmatch(r"RET(\d+)(OD|OS)", image.stem.upper())
        if match:
            image_by_eye[(str(int(match.group(1))), match.group(2))] = image

    records = []
    excluded_suspects = 0
    missing_images = 0
    for eye in ("OD", "OS"):
        path = clinical_dir / f"patient_data_{eye.lower()}.xlsx"
        rows = _xlsx_rows(path)
        headers = {}
        for row in rows[:3]:
            for column, value in row.items():
                if value.strip():
                    headers[column] = value.strip().casefold()
        id_col = next((col for col, name in headers.items() if name == "id"), None)
        diagnosis_col = next((col for col, name in headers.items() if name == "diagnosis"), None)
        if id_col is None or diagnosis_col is None:
            raise ValueError(f"PAPILA columns ID/Diagnosis not found in {path}.")

        for row in rows[3:]:
            raw_id = row.get(id_col, "").strip()
            digits = re.search(r"\d+", raw_id)
            raw_diagnosis = row.get(diagnosis_col, "").strip()
            if not digits or raw_diagnosis not in {"0", "1", "2"}:
                continue
            diagnosis = int(raw_diagnosis)
            if diagnosis == 2:
                excluded_suspects += 1
                continue
            key = (str(int(digits.group())), eye)
            image = image_by_eye.get(key)
            if image is None:
                missing_images += 1
                continue
            patient = f"RET{int(digits.group()):03d}"
            records.append(_record(image, data_root, "PAPILA", diagnosis, patient))

    if excluded_suspects:
        print(f"PAPILA: excluded {excluded_suspects} suspect eye labels (class 2).")
    if missing_images:
        print(f"PAPILA: skipped {missing_images} labeled eyes without a matching image.")
    return records


def _table_header(rows):
    for index, row in enumerate(rows[:5]):
        headers = {col: value.strip().casefold() for col, value in row.items() if value.strip()}
        image_col = next((col for col, name in headers.items() if name.replace(" ", "") == "imgname"), None)
        label_col = next((col for col, name in headers.items() if "glaucoma" in name and "label" in name), None)
        if image_col is not None and label_col is not None:
            return index, image_col, label_col
    return None


def _refuge_records(data_root):
    refuge_dir = data_root / "Refuge"
    training_dirs = [
        path for path in refuge_dir.rglob("Training400")
        if "__MACOSX" not in path.parts and (path.parent / "Test400").is_dir()
    ]
    if not training_dirs:
        raise FileNotFoundError("Could not find REFUGE/Training400 next to Test400.")
    base = training_dirs[0].parent
    records = []

    for class_dir in training_dirs[0].iterdir():
        name = class_dir.name.casefold()
        if "normal" in name or "non-glaucoma" in name:
            label = 0
        elif "glaucoma" in name:
            label = 1
        else:
            continue
        for image in sorted(class_dir.rglob("*")):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS and "__MACOSX" not in image.parts:
                records.append(_record(image, data_root, "REFUGE", label, None))

    val_dirs = [p for p in base.rglob("REFUGE-Validation400") if p.is_dir() and "__MACOSX" not in p.parts]
    val_images = {}
    for directory in val_dirs:
        for image in directory.rglob("*"):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS and "__MACOSX" not in image.parts:
                val_images.setdefault(image.name, image)
    test_images = {p.name: p for p in (base / "Test400").glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}

    added_names = set()
    for workbook in base.rglob("*.xlsx"):
        if "__MACOSX" in workbook.parts:
            continue
        rows = _xlsx_rows(workbook)
        header = _table_header(rows)
        if header is None:
            continue
        header_row, image_col, label_col = header
        for row in rows[header_row + 1:]:
            name = row.get(image_col, "").strip()
            raw_label = row.get(label_col, "").strip()
            if not name or raw_label not in {"0", "1"} or name in added_names:
                continue
            images = test_images if name.startswith("T") else val_images if name.startswith("V") else {}
            image = images.get(name)
            if image is not None:
                records.append(_record(image, data_root, "REFUGE", int(raw_label), None))
                added_names.add(name)

    if not added_names:
        raise ValueError(f"REFUGE validation/test diagnosis tables were not matched under {base}.")
    return records


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as image:
        for block in iter(lambda: image.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_refuge2(data_root, records):
    refuge2 = data_root / "REFUGE2"
    if not refuge2.is_dir():
        print("REFUGE2: directory not found; skipped.")
        return
    labeled_hashes = {
        _sha256(data_root / row["image_path"])
        for row in records if row["domain"] == "REFUGE"
    }
    duplicates = 0
    unlabeled_unique = 0
    for image in sorted(refuge2.rglob("images/*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS or "__MACOSX" in image.parts:
            continue
        if _sha256(image) in labeled_hashes:
            duplicates += 1
        else:
            unlabeled_unique += 1
    print(
        "REFUGE2: excluded from training; "
        f"{duplicates} images duplicate labeled REFUGE files, "
        f"{unlabeled_unique} unique images have no glaucoma-label file."
    )


def build_manifest(data_root):
    records = []
    datasets = (
        ("HYDR", _hydr_records),
        ("DRISHTI-GS", _drishti_records),
        ("PAPILA", _papila_records),
        ("Refuge", _refuge_records),
    )
    for directory, adapter in datasets:
        if not (data_root / directory).is_dir():
            print(f"{directory}: directory not found; skipped.")
            continue
        records.extend(adapter(data_root))
    if not records:
        raise ValueError(f"No labeled images found under {data_root}.")

    records.sort(key=lambda row: (row["domain"], row["image_path"]))
    _audit_refuge2(data_root, records)
    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare a label-verified GONet MSD manifest.")
    parser.add_argument("--data-root", default="data", help="Directory containing the dataset folders.")
    parser.add_argument("--output", default="data/manifests/gonet_msd.csv")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    records = build_manifest(data_root.absolute())
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=("image_path", "domain", "label", "patient_id"))
        writer.writeheader()
        writer.writerows(records)

    print(f"Manifest: {output}")
    for domain, rows in sorted(_group_by_domain(records).items()):
        labels = Counter(row["label"] for row in rows)
        print(f"{domain}: {len(rows)} images | GON-={labels[0]} GON+={labels[1]}")
    print(f"Total: {len(records)} images across {len(_group_by_domain(records))} domains.")


def _group_by_domain(records):
    grouped = {}
    for row in records:
        grouped.setdefault(row["domain"], []).append(row)
    return grouped


if __name__ == "__main__":
    main()

