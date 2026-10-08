# Public fundus data audit

## Datasets found and usable

| Domain | Images | GON+ | GON- | Patient/case IDs | Exact duplicates removed |
|---|---:|---:|---:|---:|---:|
| DRISHTI_GS | 101 | 70 | 31 | 0 | 0 |
| GAMMA | 99 | 50 | 49 | 99 | 1 |
| HYRD | 737 | 540 | 197 | 286 | 10 |
| PAPILA | 420 | 87 | 333 | 210 | 0 |
| REFUGE | 1200 | 120 | 1080 | 0 | 0 |

Included labeled fundus domains: **HYRD, PAPILA, DRISHTI_GS, REFUGE, GAMMA**.
`source_split` in the all-domain manifest is a deterministic, patient/case-grouped candidate split for a domain only when it is configured as a source. The training engine removes the configured target rows before using any source split, so target data does not enter training, validation, early stopping, or threshold selection.
The local image directory `HYDR/` is normalized to the official domain name `HYRD` in manifests; image paths retain the physical directory name.
REFUGE is retained as a held-out target and never enters source training or validation when selected as target.
GAMMA contributes 100 labeled training cases, of which one byte-identical image was deduplicated; 99 unique fundus images remain. It is not part of the default SSD/MSD source list.

## Datasets excluded

- **REFUGE2:** 1200 files are present; 800 are byte-identical to REFUGE and 400 are unique. No glaucoma diagnosis/label table is present locally, so no REFUGE2 row is usable. REFUGE/REFUGE2 are kept as separate dataset names.
- **KULRD:** not present locally and not included.

## Label mapping

- **HYRD:** HYRD/Labels.csv: GON+=1; GON-=0. Local folder name is HYDR.
- **PAPILA:** Clinical diagnosis: 1 (glaucoma)=1; 0 (healthy)=0; 2 (suspect) excluded.
- **DRISHTI_GS:** Image class directory: Glaucoma=1; Normal=0.
- **REFUGE:** Training class directory or REFUGE validation/test glaucoma label: 1=GON+; 0=GON-.
- **GAMMA:** Training GT one-hot grade: non=0; early or mid_advanced=1.
- Unknown labels and suspect/OHT labels are not coerced to GON-. PAPILA suspect labels are excluded explicitly.

## Integrity and duplicates

- Missing labeled images recorded by source adapters: 0.
- Missing or corrupt images after full decode check: 0.
- Exact byte-identical duplicates removed within datasets: 11.
- Exact cross-domain collisions within the usable manifest: 0; the manifest builder fails if any are found.
- Hashes are SHA-256 over original file bytes. Re-encoded near-duplicates are not detectable by this check.
- Distinct patient/case IDs linked by exact duplicate images: 7 groups. Linked IDs share one `split_group` so their rows cannot cross train/validation.

## Patient-level metadata

- **HYRD:** Available in Labels.csv; 288 patients in local source.
- **PAPILA:** Available in clinical workbooks; patient-level grouping used.
- **DRISHTI_GS:** No separate patient table; the source has one image per case, so image-level grouping is used.
- **REFUGE:** No patient ID included in the local label tables; OOD test bootstrap is image-level.
- **GAMMA:** Case ID is in the training GT workbook; not independently reconciled to the 276-person table.

## Pilot leakage checks

- Fixed pilot split: target rows all held out = True; source/target exact-hash overlap = 0; source train/validation patient leakage = 0; source train/validation split-group leakage = 0; source train/validation hash leakage = 0.

## Differences from the paper and limits

- KULRD is unavailable, so this is a public-data method reproduction and cannot reproduce the paper's KULRD numerical results.
- Local HYRD source contains 747 labeled images; the GONet paper's selected HYRD analysis set is 647 after quality/optic-disc exclusions. This pilot uses the local public labels without reapplying those unavailable filters.
- PAPILA suspect cases are excluded, not assigned to GON-.
- Exact byte hashes detect identical files, not re-encoded or geometrically transformed duplicates.
- Some byte-identical HYRD/GAMMA images carry different patient/case IDs. Duplicate rows are removed and linked IDs share one split group; the source data does not clarify whether these are copied images or the same underlying patient.
- REFUGE has no patient IDs in local label files; target confidence intervals use image-level resampling.
- Local inclusion is by valid image and binary label. The public pilot does not apply the paper's FundusQ-Net quality threshold or optic-disc-presence exclusion; this is recorded as a protocol difference.
- GAMMA mapping uses its provided non/early/mid_advanced clinical grade groups. The local labeled subset is only the 100-case training partition; one exact duplicate is removed from the usable manifest.
- GAMMA files are distributed under the license recorded in the supplied dataset README (CC BY-NC-ND); retain attribution and license when using the data package.

Manifest: `data/manifests/all_public.csv`
Machine-readable audit: `data/manifests/all_public_audit.json`
