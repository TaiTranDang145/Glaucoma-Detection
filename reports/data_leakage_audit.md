# Data Leakage & Protocol Leakage Audit

## 1. Summary of Identified Leakage Risks

| Leakage Category | Risk Level | Found in Current Repo? | Description & Root Cause | Prevention Strategy |
|---|---|---|---|---|
| **Same Image Across Splits** | HIGH (in raw data); MITIGATED (in manifest) | YES (raw HYRD & GAMMA) | 10 exact duplicate images in raw HYRD carried different patient IDs (e.g. `110_1.jpg` vs `81_1.jpg`). `all_public.csv` deduplicates them. | Filter SHA-256 byte duplicates before creating splits. |
| **Duplicate Image Between Datasets** | HIGH (REFUGE vs REFUGE2); LOW (others) | YES (800 images between REFUGE & REFUGE2) | REFUGE2 shares 800 identical fundus images with REFUGE. Cross-domain SHA-256 is 0 across the 5 usable domains in `all_public.csv`. | Keep REFUGE2 completely excluded from all manifest builds. |
| **Patient Leakage (Train vs Test)** | HIGH (if naive random split); LOW (if grouped) | POTENTIAL | In PAPILA, each patient has two eyes (`OD` and `OS`). An image-level split places OD in train and OS in test. In raw HYRD, duplicate images had different patient IDs. | Group by patient ID (and union linked duplicate IDs into `split_group`). |
| **Eye Leakage (Same Eye Across Splits)** | HIGH (if naive random split) | POTENTIAL | In HYRD, multiple captures of the same eye exist. Splitting by patient ID prevents eye leakage. | All captures/eyes of a patient must reside in the same split. |
| **Test Set Protocol Leakage** | LOW (in DINOv2 engine); HIGH (in legacy scripts) | PARTIAL | `dinov2_engine.py` isolates the test set completely; however, `src/train.py` selected thresholds without validation-only tuning. | Select model checkpoints strictly on validation loss; tune classification threshold strictly on validation ROC/F1. |
| **Pretrained Model Dataset Overlap** | **CRITICAL** | **YES** | `FunduSegmenter_OriginalImage.pth` was trained on **Drishti-GS** and **REFUGE**. Testing CDR on Drishti-GS or REFUGE constitutes direct training leakage. | Reject FunduSegmenter as a clean baseline. Train clean in-repo segmenter or use PAPILA ground-truth contours. |

---

## 2. In-Depth Analysis of Leakage Vectors

### Vector 1: Identical Images Carrying Different Patient Identifiers (HYRD & GAMMA)
During the audit of `data/manifests/all_public_audit.json`, 11 exact byte-identical duplicates (SHA-256 match) were discovered within raw dataset releases:
- **HYRD (10 duplicates):**
  - `HYDR/Images/110_1.jpg` (Patient 110) $\equiv$ `HYDR/Images/81_1.jpg` (Patient 81)
  - `HYDR/Images/110_0.jpg` (Patient 110) $\equiv$ `HYDR/Images/65_5.jpg` (Patient 65)
  - `HYDR/Images/110_2.jpg` (Patient 110) $\equiv$ `HYDR/Images/81_0.jpg` (Patient 81)
  - `HYDR/Images/239_0.jpg` (Patient 239) $\equiv$ `HYDR/Images/257_0.jpg` (Patient 257)
  - Additional collisions linking Patient 110 with Patient 65 and Patient 81.
- **GAMMA (1 duplicate):**
  - `0039.jpg` (Case 0039) $\equiv$ `0077.jpg` (Case 0077)

**Consequence of Naive Patient Splitting:**
If a split algorithm simply groups by raw `Patient` column (`GroupShuffleSplit(groups=df['Patient'])`), Patient 110 could be placed in `train` while Patient 81 is placed in `test`. Because their image files are byte-identical, the model would evaluate on an image it saw during training, causing artificial performance inflation.
**Resolution in Repo:** The manifest generator in `src/prepare_gonet_pilot.py` unions connected patient IDs into a single `split_group` and drops redundant duplicate rows. Any new single-dataset manifest must preserve this grouping.

---

### Vector 2: Bilateral Eye Correlation (PAPILA)
- In `PAPILA`, 244 patients each contribute two images: Right Eye (`OD`) and Left Eye (`OS`).
- Glaucoma pathology, optic cup enlargement, and retinal vascular features are strongly correlated between left and right eyes of the same individual.
- **Leakage Risk:** If an image-level split is used, 50% of test patients would have their fellow eye present in the training set.
- **Requirement:** A strict patient-level split (grouping by `RETxxx`) must ensure that **both OD and OS of patient `RETxxx` are assigned to the exact same partition**.

---

### Vector 3: Lack of Patient Identifiers in Benchmark Datasets (DRISHTI-GS & REFUGE)
- **DRISHTI-GS:** Contains 101 images. No patient registry or demographic table is provided. It is assumed by community convention that each image corresponds to a distinct patient, but this cannot be mathematically verified.
- **REFUGE (REFUGE 1):** Contains 1,200 images (400 train, 400 val, 400 test). No patient IDs are provided in local metadata. Patient-level clustering cannot be guaranteed. Confidence intervals on REFUGE must therefore be reported with the caveat: *image-level bootstrap resampling due to absence of public patient identifiers*.

---

### Vector 4: Cross-Dataset Overlap (REFUGE vs REFUGE2)
- REFUGE and REFUGE2 share 800 identical images.
- REFUGE2 contains no local glaucoma diagnosis tables.
- **Protocol Safeguard:** REFUGE2 must remain strictly excluded from any training or evaluation manifest to prevent uncurated duplicate contamination.

---

### Vector 5: Pretrained Model Training Contamination (FunduSegmenter)
- As documented in `docs/disc_baselines.md` and confirmed by the FunduSegmenter authors, the checkpoint `FunduSegmenter_OriginalImage.pth` was trained on:
  - Drishti-GS
  - RIM-ONE-r3
  - REFUGE Training
  - REFUGE Validation
- **Violation:** Using this checkpoint to generate OD/OC masks on Drishti-GS or REFUGE violates the premise of an unseen, independent test evaluation. The model has already memorized optic disc and cup boundaries for those specific eyes.
- **Rule:** CDR values obtained from FunduSegmenter on Drishti-GS or REFUGE **must be reported as contaminated exploratory baselines**, never as clean benchmark results.

---

### Vector 6: Target Test Contamination in Optimization & Thresholding
- In scientific protocol, the Test partition must be a **vault**:
  1. It must never participate in backpropagation (training loss).
  2. It must never participate in learning rate scheduling (ReduceLROnPlateau).
  3. It must never participate in early stopping.
  4. It must never participate in best checkpoint selection (`save_best_model`).
  5. It must never participate in decision threshold selection (e.g., Youden's $J$, optimal F1 threshold).
- **Audit Result:** `src/dinov2_engine.py` adheres to items 1–4 by evaluating the test set only after reloading the best validation checkpoint. However, threshold selection must be formally implemented to tune on Validation and freeze before evaluating on Test.
