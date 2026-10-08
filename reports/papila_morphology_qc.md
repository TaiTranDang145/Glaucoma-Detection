# PAPILA Morphology & M1 Oracle CDR Baseline Quality Control Report

## 1. Executive Summary

- **Designation Note:** **$M_1$ is designated as an "Oracle / Ground-Truth CDR Baseline"**, as CDR is derived directly from manual contours annotated by two clinical experts. It establishes the theoretical upper-bound discriminative capability of morphology alone without automated segmentation error.
- **Total Cohort Analyzed:** 420 fundus images (210 patients with strictly bilateral OD and OS eyes).
- **Masks Generated:** 1,680 lossless binary masks ($420 \times 2\text{ structures} \times 2\text{ experts}$) saved under `data/masks/papila/`.
- **Crops Generated:** 420 square high-resolution optic-disc crops (union OD bounding box + 25% margin) saved under `data/crops/papila/` (termed **Oracle Local-OD branch** for downstream $M_3$).
- **Feature Table:** `papila_morphology.csv` (and mirrored at `data/manifests/papila_morphology.csv`).
- **M1 Oracle CDR Test ROC-AUC:** **0.8039**
  - **Eye-level 95% CI:** **[0.6265, 0.9516]** (2,000 bootstrap iterations at eye level).
  - **Patient-clustered 95% CI:** **[0.5416, 0.9741]** (2,000 bootstrap iterations resampling 32 patients with replacement and taking bilateral pairs).
- **Threshold Selected on Validation (Youden's J):** **0.4068** $\to$ Test Sensitivity: **0.6154** (8/13), Test Specificity: **0.7647** (39/51), Accuracy: **0.7344** (47/64).
- **Calibration (Brier Score):** **0.1243** (Platt-calibrated via univariate Logistic Regression on Train) vs **0.1393** (raw).

---

## 2. Mask Quality Control (QC) & Anatomical Validity

### 2.1 Mask Topology and Containment QC

| Metric | Expert 1 | Expert 2 |
|---|---:|---:|
| Total Masks Evaluated | 420 Disc / 420 Cup | 420 Disc / 420 Cup |
| Empty / Zero Masks | 0 / 420 (0.0%) | 0 / 420 (0.0%) |
| Multiple Connected Components (Disc) | 0 / 420 (0.0%) | 0 / 420 (0.0%) |
| Multiple Connected Components (Cup) | 0 / 420 (0.0%) | 0 / 420 (0.0%) |
| Strict Anatomical Containment ($Cup \subseteq Disc$) | **418 / 420 (99.52%)** | **420 / 420 (100.0%)** |
| Mean % Cup Pixels Outside Disc | **0.0011%** | **0.0000%** |
| Max % Cup Pixels Outside Disc | **0.4357%** | **0.0000%** |

### 2.2 Boundary Overflow and Measurement Logic Rationale
- For **Expert 2**, 100% of cases strictly adhered to $Cup \subseteq Disc$.
- For **Expert 1**, only 2 out of 420 images exhibited minor boundary pixel overflow due to polygon boundary discretization/rounding:
  1. `RET044OD`: 14 pixels outside disc out of 136,512 cup pixels (**0.010%**).
  2. `RET119OD`: 241 pixels outside disc out of 55,313 cup pixels (**0.436%**).
- **Methodological Rationale for Vertical Height Metric:** Instead of rejecting these cases or forcefully clipping polygon contours, computing CDR via maximum vertical spans ($\frac{\text{span}(OC)_y}{\text{span}(OD)_y}$) preserves all 420 samples faithfully, is resilient to sub-pixel boundary rounding, and directly mirrors clinical slit-lamp vertical CDR evaluation.

---

## 3. Morphological Distributions & Inter-Expert Agreement

### 3.1 CDR Distribution

| Metric | Expert 1 | Expert 2 | Consensus ($CDR_{\text{mean}}$) |
|---|---:|---:|---:|
| **Minimum CDR** | 0.1024 | 0.0855 | **0.1018** |
| **Maximum CDR** | 0.8740 | 0.8887 | **0.8590** |
| **Mean CDR $\pm$ Std** | 0.3619 $\pm$ 0.1575 | 0.3583 $\pm$ 0.1581 | **0.3601 $\pm$ 0.1560** |
| **Valid Range ($0 < CDR < 1$)** | 420 / 420 (100%) | 420 / 420 (100%) | **420 / 420 (100%)** |

### 3.2 Inter-Expert Agreement and Correlation Analysis
- **Mean Absolute Difference ($|CDR_1 - CDR_2|$):** **0.0366 CDR units** (3.66 percentage points on the 0–1 scale; not relative difference).
- **Maximum Disagreement:** **0.2191 CDR units**.
- **Pearson Correlation ($r$):** **0.9539** ($p < 10^{-15}$, strong linear correlation).
- **Two-Way Random Absolute Agreement ICC(2,1):** **0.9538** (indicating excellent inter-rater reliability).
- **Bland–Altman Analysis:**
  - Mean systematic bias ($\bar{d}$): **+0.0036 CDR units** (negligible bias between experts).
  - 95% Limits of Agreement: **[-0.0903, +0.0974] CDR units**.

---

## 4. Optic-Disc Crop Verification

- **Bounding Box Policy:** Union bounding box of Expert 1 and Expert 2 OD contours + 25% margin along both axes, centered at OD centroid, cropped into 1:1 square.
- **Boundary Verification:**
  - Out-of-bounds crops: **0 / 420 (0.0%)**.
  - Minimum distance to image edge: **188 pixels**.
  - Corrupt or invalid crops: **0 / 420 (0.0%)**.
- **Protocol Classification:** Because crop coordinates use expert OD annotations across train, validation, and test sets, $M_3$ is classified as an **Oracle / Expert-guided Local OD Branch**.

---

## 5. M1 Oracle CDR Baseline Results (Frozen Test Set)

```text
============================================================
EXPERIMENT M1: ORACLE / GROUND-TRUTH CDR BASELINE
============================================================
Dataset:                     PAPILA (Frozen Test Partition)
Number of Test Patients:     32
Number of Test Images:       64 (13 GON+, 51 GON-)
Risk Score:                  CDR_mean = (CDR_exp1 + CDR_exp2) / 2

Decision Threshold:          0.4068 (selected on Validation via max Youden's J)
Test ROC-AUC:                0.8039
Eye-level 95% CI:            [0.6265, 0.9516] (2,000 iterations)
Patient-clustered 95% CI:    [0.5416, 0.9741] (2,000 cluster iterations)
Test Sensitivity:            0.6154 (8/13)
Test Specificity:            0.7647 (39/51)
Test Accuracy:               0.7344 (47/64)
Test Brier Score:            0.1243 (Platt-Calibrated) / 0.1393 (Raw)

Confusion Matrix:
  True Positive (TP):        8
  False Positive (FP):       12
  True Negative (TN):        39
  False Negative (FN):       5
============================================================
```

> [!NOTE]
> **Comparability Warning:** This $M_1$ test ROC-AUC (0.8039) must NOT be directly compared to published PAPILA CDR results from other literature (such as GONet) as if they shared the same protocol. Other works use automated segmentation models, differing train/test splits, or include suspect eyes. $M_1$ serves as our strictly controlled, internal clean morphology reference.
