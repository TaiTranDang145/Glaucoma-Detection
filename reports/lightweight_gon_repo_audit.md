# Comprehensive Repository Audit: Lightweight Clinical-Aware GON Classification

## 1. Executive Summary

This audit assesses the codebase in `/home/dekii2275/Glaucoma-Detection` against the target research roadmap:
**"Lightweight Clinical-Aware GON Classification" (GON+ vs GON-)**, which combines:
1. **Global Fundus Branch:** Lightweight CNN (e.g., MobileNetV3-Small / EfficientNet-B0) $\to z_{\text{global}}$
2. **Optic-Disc Local Branch:** OD localization/segmentation $\to$ crop OD + margin $\to$ Lightweight CNN $\to z_{\text{local}}$
3. **Clinical Morphology Branch:** OD/OC mask $\to$ CDR, RDR $\to z_{\text{clinical}}$
4. **Fusion Head:** Concatenation $[z_{\text{global}}, z_{\text{local}}, \text{CDR}, \text{RDR}] \to$ Small MLP $\to P(\text{GON+})$

### Core Audit Takeaways:
- **What is mature & reusable:** The training loop harness, data manifest builder, ImageNet/Albumentations transforms, ROC-AUC bootstrap 95% CI calculation, Brier score evaluation, and mathematical vertical CDR extraction from binary masks are solid and reusable.
- **What is missing:** MobileNetV3, EfficientNet-B0, OD bounding-box/crop generation, feature extraction/pooling, feature fusion MLP, and latency/FLOPs benchmarking are **completely missing**.
- **What is risky/unusable:** The existing OD/OC segmentation baseline relies on **FunduSegmenter**, a heavy ~300M parameter ViT-Large model whose external weights were trained directly on Drishti-GS and REFUGE. **Current CDR results from FunduSegmenter must NOT be used as a clean benchmark due to direct test leakage.**
- **RDR status:** RDR currently outputs `NaN` with status `unverified_mask_level_definition`. There is no functional mask-level RDR formula implemented.

---

## 2. Target Pipeline & Ablation Requirements

| Module | Target Architecture / Method | Output | Status in Repo |
|---|---|---|---|
| **Branch 1: Global** | MobileNetV3-Small or EfficientNet-B0 on Full Fundus | $z_{\text{global}} \in \mathbb{R}^{576}$ | **MISSING** (Only DINOv2 ViT-B & EfficientNet-B3 exist) |
| **Branch 2: Local** | OD bounding box + 25-30% margin crop $\to$ Lightweight CNN | $z_{\text{local}} \in \mathbb{R}^{576}$ | **MISSING** (No crop logic or local branch) |
| **Branch 3: Clinical** | OD/OC masks $\to$ Vertical CDR & verified RDR | $z_{\text{clinical}} \in \mathbb{R}^{2}$ | **PARTIAL** (CDR ready; RDR missing; segmenter leaky) |
| **Branch 4: Fusion** | Concatenation $[z_{\text{global}}, z_{\text{local}}, z_{\text{clinical}}] \to$ 2-layer MLP | Probability $P(\text{GON+})$ | **MISSING** |

### Planned Ablation Experiments
- **M1 (CDR Only):** Logistic regression or threshold on clean CDR.
- **M2 (Lightweight Global):** MobileNetV3-Small on full fundus.
- **M3 (Lightweight Local):** MobileNetV3-Small on OD crop.
- **M4 (Global + Local):** Concatenation of $z_{\text{global}} + z_{\text{local}} \to$ MLP.
- **M5 (Global + CDR/RDR):** Concatenation of $z_{\text{global}} + z_{\text{clinical}} \to$ MLP.
- **M6 (Global + Local + CDR/RDR):** Full multimodal fusion.
- **M7 (DINOv2 Reference):** ViT-B/14 baseline (already implemented in repo for comparison).

---

## 3. Project Inventory

| Component | Status | File | What it Currently Does | Reusable? | Current Problems / Limitations |
|---|---|---|---|---|---|
| **Dataset loader** | PARTIAL | `src/dataset.py` | Loads fundus images via `GlaucomaDataset` (HYDR) and `ManifestDataset` (generic CSV) | YES | `ManifestDataset` hardwired to 392px or 300px; no local crop/patch loader. |
| **Manifest builder** | PARTIAL | `src/prepare_gonet_pilot.py`, `src/prepare_data.py` | Aggregates datasets, normalizes aliases, checks hashes, generates `all_public.csv` | YES | Geared for cross-domain OOD splits. Unit test fails due to `'HYRD'` vs `'HYDR'` mismatch. |
| **Train/val/test split** | PARTIAL | `src/dataset.py`, `src/dinov2_engine.py` | `patient_level_split` (70/15/15) and `split_source_domains` (90/10 group split) | YES | `GroupShuffleSplit` in `dataset.py` does not stratify by class label. |
| **Patient-level split** | READY | `src/dataset.py`, `src/prepare_gonet_pilot.py` | Enforces grouping by patient ID (`split_group` unions linked IDs) | YES | Only HYRD and PAPILA supply genuine patient IDs. |
| **Image preprocessing** | READY | `src/dataset.py` | Square black padding, bilinear resize, ImageNet normalization | YES | Needs 224px / 256px resolution configuration for lightweight backbones. |
| **Augmentation** | READY | `src/dataset.py` | Flips, affine scale/rotate, brightness jitter via Albumentations | YES | Mature and robust. |
| **DINOv2 classifier** | READY | `src/models/gonet.py`, `src/dinov2_engine.py` | ViT-B/14 (`vit_base_patch14_dinov2.lvd142m`) + linear head | YES (as M7) | Heavy (86M parameters, 329MB weights). Not lightweight. |
| **SSD training** | READY | `src/train_ssd.py`, `src/dinov2_engine.py` | Trains single-source domain transfer (e.g. HYRD $\to$ REFUGE) | YES (loop logic) | Tightly bound to domain transfer rather than internal validation. |
| **MSD training** | READY | `src/train_msd.py`, `src/dinov2_engine.py` | Trains multi-source domain transfer | YES (as reference) | Domain generalization focus; not single-dataset internal evaluation. |
| **MobileNet classifier** | **MISSING** | None | Does not exist | NO | Must implement MobileNetV3-Small / Large. |
| **EfficientNet classifier** | PARTIAL | `src/models/efficientnet_b3.py` | EfficientNet-B3 with 2-layer MLP head (512-ReLU-1) | YES (adapt to B0) | Only B3 implemented (~12M params). Model factory rejects B0 or other models. |
| **OD detector** | **MISSING** | None | Does not exist | NO | No object detector (YOLO or Faster R-CNN) present. |
| **OD/OC segmenter** | **RISKY** | `src/models/fundu_segmenter_adapter.py` | Inference adapter for external FunduSegmenter | NO (clean benchmark) | Checkpoint missing locally; trained on Drishti-GS & REFUGE (direct leakage). |
| **FunduSegmenter** | **RISKY** | `src/models/fundu_segmenter_adapter.py` | ViT-Large adapter requiring external repository checkout | PARTIAL (code only) | Massive model (~300M params); severe training set leakage. |
| **CDR calculation** | READY | `src/metrics/optic_disc.py` | Computes vertical cup-to-disc ratio from largest 8-connected components | YES | Overly strict `cup_outside_disc` check marks entire image invalid on minor segmentation bleed. |
| **RDR calculation** | **BROKEN** | `src/metrics/optic_disc.py`, `src/infer_disc_baselines.py` | Returns `NaN` with status `unverified_mask_level_definition` | NO | No discrete pixel-mask formula defined or implemented. |
| **OD crop generation** | **MISSING** | None | Does not exist | NO | Must implement bbox derivation, margin expansion, cropping, and caching. |
| **Feature extraction** | **MISSING** | None | Does not exist | NO | Need forward hook or feature extractor returning $z_{\text{global}}, z_{\text{local}}$. |
| **Fusion model** | **MISSING** | None | Does not exist | NO | Must implement feature concatenation and MLP classification head. |
| **Evaluation** | READY | `src/utils.py`, `src/evaluate_disc_baselines.py` | ROC-AUC, accuracy, sensitivity, specificity, precision, F1 | YES | Threshold fixed at 0.5; needs threshold tuning on validation partition. |
| **Bootstrap CI** | READY | `src/metrics/auc.py`, `src/dinov2_engine.py` | Percentile 95% CI from 1,000 bootstrap resamples | YES | Tested and reliable. |
| **Brier score** | READY | `src/dinov2_engine.py` | Probability calibration error via scikit-learn | YES | Only present in DINOv2 engine; missing from standard `utils.py`. |
| **Inference benchmark** | **MISSING** | None | Does not exist | NO | Need scripts for FLOPs/MACs, parameter count, model size, and inference latency. |
| **Checkpoint save/load** | READY | `src/utils.py`, `src/dinov2_engine.py` | State dict saving/loading with optimizer and best metrics | YES | Mature. |
| **Prediction export** | READY | `src/infer_disc_baselines.py`, `src/dinov2_engine.py` | Exports CSV with per-image probabilities, ground truth, domain | YES | Mature. |

---

## 4. Dataset Audit

| Dataset | Root Path | Total Images | Unique Images | Patient IDs | GON+ | GON- | Suspects | OD/OC Masks Available? | Mask Format & Details | Usability Status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| **HYRD / HYDR** | `data/HYDR` | 747 | 737 | 286 | 540 | 197 | 0 | **NO** | None | Usable for Global branch only. Lacks segmentation. |
| **PAPILA** | `data/PAPILA/...` | 488 | 488 | 244 (210 non-suspect) | 87 | 333 | 68 | **YES (100%)** | Polygon contours (`.txt`) for 488 images $\times$ 2 experts $\times$ 2 structures (1,952 files). | **HIGHLY RECOMMENDED** for primary pipeline. |
| **DRISHTI-GS** | `data/DRISHTI-GS` | 101 | 101 | 0 (assumed 1/case) | 70 | 31 | 0 | **YES (100%)** | Softmap PNGs & average boundary txt files. | Small sample size (101 images). |
| **REFUGE 1** | `data/REFUGE/Refuge` | 1,200 | 1,200 | 0 (no patient IDs) | 120 | 1,080 | 0 | **YES (100%)** | Binary `.bmp` masks (1,200 images: 400 train, 400 val, 400 test). | Usable for segmentation training; lacks patient IDs. |
| **REFUGE 2** | `data/REFUGE2/REFUGE2` | 1,200 | 400 unique (800 match REFUGE 1) | 0 | Unknown | Unknown | Unknown | **YES (100%)** | 1,200 masks in `train/mask`, `val/mask`, `test/mask`. | **EXCLUDED** (No diagnosis table locally). |
| **GAMMA** | `data/GAMMA` | 100 | 99 | 99 | 50 | 49 | 0 | **NO** | Only OCT slices and grading table. | Excluded from default runs (CC BY-NC-ND license). |
| **ORIGA** | — | 0 | 0 | — | — | — | — | — | Not present in repository. | **MISSING** |
| **G1020** | — | 0 | 0 | — | — | — | — | — | Not present in repository. | **MISSING** |

---

## 5. Audit of Current CDR Pipeline

### How the Current CDR Pipeline Operates
1. **Model & Architecture:** `FunduSegmenterAdapter` loads a ViT-Large backbone (`VisionTransformer`, 24 layers, 1024 embedding width) with a `SegmenterDecoder`.
2. **Inference Execution:** Resizes fundus image to $256 \times 256$, computes logits, interpolates back to original resolution with bicubic sampling, and extracts OD (class $> 0$) and OC (class $= 2$).
3. **CDR Calculation:** `calculate_vertical_cdr` extracts the largest 8-connected component for disc and cup, computes vertical bounding-box heights, and returns $CDR = \text{height}(OC) / \text{height}(OD)$.
4. **Evaluation:** `src/evaluate_disc_baselines.py` takes the computed CDR and directly evaluates it as a continuous positive classification score for GON+ (`roc_auc_score(labels, cdr)`). **No classifier is trained on CDR.**

### Critical Leakage Finding
> [!CAUTION]
> **CURRENT CDR RESULTS MUST NOT BE USED AS A CLEAN BASELINE.**
>
> The author of FunduSegmenter explicitly stated that `FunduSegmenter_OriginalImage.pth` was trained on **Drishti-GS**, **RIM-ONE-r3**, **REFUGE Train**, and **REFUGE Validation**. Evaluating FunduSegmenter on Drishti-GS or REFUGE constitutes direct training set leakage. Furthermore, the checkpoint is not stored locally and requires an external Kaggle download.

### Reusable CDR Components
- Component isolation: `_largest_component` in `src/metrics/optic_disc.py`
- Metric calculation: `calculate_vertical_cdr` in `src/metrics/optic_disc.py`
- Bootstrap confidence interval: `bootstrap_auc_ci` in `src/metrics/auc.py`
- Segmentation visualizer: `src/visualize_disc_baselines.py`

---

## 6. OD/OC Segmentation Retraining Feasibility

Because ORIGA and G1020 are absent, the repository cannot replicate their training setups. However, **the repo contains extensive gold-standard segmentation masks**:
- **REFUGE:** 400 training and 400 validation masks.
- **PAPILA:** 488 dual-expert contour annotations.

### Proposed Lightweight Segmentation Architectures
1. **Lightweight U-Net with MobileNetV3-Small Encoder (Recommended):**
   - Parameters: ~2.5M
   - Input: $256 \times 256$ or $384 \times 384$
   - Output: 2 binary channels (OD and OC)
   - Training time: ~15 minutes on RTX 3050 for 50 epochs on REFUGE Train (400 images).
2. **YOLOv8n-seg / YOLO11n-seg:**
   - Parameters: ~3.2M
   - Advantages: Directly predicts OD/OC bounding boxes and segmentation masks simultaneously, enabling instant local crops.

---

## 7. Global Classifier Audit

- **Current State:** The repository only supports DINOv2 ViT-B/14 (`vit_base_patch14_dinov2.lvd142m`, ~86M parameters) and EfficientNet-B3 (`efficientnet_b3`, ~12M parameters).
- **MobileNetV3-Small:** Completely MISSING.
- **EfficientNet-B0:** Completely MISSING.

### Recommended Backbone Priority
1. **MobileNetV3-Small (`mobilenetv3_small_100` via `timm`):**
   - Parameters: ~1.5M - 2.5M
   - FLOPs: ~60 MFLOPs at $224 \times 224$
   - Target Global Feature Vector: $z_{\text{global}} \in \mathbb{R}^{576}$
2. **EfficientNet-B0 (`efficientnet_b0` via `timm`):**
   - Parameters: ~5.3M
   - FLOPs: ~390 MFLOPs at $224 \times 224$
   - Target Global Feature Vector: $z_{\text{global}} \in \mathbb{R}^{1280}$

### Refactoring Strategy for Global Branch
Abstract the training harness from `dinov2_engine.py` into a generic model trainer:
- Replace DINOv2 model loading with `timm.create_model(backbone, pretrained=True, num_classes=1)`.
- Use `BCEWithLogitsLoss` with positive class weighting (`pos_weight = N_neg / N_pos`).
- Retain mixed precision (`torch.amp.autocast`), gradient clipping (1.0), and `ReduceLROnPlateau`.

---

## 8. Local Optic-Disc Branch Audit

- **Current State:** Completely MISSING.
- **Required Implementation:**
  ```text
  Fundus Image + OD Mask
       ↓
  Find Disc Bounding Box: [ymin, xmin, ymax, xmax]
       ↓
  Expand with 25% - 30% Margin:
    margin_y = 0.25 * (ymax - ymin + 1)
    margin_x = 0.25 * (xmax - xmin + 1)
       ↓
  Square Crop around Centroid (padded if border reached)
       ↓
  Resize to 224 × 224 & Cache to Disk
       ↓
  MobileNetV3-Small (Feature Extractor)
       ↓
  z_local ∈ R^576
  ```

---

## 9. RDR (Rim-to-Disc Ratio) Status

- **Status in Repo:** `rdr = NaN`, `rdr_status = "unverified_mask_level_definition"`.
- **Reason:** While the GONet preprint references rim measurements, translating continuous neuroretinal rim geometry into discrete pixel mask calculations requires a clear mathematical definition:
  - Minimum neuroretinal rim thickness along radial rays from the cup center to the disc boundary:
    $$\text{RDR} = \frac{\min_{\theta} (R_{\text{disc}}(\theta) - R_{\text{cup}}(\theta))}{\text{Diameter}_{\text{disc}}}$$
  - Without an empirical validation of this discrete measurement against expert clinical annotations, RDR cannot be assumed correct.
- **Recommendation:** Keep RDR optional in Phase 1; focus Phase 1 clinical morphology strictly on **validated vertical CDR**. Introduce verified radial RDR in Phase 2.

---

## 10. Feature Fusion Architecture

- **Status:** Completely MISSING.
- **Target Design:**
  ```text
  z_global (576) ──┐
  z_local  (576) ──┼── Concatenation [1153 or 1154 dims]
  CDR (norm)     ──┤
  [RDR (norm)]   ──┘
         ↓
  Linear(dim, 64) → BatchNorm1d → ReLU → Dropout(0.2)
         ↓
  Linear(64, 1) → Sigmoid → P(GON+)
  ```
- **Crucial Normalization Rule:** CDR (and RDR) must be standardized ($\mu, \sigma$) using statistics **fitted strictly on the training partition**. Validation and test partitions must use the training scaler to prevent data leakage.

---

## 11. Recommended First Dataset

### Evaluation Matrix

| Criteria | HYRD (737 images) | PAPILA (420 images) | REFUGE (1,200 images) |
|---|---|---|---|
| **Patient Identifiers** | YES (286 patients) | **YES (210 patients)** | NO |
| **Bilateral Eyes (OD/OS)** | Implicit/Mixed | **YES (Explicit OD & OS)** | NO |
| **Segmentation Available** | NO | **YES (100% Expert Contours)** | YES (100% Masks) |
| **Ground-Truth CDR Available** | NO | **YES (Dual-expert GT)** | YES |
| **Independent Leakage Risk** | Low (Internal split) | **ZERO (Clean public dataset)** | High (Seen by FunduSegmenter) |
| **Class Balance** | 73% GON+, 27% GON- | **21% GON+, 79% GON-** | 10% GON+, 90% GON- |

### Recommendation
**PAPILA** is the **optimal primary dataset** for the initial development and validation:
1. It contains **100% ground-truth OD and OC contours from two clinical experts**. This allows computing **clean, noise-free ground-truth CDR** and perfect OD crops without depending on any external segmentation model.
2. It has verified patient IDs with explicit bilateral labels (`OD` and `OS`), enabling a leak-free patient-stratified split (70/15/15).
3. Uniform high resolution ($2576 \times 1934$).

---

## 12. Hardware & Runtime Estimates

Runtime benchmarked against actual local run on RTX 3050 Laptop GPU (DINOv2 took 8 minutes/epoch for 2,126 images):

| Stage | Operations | RTX 3050 Laptop GPU | Kaggle T4 / P100 |
|---|---|---|---|
| **Data Preparation** | Parse PAPILA contours $\to$ binary masks + patient split | ~1 minute | ~1 minute |
| **OD Crop & CDR Caching** | Crop 420 images + compute CDR | ~1 - 2 minutes | ~1 - 2 minutes |
| **Global Branch (MobileNetV3)** | 30 epochs on ~300 train images (batch 16) | ~1 - 2 minutes | ~45 seconds |
| **Local Branch (MobileNetV3)** | 30 epochs on ~300 cropped images | ~1 - 2 minutes | ~45 seconds |
| **Feature Extraction** | Extract $z_{\text{global}}$ & $z_{\text{local}}$ for all 420 images | ~20 seconds | ~10 seconds |
| **Fusion MLP Training** | 50 epochs on extracted feature vectors | ~10 seconds | ~5 seconds |
| **Evaluation & Bootstrap** | ROC-AUC, 1,000 bootstrap CI, Brier score, latency benchmark | ~30 seconds | ~15 seconds |
| **Total Pipeline Runtime** | End-to-end execution on PAPILA | **~8 to 12 minutes** | **~5 to 8 minutes** |
