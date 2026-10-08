# Gap Analysis: Lightweight Clinical-Aware GON Classification

## 1. Prioritized Gap Matrix

| Required Component | Current Status | Can Reuse | Missing Work Needed | Priority | Estimated Effort |
|---|---|---|---|---|---|
| **Clean Single-Dataset Manifest (PAPILA)** | PARTIAL | `src/prepare_gonet_pilot.py` | Create a clean, single-dataset manifest script for PAPILA (excluding suspects, linking patient IDs, 70/15/15 split). Fix `'HYRD'` vs `'HYDR'` test bug. | **P0** | ~1 - 2 hours |
| **Patient-Level Stratified Split** | PARTIAL | `src/dataset.py`, `src/dinov2_engine.py` | Implement stratified patient grouping where both eyes (`OD` and `OS`) of each patient remain strictly within the same fold while preserving label distribution. | **P0** | ~2 hours |
| **Clean OD/OC Masks & CDR Pipeline** | PARTIAL | `src/metrics/optic_disc.py` | Implement direct contour-to-mask rasterization for PAPILA dual-expert annotations; extract gold-standard CDR; filter invalid segmentations gracefully. | **P0** | ~2 - 3 hours |
| **Global MobileNetV3-Small Classifier** | **MISSING** | `src/dinov2_engine.py` (loop) | Add `mobilenetv3_small_100` to `src/models/`, configure 224px transforms, adapt training harness with BCEWithLogitsLoss and positive weighting. | **P1** | ~2 - 3 hours |
| **Local Optic-Disc Crop Generator** | **MISSING** | None | Implement OD bounding box detection from masks, add 25% margin, center crop, resize to 224px, and cache to `crops/` directory. | **P1** | ~2 - 3 hours |
| **Local MobileNetV3-Small Classifier** | **MISSING** | Global branch code | Train lightweight CNN branch on the cropped optic-disc patches; extract $z_{\text{local}}$. | **P1** | ~1 - 2 hours |
| **Feature Extraction & Fusion Head** | **MISSING** | None | Build feature extractor for $z_{\text{global}}$ and $z_{\text{local}}$; fit training-only scaler for CDR; build 2-layer MLP fusion classifier. | **P1** | ~3 - 4 hours |
| **Threshold Tuning & Evaluation Suite** | PARTIAL | `src/utils.py`, `src/metrics/auc.py` | Implement validation-set threshold selection (optimal F1 / Youden's $J$ index) applied to test set; report ROC-AUC, 95% bootstrap CI, Brier score, sensitivity, and specificity. | **P1** | ~2 hours |
| **Efficiency & Latency Benchmark** | **MISSING** | None | Script to calculate parameter count, disk size (MB), FLOPs/MACs (via `thop` or `fvcore`), and inference latency (ms/image on CPU & GPU). | **P2** | ~2 hours |
| **Radial RDR Implementation** | **MISSING** | None | Formulate mathematically verified radial ray rim-to-disc ratio; evaluate against expert contours. | **P2** | ~3 - 4 hours |
| **Clean Lightweight Segmenter Training** | **MISSING** | None | Train a lightweight U-Net (MobileNetV3 encoder) on REFUGE or PAPILA to replace external FunduSegmenter for unannotated images. | **P2** | ~4 - 6 hours |
| **DINOv2 Strong Reference (M7)** | READY | `src/train_msd.py`, `src/dinov2_engine.py` | Run DINOv2 baseline on the exact same patient split for direct numerical comparison. | **P3** | ~1 hour |

---

## 2. Priority Level Definitions

- **P0 (Critical Foundation):** Must be completed before running any model training. Focuses on data hygiene, leakage prevention, and reliable ground truth.
- **P1 (Core Research Pipeline):** The minimal set of components required to execute the target ablation matrix ($M_1$ through $M_6$).
- **P2 (Clinical & Efficiency Enhancements):** Model efficiency profiling, mathematical validation of RDR, and training an independent in-house segmenter.
- **P3 (Benchmarking & Extension):** Running heavy foundation model baselines (DINOv2) and exploring cross-dataset domain shifts.

---

## 3. Detailed Component Breakdown

### P0.1: Clean Single-Dataset Manifest (PAPILA)
- **Problem:** The repository currently has `all_public.csv` (2,557 rows) and `gonet_pilot.csv` (2,458 rows), both designed for multi-domain leave-one-dataset-out experiments.
- **Action:** Write `scripts/prepare_papila_manifest.py` (or adapter in `prepare_data.py`) to parse PAPILA's 488 fundus images:
  - Exclude class 2 suspects (68 eyes) $\to$ retain 420 binary cases (87 GON+, 333 GON-).
  - Extract patient ID (`RETxxx`) and eye laterality (`OD` vs `OS`).
  - Output clean manifest: `data/manifests/papila_binary.csv`.

### P0.2: Stratified Patient-Level Split
- **Problem:** Standard `GroupShuffleSplit` does not guarantee balanced class representation between folds.
- **Action:** Implement a stratified group partition algorithm:
  - Partition 210 patients into 70% Train (~146 patients, ~294 images), 15% Val (~32 patients, ~63 images), 15% Test (~32 patients, ~63 images).
  - Ensure both eyes of each patient remain in the same split.
  - Optimize the split to minimize the difference in positive ratio between splits.

### P1.1: Local Optic-Disc Crop Pipeline
- **Problem:** No cropping logic exists in the repository.
- **Action:**
  1. For each image, load the ground-truth OD mask (rasterized from `RET..._disc_exp1.txt`).
  2. Compute bounding box coordinates: $(y_{\min}, x_{\min}, y_{\max}, x_{\max})$.
  3. Expand box by $25\%$ margin along both dimensions.
  4. Crop square patch, apply black padding if boundaries exceed image dimensions, and resize to $224 \times 224$.
  5. Cache crops to `data/crops/papila/`.

### P1.2: Model Factory Refactor & Lightweight Backbones
- **Problem:** `src/models/__init__.py` throws `ValueError` for any model other than `efficientnet_b3` or `gonet`.
- **Action:**
  - Support `mobilenetv3_small_100` and `efficientnet_b0` dynamically via `timm`.
  - Expose a clean feature-extraction interface returning penultimate embedding vectors ($z_{\text{global}}, z_{\text{local}}$).

### P1.3: Multimodal Feature Fusion
- **Problem:** No fusion model exists.
- **Action:**
  - Build `FusionMLP` accepting concatenated vector $[z_{\text{global}}, z_{\text{local}}, \text{CDR}_{\text{scaled}}]$.
  - Implement a training pipeline that trains the lightweight MLP head on extracted features or fine-tunes end-to-end.
