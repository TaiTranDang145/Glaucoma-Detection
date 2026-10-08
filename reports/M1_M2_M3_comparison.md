# Step 3 Milestone Report: M1 vs M2 vs M3 Benchmark Comparison

## 1. Executive Summary

In Step 3, we implemented, trained, and benchmarked the lightweight unimodal branches for Glaucomatous Optic Neuropathy (GON) detection on the PAPILA dataset:
1. **$M_1$ (Oracle CDR Baseline):** Consensus Vertical CDR derived directly from expert manual contours.
2. **$M_2$ (Global Fundus Branch):** Full fundus image resized with aspect-ratio preserving square padding $\to$ **MobileNetV3-Small** (1.52M parameters).
3. **$M_3$ (Oracle Local-OD Branch):** Expert-guided Optic Disc crop (union bbox + 25% margin) $\to$ **MobileNetV3-Small** (1.52M parameters).

All deep learning models ($M_2, M_3$) were trained strictly on **GPU (NVIDIA GeForce RTX 3050 Laptop GPU)** under an **identical experimental setup** (same patient-stratified split, same augmentations, same optimizer, same scheduler, and same validation selection criterion) to ensure differences reflect visual region characteristics rather than hyperparameter discrepancies.

---

## 2. Benchmark Comparison Table (Frozen Test Partition, N = 64 eyes / 32 patients)

| Model | Architecture | Visual Input Domain | Parameters | Test ROC-AUC | Eye-Level 95% CI | Patient-Clustered 95% CI | Sensitivity | Specificity | Accuracy | Brier Score (Calibrated) |
|---|---|---|---:|---:|:---:|:---:|---:|---:|---:|---:|
| **$M_1$** | Clinical Morphology Rule | Expert Consensus CDR | 0 | **0.8039** | [0.6265, 0.9516] | [0.5416, 0.9741] | 0.6154 (8/13) | 0.7647 (39/51) | 0.7344 | 0.1243 |
| **$M_2$** | MobileNetV3-Small | Global Full Fundus | 1,518,881 | **0.8371** | [0.6728, 0.9574] | [0.6955, 0.9627] | **0.7692** (10/13) | **0.8824** (45/51) | **0.8594** | **0.1237** |
| **$M_3$** | MobileNetV3-Small | Oracle Local OD Crop | 1,518,881 | **0.7481** | [0.5915, 0.8800] | [0.6362, 0.8574] | **0.7692** (10/13) | 0.6078 (31/51) | 0.6406 | 0.1684 |

*Note: All decision thresholds were determined exclusively on the validation set using Youden's Index ($J = \text{Sens} + \text{Spec} - 1$) and locked prior to evaluating the test set. Calibrated Brier scores were fitted via Platt scaling (univariate logistic regression) on training predictions.*

---

## 3. Detailed Experimental Controls & Hardware Setup

Both $M_2$ and $M_3$ shared an identical training protocol:
- **Hardware Device:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (`device="cuda"`).
- **Partition:**
  - Train: 294 images (61 GON+, 233 GON-) across 147 patients.
  - Val: 62 images (13 GON+, 49 GON-) across 31 patients.
  - Test: 64 images (13 GON+, 51 GON-) across 32 patients (permanently frozen).
- **Input Preprocessing:**
  - Aspect-ratio preserving square canvas padding (black margin).
  - Target resolution: $224 \times 224 \times 3$.
  - ImageNet normalization: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
- **Data Augmentation Policy (Train only):**
  - Horizontal Flip ($p=0.5$), Vertical Flip ($p=0.5$).
  - Affine transformation: scale $\in [0.9, 1.1]$, rotation $\in [-15^\circ, +15^\circ]$ ($p=0.5$).
  - Random Brightness ($\pm 20\%$) & Contrast ($\pm 10\%$) ($p=0.5$).
- **Loss Function:**
  - `nn.BCEWithLogitsLoss(pos_weight=3.8197)` ($N_{\text{neg}} / N_{\text{pos}} = 233 / 61$).
- **Optimization:**
  - Optimizer: `AdamW` (learning rate $= 10^{-4}$, weight decay $= 10^{-2}$).
  - Scheduler: `CosineAnnealingLR` ($T_{\max}=35$, $\eta_{\min}=10^{-6}$).
  - Batch size: 16.
- **Model Checkpoint Selection:**
  - Highest validation ROC-AUC (`best_val_auc`), with tie-breaker on lower validation BCE loss.
  - $M_2$ Best Val Epoch: Epoch 11 (`val_auc` = 0.8587, `val_loss` = 1.6224).
  - $M_3$ Best Val Epoch: Epoch 11 (`val_auc` = 0.8791, `val_loss` = 2.3158).

---

## 4. Key Clinical & Technical Insights

1. **Global vs Local Representations ($M_2$ vs $M_3$):**
   - $M_2$ (Full Fundus) achieved **0.8371 ROC-AUC**, outperforming $M_3$ (0.7481 ROC-AUC) by **+0.089 AUC**.
   - Specificity on full fundus is substantially higher (0.8824 vs 0.6078), demonstrating that peripheral retina context (such as retinal nerve fiber bundle defects, vascular arcades, and macro-architecture) prevents false-positive classifications that happen when zooming purely into the optic disc.
   - However, $M_3$ achieves the same high sensitivity (0.7692, 10/13 GON+ detected) as $M_2$, indicating that local OD morphology is rich in discriminatory cues for detecting true disease.

2. **Deep Learning vs Clinical Morphology Baseline ($M_2$ vs $M_1$):**
   - $M_2$ (0.8371) surpasses the Oracle CDR baseline $M_1$ (0.8039) on the frozen test set.
   - This indicates that a lightweight CNN (1.52M parameters) trained directly on full fundus images extracts subtle textural and structural features beyond a single vertical cup-to-disc ratio.

3. **Readiness for Feature Fusion (Step 4 Preview):**
   - Both $M_2$ and $M_3$ provide distinct, non-redundant perspectives:
     - $M_2$ provides wide-field retinal context and high specificity (0.8824).
     - $M_3$ provides magnified disc-margin details.
     - $M_1$ provides clinical CDR rule interpretability.
   - In Step 4, we will extract the 1024-dimensional pre-logits features from both frozen branches and fuse them ($M_4: \text{Global} + \text{Local}$, $M_5: \text{Global} + \text{Local} + \text{CDR}$).

---

## 5. Artifact Ledger

- **Trained Model Checkpoints:**
  - `outputs/models/M2_global/best_model.pt`
  - `outputs/models/M3_local/best_model.pt`
- **Training Histories:**
  - `outputs/models/M2_global/history.json`
  - `outputs/models/M3_local/history.json`
- **Test Predictions:**
  - `results/M2_global_test_predictions.csv`
  - `results/M3_local_test_predictions.csv`
- **Metrics JSON:**
  - `results/M2_global_metrics.json`
  - `results/M3_local_metrics.json`
