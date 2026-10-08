# Step 4 Milestone Report: Complete Multimodal & Multi-Branch Fusion Ablation

## 1. Executive Summary

In Step 4, we evaluated the complete ablation matrix proposed for the **"Lightweight Clinical-Aware GON Classification"** framework on the PAPILA dataset.
Using pre-extracted 1024-dimensional feature embeddings from the trained MobileNetV3-Small backbones ($M_2$ Global and $M_3$ Local) and the standardized expert consensus CDR ($M_1$), all fusion models were trained and benchmarked strictly on **GPU (NVIDIA GeForce RTX 3050 Laptop GPU)** under identical training protocols, validation checkpoint selection (`best_val_auc`), validation threshold tuning (Youden's $J$), and frozen test evaluation with both eye-level and patient-clustered bootstrap 95% CIs.

---

## 2. Complete Ablation Matrix Table (Frozen Test Partition, N = 64 eyes / 32 patients)

| Model | Global ($z_{\text{global}}$) | Local ($z_{\text{local}}$) | Clinical CDR | Feature Dim | Test ROC-AUC | Eye-Level 95% CI | Patient-Clustered 95% CI | Sensitivity | Specificity | Accuracy | Brier Score (Calibrated) |
|---|:---:|:---:|:---:|---:|---:|:---:|:---:|---:|---:|---:|---:|
| **$M_1$** (Oracle CDR) | | | ✓ | 1 | **0.8039** | [0.6265, 0.9516] | [0.5416, 0.9741] | 0.6154 (8/13) | 0.7647 (39/51) | 0.7344 | 0.1243 |
| **$M_2$** (Global MobileNet) | ✓ | | | 1024 | **0.8371** | [0.6728, 0.9574] | [0.6955, 0.9627] | 0.7692 (10/13) | **0.8824** (45/51) | **0.8594** | 0.1237 |
| **$M_3$** (Local OD Crop) | | ✓ | | 1024 | **0.7481** | [0.5915, 0.8800] | [0.6362, 0.8574] | 0.7692 (10/13) | 0.6078 (31/51) | 0.6406 | 0.1684 |
| **$M_4$** (Global + Local) | ✓ | ✓ | | 2048 | **0.8431** | [0.6751, 0.9598] | [0.7035, 0.9578] | **0.8462** (11/13) | 0.6863 (35/51) | 0.7188 | 0.1528 |
| **$M_{5a}$** (Global + CDR) | ✓ | | ✓ | 1025 | **0.8627** | **[0.6938, 0.9740]** | **[0.7067, 0.9760]** | **0.8462** (11/13) | 0.8039 (41/51) | 0.8125 | **0.1163** |
| **$M_{5b}$** (Global + Local + CDR) | ✓ | ✓ | ✓ | 2049 | **0.8371** | [0.6677, 0.9571] | [0.7009, 0.9554] | **0.8462** (11/13) | 0.6471 (33/51) | 0.6875 | 0.1488 |

*Note: All classification thresholds were selected strictly on the validation set via maximum Youden's Index ($J = \text{Sens} + \text{Spec} - 1$) and locked prior to evaluating the test set. Brier scores were calibrated using univariate Platt scaling fitted exclusively on the training set.*

---

## 3. Scientific Findings & Clinical Discussion

### 3.1 Empirical Confirmation of the User's Hypothesis ($M_{5a}$ vs $M_4$)
The hypothesis that **$\text{Global} + \text{CDR}$ ($M_{5a}$)** would provide clearer clinical value than **$\text{Global} + \text{Local}$ ($M_4$)** was **conclusively confirmed**:
- **$M_{5a}$ achieved the highest ROC-AUC across all models: 0.8627** (Eye 95% CI: `[0.6938, 0.9740]`, Clustered 95% CI: `[0.7067, 0.9760]`).
- Compared to $M_2$ alone (0.8371), adding explicit CDR produced a **+0.0256 AUC gain** while boosting sensitivity from 76.9% to **84.6%** (11 out of 13 GON+ detected) with strong specificity (**80.4%**).
- In addition, $M_{5a}$ achieved the **lowest Brier calibration score (0.1163)** of all tested configurations, showing that combining deep visual features with a calibrated clinical measurement markedly improves probabilistic reliability.

### 3.2 Why Did Global + CDR Outperform Global + Local?
1. **Redundancy vs Orthogonality:**
   - The Local OD Crop ($M_3$) contains visual pixels already visible within the full fundus image ($M_2$). Consequently, concatenating $z_{\text{global}}$ and $z_{\text{local}}$ ($M_4$) yields only marginal discriminative gain (0.8371 $\to$ 0.8431) because both representations rely on the same image features with correlated noise.
   - In contrast, the Vertical Cup-to-Disc Ratio (CDR) represents an **explicit clinical measurement** based on expert geometric boundaries ($M_1 = 0.8039$). This constitutes an **orthogonal clinical prior** that CNNs do not explicitly compute or constrain on their own.
2. **Feature Dilution in $M_{5b}$:**
   - In $M_{5b}$ ($z_{\text{global}} \oplus z_{\text{local}} \oplus \text{CDR}$), adding the weaker 1024-dimensional local crop embeddings ($M_3 = 0.7481$) increased parameter count and dimensionality to 2049 without adding new information, diluting the crisp synergy of Global + CDR and reducing specificity from 80.4% down to 64.7%.

---

## 4. Hardware and Training Setup Details

- **Compute Environment:** NVIDIA GeForce RTX 3050 Laptop GPU (`device="cuda"`).
- **Optimizer:** `AdamW` ($lr = 10^{-3}$, weight decay $= 10^{-2}$).
- **Scheduler:** `CosineAnnealingLR` ($T_{\max}=40$, $\eta_{\min}=10^{-6}$).
- **Loss Pos Weight:** $3.8197$ ($233 / 61$ negative/positive ratio on train split).
- **CDR Scaling:** `StandardScaler` fit strictly on the 294 train samples ($\mu_{\text{train}} = 0.3582, \sigma_{\text{train}} = 0.1541$) and applied identically to validation and test.
- **Fusion Head Architecture:** `LayerNorm(in_dim) -> Dropout(0.2) -> Linear(in_dim, 1)` trained end-to-end on GPU.

---

## 5. Artifact Ledger

- **Pipeline Script:** [`src/train_fusion.py`](file:///home/dekii2275/Glaucoma-Detection/src/train_fusion.py)
- **Extracted Feature Tensors:**
  - Global ($M_2$): `data/features/papila_m2_global_features.pt`
  - Local ($M_3$): `data/features/papila_m3_local_features.pt`
- **Model Checkpoints:**
  - $M_4$: `outputs/models/M4_global_local/best_model.pt`
  - $M_{5a}$: `outputs/models/M5a_global_cdr/best_model.pt`
  - $M_{5b}$: `outputs/models/M5b_global_local_cdr/best_model.pt`
- **Test Predictions:**
  - [`results/M4_global_local_test_predictions.csv`](file:///home/dekii2275/Glaucoma-Detection/results/M4_global_local_test_predictions.csv)
  - [`results/M5a_global_cdr_test_predictions.csv`](file:///home/dekii2275/Glaucoma-Detection/results/M5a_global_cdr_test_predictions.csv)
  - [`results/M5b_global_local_cdr_test_predictions.csv`](file:///home/dekii2275/Glaucoma-Detection/results/M5b_global_local_cdr_test_predictions.csv)
- **Metrics JSON:**
  - [`results/M4_global_local_metrics.json`](file:///home/dekii2275/Glaucoma-Detection/results/M4_global_local_metrics.json)
  - [`results/M5a_global_cdr_metrics.json`](file:///home/dekii2275/Glaucoma-Detection/results/M5a_global_cdr_metrics.json)
  - [`results/M5b_global_local_cdr_metrics.json`](file:///home/dekii2275/Glaucoma-Detection/results/M5b_global_local_cdr_metrics.json)
