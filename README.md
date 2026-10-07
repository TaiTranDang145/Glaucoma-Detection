# 🔬 Glaucoma Detection from Deep Fundus Images

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-ODC_By-green?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-HYGD-orange?style=flat-square)

**GONet reproduction: DINOv2 ViT-B/14 fine-tuned across multiple fundus-image domains**

[Overview](#-overview) • [Dataset](#-dataset) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Citation](#-citation)

</div>

---

## 📋 Overview

This project contains a GONet reproduction and keeps the original EfficientNet-B3 HYDR baseline. GONet uses a DINOv2 ViT-B/14 backbone for binary **Glaucomatous Optic Neuropathy (GON)** classification:

- `GON+` — Glaucomatous (positive)
- `GON-` — Non-glaucomatous (negative)

### Key Features

- ✅ DINOv2 ViT-B/14 with multi-source leave-one-domain-out evaluation
- ✅ Paper preprocessing: black square padding, 392×392 resize, ImageNet normalization
- ✅ Label-verified manifests for HYDR, DRISHTI-GS, PAPILA, and REFUGE
- ✅ Comprehensive evaluation: AUC-ROC, F1, Sensitivity, Specificity
- ✅ Grad-CAM visualizations for model explainability
- ✅ Full experiment logging with **TensorBoard**
- ✅ Reproducible training with fixed random seeds

---

## 📁 Dataset

The MSD manifest is prepared from the local folders below. GAMMA is intentionally excluded.

| Folder | Domain | Label source |
|--------|--------|--------------|
| `data/HYDR/` | HYDR | `Labels.csv` |
| `data/DRISHTI-GS/` | DRISHTI-GS | `Images/GLAUCOMA` and `Images/NORMAL` class folders |
| `data/PAPILA/` | PAPILA | OD/OS clinical workbooks; suspect class 2 is excluded |
| `data/Refuge/` | REFUGE | training class folders and validation/test label workbooks |

Run the preparation command to create `data/manifests/gonet_msd.csv`. In the current local copy, REFUGE2 has 800 images that are exact REFUGE duplicates and 400 unique images without a diagnosis-label file; it is therefore excluded as a separate domain to prevent leakage. The script rechecks and prints these counts when it prepares the manifest.

The free preprint describes FundusQ-Net quality filtering and LUNet optic-disc filtering. Their model weights are not part of this repository, so this first implementation does not claim to reproduce those filters. HYDR's supplied quality score is retained out of that filter because it is not a verified FundusQ-Net score.

Patient-level source splitting uses IDs supplied by HYDR and PAPILA. DRISHTI-GS contains one image per subject. The local REFUGE annotations do not include a patient crosswalk, so its source-domain validation split is image-level and cannot guarantee separation of both eyes from the same person.

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (optional, for GPU training)

### Setup

```bash
# Clone repository
git clone https://github.com/your-username/glaucoma-detection.git
cd glaucoma-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Exploratory Data Analysis

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Prepare data and train GONet (MSD)

```bash
python src/prepare_data.py --data-root data
python src/train_msd.py --config configs/gonet_msd.yaml

# Run one leave-one-domain-out fold
python src/train_msd.py --config configs/gonet_msd.yaml --target-domain PAPILA
```

#### Train on Kaggle

The Kaggle notebook supports the three labeled inputs available for this run: HYDR, DRISHTI-GS, and PAPILA. It creates a common `DATA_ROOT` from their separate Kaggle mounts and runs three leave-one-domain-out folds. Original REFUGE is optional; REFUGE2 is not used as a replacement because it has no glaucoma labels. Run [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) after pushing the current repository code to GitHub `main`:

```bash
python src/prepare_data.py \
  --data-root "$DATA_ROOT" \
  --output /kaggle/working/gonet_msd.csv

python src/train_msd.py --config configs/gonet_msd.yaml \
  --data-root "$DATA_ROOT" \
  --manifest /kaggle/working/gonet_msd.csv \
  --output-dir /kaggle/working/gonet_msd_runs \
  --batch-size 16 --num-workers 2
```

`/kaggle/input` is read-only; checkpoints, metrics, and combined per-image held-out probabilities (`gonet_predictions.csv`) are written under `/kaggle/working/gonet_msd_runs`. Enable Kaggle Internet for package installation and the first download of pretrained DINOv2 weights. If GPU memory runs out, lower `--batch-size` (the paper setting is 16).

For a domain-balance ablation on the PAPILA held-out fold, run the same manifest and settings with `--target-domain PAPILA --domain-balanced-sampling` and a separate output directory, for example `/kaggle/working/gonet_msd_domain_balanced`. This samples source images with replacement so each source domain has equal total sampling weight; it changes training only, leaving validation, target evaluation, and the default baseline unchanged. On Kaggle, set `DOMAIN_BALANCED = True` in the notebook training cell to enable this run.

GONet fully fine-tunes the pretrained DINOv2 backbone and binary classification head. Its training uses batch size 16, Adam, learning rate `1e-6`, weight decay `0.04`, a 10× learning-rate reduction after four validation-loss epochs without improvement, and early stopping/checkpoint selection by validation loss. These optimizer settings are adapted from [DRStageNet, arXiv:2312.14891v1](https://arxiv.org/abs/2312.14891), which addresses diabetic-retinopathy grade regression; this GONet implementation retains binary BCE loss and a one-layer classification head rather than DRStageNet's MSE loss and regression head. The cited paper does not specify an early-stopping patience, so this implementation uses 10 epochs.

The first pretrained run downloads DINOv2 weights through `timm`. The GONet preprint specifies the DINOv2 ViT-B backbone and MSD protocol but does not list the exact binary fine-tuning hyperparameters or classifier head. KULRD was a source domain in the paper but is private/unavailable here; this implementation trains on the available labeled domains. The reproduction is based on the [free arXiv preprint](https://arxiv.org/abs/2502.19514).

The run writes per-domain best checkpoints and `outputs/gonet_msd/ood_metrics.csv` with AUC, a bootstrap 95% AUC interval, and Brier score. It also writes `gonet_predictions.csv` with `image_path`, `domain`, and `gonet_prob`, ready for comparison against the CDR baseline.

### 3. Train the EfficientNet-B3 baseline

```bash
# Using default config
python src/train.py --config configs/efficientnet_b3.yaml

# Override specific parameters
python src/train.py --config configs/efficientnet_b3.yaml \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4
```

### 4. Evaluate the EfficientNet-B3 baseline

```bash
python src/evaluate.py \
    --config configs/efficientnet_b3.yaml \
    --checkpoint outputs/checkpoints/best_model.pth
```

### 5. Grad-CAM Visualization (EfficientNet-B3)

```bash
python src/gradcam.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --image_path data/HYDR/Images/188_1.jpg
```

### 6. Monitor Training

```bash
tensorboard --logdir outputs/logs/
```

---

## 📊 Results

> Results will be updated after training on full dataset.

| Metric | Value |
|--------|-------|
| AUC-ROC | — |
| AUC 95% CI | — |
| Brier score | — |
| Accuracy | — |
| Sensitivity (Recall) | — |
| Specificity | — |
| F1-Score | — |
| Precision | — |

### Training Curves

*(Generated after training — see `outputs/figures/`)*

---

## 🗂️ Project Structure

```
glaucoma-detection/
├── data/
│   └── HYDR/
│       ├── Images/              ← Prepared fundus images
│       ├── Labels.csv           ← Matching annotations
│
├── src/
│   ├── dataset.py               ← HYDR and manifest-backed datasets
│   ├── prepare_data.py          ← Dataset adapters and common manifest
│   ├── models/
│   │   ├── __init__.py          ← Architecture factory
│   │   ├── efficientnet_b3.py   ← EfficientNet-B3 baseline
│   │   └── gonet.py             ← DINOv2 GONet
│   ├── losses.py                ← Shared binary losses
│   ├── train.py                 ← EfficientNet training loop
│   ├── train_msd.py             ← GONet leave-one-domain-out loop
│   ├── evaluate.py              ← EfficientNet evaluation
│   ├── gradcam.py               ← EfficientNet Grad-CAM
│   └── utils.py                 ← Metrics and helpers
│
├── configs/
│   ├── efficientnet_b3.yaml     ← HYDR baseline settings
│   └── gonet_msd.yaml           ← GONet MSD settings
│
├── notebooks/
│   ├── 01_eda.ipynb             ← Exploratory Data Analysis
│   └── 02_inference.ipynb       ← Inference & visualization
│
├── outputs/
│   ├── checkpoints/             ← Saved model weights
│   ├── logs/                    ← TensorBoard logs
│   └── figures/                 ← Plots & Grad-CAM outputs
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📖 Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{hygd2025,
  author    = {Abramovich, O. and Pizem, H. and Fhima, J. and Berkowitz, E. and Gofrit, B. and Meisel, M. and others},
  title     = {Hillel Yaffe Glaucoma Dataset (HYGD) (version 1.0.0)},
  year      = {2025},
  publisher = {PhysioNet},
  doi       = {10.13026/xxxxx}
}

@article{gonet2025,
  author  = {Abramovich, O. and Pizem, H. and Fhima, J. and Berkowitz, E. and Gofrit, B. and Meisel, M. and others},
  title   = {GONet: A Generalizable Deep Learning Model for Glaucoma Detection},
  journal = {arXiv},
  year    = {2025}
}
```

If using quality scores, also cite:

```bibtex
@article{fundusqnet2023,
  author  = {Abramovich, O. and Pizem, H. and Van Eijgen, J. and Oren, I. and Melamed, J. and Stalmans, I. and others},
  title   = {FundusQ-Net: A regression quality assessment deep learning algorithm for fundus images quality grading},
  journal = {Computer Methods and Programs in Biomedicine},
  volume  = {239},
  pages   = {107522},
  year    = {2023}
}
```

---

## 📄 License

This project is licensed under the [ODC Attribution License (ODC-By)](https://opendatacommons.org/licenses/by/1-0/).

---

## 📬 Contact

For dataset inquiries: [orabramovich@campus.technion.ac.il](mailto:orabramovich@campus.technion.ac.il)

---

<div align="center">
Made with ❤️ for medical AI research
</div>
