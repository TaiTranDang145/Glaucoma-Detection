# 🔬 Glaucoma Detection from Deep Fundus Images

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-ODC_By-green?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-HYGD-orange?style=flat-square)

**Binary classification of Glaucomatous Optic Neuropathy (GON) from Deep Fundus Images using EfficientNet-B3**

[Overview](#-overview) • [Dataset](#-dataset) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Citation](#-citation)

</div>

---

## 📋 Overview

This project implements a deep learning pipeline for automated detection of **Glaucomatous Optic Neuropathy (GON)** from Deep Fundus Images (DFIs). Using transfer learning with **EfficientNet-B3**, the model classifies each fundus image as either:

- `GON+` — Glaucomatous (positive)
- `GON-` — Non-glaucomatous (negative)

### Key Features

- ✅ Transfer learning with **EfficientNet-B3** (ImageNet pretrained)
- ✅ **Weighted loss** to handle class imbalance (540 GON+ vs 197 GON− in the prepared dataset)
- ✅ **Quality-score-aware** filtering & sample weighting
- ✅ Comprehensive evaluation: AUC-ROC, F1, Sensitivity, Specificity
- ✅ Grad-CAM visualizations for model explainability
- ✅ Full experiment logging with **TensorBoard**
- ✅ Reproducible training with fixed random seeds

---

## 📁 Dataset

**Hillel Yaffe Glaucoma Dataset (HYGD)** — the prepared working dataset contains 737 images for 286 patients. The images and matching labels are already cleaned and are loaded directly from `data/HYDR/`.

```
data/
└── HYDR/
    ├── Images/             ← 737 prepared DFIs
    └── Labels.csv          ← Labels for the prepared images
```

| Column | Description |
|--------|-------------|
| `Image Name` | Filename (e.g. `188_1.jpg`) |
| `Patient` | Unique patient ID |
| `Label` | `GON+` or `GON-` |
| `Quality Score` | Image quality score (observed range: 2.04–7.69) |

| Split | GON+ | GON- | Total |
|-------|------|------|-------|
| Train | 376 | 136 | 512 |
| Val | 69 | 34 | 103 |
| Test | 89 | 27 | 116 |

Counts above use the prepared, already-deduplicated dataset after the configured quality filter (`Quality Score >= 3`) and patient-level split (seed 42). Training reads that dataset directly. `GroupShuffleSplit` keeps patients disjoint but does not stratify by label.

### Download Dataset

1. Obtain the prepared HYGD image set and its matching `Labels.csv` (the upstream [HYGD dataset](https://www.kaggle.com/datasets/augieaditama/hillel-yaffe-glaucoma-dataset-hygd) is the source dataset).
2. Put the already-cleaned `Images/` folder and matching `Labels.csv` directly in `data/HYDR/`. The training config reads them there; no deduplication step is needed.

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

### 2. Train the Model

```bash
# Using default config
python src/train.py --config configs/efficientnet_b3.yaml

# Override specific parameters
python src/train.py --config configs/efficientnet_b3.yaml \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4
```

### 3. Evaluate

```bash
python src/evaluate.py \
    --config configs/efficientnet_b3.yaml \
    --checkpoint outputs/checkpoints/best_model.pth
```

### 4. Grad-CAM Visualization

```bash
python src/gradcam.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --image_path data/HYDR/Images/188_1.jpg
```

### 5. Monitor Training

```bash
tensorboard --logdir outputs/logs/
```

---

## 📊 Results

> Results will be updated after training on full dataset.

| Metric | Value |
|--------|-------|
| AUC-ROC | — |
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
│   ├── dataset.py               ← PyTorch Dataset & DataLoader
│   ├── model.py                 ← EfficientNet-B3 model definition
│   ├── train.py                 ← Training loop
│   ├── evaluate.py              ← Evaluation & metrics
│   ├── gradcam.py               ← Grad-CAM visualization
│   └── utils.py                 ← Helper functions
│
├── configs/
│   └── efficientnet_b3.yaml     ← Hyperparameters & settings
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
