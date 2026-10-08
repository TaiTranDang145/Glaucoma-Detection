## Overview

This dataset contains 747 Deep Fundus Images (DFIs) annotated for glaucomatous optic neuropathy (GON) and accompanied by quality scores.
The dataset is structured into an image folder and a labels file providing metadata and annotations.

## Dataset Structure

### 1. Images

- Located in the `Images/` folder.
- Contains 747 DFIs in JPG format with a 1:1 aspect ratio.
- Image naming convention: `x_y.jpg`
  - `x`: Patient ID (starting from 1)
  - `y`: Image number per patient (starting from 0)
  - Example: `188_1.jpg` → Second image of patient 188.

### 2. Labels File (`Labels.csv`)

A CSV file containing metadata and annotations for each image. It includes the following columns:

| Column Name   | Description                                                                |
| ------------- | -------------------------------------------------------------------------- |
| Image Name    | Filename of the DFI (e.g., `188_1.jpg`).                                   |
| Patient       | Unique patient ID.                                                         |
| Label         | Binary classification: `GON+` (glaucomatous) or `GON-` (non-glaucomatous). |
| Quality Score | Score ranging from 1 to 10.              				     |

## Dataset Statistics

- Total Images: 747
- Glaucomatous DFIs (GON+): 548
- Non-Glaucomatous DFIs (GON-): 199

## Citation

If you use this dataset in your research, please cite the following:

"Abramovich O, Pizem H, Fhima J, Berkowitz E, Gofrit B, Meisel M, et al. (2025). Hillel Yaffe Glaucoma Dataset (HYGD) (version 1.0.0). PhysioNet. https://doi.org/10.13026/*****."
"Abramovich O, Pizem H, Fhima J, Berkowitz E, Gofrit B, Meisel M, et al. GONet: A Generalizable Deep Learning Model for Glaucoma Detection [Internet]. arXiv.org. 2025."

Additionally, if you use the quality scores in your research, please cite the following:
"Abramovich O, Pizem H, Eijgen JV, Oren I, Melamed J, Stalmans I, et al. FundusQ-Net: A regression quality assessment deep learning algorithm for fundus images quality grading. Comput Methods Programs Biomed. 2023;239:Art. no. 107522."

## Contact

For any inquiries, please contact the corresponding author at orabramovich@campus.technion.ac.il.

