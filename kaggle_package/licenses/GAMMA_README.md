---
language:
- en
license: other # Using 'other' for CC BY-NC-ND (Non-Commercial, No Derivs)
tags:
- medical
- ophthalmology
- fundus-image
- oct-volume
- multi-modal # Key feature: combining 2D fundus and 3D OCT
- image-classification
- image-segmentation
- glaucoma-grading
- optic-disc-segmentation
task_categories:
- image-classification
- image-segmentation
- object-detection # Includes Fovea Localization (coordinates)
task_ids:
- multi-class-image-classification # Glaucoma Grading (Normal/Early/Progressive)
- semantic-segmentation # OD/OC Segmentation
pretty_name: GAMMA (Glaucoma grading from Multi-Modality imAges) Challenge Dataset
size_categories:
- 100<n<1K # 300 paired samples (under 1000)
annotations_creators:
- expert-generated
source_datasets:
- original
source_data_urls:
- https://gamma.grand-challenge.org/
- https://arxiv.org/abs/2202.06511
---

# GAMMA — Glaucoma grading from Multi-Modality imAges (Challenge dataset)

<table align="center">
    <tr>
        <td width="100%" align="center">
            <img src="rm_images/Merged_Fundus_Images_with_Captions.jpg" alt="Merged Dataset Samples" style="max-width: 100%; height: auto;">
            <br>
            <p><strong>Image:</strong> Dataset Samples.</p>
        </td>
    </tr>
</table>

## Short description
GAMMA is the first **public multi-modality glaucoma grading** dataset that pairs **2D color fundus photographs** with **3D OCT volumes** for each sample. It was released as part of the GAMMA challenge (OMIA8 / MICCAI 2021) to encourage algorithms that combine fundus and OCT information for automatic glaucoma grading.

---

## What the dataset contains
- **Paired modalities:** one macula/optic-disc centered 2D color fundus image **and** one 3D OCT volume (macula-centered) per sample.  
- **Samples:** **300 paired samples** (fundus + OCT) corresponding to **276 patients**.  
- **Labeling / ground truth:** each sample has a glaucoma grade (normal / early / progressive), derived from visual field mean deviation (MD) criteria; auxiliary labels include **optic disc & cup (OD/OC) segmentation masks** and **fovea coordinates** on the fundus images.  
- **Demographics:** 276 Chinese patients, age range 19–77, mean ≈ 40.6 years; female ≈ 42%.  
- **Balanced classes:** glaucoma ~50% of samples; within glaucoma: ~52% early, ~29% intermediate, ~19% advanced (intermediate+advanced grouped as “progressive” in challenge tasks).  
- **Acquisition devices:** OCT volumes acquired using **Topcon DRI OCT Triton**; fundus images captured by **KOWA** and **Topcon TRC-NW400** cameras (macula or midpoint between disc and macula).  
- **OCT spec:** 3×3 mm en-face FOV; each volume contains 256 B-scans (cross-sectional frames).  
- **Image quality:** manually checked; dataset split into three challenge sets (training, preliminary, final) with ~100 pairs per set.  
- **License / access:** publicly available via the GAMMA grand-challenge page; dataset distributed under **CC BY-NC-ND** (Attribution-NonCommercial-NoDerivs).  
- **Official dataset page / access:** https://gamma.grand-challenge.org/

---

## Intended tasks
Primary:
- **Glaucoma grading** from paired fundus + OCT (predict: normal / early-glaucoma / progressive-glaucoma).

Auxiliary:
- **OD/OC segmentation** (optic disc and optic cup masks on fundus images).  
- **Fovea localization** (x,y coordinates).  

Researchers may optionally use the auxiliary tasks to boost the main grading performance.

---

## Dataset structure (typical)

```text
GAMMA/
├── images/
│ ├── fundus/ # fundus images (JPEG/PNG)
│ │ ├── sample_0001_fundus.jpg
│ │ └── ...
│ └── oct/ # OCT volumes (folder or volume files per sample)
│ ├── sample_0001_oct/ # 256 B-scans or a volume file (format described in README_original)
│ └── ...
├── labels/
│ ├── grades.csv # sample_id, grade (normal/early/progressive), MD values, other clinical metadata
│ ├── fovea_coords.csv # sample_id, x, y
│ └── od_oc_masks/ # per-sample masks (optional; may be in separate archive)
│ ├── sample_0001_od.png
│ └── ...
└── README_original.txt
```

---

## How samples were graded
Glaucoma grading ground truth was determined using **visual field mean deviation (MD)** thresholds from visual field tests performed the same day as OCT:
- **Early:** MD > −6 dB  
- **Intermediate:** −12 dB < MD ≤ −6 dB  
- **Advanced:** MD ≤ −12 dB  
For the main challenge, intermediate + advanced were grouped as **progressive-glaucoma**.

---

## Size & splits
- **Total paired samples:** **300** (fundus + OCT)  
- **Patients:** 276 (some bilateral samples)  
- **Class distribution:** ~50% glaucoma / 50% non-glaucoma; within glaucoma: early ≈ 52%, intermediate ≈ 28.7%, advanced ≈ 19.3%  
- **Challenge splits:** approximately **100 pairs** for training, 100 for preliminary, 100 for final test (samples from each category distributed across splits).

---

## Recommended uses & notes
- Use paired modalities (fundus + OCT) for multimodal fusion models — combining morphological cues (fundus OD/OC, vCDR) and structural OCT features (RNFL thickness) improves grading.  
- Auxiliary tasks (OD/OC masks, fovea) are provided to support explainability and localized feature extraction.  
- Respect the **CC BY-NC-ND** license for redistribution and commercial restrictions.

---

## Citation / sources
Please cite the GAMMA challenge paper and dataset when using the data:

- Wu J., Fang H., Li F., Fu H., Lin F., et al., **“GAMMA challenge: Glaucoma grAding from Multi-Modality imAges.”** (paper / challenge summary). arXiv:2202.06511; journal: *Medical Image Analysis* (2023). DOI: 10.1016/j.media.2023.102938.  
- Official dataset page (host & download): **https://gamma.grand-challenge.org/**

Primary references used to prepare this README:  
- arXiv / GAMMA challenge paper: https://arxiv.org/abs/2202.06511  
- Final journal version / PubMed entry: https://pubmed.ncbi.nlm.nih.gov/37806020/  
- GAMMA challenge (Grand Challenge) dataset page: https://gamma.grand-challenge.org/

---