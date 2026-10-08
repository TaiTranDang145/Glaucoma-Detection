# Model & Checkpoint Provenance Audit

## 1. Overview of Evaluated Models and Weights

This audit reviews all pretrained checkpoints and model weights referenced or present in the repository, assessing their training origin, license, and risk of data contamination against our evaluation benchmarks.

| Model / Checkpoint | Source | Pretraining Dataset | Fine-Tuning Dataset | Datasets Seen During Training | License / URL | Overlap with Evaluation Data | Risk Level |
|---|---|---|---|---|---|---|---|
| **FunduSegmenter** (`FunduSegmenter_OriginalImage.pth`) | External GitHub (`JusticeZzy/FunduSegmenter`) | VisionTransformer ImageNet-22k (assumed) | Drishti-GS, RIM-ONE-r3, REFUGE Train, REFUGE Validation | **Drishti-GS, REFUGE, RIM-ONE-r3** | CC BY-NC 4.0 | **DIRECT OVERLAP** with Drishti-GS & REFUGE | **HIGH / UNKNOWN TRAINING PROVENANCE** |
| **DINOv2 ViT-B/14** (`vit_base_patch14_dinov2.lvd142m`) | `timm` / Meta AI | LVD-142M (142M uncurated natural images) | None (self-supervised SSL) | None (general web crawl) | Apache 2.0 | Negligible clinical overlap | **LOW** |
| **GONet Pilot Checkpoint** (`outputs/gonet_msd_.../best_model.pth`) | Locally trained on RTX 3050 (Oct 7, 2026) | DINOv2 ViT-B/14 on LVD-142M | HYRD, PAPILA, REFUGE (MSD pilot) | HYRD, PAPILA, REFUGE | Internal repo artifact | DRISHTI-GS held out (target); other 3 seen | **LOW** for DRISHTI-GS; **HIGH** for HYRD/PAPILA/REFUGE |
| **EfficientNet-B3** (`efficientnet_b3`) | `timm` / Google AutoML | ImageNet-1k | None (in codebase) / HYGD (in notebook) | ImageNet-1k | Apache 2.0 | Negligible | **LOW** |
| **MobileNetV3-Small** (Planned target) | `timm` / PyTorch Hub | ImageNet-1k | Planned: PAPILA Train (or HYRD Train) | ImageNet-1k | Apache 2.0 | Zero clinical overlap | **LOW (Clean)** |

---

## 2. In-Depth Provenance Profiles

### 1. FunduSegmenter (`FunduSegmenter_OriginalImage.pth`)
- **Architecture:** VisionTransformer ViT-Large/16 (24 transformer layers, embedding dimension 1,024, MLP dimension 4,096, 16 attention heads) + SegmenterDecoder (2 layers, 16 heads, 1,024 dim) + Pre/Post convolutional adapters.
- **Estimated Parameter Count:** $\sim 300\text{M}$ parameters.
- **Physical Checkpoint File:** **NOT stored in repository.** Referenced in `configs/disc_baselines.yaml` and loaded via `src/models/fundu_segmenter_adapter.py`. Requires manual download from author's Google Drive or Kaggle dataset mount.
- **Training Origin & Datasets:**
  According to the official repository documentation (Section 6, Evaluation Only):
  > *"The checkpoint FunduSegmenter_OriginalImage.pth is trained on Drishti-GS, RIM-ONE-r3, REFUGE train, and REFUGE validation."*
- **Provenance Verdict:**
  **UNKNOWN / CONTAMINATED TRAINING PROVENANCE for Benchmark Evaluation.**
  Because the author pooled multiple public fundus benchmarks to train this model, using it to evaluate optic disc and cup segmentation on either REFUGE or Drishti-GS violates independent evaluation integrity. The model is evaluating on images from its own training/validation distribution.
- **Clinical Usability:**
  Can only be used for qualitative exploratory sanity checks or out-of-domain exploratory inference on datasets never seen by the author (e.g., PAPILA), but must never be claimed as a clean scientific baseline.

---

### 2. DINOv2 ViT-B/14 (`vit_base_patch14_dinov2.lvd142m`)
- **Architecture:** Vision Transformer Base with patch size 14 ($392 \times 392$ native input resolution $\to 28 \times 28$ patch grid).
- **Parameters:** $86,131,201$ parameters (backbone $86.1\text{M} + 768$ head parameters).
- **Source:** Loaded dynamically via `timm.create_model('vit_base_patch14_dinov2.lvd142m')`.
- **Pretraining Dataset:** LVD-142M dataset, a curated web dataset of 142 million natural images filtered for visual diversity. Pretraining was entirely self-supervised (DINOv2 loss, iBOT masked image modeling loss) without task-specific labels.
- **Overlap Risk:**
  **LOW.** LVD-142M is a general-domain natural image dataset. While web crawls can theoretically contain public web pages, DINOv2 was not trained with glaucoma labels or fundus segmentation objectives.
- **Role in Research:**
  Serves as the **strong heavy reference baseline (M7)** representing large-scale foundation model fine-tuning.

---

### 3. Locally Trained GONet Pilot (`outputs/gonet_msd_rtx3050_pilot/DRISHTI_GS/best_model.pth`)
- **Architecture:** DINOv2 ViT-B/14 with single linear classification head.
- **File Details:** Size: $329\text{MB}$, created Oct 7, 2026.
- **Training Configuration:**
  - Source training: `HYRD`, `PAPILA`, `REFUGE` (2,126 train images, 241 validation images).
  - Target domain: `DRISHTI_GS` (101 images held out completely).
  - Outcome: Best validation AUC 0.8549 (Epoch 1); OOD DRISHTI_GS AUC was 0.2839 (95% CI 0.1707–0.4092), reflecting severe cross-domain feature inversion under zero-shot target evaluation.
- **Provenance Verdict:**
  Cleanly isolated for evaluating DRISHTI_GS. However, because it was trained on HYRD, PAPILA, and REFUGE, it **cannot** be used to evaluate those three source domains.

---

### 4. ImageNet-Pretrained Lightweight Backbones (MobileNetV3 / EfficientNet-B0)
- **Source:** Standard PyTorch / `timm` official distributions.
- **Pretraining:** ImageNet-1k (1,000 object categories, 1.28 million natural photos).
- **Risk Level:** **LOW / ZERO CLINICAL OVERLAP.**
- **Provenance Verdict:**
  Standard scientific starting point for transfer learning in medical imaging. Clean, reproducible, and universally accepted.
