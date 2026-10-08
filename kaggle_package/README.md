# Public-data DINOv2 GON baseline

This package implements a **public-data method reproduction of DINOv2 single-source and multi-source domain training for GON detection**. It is not an exact numerical reproduction of GONet because KULRD is not available locally and the local public datasets do not reproduce every paper exclusion/filter.

## Upload and run on Kaggle

1. Create or use two Kaggle Datasets and upload `glaucoma_dinov2_code.zip` and `glaucoma_public_data.zip` as files. Attach both datasets to a Kaggle Notebook.
2. Select a GPU accelerator and enable Internet so `timm` can download the pretrained DINOv2 ViT-B/14 checkpoint the first time.
3. Open `glaucoma_dinov2_baseline.ipynb`. Its path-detection cell finds and extracts both archives into `/kaggle/working/glaucoma_dinov2_workspace`.
4. Run the notebook from top to bottom. The first run defaults to `FAST_DEBUG=True` (2 epochs and up to 8 images per split/domain). This checks the Kaggle path and model wiring; it is not a performance result.
5. For a full run, change only the configuration cell to `FAST_DEBUG=False` and `FULL_TRAIN=True`. This uses the configured 20-epoch maximum, batch size 16, and full selected source/target datasets. CUDA OOM fallback splits batches into smaller gradient-accumulated microbatches.

The same configuration cell controls `MODE`, `SOURCE_DOMAINS`, and `TARGET_DOMAIN`. Example SSD: `MODE="SSD"`, `SOURCE_DOMAINS=["HYRD"]`, `TARGET_DOMAIN="REFUGE"`. Example MSD: `MODE="MSD"`, `SOURCE_DOMAINS=["HYRD", "PAPILA", "DRISHTI_GS"]`, `TARGET_DOMAIN="REFUGE"`.

The notebook writes a checkpoint, `history.csv`, `predictions.csv`, `metrics.json`, and a ROC plot under `/kaggle/working/glaucoma_dinov2_workspace/outputs/`. Target rows are not used for training, validation, early stopping, or threshold selection. Target evaluation starts after the best source-validation checkpoint is loaded.

## Package contents

- `configs/dinov2_baseline.yaml`: shared SSD/MSD defaults.
- `src/`: model, dataset, training, evaluation, and data-audit code.
- `manifests/all_public.csv`: 2,557 verified labeled images and deterministic source-split candidates for each domain.
- `manifests/gonet_pilot.csv`: fixed HYRD + PAPILA + DRISHTI_GS source / REFUGE target pilot.
- `reports/`: data and model audit reports.
- `glaucoma_public_data.zip`: only the unique fundus images referenced by the all-public manifest, plus available source README/license materials. OCT volumes, masks, unrelated data, and duplicate image bytes are excluded.

The data archive is about 2.45 GB before ZIP overhead. Its image paths retain the source directory structure (the local `HYDR/` folder is called `HYRD` in the manifest domain column). GAMMA includes only 99 unique labeled training fundus photographs, not OCT or masks, and is excluded from the default source list.

## License and attribution

Dataset terms remain those of the original providers. The local GAMMA README and license identify CC BY-NC-ND; retain its attribution and non-commercial/no-derivatives conditions. The HYRD license and README and the PAPILA README are included. No separate license documents were present in the local PAPILA, DRISHTI-GS, or REFUGE folders; consult their original distribution terms before redistribution or use beyond the intended research setting. Uploading the data archive to Kaggle makes those image files available through the attached dataset, so follow each provider's terms.

See `reports/data_audit.md` for counts, label rules, duplicate handling, and excluded datasets, and `reports/dinov2_audit.md` for protocol choices and paper differences.
