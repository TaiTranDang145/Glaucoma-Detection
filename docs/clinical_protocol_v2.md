# Clinical OOF protocol v2

This is the corrected training and inference path after the [15 October audit](../reports/research_claims_audit_2026-10-15.md). The historical CSV/JSON results are preserved. They are not performance measurements for v2.

## Patient roles

The original PAPILA train/validation/test split stays fixed. Before any fitting, one fold of a five-fold `StratifiedGroupKFold` over the source training patients is reserved for calibration (split seed 42). All eyes of a patient have one role.

| Role | Images | Patients | Positive eyes | Allowed use |
|---|---:|---:|---:|---|
| Training | 236 | 118 | 49 | Fit global backbone, segmenters and fusion head |
| Calibration | 58 | 29 | 12 | Fit Platt mapping after all predictor selection ends |
| Source validation | 62 | 31 | 13 | Choose final checkpoints and raw-probability decision threshold |
| Source test | 64 | 32 | 13 | Evaluate the frozen predictor |

Calibration patients are excluded from **every** model-training and checkpoint-selection stage. Reusing the old M2 checkpoint trained on all 294 source-training images would violate this rule. V2 trains a fresh global backbone for each requested seed. The smaller fitting cohort changes the experiment: compare all future baselines using these same roles. The small calibration set is a limitation; it does not guarantee calibrated predictions on another domain.

## Nested OOF CDR

For each of five outer folds of the 236 training images:

1. Divide the outer-training patients into inner-training and inner-validation using the first fold of a three-fold stratified group split.
2. Select the segmenter epoch using only inner-validation Dice.
3. Start from the same initialization seed and refit on all outer-training patients for that many epochs. Keep the original cosine schedule horizon so epoch selection does not change the learning-rate trajectory.
4. Save the refit checkpoint, then predict the outer-held-out images. Their masks do not enter fitting or epoch selection.

The inner split is an epoch-selection holdout, not a claim of exhaustive nested cross-validation. The final deployable segmenter fits the training role and selects its checkpoint on source validation. It predicts CDR for calibration, validation, test and future images; the fusion-training role uses only OOF CDR. The CDR scaler is fitted on training OOF CDR only.

Global embeddings come from the selected global model. A cache must record its checkpoint SHA-256, preprocessing, unique image IDs and patient/label/role metadata. Loading reorders by ID and rejects missing IDs, duplicates, altered labels/roles or another checkpoint. V2 never loads the historical unprovenanced feature cache.

## One preprocessing definition

Both training and inference decode through PIL RGB, center-pad to square with black pixels and resize once using OpenCV linear interpolation:

- Global branch: 224×224, then ImageNet normalization.
- Segmenter: 256×256, then the same normalization.
- Segmentation masks: existing square-padded 256px consensus masks; augmentation is applied jointly with the image.

The branches receive separately prepared tensors. V2 does not feed 256px inputs to the global backbone and does not resize an already normalized segmenter tensor to create the global input.

## Calibration and threshold

The head is LayerNorm → Dropout(0.2) → Linear(1025,1), combining a 1024-dimensional embedding and normalized predicted CDR. After validation-based head selection:

- Fit logistic Platt scaling on logits of the reserved calibration patients. Save the coefficient/intercept and exact fitting IDs; no sklearn pickle is needed for inference.
- Choose a finite Youden threshold using **raw sigmoid probabilities on source validation**. The bundle records that the threshold applies to `prob_raw`.
- Export raw/calibrated probabilities and the binary decision separately. Calibration-role metrics are fitted-sample diagnostics, not independent validation results.

## Commands

First validate source hashes, role separation and fold assignments without training:

```bash
NO_ALBUMENTATIONS_UPDATE=1 venv/bin/python src/run_step6_oof_pipeline.py \
  --plan-only --output-dir outputs/clinical_oof_v2_plan
```

Run training in another empty directory, initially with one seed:

```bash
NO_ALBUMENTATIONS_UPDATE=1 venv/bin/python src/run_step6_oof_pipeline.py \
  --output-dir outputs/clinical_oof_v2_run01 --seeds 42 --device cuda
```

Use `--device cpu` if CUDA is unavailable. `--no-pretrained` disables pretrained downloads for synthetic code verification; it changes initialization and must not silently replace ImageNet initialization in a research comparison. Epochs, learning rates, worker count and CPU thread count are explicit CLI settings (`--help`). An existing nonempty output directory is refused; historical models/results are never overwritten. Prepared plan directories and training directories are separate runs, with the deterministic split regenerated from the recorded configuration.

After training, external inference requires the actual bundle:

```bash
NO_ALBUMENTATIONS_UPDATE=1 venv/bin/python src/evaluate_zero_shot_refuge.py \
  --bundle outputs/clinical_oof_v2_run01/seed_42/bundle.pt \
  --output-dir results/refuge_oof_v2_run01 --device cuda
```

There is no fallback to an oracle-trained head or a recomputed target-domain scaler. The evaluator reads labels/masks for evaluation after prediction; it does not fit any component. Training400 mask lookup includes diagnosis subdirectories. CDR evaluation is explicitly against native-resolution GT. This evaluator does not invent or copy DINOv2 scores; that baseline requires its own independent, traceable run.

## Saved artifacts

Each run records `protocol.json`, `roles.csv` and `fold_plan.json`. For each seed:

```text
seed_42/
  oof/fold_0/selection.pt       # inner-validation selected checkpoint and history
  oof/fold_0/refit.pt           # outer-train fixed-epoch checkpoint
  oof/fold_0/heldout_predictions.csv
  ...                         # all five folds
  oof/fold_evidence.json
  oof/oof_predictions.csv
  global.pt
  segmenter.pt
  head.pt
  global_features.pt
  cdr_predictions.csv
  bundle.pt
  bundle.json                 # human-readable metadata, no tensor weights
  predictions.csv
  summary.json
```

`bundle.pt` includes all inference weights, normalization/postprocessing, OOF CDR scaler, calibration coefficients, threshold semantics, seed/config, roles/fold mapping, component/fold checkpoint hashes, source code/input fingerprints and library/environment information. Each image-model checkpoint records the initial-state hash, selected epoch and allowed training/selection patients. Loading uses `torch.load(..., weights_only=True)` and rejects the legacy checkpoint format and inconsistent calibration provenance.

## Verification and remaining work

```bash
NO_ALBUMENTATIONS_UPDATE=1 venv/bin/python -m unittest discover -s tests -v
```

Tests cover patient leakage, nested split/refit contracts, feature reordering/provenance, matching image preprocessing, held-out calibration and isolation from test labels. Integration tests train actual MobileNetV3/UNet/head models briefly on synthetic images, export/reload the bundle and run external inference. Temporary synthetic weights are deleted; they are not research models or new PAPILA/REFUGE metrics.

The implementation and source plan can be verified without GPU. A full PAPILA retraining and REFUGE benchmark remain separate execution steps; the audit's old AUC values must not be relabeled as corrected v2 results.
