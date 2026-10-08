# DINOv2 GON baseline audit

## Decision

**READY FOR PILOT TRAINING** as a public-data method reproduction. The package is not an exact numerical reproduction of GONet: KULRD is absent locally, and the public datasets below do not reproduce the paper's complete inclusion criteria or training population.

The implementation follows the GONet preprocessing, DINOv2 ViT-B backbone, binary GON task, source-only validation, and OOD target evaluation. The selected architecture and preprocessing are described in the [GONet preprint](https://arxiv.org/abs/2502.19514). The optimizer settings align with the fine-tuning recipe reported by [DRStageNet](https://arxiv.org/abs/2312.14891), which GONet cites for its hyperparameters; some details that the papers leave unspecified are explicitly recorded below.

## Previous behavior and corrections

| Area | Previous behavior / issue | Audited implementation |
|---|---|---|
| Dataset path | The REFUGE adapter expected a `Refuge/` directory, while this checkout contains `REFUGE/`; on this case-sensitive filesystem the old manifest builder could omit REFUGE. | Resolve the REFUGE directory case-insensitively. The rebuilt manifest contains all 1,200 labeled REFUGE images. |
| Label and image audit | Label adapters did not produce one audited manifest spanning the local public datasets, and there was no complete decode/hash report. | Explicit dataset adapters, label rules, missing-file checks, full Pillow decode checks, and SHA-256 duplicate checks generate `data/manifests/all_public.csv` and `reports/data_audit.md`. |
| Duplicate/patient grouping | Patient IDs alone did not connect exact duplicate images when source records gave those images different IDs. | Remove same-domain byte duplicates and union linked patient/case IDs into `split_group`; source splits group on that field. Cross-domain exact hash collisions fail the manifest build. |
| Split selection | One non-stratified group split was sampled for combined source rows; source domain boundaries were not independently controlled. | Make a reproducible, approximately stratified 90/10 group split per source domain. Candidate `source_split` values are used only after the selected target rows have been filtered out. |
| Target isolation | The old script constructed the target loader before source training, although it did not use target results for optimization. | The shared engine does not construct the target loader or read its labels for scoring until the best source-validation checkpoint has been selected and reloaded. Source/target domain overlap is rejected. |
| Transform details | The old DINO transform relied on library defaults for interpolation and brightness/contrast, and the data-loader worker RNG was not explicitly seeded. | RGB conversion, one square black pad, one linear resize, explicit transform parameters, and seeded workers are used. Contrast change is set to zero so the configured color augmentation changes brightness only. |
| SSD/MSD implementation | There was no separate SSD entry point and training/evaluation logic was embedded in the MSD script. | `src/train_ssd.py` and `src/train_msd.py` call the same DINOv2 engine; only the source-domain list and run name differ. The former MSD script is preserved as `src/train_msd_legacy.py`. |
| Bootstrap edge cases | A bootstrap with a one-class sample was skipped, but the old result did not report the number of valid samples and could serialize undefined bounds. | AUC bootstrap skips one-class samples, reports valid/requested counts, and returns JSON-safe null bounds when none are valid. |

## Model and training protocol

| Setting | Pilot value |
|---|---|
| Backbone / pretrained weights | `timm` model `vit_base_patch14_dinov2.lvd142m`, DINOv2 ViT-B/14 pretrained on LVD-142M |
| Input resolution | 392 × 392, a 28 × 28 patch grid for ViT-B/14 |
| Classifier | Dropout 0.0 followed by one linear layer from 768 features to one logit; the paper does not specify the binary head in enough detail to reproduce it exactly |
| Fine-tuning | Entire backbone and head are trainable; the local smoke model has 86,131,201 trainable parameters |
| Loss | Unweighted `BCEWithLogitsLoss`; no class sampler or class weight |
| Optimizer | Adam, learning rate `1e-6`, weight decay `0.04` |
| Scheduler | `ReduceLROnPlateau`, validation BCE loss, patience 4, factor 0.1 |
| Epoch/batch limit | Up to 20 epochs, batch size 16 |
| Early stopping / checkpoint | Stop after 10 epochs without a lower source-validation BCE loss; retain the lowest-loss source-validation checkpoint |
| Gradient handling | Gradient norm clipped at 1.0, consistent with the existing repository training loop |
| Mixed precision | Enabled only on CUDA; full precision on CPU |
| Device selection | Config requests CUDA by default; the engine stops with a clear error if CUDA is unavailable instead of silently starting CPU fine-tuning. CPU is available only when explicitly selected for a deliberate smoke test. |
| Reproducibility | Seed 42 for Python, NumPy, PyTorch, CUDA, data-loader shuffling, and workers; software/GPU versions are written to `metrics.json` |
| Target threshold | None. No threshold is tuned on target test data, and sensitivity/specificity are not reported. |

The GONet preprint specifies DINOv2 ViT-B and says its fine-tuning hyperparameters follow Men et al.; it does not give all numeric details of the final GONet head, augmentation probabilities/ranges, or early-stopping patience. This implementation makes those choices visible in `configs/dinov2_baseline.yaml`: 20 maximum epochs, patience 10, 0.5 augmentation probabilities, zoom 0.9–1.1, rotation ±15°, and brightness limit 0.2. These are documented implementation settings, not claims that every value is quoted from GONet. The DRStageNet recipe reports batch 16, Adam, learning rate `1e-6`, weight decay `0.04`, a tenfold scheduler drop after four stalled validation epochs, early stopping, and best-validation-loss checkpointing.

## Preprocessing and augmentation

1. Open with Pillow and convert to RGB (no BGR path).
2. Add black padding to the longer side to make a square; this does not crop the optic disc.
3. Resize once to 392 × 392 using linear interpolation.
4. During training only: horizontal flip (0.5), vertical flip (0.5), affine zoom (0.9–1.1) and rotation (−15° to +15°) with linear interpolation and constant-black borders, and brightness change (limit 0.2, probability 0.5). The GONet paper lists brightness, zoom, rotation, and both flips; it does not specify these numeric ranges/probabilities.
5. Normalize once using ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`, then convert to a PyTorch tensor.

Validation and target transforms perform only the same RGB conversion, square padding, one resize, one normalization, and tensor conversion. They are deterministic. No crop, blur, dropout, contrast, saturation, or hue augmentation is used in this DINOv2 path. The older 300-pixel HYGD pipeline remains in `get_transforms`; the new DINOv2 engine calls only `get_msd_transforms`.

## SSD and MSD split protocol

- **SSD:** one explicit source domain, e.g. `HYRD`; one explicit unseen target, e.g. `REFUGE`.
- **MSD:** an explicit list of source domains, e.g. `HYRD`, `PAPILA`, and `DRISHTI_GS`; one explicit unseen target. Both modes use the same model, preprocessing, augmentation, optimizer, loss, scheduler, seed, checkpoint selection, and target metrics.
- Each domain has a deterministic candidate 90/10 split. For a configured source, validation rows are selected by patient/case `split_group`; if unavailable, use a supplied patient ID, then an image/case group. PAPILA and HYRD use patient groups. DRISHTI-GS has no patient table and is split by individual image/case. Candidate target splits in the all-domain manifest are ignored for the selected target.
- The default multi-source list intentionally excludes GAMMA because it contributes only the 99 local training cases and has a distinct non-commercial/no-derivatives license. It remains available in the manifest for an explicit experiment.
- The engine fits, schedules, stops, and selects weights using source rows only. Target predictions and labels are read after checkpoint selection. Metrics are ROC-AUC, Brier score, and a 95% percentile interval from 1,000 bootstrap resamples of 95% of target images sampled with replacement. One-class resamples are skipped and their valid count is reported. REFUGE patient identifiers are unavailable, so target bootstrap is image-level.

This is a single configured SSD/MSD experiment per invocation, rather than the complete seven-domain leave-one-domain-out experiment in GONet. Run separate invocations to evaluate additional targets. The notebook defaults to a short debug run and uses the same functions for SSD and MSD.

## Smoke test

The authorized synthetic smoke test ran one CPU epoch with one synthetic source and one synthetic unseen target. It confirmed finite loss, nonzero gradients, a saved and reloaded checkpoint, inference probabilities in `[0, 1]`, AUC calculation, Brier score calculation, and prediction CSV fields. The test used random initialization and tiny synthetic images; it verifies plumbing only and is not model performance. Local CUDA is unavailable, so CUDA mixed precision and OOM fallback still need confirmation on Kaggle GPU.

The completed smoke predates the later prepared-`source_split` selection and the notebook's separate checkpoint-load/evaluation interface. A follow-up CPU run reached the prepared source split but was interrupted before an epoch completed. Static compilation, source-split audit, notebook parsing, package imports, and ZIP CRC checks pass; the exact final training/evaluation flow still needs its short debug run on Kaggle GPU before a full run.

## Known differences and limits

- KULRD is unavailable locally, and the local HYRD data contains 747 labeled rows (737 unique images after exact duplicate removal), while the paper's selected HYRD set is 647 after quality/optic-disc exclusions. This package does not recreate those exclusions.
- The paper's main analysis excludes GON suspects, images with FundusQ-Net quality below 5, and images missing a complete optic disc. PAPILA suspects are explicitly excluded here, but the package does not apply the paper's FundusQ-Net or optic-disc filters to any domain.
- Local PAPILA has 420 non-suspect images versus the paper's selected 393; local DRISHTI-GS has 101 versus 89 selected; local REFUGE has 1,200 versus 1,199 selected; local GAMMA has only 100 labeled training cases (99 unique) versus 299 selected. These differences are recorded in `reports/data_audit.md`.
- REFUGE2 has 1,200 local images: 800 are exact byte matches to REFUGE, 400 are unique, and no glaucoma label table is present. It is excluded and remains a separate dataset name.
- The dataset adapters can detect byte-identical files, not re-encoded or transformed near-duplicates. Exact duplicates within a domain are dropped; no exact cross-domain collisions were found.
- DRISHTI-GS and REFUGE do not include usable patient ID tables in the local copy. GAMMA case IDs are not independently reconciled to the 276-person table.
- The local GAMMA README states CC BY-NC-ND. Attribution and license terms are included with the data package; it is not a default source domain.
- Mixed precision on CUDA is a Kaggle resource choice and can cause small numeric differences from full precision. The model weights are downloaded by `timm` when pretrained initialization is enabled; Kaggle Internet must be enabled on first use.
- The optimizer/loss and classifier head are a reasonable binary classification implementation. Exact numeric reproduction should not be expected without the original KULRD data and every original filtering/checkpoint detail.

## Conclusion

**READY FOR PILOT TRAINING.** Use `python src/train_ssd.py --source HYRD --target REFUGE --config configs/dinov2_baseline.yaml` for the initial SSD run (`--target-domain` is also accepted), or the notebook for a Kaggle debug/full run. Results should be described as a **public-data method reproduction of DINOv2 single-source and multi-source domain training for GON detection**, not as an exact GONet reproduction.
