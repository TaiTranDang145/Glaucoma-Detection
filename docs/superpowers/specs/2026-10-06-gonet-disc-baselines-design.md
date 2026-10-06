# GONet CDR/RDR Baselines Design

## Goal

Add a reproducible disc-feature baseline around the existing GONet project. Given fundus images and a compatible optic-disc/optic-cup segmenter, the pipeline will save per-image CDR results, evaluate CDR as a direct GON+ score, and make review visualizations. It will accept an optional per-image GONet/DINOv2 probability CSV for an AUC comparison. The existing GONet classifier and training flow are out of scope.

## Current project constraints

- `src/prepare_data.py` creates a manifest with `image_path`, `domain`, integer `label` (`1 = GON+`, `0 = GON-`), and `patient_id`.
- `src/train_msd.py` has an AUC bootstrap helper, but its prediction path does not currently write per-image probabilities. Evaluation may accept a probability CSV when one exists; this change will not alter classifier training or prediction code to create one.
- No LUNet OD/OC model implementation or corresponding checkpoint is present in the repository. The public LUNet reference is for artery/vein segmentation, while GONet says it retrained LUNet for OD/OC. The design therefore requires a compatible OD/OC segmenter factory and checkpoint supplied by the user. It will not load artery/vein weights as an OD/OC model or substitute a different network.
- Local DRISHTI-GS, REFUGE, and REFUGE2 folders contain segmentation annotations, but those are ground truth and will not be presented as LUNet predictions.
- REFUGE records have no `patient_id` in the current manifest. Export an empty value when the source metadata does not provide an ID; do not derive one from filenames.

## Architecture and data flow

1. Add `src/metrics/optic_disc.py` for segmentation-mask validation and feature calculations. Its public CDR function accepts two same-sized 2D binary masks and reports both a numeric value and a status.
2. Add `src/infer_disc_baselines.py`. It reads the existing manifest, resolves images under `data_root`, invokes a user-supplied segmenter factory with its checkpoint/device configuration, calculates features, saves binary OD/OC masks for later review, and writes one CSV row per manifest image.
3. Add `src/evaluate_disc_baselines.py`. It filters invalid/non-finite scores, computes per-domain ROC/AUROC for CDR, and optionally joins an external `gonet_prob` CSV to produce rows in a `model, domain, auc` comparison table. Reuse the existing bootstrap AUC helper for confidence intervals where appropriate.
4. Add `src/visualize_disc_baselines.py`. It reads the result CSV and saved masks, then writes a small set of fundus overlays with OD/OC contours and the computed values/status.
5. Add `tests/test_optic_disc_metrics.py` using the standard-library `unittest` runner so the mask metric tests do not add a test dependency.
6. Add `configs/disc_baselines.yaml` for manifest/data/output paths, segmenter factory/checkpoint, device, optional probability CSV, and visualization sample count, following the existing OmegaConf configuration pattern.

The segmenter integration will be an explicit importable factory contract rather than an assumed LUNet implementation: the factory receives checkpoint/device settings and returns an object whose prediction method accepts a fundus image and returns OD and OC binary masks in the original image coordinate system. The CLI will fail with a clear message if the factory or checkpoint is absent. Preprocessing and output-channel mapping remain the supplied segmenter's responsibility because neither is available in this repository. The assumption that each anatomical mask is represented by its largest connected component will be documented and tested; isolated one-pixel components will be ignored.

## CDR behavior

For valid masks, use:

`vertical_CDR = (max_y(cup foreground) - min_y(cup foreground) + 1) / (max_y(disc foreground) - min_y(disc foreground) + 1)`.

The `+1` counts both boundary rows in a discrete mask. Reject mismatched shapes, non-2D inputs, empty masks, a zero-height disc, or any cup foreground outside the disc. Invalid input produces `NaN` and a machine-readable status; it is not silently clipped. The implementation will use the largest connected component for each mask so isolated one-pixel speckles do not change the anatomical extent. An all-noise/single-pixel-only mask remains invalid.

## RDR behavior

Do not calculate RDR yet. The GONet paper refers to the narrowest neuroretinal rim width and the cited RDR paper explains its circle-center geometry, but the repository has no pixel-mask implementation to reproduce. Choosing a center, ray, boundary-distance rule, or denominator for arbitrary discrete LUNet masks would add an unverified method detail. Export an `rdr` column as `NaN` with an `rdr_status` explaining that the mask-level computation is unverified, and include a TODO pointing to reference [10]. Do not plot an RDR ROC or report an RDR AUC until that method is confirmed. If enabled later, lower RDR will map to higher GON+ risk, so the ROC score will be `-rdr`.

## Output and evaluation

The results CSV will include the requested fields:

`image_path, patient_id, domain, ground_truth, cdr, rdr, segmentation_valid`

It will also contain machine-readable segmentation/metric status and OD/OC mask paths to support auditing and visualization. `ground_truth` will preserve the readable `GON+` / `GON-` mapping. Invalid segmentations have `segmentation_valid = false`; their CDR is `NaN` and they are excluded from ROC/AUC. Missing patient IDs remain empty.

Evaluation writes per-domain ROC plots and an AUC table with `model, domain, auc` columns. It reports CDR whenever a domain contains both labels and at least one valid finite score. It adds a GONet/DINOv2 row only when the optional probability CSV is supplied and successfully joined on `image_path` and `domain`. RDR is omitted from numerical evaluation while it remains unimplemented.

## Tests

Cover: a normal nested disc/cup mask, an empty mask, a cup not contained in the disc (including a cup larger than the disc), an isolated single-pixel component beside a valid mask, and a synthetic mask with a known vertical CDR. Also verify that invalid cases return `NaN` and a specific status.

## Non-goals and caveats

- Do not train or implement a substitute segmentation network in this change.
- Do not modify the GONet/DINOv2 classifier, its training loop, or its checkpoint format.
- Do not claim the segmentation baseline is runnable end-to-end until a compatible OD/OC segmenter factory and checkpoint are supplied.
- Do not treat local expert masks as predictions.
- RDR remains explicitly unavailable until the mask-level calculation is confirmed against reference [10] or its implementation.

## References

- GONet free preprint: https://arxiv.org/abs/2502.19514
- Reference [10], *Rim-to-Disc Ratio Outperforms Cup-to-Disc Ratio for Glaucoma Prescreening*: https://doi.org/10.1038/s41598-019-43385-2
- Reference [10] supporting information: https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-019-43385-2/MediaObjects/41598_2019_43385_MOESM1_ESM.pdf
