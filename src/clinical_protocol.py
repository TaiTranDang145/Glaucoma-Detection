"""Patient-safe data contracts and self-contained inference for clinical OOF v2."""

import hashlib
import importlib.metadata
import json
import platform
from functools import lru_cache
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

from src.models.mobilenetv3 import MobileNetV3Classifier
from src.models.segmenter import MobileNetV3UNet
from src.train_segmenter import compute_predicted_cdr


PREPROCESS = {
    "geometry": "center_square_pad_then_resize", "padding_rgb": [0, 0, 0],
    "resize_interpolation": "opencv_linear", "decoder": "PIL_RGB",
    "global_size": 224, "segmenter_size": 256,
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "mask_threshold": 0.5, "cdr": "largest_component_cup_inside_disc_vertical_span",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def environment_info():
    packages = ["torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn",
                "scipy", "albumentations", "opencv-python", "Pillow"]
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return {"python": platform.python_version(), "packages": versions,
            "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}


def validate_manifest(frame):
    required = {"image_id", "patient_id", "label", "split", "path"}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError("Manifest requires non-null image_id, patient_id, label, split, path")
    if frame.empty or frame.image_id.duplicated().any() or frame.path.duplicated().any():
        raise ValueError("Manifest must contain unique image IDs and paths")
    if not set(frame.label).issubset({0, 1}) or set(frame.split) != {"train", "val", "test"}:
        raise ValueError("Expected binary labels and train/val/test partitions")
    if (frame.groupby("patient_id").split.nunique() != 1).any():
        raise ValueError("Patient overlap across source partitions")
    if "sha256" in frame and frame.sha256.duplicated().any():
        raise ValueError("Duplicate image hash in source manifest")
    for _, part in frame.groupby("split"):
        require_two_classes(part)


def require_two_classes(frame):
    if set(frame.label) != {0, 1}:
        raise ValueError("Each training/selection/calibration partition needs both classes")


def require_disjoint(*frames):
    for i, left in enumerate(frames):
        for right in frames[i + 1:]:
            if set(left.patient_id) & set(right.patient_id) or set(left.image_id) & set(right.image_id):
                raise ValueError("Patient or image overlap between protocol roles")


def validate_roles(frame):
    if "role" not in frame or set(frame.role) != {"train", "val", "calibration", "test"}:
        raise ValueError("Expected train/val/calibration/test roles")
    if frame.image_id.duplicated().any():
        raise ValueError("Duplicate image IDs in protocol roles")
    parts = [frame[frame.role == role] for role in ["train", "val", "calibration", "test"]]
    require_disjoint(*parts)
    for part in parts:
        require_two_classes(part)
    expected_split = frame.role.replace({"calibration": "train"})
    if not np.array_equal(frame.split, expected_split):
        raise ValueError("Protocol roles do not preserve the source splits")


def assign_roles(frame, seed=42, calibration_folds=5):
    """Reserve calibration patients before any model is fitted or selected."""
    validate_manifest(frame)
    result = frame.copy().reset_index(drop=True)
    result["role"] = result.split
    source_train = result[result.split == "train"]
    splitter = StratifiedGroupKFold(calibration_folds, shuffle=True, random_state=seed)
    _, held = next(splitter.split(source_train, source_train.label, source_train.patient_id))
    result.loc[source_train.iloc[held].index, "role"] = "calibration"
    parts = [result[result.role == role] for role in ["train", "val", "calibration", "test"]]
    require_disjoint(*parts)
    for part in parts:
        require_two_classes(part)
    validate_roles(result)
    return result


def nested_fold_plan(train, outer_folds=5, inner_folds=3, seed=42):
    """Outer held-out patients never appear in inner epoch selection."""
    train = train.reset_index(drop=True)
    if "role" in train and set(train.role) != {"train"}:
        raise ValueError("OOF folds may contain only model-training patients")
    plans = []
    outer = StratifiedGroupKFold(outer_folds, shuffle=True, random_state=seed)
    for fold, (fit_indices, held_indices) in enumerate(outer.split(train, train.label, train.patient_id)):
        fit = train.iloc[fit_indices]
        held = train.iloc[held_indices]
        inner = StratifiedGroupKFold(inner_folds, shuffle=True, random_state=seed + fold + 1)
        inner_train, inner_val = next(inner.split(fit, fit.label, fit.patient_id))
        selection_train, selection_val = fit.iloc[inner_train], fit.iloc[inner_val]
        require_disjoint(selection_train, selection_val, held)
        for part in [selection_train, selection_val, held]:
            require_two_classes(part)
        plans.append({"fold": fold, "selection_train_ids": selection_train.image_id.tolist(),
                      "selection_val_ids": selection_val.image_id.tolist(),
                      "refit_train_ids": fit.image_id.tolist(), "heldout_ids": held.image_id.tolist()})
    held_ids = [i for fold in plans for i in fold["heldout_ids"]]
    if len(held_ids) != len(train) or set(held_ids) != set(train.image_id):
        raise ValueError("OOF folds do not cover training images exactly once")
    return plans


def prepare_rgb(image, size):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("Expected non-empty uint8 RGB image")
    h, w = image.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    square = cv2.copyMakeBorder(image, top, side - h - top, left, side - w - left,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_LINEAR)


def make_transform(training=False, seed=42):
    transforms = []
    if training:
        transforms += [A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5),
                       A.Affine(scale=(.9, 1.1), rotate=(-15, 15), border_mode=cv2.BORDER_CONSTANT, p=.5),
                       A.RandomBrightnessContrast(brightness_limit=.15, contrast_limit=.15, p=.4)]
    transforms += [A.Normalize(mean=PREPROCESS["mean"], std=PREPROCESS["std"]), ToTensorV2()]
    return A.Compose(transforms, seed=seed)


def inference_inputs(rgb):
    transform = make_transform()
    return (transform(image=prepare_rgb(rgb, 224))["image"],
            transform(image=prepare_rgb(rgb, 256))["image"])


@lru_cache(maxsize=512)
def _prepared_image(path, size, modified_ns, file_size):
    # File metadata participates in the cache key; augmentation always gets a copy.
    with Image.open(path) as image:
        prepared = prepare_rgb(np.asarray(image.convert("RGB")), size)
    prepared.setflags(write=False)
    return prepared


class ClinicalImageDataset(Dataset):
    def __init__(self, frame, branch, training=False, mask_dir=None, seed=42):
        if branch not in {"global", "segmenter"}:
            raise ValueError("Unknown image branch")
        self.frame = frame.reset_index(drop=True)
        self.branch = branch
        self.size = PREPROCESS[f"{branch}_size"]
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.transform = make_transform(training, seed)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(row.path)
        stat = path.stat()
        rgb = _prepared_image(str(path), self.size, stat.st_mtime_ns, stat.st_size).copy()
        payload = {"image": rgb}
        if self.mask_dir is not None:
            masks = [cv2.imread(str(self.mask_dir / f"{row.image_id}_{structure}_consensus.png"), 0)
                     for structure in ["disc", "cup"]]
            if any(m is None or m.shape != (256, 256) for m in masks):
                raise ValueError(f"Missing or invalid 256px mask for {row.image_id}")
            payload["mask"] = np.stack(masks, axis=-1)
        transformed = self.transform(**payload)
        result = {"image": transformed["image"], "image_id": row.image_id,
                  "label": torch.tensor(float(row.label), dtype=torch.float32)}
        if "mask" in transformed:
            result["mask"] = transformed["mask"].permute(2, 0, 1).float() / 255
        return result


def aligned_features(cache, frame, expected_checkpoint_sha):
    ids = cache.get("image_ids", [])
    values = cache.get("features")
    if not isinstance(values, torch.Tensor) or values.ndim != 2 or len(ids) != len(values):
        raise ValueError("Invalid feature tensor or ID count")
    if len(set(ids)) != len(ids) or frame.image_id.duplicated().any() or set(ids) != set(frame.image_id):
        raise ValueError("Feature IDs duplicated, missing or unexpected")
    if cache.get("checkpoint_sha256") != expected_checkpoint_sha or cache.get("preprocessing") != PREPROCESS:
        raise ValueError("Feature provenance or preprocessing mismatch; regenerate features")
    metadata = pd.DataFrame(cache.get("row_metadata", []))
    if "image_id" not in metadata or metadata.image_id.duplicated().any() or set(metadata.image_id) != set(frame.image_id):
        raise ValueError("Missing or invalid feature row metadata")
    metadata = metadata.set_index("image_id").loc[frame.image_id]
    for key in ["patient_id", "label", "role"]:
        if key not in metadata or not np.array_equal(metadata[key].to_numpy(), frame[key].to_numpy()):
            raise ValueError(f"Feature row metadata mismatch: {key}")
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite feature values")
    order = {image_id: i for i, image_id in enumerate(ids)}
    return values[[order[i] for i in frame.image_id]]


class FusionHead(nn.Sequential):
    def __init__(self, in_features=1025, dropout=.2):
        super().__init__(nn.LayerNorm(in_features), nn.Dropout(dropout), nn.Linear(in_features, 1))


def fit_calibrator(logits, calibration_frame, fit_frame, selection_frame):
    """Fit only on patients unseen by every predictor-training/selection stage."""
    require_disjoint(calibration_frame, fit_frame, selection_frame)
    if "role" not in calibration_frame or set(calibration_frame.role) != {"calibration"}:
        raise ValueError("Calibration requires its reserved protocol role")
    require_two_classes(calibration_frame)
    values = np.asarray(logits).reshape(-1)
    if len(values) != len(calibration_frame) or not np.isfinite(values).all():
        raise ValueError("Invalid calibration logits")
    model = LogisticRegression(solver="lbfgs").fit(values[:, None], calibration_frame.label)
    return {"method": "platt_logistic_regression", "coefficient": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]), "fit_image_ids": calibration_frame.image_id.tolist(),
            "fit_patient_ids": sorted(calibration_frame.patient_id.unique().tolist())}


def calibrated_probability(logits, calibrator):
    return expit(np.asarray(logits) * calibrator["coefficient"] + calibrator["intercept"])


def select_threshold(labels, logits):
    fpr, tpr, thresholds = roc_curve(labels, expit(logits))
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_checkpoint(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": cpu_state(model), "metadata": metadata}, path)


def validate_bundle_metadata(scaler, calibrator, threshold, metadata):
    if scaler["scale"] <= 0 or not np.isfinite([scaler["mean"], scaler["scale"], threshold,
                                               calibrator["coefficient"], calibrator["intercept"]]).all():
        raise ValueError("Invalid scaler or threshold")
    if not 0 <= threshold <= 1 or not metadata.get("training_complete"):
        raise ValueError("Bundle is incomplete or has an invalid probability threshold")
    roles = metadata["roles"]
    train = set(roles["train"]["patient_ids"])
    val = set(roles["val"]["patient_ids"])
    calibration = set(roles["calibration"]["patient_ids"])
    test = set(roles["test"]["patient_ids"])
    sets = [train, val, calibration, test]
    if any(sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("Bundle records patient leakage")
    if set(calibrator["fit_patient_ids"]) != calibration:
        raise ValueError("Calibrator provenance does not match reserved calibration patients")
    if set(calibrator["fit_image_ids"]) != set(roles["calibration"]["image_ids"]):
        raise ValueError("Calibrator image IDs do not match reserved calibration images")
    if set(scaler["fit_image_ids"]) != set(roles["train"]["image_ids"]) or scaler.get("feature") != "cdr_pred_oof":
        raise ValueError("Scaler must be fitted only to training OOF CDR")


def save_bundle(path, global_model, segmenter, head, scaler, calibrator, threshold, metadata):
    validate_bundle_metadata(scaler, calibrator, threshold, metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "clinical_oof_v2", "preprocessing": PREPROCESS,
               "global_state": cpu_state(global_model), "segmenter_state": cpu_state(segmenter),
               "head_state": cpu_state(head), "scaler": scaler, "calibrator": calibrator,
               "threshold": {"value": float(threshold), "score": "prob_raw", "selection_role": "val"},
               "metadata": metadata, "environment": environment_info()}
    torch.save(payload, path)
    write_json(path.with_suffix(".json"), {key: value for key, value in payload.items() if not key.endswith("_state")})
    return payload


class ClinicalBundle:
    def __init__(self, payload, device="cpu"):
        if payload.get("format") != "clinical_oof_v2" or payload.get("preprocessing") != PREPROCESS:
            raise ValueError("Expected clinical_oof_v2 bundle with supported preprocessing")
        if payload["threshold"].get("score") != "prob_raw" or payload["threshold"].get("selection_role") != "val":
            raise ValueError("Unsupported threshold score or selection role")
        validate_bundle_metadata(payload["scaler"], payload["calibrator"], payload["threshold"]["value"], payload["metadata"])
        self.payload, self.device = payload, torch.device(device)
        self.global_model = MobileNetV3Classifier(pretrained=False).to(self.device).eval()
        self.segmenter = MobileNetV3UNet(pretrained=False).to(self.device).eval()
        self.head = FusionHead().to(self.device).eval()
        for model, key in [(self.global_model, "global_state"), (self.segmenter, "segmenter_state"), (self.head, "head_state")]:
            model.load_state_dict(payload[key])

    @classmethod
    def load(cls, path, device="cpu"):
        return cls(torch.load(path, map_location="cpu", weights_only=True), device)

    @torch.inference_mode()
    def predict_tensors(self, global_images, segmenter_images):
        if global_images.shape[1:] != (3, 224, 224) or segmenter_images.shape[1:] != (3, 256, 256):
            raise ValueError("Expected global Bx3x224x224 and segmenter Bx3x256x256")
        if len(global_images) != len(segmenter_images):
            raise ValueError("Branch batch sizes differ")
        if not torch.isfinite(global_images).all() or not torch.isfinite(segmenter_images).all():
            raise ValueError("Non-finite image tensor")
        g, s = global_images.to(self.device), segmenter_images.to(self.device)
        probs = torch.sigmoid(self.segmenter(s)).cpu().numpy()
        cdr = np.asarray([compute_predicted_cdr(p[0] >= .5, p[1] >= .5)[0] for p in probs])
        scaler = self.payload["scaler"]
        normalized = torch.tensor((cdr - scaler["mean"]) / scaler["scale"], dtype=torch.float32, device=self.device)[:, None]
        logits = self.head(torch.cat([self.global_model.extract_features(g), normalized], dim=1)).flatten().cpu().numpy()
        return {"logit": logits, "prob_raw": expit(logits), "cdr_pred": cdr,
                "prob_calibrated": calibrated_probability(logits, self.payload["calibrator"]),
                "prediction_binary": (expit(logits) >= self.payload["threshold"]["value"]).astype(int),
                "m2_prob": torch.sigmoid(self.global_model(g)).flatten().cpu().numpy()}

    def predict_rgb(self, rgb):
        g, s = inference_inputs(rgb)
        return {key: value[0].item() for key, value in self.predict_tensors(g[None], s[None]).items()}
