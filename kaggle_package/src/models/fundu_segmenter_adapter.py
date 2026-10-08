"""Adapter for the public FunduSegmenter OD/OC checkpoint."""

import importlib
import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.nn import functional as F


IMAGE_SIZE = 256
IMAGE_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def decode_class_map(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map FunduSegmenter's labels (background, rim, cup) to OD/OC masks."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("FunduSegmenter class map must be 2D.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("FunduSegmenter class map must contain integer class IDs.")
    if not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("FunduSegmenter class IDs must be in {0, 1, 2}.")

    disc_mask = (labels > 0).astype(np.uint8)
    cup_mask = (labels == 2).astype(np.uint8)
    return disc_mask, cup_mask


def preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
    """Apply FunduSegmenter's 256px validation resize and ImageNet normalization."""
    image_rgb = np.asarray(image_rgb)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")
    if image_rgb.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image.")

    resized = cv2.resize(
        image_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    image = resized.astype(np.float32) / 255.0
    image = (image - IMAGE_MEAN) / IMAGE_STD
    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


class FunduSegmenterAdapter:
    """Expose FunduSegmenter predictions through the disc-baseline contract."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def predict_masks(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return binary OD/OC masks, resized to the input image dimensions."""
        image_rgb = np.asarray(image_rgb)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape (height, width, 3).")
        original_size = image_rgb.shape[:2]
        image = preprocess_image(image_rgb).to(self.device)

        with torch.inference_mode():
            logits = self.model(image)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
                raise ValueError("FunduSegmenter must return logits with shape (B, classes, H, W).")
            if logits.shape[0] != 1 or logits.shape[1] != 3:
                raise ValueError("FunduSegmenter must return one 3-class prediction per image.")
            logits = F.interpolate(logits, size=original_size, mode="bicubic")
            labels = torch.softmax(logits, dim=1).argmax(dim=1)[0].cpu().numpy()

        return decode_class_map(labels)


def make_segmenter(checkpoint: str, device: str) -> FunduSegmenterAdapter:
    """Build the official FunduSegmenter architecture and load its checkpoint.

    Set ``FUNDUS_SEGMENTER_REPO`` to a checkout of the authors' repository.
    The official model implementation is loaded lazily so CDR evaluation and
    mask utilities do not require its optional architecture dependencies.
    """
    repo_path = _fundu_segmenter_repo()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    try:
        FunduSegmenter = importlib.import_module("model.fundusegmenter").FunduSegmenter
        VisionTransformer = importlib.import_module("model.modules.vit").VisionTransformer
        SegmenterDecoder = importlib.import_module("model.modules.segmenter_decoder").SegmenterDecoder
        adapters = importlib.import_module("model.modules.adapters")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "Could not import FunduSegmenter architecture from {}. Install its "
            "runtime dependencies and check that this is the authors' repository.".format(repo_path)
        ) from exc

    pre_adapter = adapters.PreAdapter(input_channel=3, mid_channel=64, norm="bn")
    encoder = VisionTransformer(
        image_size=(224, 224),
        patch_size=16,
        n_layers=24,
        d_model=1024,
        d_ff=4096,
        n_heads=16,
        n_cls=1000,
        dropout=0.0,
        drop_path_rate=0.0,
        distilled=False,
        channels=3,
    )
    decoder = SegmenterDecoder(
        n_cls=3,
        patch_size=16,
        d_encoder=1024,
        n_layers=2,
        n_heads=16,
        d_model=1024,
        d_ff=4096,
        drop_path_rate=0.0,
        dropout=0.1,
    )
    post_adapter = adapters.PostAdapter(
        input_channel=3,
        mid_channel=64,
        output_channel=3,
        norm="bn",
    )
    model = FunduSegmenter(
        pre_adapter=pre_adapter,
        encoder=encoder,
        decoder=decoder,
        post_adapter=post_adapter,
    )

    checkpoint_path = Path(checkpoint)
    try:
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with PyTorch versions predating weights_only
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint_state, dict) or "model" not in checkpoint_state:
        raise ValueError("Expected the official FunduSegmenter checkpoint with a 'model' state dict.")
    model.load_state_dict(checkpoint_state["model"], strict=True)

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for FunduSegmenter, but no CUDA device is available.")
    return FunduSegmenterAdapter(model, device=device)


def _fundu_segmenter_repo() -> Path:
    configured = os.environ.get("FUNDUS_SEGMENTER_REPO")
    if configured:
        repo_path = Path(configured).expanduser().absolute()
        if (repo_path / "model" / "fundusegmenter.py").is_file():
            return repo_path
        raise FileNotFoundError(
            "FUNDUS_SEGMENTER_REPO does not contain model/fundusegmenter.py: {}".format(repo_path)
        )

    candidates = (
        Path.cwd() / "FunduSegmenter",
        Path("/kaggle/working/FunduSegmenter"),
    )
    for repo_path in candidates:
        if (repo_path / "model" / "fundusegmenter.py").is_file():
            return repo_path.absolute()
    raise FileNotFoundError(
        "FunduSegmenter source is missing. Clone https://github.com/JusticeZzy/FunduSegmenter "
        "and set FUNDUS_SEGMENTER_REPO to that checkout."
    )
