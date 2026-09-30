"""
gradcam.py
----------
Grad-CAM visualization for the trained Glaucoma Classifier.

Highlights the discriminative regions in a fundus image that the model
focuses on when predicting GON+ or GON-.

Usage:
    python src/gradcam.py \
        --checkpoint outputs/checkpoints/best_model.pth \
        --image_path data/HYDR/Images/188_1.jpg \
        --config configs/efficientnet_b3.yaml

    # Visualize multiple images from test set
    python src/gradcam.py \
        --checkpoint outputs/checkpoints/best_model.pth \
        --config configs/efficientnet_b3.yaml \
        --batch 8
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf

from model import build_model, GlaucomaClassifier
from utils import set_seed, load_checkpoint
from dataset import get_transforms, LABEL_MAP


# ── Grad-CAM implementation ───────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for EfficientNet-B3.

    Hooks the last convolutional block to extract feature maps and gradients.

    Args:
        model (GlaucomaClassifier): Trained model.
        target_layer (str): Name of layer to hook. Default: last conv block.
    """

    def __init__(self, model: GlaucomaClassifier, target_layer: Optional[str] = None):
        self.model  = model
        self.model.eval()

        self._features = None
        self._gradients = None

        # Hook the last convolutional block of EfficientNet-B3
        target = (
            model.backbone.blocks[-1]  # Last block of EfficientNet
            if target_layer is None
            else dict(model.named_modules())[target_layer]
        )

        self._fwd_hook = target.register_forward_hook(self._save_features)
        self._bwd_hook = target.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        self._features = output.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Computes Grad-CAM heatmap for one image.

        Args:
            image_tensor: Shape (1, 3, H, W) on the model's device.

        Returns:
            np.ndarray: Normalized heatmap in [0, 1], shape (H, W).
        """
        self.model.zero_grad()

        logit = self.model(image_tensor)          # (1, 1)
        score = logit[0, 0]
        score.backward()

        # Global average pooling of gradients → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._features).sum(dim=1).squeeze(0)      # (H', W')
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Overlay heatmap on image ──────────────────────────────────────────────────

def overlay_heatmap(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap on the original image.

    Args:
        original_img: RGB image (H, W, 3) uint8.
        heatmap: Normalized heatmap (H', W') in [0, 1].
        alpha: Blend factor for heatmap overlay.
        colormap: OpenCV colormap.

    Returns:
        Blended RGB image (H, W, 3) uint8.
    """
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_color   = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), colormap
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlaid = np.uint8(alpha * heatmap_color + (1 - alpha) * original_img)
    return overlaid


# ── Single image visualization ────────────────────────────────────────────────

def visualize_single(
    model: GlaucomaClassifier,
    image_path: str,
    image_size: int,
    device: torch.device,
    save_path: Optional[str] = None,
    true_label: Optional[str] = None,
) -> None:
    """Runs Grad-CAM on one image and shows/saves the result."""
    # Load and preprocess
    img_rgb = np.array(Image.open(image_path).convert("RGB"))
    tf      = get_transforms("test", image_size)
    tensor  = tf(image=img_rgb)["image"].unsqueeze(0).to(device)

    # Grad-CAM
    gradcam = GradCAM(model)
    heatmap = gradcam(tensor)
    gradcam.remove_hooks()

    prob = torch.sigmoid(model(tensor)).item()
    pred = "GON+" if prob >= 0.5 else "GON−"

    overlay = overlay_heatmap(img_rgb, heatmap)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(img_rgb);        axes[0].set_title("Original Image",  fontsize=13)
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM Heatmap", fontsize=13)
    axes[2].imshow(overlay);        axes[2].set_title("Overlay",         fontsize=13)

    for ax in axes:
        ax.axis("off")

    title = f"Prediction: {pred} (p={prob:.3f})"
    if true_label:
        title += f"  |  Ground Truth: {true_label}"
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[GradCAM] Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--image_path",  type=str, default=None)
    parser.add_argument("--batch",       type=int, default=None,
                        help="Visualize N random test images")
    parser.add_argument("--save_dir",    type=str, default="outputs/figures/gradcam")
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = OmegaConf.load(args.config)
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")

    set_seed(cfg.experiment.seed)

    model = build_model(
        model_name=cfg.model.architecture,
        pretrained=False,
        dropout_rate=cfg.model.dropout_rate,
        device=str(device),
    )
    load_checkpoint(args.checkpoint, model, device=str(device))

    if args.image_path:
        visualize_single(
            model=model,
            image_path=args.image_path,
            image_size=cfg.dataset.image_size,
            device=device,
            save_path=f"{args.save_dir}/{Path(args.image_path).stem}_gradcam.png",
        )

    elif args.batch:
        import pandas as pd, random
        df = pd.read_csv(cfg.paths.labels_csv)
        df = df[df["Quality Score"] >= cfg.dataset.min_quality_score]
        samples = df.sample(min(args.batch, len(df)), random_state=42)
        for _, row in samples.iterrows():
            img_path = f"{cfg.paths.images_dir}/{row['Image Name']}"
            save_path = f"{args.save_dir}/{Path(img_path).stem}_gradcam.png"
            visualize_single(
                model=model,
                image_path=img_path,
                image_size=cfg.dataset.image_size,
                device=device,
                save_path=save_path,
                true_label=row["Label"],
            )

    else:
        print("Provide --image_path or --batch N")


if __name__ == "__main__":
    main()
