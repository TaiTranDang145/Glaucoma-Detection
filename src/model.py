"""
model.py
--------
EfficientNet-B3 model for binary glaucoma classification (GON+ vs GON-).

Architecture:
    - Backbone : EfficientNet-B3 (ImageNet pretrained via timm)
    - Head     : Dropout → Linear(1280 → 512) → ReLU → Dropout → Linear(512 → 1)
    - Output   : Raw logit (use with BCEWithLogitsLoss) or sigmoid probability
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class GlaucomaClassifier(nn.Module):
    """
    EfficientNet-B3 based binary classifier for Glaucomatous Optic Neuropathy.

    Args:
        model_name (str): timm model name. Default: 'efficientnet_b3'.
        pretrained (bool): Use ImageNet pretrained weights. Default: True.
        dropout_rate (float): Dropout probability in classifier head. Default: 0.3.
        num_classes (int): Number of output classes. Default: 1 (binary).
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        num_classes: int = 1,
    ):
        super().__init__()

        # Load backbone without classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # Remove default head
            global_pool="avg",
        )
        in_features = self.backbone.num_features  # 1536 for EfficientNet-B3

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

        # Initialize head weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for Linear layers in the head."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Raw logits of shape (B, 1). Apply sigmoid for probability.
        """
        features = self.backbone(x)           # (B, in_features)
        logits = self.classifier(features)    # (B, 1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns sigmoid probability (0–1 range).

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Probabilities of shape (B,).
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(1)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (for linear probing)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (for full fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[Model] Backbone unfrozen.")

    def count_parameters(self) -> dict:
        """Returns total and trainable parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ── Loss ──────────────────────────────────────────────────────────────────────

def build_criterion(
    pos_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """
    Builds Binary Cross-Entropy loss with optional class weighting
    and label smoothing.

    Args:
        pos_weight (Tensor, optional): Weight for GON+ class.
                                       Computed as n_neg / n_pos.
        label_smoothing (float): Label smoothing factor (0.0 = disabled).

    Returns:
        nn.Module: Loss function.
    """
    if label_smoothing > 0.0:
        # Manual label smoothing wrapper
        return LabelSmoothingBCE(pos_weight=pos_weight, smoothing=label_smoothing)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


class LabelSmoothingBCE(nn.Module):
    """
    BCE with logits + label smoothing for binary classification.

    Smoothed targets: positive → (1 - ε), negative → ε.
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        smoothing: float = 0.1,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        targets_smooth = targets.float() * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits.squeeze(1), targets_smooth)


# ── Builder ───────────────────────────────────────────────────────────────────

def build_model(
    model_name: str = "efficientnet_b3",
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    device: str = "cuda",
) -> GlaucomaClassifier:
    """
    Convenience factory to build and move model to device.

    Args:
        model_name (str): timm model name.
        pretrained (bool): ImageNet pretrained.
        dropout_rate (float): Dropout rate.
        device (str): 'cuda' or 'cpu'.

    Returns:
        GlaucomaClassifier on specified device.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = GlaucomaClassifier(
        model_name=model_name,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    ).to(device)

    params = model.count_parameters()
    print(f"[Model] Architecture : {model_name}")
    print(f"[Model] Total params : {params['total']:,}")
    print(f"[Model] Trainable    : {params['trainable']:,}")
    print(f"[Model] Device       : {device}")

    return model
