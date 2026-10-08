"""MobileNetV3-Small classifier for lightweight GON classification.

Supports:
- Binary logit classification head
- Pre-logits embedding extraction (1024-dim) for downstream clinical/multi-modal fusion
- Spatial convolutional feature extraction (576-dim x 7 x 7) for explainability / Grad-CAM
"""

from typing import Dict
import timm
import torch
import torch.nn as nn


class MobileNetV3Classifier(nn.Module):
    """MobileNetV3-Small binary classifier."""

    def __init__(
        self,
        model_name: str = "mobilenetv3_small_100",
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        num_classes: int = 1,
    ):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (B, num_classes)."""
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the 1024-dimensional pre-logits feature vector."""
        feat = self.backbone.forward_features(x)
        return self.backbone.forward_head(feat, pre_logits=True)

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the 576-channel spatial feature map (B, 576, H/32, W/32)."""
        return self.backbone.forward_features(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes sigmoid probabilities for binary classification."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(-1)

    def count_parameters(self) -> Dict[str, int]:
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
