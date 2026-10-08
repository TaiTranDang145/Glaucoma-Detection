"""GONet: DINOv2 ViT-B/14 fine-tuned for binary GON classification."""

import torch
import torch.nn as nn
import timm


DINO_MODEL = "vit_base_patch14_dinov2.lvd142m"


class GONet(nn.Module):
    """The preprint specifies DINOv2 ViT-B; its exact binary head is unspecified.

    A single linear classifier is used as the minimal fine-tuning head.
    """

    def __init__(self, pretrained=True, dropout_rate=0.0, image_size=392):
        super().__init__()
        self.backbone = timm.create_model(
            DINO_MODEL, pretrained=pretrained, num_classes=0, img_size=image_size
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.num_features, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(images))
