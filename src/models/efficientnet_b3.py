"""EfficientNet-B3 classifier retained for the original HYDR experiment."""

import torch
import torch.nn as nn
import timm


class GlaucomaClassifier(nn.Module):
    def __init__(self, model_name="efficientnet_b3", pretrained=True, dropout_rate=0.3, num_classes=1):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes),
        )
        for layer in self.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, images):
        return self.classifier(self.backbone(images))

    def predict_proba(self, images):
        with torch.no_grad():
            return torch.sigmoid(self(images)).squeeze(1)

    def count_parameters(self):
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
        }
