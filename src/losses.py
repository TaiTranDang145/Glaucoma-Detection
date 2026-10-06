"""Binary classification losses shared by the model training scripts."""

from typing import Optional

import torch
import torch.nn as nn


class LabelSmoothingBCE(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        smoothed = targets.float() * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, smoothed)


def build_criterion(pos_weight=None, label_smoothing=0.0):
    if label_smoothing > 0:
        return LabelSmoothingBCE(pos_weight=pos_weight, smoothing=label_smoothing)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
