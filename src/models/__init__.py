"""Model factory; model implementations live in one file per architecture."""

import torch

from .efficientnet_b3 import GlaucomaClassifier
from .gonet import DINO_MODEL, GONet
from .mobilenetv3 import MobileNetV3Classifier


def build_model(model_name="mobilenetv3_small", pretrained=True, dropout_rate=0.2, device="cuda", image_size=224):
    name = model_name.casefold()
    if name in {"gonet", "dinov2", DINO_MODEL.casefold()}:
        model = GONet(pretrained=pretrained, dropout_rate=dropout_rate, image_size=image_size)
    elif name.startswith("efficientnet_b3"):
        model = GlaucomaClassifier(
            model_name=model_name, pretrained=pretrained, dropout_rate=dropout_rate
        )
    elif "mobilenetv3" in name or name in {"m2", "m3", "mobilenet"}:
        actual_name = "mobilenetv3_small_100" if "small" in name or name in {"m2", "m3", "mobilenet"} else model_name
        model = MobileNetV3Classifier(
            model_name=actual_name, pretrained=pretrained, dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"[Model] Architecture : {model_name}")
    print(f"[Model] Parameters   : {params:,} ({trainable:,} trainable)")
    print(f"[Model] Device       : {device}")
    return model
