"""Lightweight U-Net with MobileNetV3-Small encoder for Optic Disc & Cup segmentation.

Produces 2-channel binary logits:
- Channel 0: Optic Disc (OD)
- Channel 1: Optic Cup (OC)

Total params: ~1.18M
Input resolution: (B, 3, H, W)
Output resolution: (B, 2, H, W)
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double 3x3 Conv + BN + ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileNetV3UNet(nn.Module):
    """Lightweight U-Net with pre-trained MobileNetV3-Small encoder."""

    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        # Encoder: returns feature maps at 5 scales
        # Stage 0: 16 ch (stride 2)
        # Stage 1: 16 ch (stride 4)
        # Stage 2: 24 ch (stride 8)
        # Stage 3: 48 ch (stride 16)
        # Stage 4: 576 ch (stride 32)
        self.encoder = timm.create_model(
            "mobilenetv3_small_100",
            features_only=True,
            pretrained=pretrained,
        )

        # Decoder stages
        self.up4 = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = ConvBlock(128 + 48, 64)

        self.up3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = ConvBlock(64 + 24, 48)

        self.up2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.conv2 = ConvBlock(48 + 16, 32)

        self.up1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv1 = ConvBlock(32 + 16, 24)

        self.up0 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.conv0 = ConvBlock(24, 16)

        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass
        feats = self.encoder(x)
        f0, f1, f2, f3, f4 = feats

        # Decoder forward pass
        # Stage 4 -> 3
        d4 = F.interpolate(self.up4(f4), size=f3.shape[2:], mode="bilinear", align_corners=False)
        d4 = torch.cat([d4, f3], dim=1)
        d4 = self.conv4(d4)

        # Stage 3 -> 2
        d3 = F.interpolate(self.up3(d4), size=f2.shape[2:], mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, f2], dim=1)
        d3 = self.conv3(d3)

        # Stage 2 -> 1
        d2 = F.interpolate(self.up2(d3), size=f1.shape[2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, f1], dim=1)
        d2 = self.conv2(d2)

        # Stage 1 -> 0
        d1 = F.interpolate(self.up1(d2), size=f0.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, f0], dim=1)
        d1 = self.conv1(d1)

        # Stage 0 -> original input resolution
        d0 = F.interpolate(self.up0(d1), size=x.shape[2:], mode="bilinear", align_corners=False)
        d0 = self.conv0(d0)

        # Logits
        logits = self.final_conv(d0)
        return logits


if __name__ == "__main__":
    model = MobileNetV3UNet(pretrained=False, num_classes=2)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MobileNetV3-UNet successfully instantiated!")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total params: {total_params:,} ({total_params/1e6:.3f}M)")
