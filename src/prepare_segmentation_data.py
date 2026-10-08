"""Pre-cache square-padded 256x256 masks for PAPILA.

Generates:
data/cache/masks_256/{image_id}_disc_exp1.png
data/cache/masks_256/{image_id}_cup_exp1.png
data/cache/masks_256/{image_id}_disc_exp2.png
data/cache/masks_256/{image_id}_cup_exp2.png
data/cache/masks_256/{image_id}_disc_consensus.png (mean of exp1 & exp2)
data/cache/masks_256/{image_id}_cup_consensus.png  (mean of exp1 & exp2)
"""

import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def square_pad_and_resize(mask: np.ndarray, target_size: int = 256) -> np.ndarray:
    h, w = mask.shape
    side = max(h, w)
    pad_top = (side - h) // 2
    pad_left = (side - w) // 2
    padded = cv2.copyMakeBorder(
        mask,
        pad_top,
        side - h - pad_top,
        pad_left,
        side - w - pad_left,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return resized


def main():
    print("=== Caching Square-Padded 256x256 Masks ===")
    manifest_p = Path("data/manifests/papila_morphology.csv")
    df = pd.read_csv(manifest_p)

    out_dir = Path("data/cache/masks_256")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        img_id = row["image_id"]

        d1 = cv2.imread(f"data/masks/papila/exp1/{img_id}_disc.png", cv2.IMREAD_GRAYSCALE)
        c1 = cv2.imread(f"data/masks/papila/exp1/{img_id}_cup.png", cv2.IMREAD_GRAYSCALE)
        d2 = cv2.imread(f"data/masks/papila/exp2/{img_id}_disc.png", cv2.IMREAD_GRAYSCALE)
        c2 = cv2.imread(f"data/masks/papila/exp2/{img_id}_cup.png", cv2.IMREAD_GRAYSCALE)

        d1_256 = square_pad_and_resize(d1, 256)
        c1_256 = square_pad_and_resize(c1, 256)
        d2_256 = square_pad_and_resize(d2, 256)
        c2_256 = square_pad_and_resize(c2, 256)

        # Soft consensus in uint8: 0, 128 (one expert), 255 (both experts)
        d_cons = np.clip(((d1_256.astype(np.float32) + d2_256.astype(np.float32)) / 2.0).round(), 0, 255).astype(np.uint8)
        c_cons = np.clip(((c1_256.astype(np.float32) + c2_256.astype(np.float32)) / 2.0).round(), 0, 255).astype(np.uint8)

        cv2.imwrite(str(out_dir / f"{img_id}_disc_exp1.png"), d1_256)
        cv2.imwrite(str(out_dir / f"{img_id}_cup_exp1.png"), c1_256)
        cv2.imwrite(str(out_dir / f"{img_id}_disc_exp2.png"), d2_256)
        cv2.imwrite(str(out_dir / f"{img_id}_cup_exp2.png"), c2_256)
        cv2.imwrite(str(out_dir / f"{img_id}_disc_consensus.png"), d_cons)
        cv2.imwrite(str(out_dir / f"{img_id}_cup_consensus.png"), c_cons)

    print(f"Successfully cached 256x256 masks for {len(df)} images in {out_dir}")


if __name__ == "__main__":
    main()
