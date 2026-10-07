"""
dataset.py
----------
PyTorch Dataset and DataLoader for the Hillel Yaffe Glaucoma Dataset (HYGD).

Handles:
- Patient-level train/val/test splitting (no data leakage)
- Quality-score-based filtering and sample weighting
- Augmentation pipeline via Albumentations
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupShuffleSplit

try:
    from .domain_sampling import make_domain_balanced_sampler
except ImportError:  # Support `python src/train_msd.py` imports from Kaggle.
    from domain_sampling import make_domain_balanced_sampler


# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL_MAP = {"GON+": 1, "GON-": 0}


# ── Augmentation pipelines ────────────────────────────────────────────────────

def get_transforms(split: str, image_size: int = 300) -> A.Compose:
    """
    Returns Albumentations augmentation pipeline.

    Args:
        split (str): One of 'train', 'val', 'test'.
        image_size (int): Target image size (square).

    Returns:
        A.Compose: Augmentation pipeline.
    """
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(
                max_holes=4, max_height=20, max_width=20,
                p=0.1
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:  # val / test
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class GlaucomaDataset(Dataset):
    """
    PyTorch Dataset for the HYGD glaucoma fundus image dataset.

    Args:
        dataframe (pd.DataFrame): Filtered subset of Labels.csv.
        images_dir (str | Path): Path to the Images/ folder.
        transform (A.Compose): Albumentations transform pipeline.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        images_dir: str,
        transform: Optional[A.Compose] = None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path = self.images_dir / row["Image Name"]

        # Load image as RGB numpy array
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        label = LABEL_MAP[row["Label"]]
        return image, label, row["Image Name"]


# ── Data splitting ────────────────────────────────────────────────────────────

def patient_level_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset at the patient level to prevent data leakage.

    Args:
        df (pd.DataFrame): Full Labels.csv dataframe.
        val_size (float): Fraction for validation.
        test_size (float): Fraction for test.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    patients = df["Patient"].values

    # First split: separate test set
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(df, groups=patients))

    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    df_test     = df.iloc[test_idx].reset_index(drop=True)

    # Second split: separate val from train
    adjusted_val = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=adjusted_val, random_state=seed
    )
    train_idx, val_idx = next(
        gss_val.split(df_trainval, groups=df_trainval["Patient"].values)
    )

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val   = df_trainval.iloc[val_idx].reset_index(drop=True)

    return df_train, df_val, df_test


# ── WeightedRandomSampler ─────────────────────────────────────────────────────

def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler to handle class imbalance.

    Each sample's weight is inversely proportional to its class frequency.
    Optionally scales by quality score.

    Args:
        df (pd.DataFrame): Training subset dataframe.

    Returns:
        WeightedRandomSampler instance.
    """
    labels = df["Label"].map(LABEL_MAP).values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts

    # Per-sample weight based on class
    sample_weights = class_weights[labels]

    # Optionally boost weights by quality score (higher quality → higher weight)
    if "Quality Score" in df.columns:
        quality = df["Quality Score"].fillna(5).values
        quality_norm = quality / quality.max()          # normalize to [0, 1]
        sample_weights = sample_weights * quality_norm

    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ── DataLoaders factory ───────────────────────────────────────────────────────

def build_dataloaders(
    labels_csv: str,
    images_dir: str,
    image_size: int = 300,
    batch_size: int = 32,
    num_workers: int = 4,
    min_quality_score: int = 3,
    val_size: float = 0.15,
    test_size: float = 0.15,
    use_weighted_sampler: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Full pipeline: load CSV → filter → split → build DataLoaders.

    Args:
        labels_csv (str): Path to Labels.csv.
        images_dir (str): Path to Images/ folder.
        image_size (int): Target image size.
        batch_size (int): Batch size.
        num_workers (int): DataLoader workers.
        min_quality_score (int): Drop images below this quality threshold.
        val_size (float): Validation fraction.
        test_size (float): Test fraction.
        use_weighted_sampler (bool): Use WeightedRandomSampler for training.
        seed (int): Random seed.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, split_info_dict).
    """
    df = pd.read_csv(labels_csv)

    # Validate required columns
    required_cols = {"Image Name", "Patient", "Label", "Quality Score"}
    assert required_cols.issubset(df.columns), (
        f"Labels.csv missing columns: {required_cols - set(df.columns)}"
    )

    # Filter by quality score
    n_before = len(df)
    df = df[df["Quality Score"] >= min_quality_score].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[Dataset] Dropped {n_dropped} images with quality < {min_quality_score}")

    # Patient-level split
    df_train, df_val, df_test = patient_level_split(
        df, val_size=val_size, test_size=test_size, seed=seed
    )

    # Transforms
    train_tf = get_transforms("train", image_size)
    val_tf   = get_transforms("val",   image_size)
    test_tf  = get_transforms("test",  image_size)

    # Datasets
    train_ds = GlaucomaDataset(df_train, images_dir, transform=train_tf)
    val_ds   = GlaucomaDataset(df_val,   images_dir, transform=val_tf)
    test_ds  = GlaucomaDataset(df_test,  images_dir, transform=test_tf)

    # Sampler
    sampler = make_weighted_sampler(df_train) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    split_info = {
        "train": {"total": len(df_train), "GON+": (df_train["Label"] == "GON+").sum(), "GON-": (df_train["Label"] == "GON-").sum()},
        "val":   {"total": len(df_val),   "GON+": (df_val["Label"]   == "GON+").sum(), "GON-": (df_val["Label"]   == "GON-").sum()},
        "test":  {"total": len(df_test),  "GON+": (df_test["Label"]  == "GON+").sum(), "GON-": (df_test["Label"]  == "GON-").sum()},
    }

    print("\n[Dataset] Split Summary:")
    print(f"  Train → {split_info['train']['total']} images "
          f"(GON+: {split_info['train']['GON+']}, GON-: {split_info['train']['GON-']})")
    print(f"  Val   → {split_info['val']['total']} images "
          f"(GON+: {split_info['val']['GON+']}, GON-: {split_info['val']['GON-']})")
    print(f"  Test  → {split_info['test']['total']} images "
          f"(GON+: {split_info['test']['GON+']}, GON-: {split_info['test']['GON-']})\n")

    return train_loader, val_loader, test_loader, split_info


# ── Paper preprocessing and multi-source manifest ───────────────────────────

def get_msd_transforms(split: str, image_size: int = 392) -> A.Compose:
    """Paper-sized transforms; square black padding is applied by the dataset."""
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transforms = [A.Resize(image_size, image_size)]
    if split == "train":
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ])
    transforms.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return A.Compose(transforms)


class ManifestDataset(Dataset):
    """Dataset for prepare_data.py's common CSV schema."""

    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = self.root_dir / row["image_path"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        side = max(width, height)
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(image, ((side - width) // 2, (side - height) // 2))
        image = np.array(canvas)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, int(row["label"]), row["image_path"]


def build_manifest_loader(
    dataframe,
    root_dir,
    image_size=392,
    batch_size=16,
    num_workers=4,
    split="train",
    domain_balanced_sampling=False,
):
    dataset = ManifestDataset(
        dataframe,
        root_dir,
        transform=get_msd_transforms(split, image_size),
    )
    sampler = None
    if split == "train" and domain_balanced_sampling:
        sampler = make_domain_balanced_sampler(dataframe["domain"].tolist())

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(split == "train" and sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
