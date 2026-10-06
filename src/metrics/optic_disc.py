"""Optic-disc and optic-cup measurements derived from binary masks."""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CDRResult:
    """Vertical cup-to-disc ratio and segmentation validity."""

    cdr: float
    segmentation_valid: bool
    status: str


def calculate_vertical_cdr(optic_disc_mask: np.ndarray, optic_cup_mask: np.ndarray) -> CDRResult:
    """Calculate vertical CDR as inclusive cup height divided by disc height.

    Each mask's largest 8-connected component is treated as the anatomical
    structure. Smaller components, such as isolated segmentation speckles, are
    ignored. Foreground is any non-zero pixel.
    """
    disc_source = np.asarray(optic_disc_mask)
    cup_source = np.asarray(optic_cup_mask)
    if disc_source.ndim != 2 or cup_source.ndim != 2:
        return _invalid("invalid_dimensions")
    if disc_source.shape != cup_source.shape:
        return _invalid("shape_mismatch")
    if not _is_binary_mask(disc_source) or not _is_binary_mask(cup_source):
        return _invalid("invalid_mask_values")

    disc = disc_source != 0
    cup = cup_source != 0
    if not disc.any():
        return _invalid("empty_disc")
    if not cup.any():
        return _invalid("empty_cup")

    disc, disc_area = _largest_component(disc)
    cup, cup_area = _largest_component(cup)
    if disc_area <= 1 or cup_area <= 1:
        return _invalid("single_pixel_mask")
    if np.any(cup & ~disc):
        return _invalid("cup_outside_disc")

    disc_rows = np.flatnonzero(disc.any(axis=1))
    cup_rows = np.flatnonzero(cup.any(axis=1))
    disc_height = int(disc_rows[-1] - disc_rows[0] + 1)
    cup_height = int(cup_rows[-1] - cup_rows[0] + 1)
    if disc_height <= 0:
        return _invalid("empty_disc")
    return CDRResult(cup_height / disc_height, True, "ok")


def _invalid(status: str) -> CDRResult:
    return CDRResult(float("nan"), False, status)


def _largest_component(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return an 8-connected foreground mask containing its largest component."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool), 0
    component_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component_label, cv2.CC_STAT_AREA])
    return labels == component_label, area


def _is_binary_mask(mask: np.ndarray) -> bool:
    """Accept finite binary masks encoded as 0/1 or 0/255."""
    try:
        finite = np.isfinite(mask)
        binary = (mask == 0) | (mask == 1) | (mask == 255)
    except (TypeError, ValueError):
        return False
    return bool(finite.all() and binary.all())
