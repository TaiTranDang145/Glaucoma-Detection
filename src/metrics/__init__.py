"""Metrics computed from optic-disc segmentation masks."""

from .optic_disc import CDRResult, calculate_vertical_cdr

__all__ = ["CDRResult", "calculate_vertical_cdr"]
