# phase4_finetuning/utils/__init__.py
"""Utility modules for phase4 finetuning."""

from .augmentations import create_cotton_leaf_augmentation
from .losses import FocalLoss, CombinedLoss
from .metrics import compute_metrics

# EMA requires timm, import conditionally
try:
    from .ema import EMA
    __all__ = [
        "create_cotton_leaf_augmentation",
        "FocalLoss",
        "CombinedLoss",
        "compute_metrics",
        "EMA",
    ]
except ImportError:
    __all__ = [
        "create_cotton_leaf_augmentation",
        "FocalLoss",
        "CombinedLoss",
        "compute_metrics",
    ]