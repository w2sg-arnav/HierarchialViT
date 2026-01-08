# hvit/data/__init__.py
"""
Data loading and augmentation utilities.

This module contains:
- Dataset classes for SAR-CLD-2024 cotton disease dataset
- Augmentation pipelines for pre-training and fine-tuning
"""

from hvit.data.dataset import (
    BaseSARCLD2024Dataset,
    SARCLD2024Dataset,
    SARCLD_CLASSES,
)
from hvit.data.augmentations import (
    SimCLRAugmentation,
    EnhancedColorJitter,
    LightingVariation,
    GaussianNoise,
    get_train_transforms,
    get_val_transforms,
)

__all__ = [
    # Datasets
    "BaseSARCLD2024Dataset",
    "SARCLD2024Dataset",
    "SARCLD_CLASSES",
    # Augmentations
    "SimCLRAugmentation",
    "EnhancedColorJitter",
    "LightingVariation",
    "GaussianNoise",
    "get_train_transforms",
    "get_val_transforms",
]
