# hvit/__init__.py
"""
HViT (Hierarchical Vision Transformer) for Cotton Disease Classification.

This package provides a complete implementation of the Disease-Aware Hierarchical
Vision Transformer (HViT) with multi-scale feature extraction and optional
spectral data fusion.

Package Structure:
- models: HViT architecture and baseline models
- data: Dataset loaders and augmentation pipelines  
- training: Pre-training and fine-tuning trainers
- utils: Utilities for EMA, metrics, and logging
"""

from hvit.models import (
    DiseaseAwareHVT,
    create_disease_aware_hvt,
    DiseaseFocusedCrossAttention,
    InceptionV3Baseline,
)
from hvit.data import (
    BaseSARCLD2024Dataset,
    SARCLD2024Dataset,
    SARCLD_CLASSES,
)

__version__ = "1.0.0"
__author__ = "HViT Research Team"

__all__ = [
    # Models
    "DiseaseAwareHVT",
    "create_disease_aware_hvt",
    "DiseaseFocusedCrossAttention",
    "InceptionV3Baseline",
    # Data
    "BaseSARCLD2024Dataset",
    "SARCLD2024Dataset",
    "SARCLD_CLASSES",
    # Version
    "__version__",
]
