# hvit/models/__init__.py
"""
HViT model architectures.

This module contains:
- DiseaseAwareHVT: Main hierarchical vision transformer with disease-focused attention
- DiseaseFocusedCrossAttention: Cross-attention module for RGB-spectral fusion
- InceptionV3Baseline: Baseline comparison model
"""

from hvit.models.hvt import (
    DiseaseAwareHVT,
    create_disease_aware_hvt,
    PatchEmbed,
    PatchMerging,
    Attention,
    TransformerBlock,
    HVTStage,
    MAEPredictionHead,
)
from hvit.models.dfca import DiseaseFocusedCrossAttention
from hvit.models.baseline import InceptionV3Baseline

__all__ = [
    # Main model
    "DiseaseAwareHVT",
    "create_disease_aware_hvt",
    # Building blocks
    "PatchEmbed",
    "PatchMerging", 
    "Attention",
    "TransformerBlock",
    "HVTStage",
    "MAEPredictionHead",
    # Fusion module
    "DiseaseFocusedCrossAttention",
    # Baseline
    "InceptionV3Baseline",
]
