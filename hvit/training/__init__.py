# hvit/training/__init__.py
"""
Training utilities for pre-training and fine-tuning.

This module contains:
- Pretrainer: SimCLR-based self-supervised pre-training
- Finetuner: Supervised fine-tuning with advanced techniques
- Loss functions for contrastive and classification tasks
"""

from hvit.training.pretrainer import (
    Pretrainer,
    get_cosine_schedule_with_warmup,
)
from hvit.training.finetuner import EnhancedFinetuner
from hvit.training.losses import (
    InfoNCELoss,
    FocalLoss,
    CombinedLoss,
)

__all__ = [
    # Trainers
    "Pretrainer",
    "EnhancedFinetuner",
    # Schedulers
    "get_cosine_schedule_with_warmup",
    # Losses
    "InfoNCELoss",
    "FocalLoss",
    "CombinedLoss",
]
