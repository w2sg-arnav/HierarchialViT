# phase4_finetuning/dataset.py
"""Dataset module for phase4 finetuning.

This module re-exports the shared SARCLD2024Dataset for backward compatibility.
"""

from typing import Callable, Optional, Tuple

# Import from shared module
from shared.dataset import (
    SARCLD_CLASSES,
    BaseSARCLD2024Dataset,
    SARCLD2024Dataset,
)

__all__ = [
    "SARCLD_CLASSES",
    "BaseSARCLD2024Dataset",
    "SARCLD2024Dataset",
]