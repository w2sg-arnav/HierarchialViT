# hvit/utils/__init__.py
"""
Utility functions and classes.

This module contains:
- EMA: Exponential Moving Average for model weights
- Metrics: Evaluation metrics computation
- Logging: Logging setup utilities
"""

from hvit.utils.ema import EMA
from hvit.utils.metrics import compute_metrics
from hvit.utils.logging_setup import setup_logging

__all__ = [
    "EMA",
    "compute_metrics",
    "setup_logging",
]
