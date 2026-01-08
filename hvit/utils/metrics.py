# hvit/utils/metrics.py
"""Evaluation metrics for classification tasks.

This module provides utilities for computing classification metrics
including accuracy, F1 score, precision, and recall.
"""

import numpy as np
import torch
from typing import Dict, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

# Optional sklearn import
try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available. Metrics computation will be limited.")


def compute_metrics(
    preds: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    num_classes: int,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute evaluation metrics for multi-class classification.
    
    Args:
        preds: Predicted class indices or logits.
        labels: Ground truth labels.
        num_classes: Total number of classes.
        class_names: Optional class names for per-class metrics.
    
    Returns:
        Dictionary containing:
        - accuracy: Overall accuracy
        - f1_macro: Macro-averaged F1 score
        - f1_weighted: Weighted F1 score
        - precision_macro: Macro-averaged precision
        - precision_weighted: Weighted precision
        - recall_macro: Macro-averaged recall
        - recall_weighted: Weighted recall
        - f1_<class_name>: Per-class F1 (if class_names provided)
    """
    if not SKLEARN_AVAILABLE:
        logger.error("sklearn required for compute_metrics but not available.")
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0
        }

    if len(preds) == 0 or len(labels) == 0:
        logger.warning("Empty predictions or labels. Returning zero metrics.")
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "precision_macro": 0.0,
            "precision_weighted": 0.0,
            "recall_macro": 0.0,
            "recall_weighted": 0.0
        }

    # Convert tensors to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Handle logits (2D) by taking argmax
    if preds.ndim == 2:
        preds = np.argmax(preds, axis=1)

    labels = labels.flatten()

    # Ensure integer types
    if not np.issubdtype(preds.dtype, np.integer):
        preds = preds.astype(int)
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(int)

    target_labels = list(range(num_classes))

    metrics: Dict[str, float] = {}
    
    # Accuracy
    metrics["accuracy"] = float(accuracy_score(labels, preds))

    # F1 scores
    metrics["f1_macro"] = float(f1_score(
        labels, preds, average='macro', zero_division=0, labels=target_labels
    ))
    metrics["f1_weighted"] = float(f1_score(
        labels, preds, average='weighted', zero_division=0, labels=target_labels
    ))

    # Precision
    metrics["precision_macro"] = float(precision_score(
        labels, preds, average='macro', zero_division=0, labels=target_labels
    ))
    metrics["precision_weighted"] = float(precision_score(
        labels, preds, average='weighted', zero_division=0, labels=target_labels
    ))

    # Recall
    metrics["recall_macro"] = float(recall_score(
        labels, preds, average='macro', zero_division=0, labels=target_labels
    ))
    metrics["recall_weighted"] = float(recall_score(
        labels, preds, average='weighted', zero_division=0, labels=target_labels
    ))

    # Per-class F1 scores
    if class_names is not None and len(class_names) == num_classes:
        per_class_f1 = f1_score(
            labels, preds, average=None, zero_division=0, labels=target_labels
        )
        
        # Ensure correct length
        if len(per_class_f1) != num_classes:
            temp_f1 = np.zeros(num_classes)
            for i, val in enumerate(per_class_f1):
                if i < num_classes:
                    temp_f1[i] = val
            per_class_f1 = temp_f1

        for i, name in enumerate(class_names):
            clean_name = name.replace(' ', '_')
            if i < len(per_class_f1):
                metrics[f"f1_{clean_name}"] = float(per_class_f1[i])
            else:
                metrics[f"f1_{clean_name}"] = 0.0

    return metrics


__all__ = ["compute_metrics"]
