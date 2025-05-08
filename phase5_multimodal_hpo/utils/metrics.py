# phase5_multimodal_hpo/utils/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torch
import logging
from typing import Optional, List, Dict # Added Dict

logger = logging.getLogger(__name__)

def compute_metrics(preds, labels, num_classes: int, class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Compute evaluation metrics for multi-class classification.
    
    Args:
        preds: Predicted class indices (numpy array or torch tensor, 1D or 2D [Logits/Probs])
        labels: Ground truth labels (numpy array or torch tensor, 1D)
        num_classes (int): Total number of classes for averaging purposes.
        class_names (list, optional): List of class names for detailed reporting.
    
    Returns:
        dict: Dictionary containing accuracy, F1 (macro/weighted), Precision (macro/weighted), Recall (macro/weighted).
              Optionally includes per-class F1 if class_names are provided.
    """
    default_metrics = {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, 
                       "precision_macro": 0.0, "precision_weighted": 0.0,
                       "recall_macro": 0.0, "recall_weighted": 0.0}
                       
    if preds is None or labels is None or len(preds) == 0 or len(labels) == 0:
        logger.warning("Empty predictions or labels received in compute_metrics. Returning zero metrics.")
        return default_metrics
    if len(preds) != len(labels):
         logger.error(f"Mismatch in length of predictions ({len(preds)}) and labels ({len(labels)}). Cannot compute metrics.")
         return default_metrics


    if isinstance(preds, torch.Tensor): preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
    
    if preds.ndim == 2: preds = np.argmax(preds, axis=1)
    
    labels = labels.flatten().astype(int)
    preds = preds.flatten().astype(int)
    
    present_labels = np.unique(labels)
    target_labels = list(range(num_classes)) 

    metrics = {}
    try:
        metrics["accuracy"] = accuracy_score(labels, preds)
        metrics["f1_macro"] = f1_score(labels, preds, average='macro', zero_division=0, labels=target_labels)
        metrics["f1_weighted"] = f1_score(labels, preds, average='weighted', zero_division=0, labels=target_labels)
        metrics["precision_macro"] = precision_score(labels, preds, average='macro', zero_division=0, labels=target_labels)
        metrics["precision_weighted"] = precision_score(labels, preds, average='weighted', zero_division=0, labels=target_labels)
        metrics["recall_macro"] = recall_score(labels, preds, average='macro', zero_division=0, labels=target_labels)
        metrics["recall_weighted"] = recall_score(labels, preds, average='weighted', zero_division=0, labels=target_labels)

        if class_names is not None and len(class_names) == num_classes:
            per_class_f1 = f1_score(labels, preds, average=None, zero_division=0, labels=target_labels)
            for i, name in enumerate(class_names):
                 safe_name = name.replace(' ', '_') # Make name filesystem/dict key safe
                 metrics[f"f1_{safe_name}"] = per_class_f1[i]

    except Exception as e:
        logger.error(f"Error computing metrics: {e}", exc_info=True)
        return default_metrics # Return default values on error

    return metrics