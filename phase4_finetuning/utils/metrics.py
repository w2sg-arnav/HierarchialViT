# phase4_finetuning/utils/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torch
import logging
from typing import Optional, List # Added List

logger = logging.getLogger(__name__)

def compute_metrics(preds, labels, num_classes: int, class_names: Optional[List[str]] = None):
    """
    Compute evaluation metrics for multi-class classification.
    
    Args:
        preds: Predicted class indices (numpy array or torch tensor, 1D or 2D)
        labels: Ground truth labels (numpy array or torch tensor, 1D)
        num_classes (int): Total number of classes for averaging purposes.
        class_names (list, optional): List of class names for detailed reporting.
    
    Returns:
        dict: Dictionary containing accuracy, macro F1, weighted F1,
              macro precision, weighted precision, macro recall, weighted recall.
              Optionally includes per-class F1 if class_names are provided.
    """
    if len(preds) == 0 or len(labels) == 0:
        logger.warning("Empty predictions or labels received in compute_metrics. Returning zero metrics.")
        return {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, 
                "precision_macro": 0.0, "precision_weighted": 0.0,
                "recall_macro": 0.0, "recall_weighted": 0.0}

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    if preds.ndim == 2:
        preds = np.argmax(preds, axis=1)
    
    labels = labels.flatten()

    if not np.issubdtype(preds.dtype, np.integer):
        # logger.warning(f"Predictions dtype is {preds.dtype}. Casting to int.")
        preds = preds.astype(int)
    if not np.issubdtype(labels.dtype, np.integer):
        # logger.warning(f"Labels dtype is {labels.dtype}. Casting to int.")
        labels = labels.astype(int)

    present_labels = np.unique(labels)
    target_labels = list(range(num_classes)) 

    metrics = {}
    metrics["accuracy"] = accuracy_score(labels, preds)
    
    metrics["f1_macro"] = f1_score(labels, preds, average='macro', zero_division=0, labels=target_labels)
    metrics["f1_weighted"] = f1_score(labels, preds, average='weighted', zero_division=0, labels=target_labels)
    
    metrics["precision_macro"] = precision_score(labels, preds, average='macro', zero_division=0, labels=target_labels)
    metrics["precision_weighted"] = precision_score(labels, preds, average='weighted', zero_division=0, labels=target_labels)
    
    metrics["recall_macro"] = recall_score(labels, preds, average='macro', zero_division=0, labels=target_labels)
    metrics["recall_weighted"] = recall_score(labels, preds, average='weighted', zero_division=0, labels=target_labels)

    if class_names is not None and len(class_names) == num_classes:
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0, labels=target_labels)
        # Ensure per_class_f1 has length num_classes even if some classes absent
        if len(per_class_f1) != num_classes: 
             temp_f1 = np.zeros(num_classes)
             present_mask = np.isin(target_labels, present_labels)
             present_indices_in_f1 = np.where(present_mask)[0]
             # Map calculated f1 scores to the correct class index
             f1_map_idx = 0
             for target_idx in target_labels:
                 if target_idx in present_labels:
                    if f1_map_idx < len(per_class_f1):
                         temp_f1[target_idx] = per_class_f1[f1_map_idx]
                         f1_map_idx +=1
                 # else: temp_f1[target_idx] remains 0
             per_class_f1 = temp_f1


        for i, name in enumerate(class_names):
            # Check if index i is within bounds of per_class_f1
            if i < len(per_class_f1):
                 metrics[f"f1_{name.replace(' ', '_')}"] = per_class_f1[i]
            else:
                 metrics[f"f1_{name.replace(' ', '_')}"] = 0.0 # Should not happen with labels=target_labels

    return metrics