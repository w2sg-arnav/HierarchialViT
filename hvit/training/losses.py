# hvit/training/losses.py
"""Loss functions for pre-training and fine-tuning.

This module provides:
- InfoNCELoss: Contrastive loss for SimCLR pre-training
- FocalLoss: Class-imbalance aware loss for fine-tuning
- CombinedLoss: Blended CE + Focal loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for self-supervised learning.
    
    Uses cosine similarity to maximize agreement between positive pairs
    (different augmentations of the same image) while minimizing agreement
    with negative pairs (different images).
    
    Args:
        temperature: Temperature scaling factor for softmax.
    """
    
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        logger.info(f"InfoNCELoss initialized with temperature: {self.temperature}")

    def forward(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor
    ) -> torch.Tensor:
        """Compute InfoNCE loss.
        
        Args:
            features1: First view features (B, D).
            features2: Second view features (B, D).
        
        Returns:
            Scalar loss tensor.
        """
        if not (torch.is_tensor(features1) and torch.is_tensor(features2)):
            raise TypeError("Inputs must be PyTorch tensors.")
        if features1.shape != features2.shape:
            raise ValueError(
                f"Input feature tensors must have the same shape. "
                f"Got {features1.shape} and {features2.shape}"
            )
        if features1.ndim != 2:
            raise ValueError(
                f"Input features should be 2D tensors (Batch, Dim). Got {features1.ndim} dims."
            )
        if features1.shape[0] == 0:
            logger.warning("InfoNCELoss received empty batch of features. Returning 0 loss.")
            return torch.tensor(0.0, device=features1.device, requires_grad=True)

        z_i = F.normalize(features1, p=2, dim=1)
        z_j = F.normalize(features2, p=2, dim=1)
        batch_size = z_i.shape[0]

        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        labels = torch.arange(batch_size, device=z_i.device, dtype=torch.long)

        loss_ij = F.cross_entropy(sim_matrix, labels)
        loss_ji = F.cross_entropy(sim_matrix.T, labels)
        loss = (loss_ij + loss_ji) / 2.0

        if not torch.isfinite(loss):
            logger.warning(
                f"InfoNCELoss: Non-finite loss: {loss.item()}. "
                f"Temp: {self.temperature}. "
                f"SimMat min/max: {sim_matrix.min().item()}/{sim_matrix.max().item()}"
            )
        return loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    Reduces the loss for well-classified examples, focusing training
    on hard-to-classify examples.
    
    Args:
        alpha: Weighting factor for the positive class.
        gamma: Focusing parameter (higher = more focus on hard examples).
        reduction: Reduction method ('mean', 'sum', or 'none').
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        logger.info(f"FocalLoss initialized with alpha={alpha}, gamma={gamma}.")

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Predicted logits (B, C).
            targets: Ground truth labels (B,).
        
        Returns:
            Loss tensor.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss_val = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss_val.mean()
        elif self.reduction == 'sum':
            return focal_loss_val.sum()
        return focal_loss_val


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy and Focal Loss.
    
    Blends standard classification loss with focal loss for handling
    class imbalance while maintaining training stability.
    
    Args:
        num_classes: Number of target classes.
        smoothing: Label smoothing factor for CE loss.
        focal_alpha: Alpha parameter for focal loss.
        focal_gamma: Gamma parameter for focal loss.
        ce_weight: Weight for cross-entropy component.
        focal_weight: Weight for focal loss component.
        class_weights_tensor: Optional per-class weights.
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        ce_weight: float = 0.5,
        focal_weight: float = 0.5,
        class_weights_tensor: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=smoothing,
            weight=class_weights_tensor
        )
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma
        )
        logger.info(
            f"CombinedLoss initialized with CE weight={ce_weight}, "
            f"Focal weight={focal_weight}, Smoothing={smoothing}."
        )
        if abs(ce_weight + focal_weight - 1.0) > 1e-6 and (ce_weight + focal_weight > 0):
            logger.warning(
                f"CombinedLoss weights (CE:{ce_weight}, Focal:{focal_weight}) do not sum to 1.0."
            )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            inputs: Predicted logits (B, C).
            targets: Ground truth labels (B,).
        
        Returns:
            Weighted sum of CE and focal losses.
        """
        loss = torch.tensor(0.0, device=inputs.device)
        if self.ce_weight > 0:
            loss = loss + self.ce_weight * self.ce_loss(inputs, targets)
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self.focal_loss(inputs, targets)
        return loss


__all__ = [
    "InfoNCELoss",
    "FocalLoss",
    "CombinedLoss",
]
