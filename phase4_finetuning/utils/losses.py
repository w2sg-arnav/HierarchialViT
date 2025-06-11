# phase4_finetuning/utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reduces the loss for well-classified examples, focusing on hard-to-classify examples.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        logger.info(f"FocalLoss initialized with alpha={alpha}, gamma={gamma}.")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss_val = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss_val.mean()
        elif self.reduction == 'sum':
            return focal_loss_val.sum()
        return focal_loss_val

class CombinedLoss(nn.Module):
    """
    Combines Cross-Entropy with Focal Loss.
    This allows for a blend of standard classification loss with a focus on hard examples.
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1, focal_alpha: float = 0.25, 
                 focal_gamma: float = 2.0, ce_weight: float = 0.5, focal_weight: float = 0.5, 
                 class_weights_tensor: Optional[torch.Tensor] = None):
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
        logger.info(f"CombinedLoss initialized with CE weight={ce_weight}, Focal weight={focal_weight}, Smoothing={smoothing}.")
        if abs(ce_weight + focal_weight - 1.0) > 1e-6 and (ce_weight + focal_weight > 0):
            logger.warning(f"CombinedLoss weights (CE:{ce_weight}, Focal:{focal_weight}) do not sum to 1.0.")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=inputs.device)
        if self.ce_weight > 0:
            loss += self.ce_weight * self.ce_loss(inputs, targets)
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal_loss(inputs, targets)
        return loss