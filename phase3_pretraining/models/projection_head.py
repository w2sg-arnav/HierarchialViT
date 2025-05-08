# phase3_pretraining/models/projection_head.py
import torch
import torch.nn as nn
import logging

# PROJECTION_DIM, PROJECTION_HIDDEN_DIM will be from config, passed during init
# from ..config import PROJECTION_DIM, PROJECTION_HIDDEN_DIM 

logger = logging.getLogger(__name__)

class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning (e.g., SimCLR).
    Maps backbone features to a lower-dimensional space for loss calculation.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Often beneficial in projection heads
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        logger.info(f"ProjectionHead initialized: in_dim={in_dim}, hidden_dim={hidden_dim}, out_dim={out_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is expected to be [Batch, InDim] (features after global pooling)
        if x.ndim != 2:
            # Example: if x is [B, L, C] from transformer, it needs pooling first
            # logger.warning(f"ProjectionHead input has {x.ndim} dims, expected 2 (B, C). Applying mean pooling over dim 1.")
            # x = x.mean(dim=1) # This should be done in the model backbone before projection head
            pass # Assume x is already [B, C]
        return self.head(x)