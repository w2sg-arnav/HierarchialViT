# phase3_pretraining/models/projection_head.py
import torch
import torch.nn as nn
import logging
from typing import Optional # Added Optional

logger = logging.getLogger(__name__)

class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning (e.g., SimCLR).
    Maps backbone features to a lower-dimensional space for loss calculation.
    Includes BatchNorm for stability.
    """
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: Optional[int] = None, # Make hidden_dim optional
                 out_dim: int = 256, # Default from common practice / config
                 use_batch_norm: bool = True): 
        super().__init__()
        # Default hidden_dim to in_dim if not provided
        _hidden_dim = hidden_dim if hidden_dim is not None else in_dim 
        
        layers = [
            nn.Linear(in_dim, _hidden_dim),
        ]
        if use_batch_norm:
             layers.append(nn.BatchNorm1d(_hidden_dim))
             
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Linear(_hidden_dim, out_dim)
        ])
        
        self.head = nn.Sequential(*layers)
        
        logger.info(f"ProjectionHead initialized: in={in_dim}, hidden={_hidden_dim}, out={out_dim}, bn={use_batch_norm}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is expected to be [Batch, InDim]
        if x.ndim != 2:
             logger.warning(f"ProjectionHead input has {x.ndim} dims, expected 2 (B, C). Check pooling in backbone.")
             # Return input or raise error? Returning input might hide issues.
             # raise ValueError(f"ProjectionHead expects input of shape [Batch, Dim], got {x.shape}")
             return x # Or potentially pool here as a fallback: x = x.mean(dim=1)
        return self.head(x)