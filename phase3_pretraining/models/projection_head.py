# phase3_pretraining/models/projection_head.py
import torch
import torch.nn as nn
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ProjectionHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: Optional[int] = None,
                 out_dim: int = 256, # Typical SimCLR projection dimension
                 use_batch_norm: bool = True):
        super().__init__()
        # If hidden_dim is None or 0, use in_dim (effectively making it a 2-layer MLP if hidden=in)
        # Or, one could argue for a simpler direct linear layer if hidden_dim is explicitly set low.
        # For SimCLR, a hidden layer is common.
        _actual_hidden_dim = hidden_dim if hidden_dim is not None and hidden_dim > 0 else in_dim

        layers = [nn.Linear(in_dim, _actual_hidden_dim)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(_actual_hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(_actual_hidden_dim, out_dim))
        # SimCLR does not typically use BatchNorm after the final projection layer.

        self.head = nn.Sequential(*layers)
        logger.info(f"ProjectionHead initialized: In={in_dim}, Hidden={_actual_hidden_dim}, Out={out_dim}, BatchNorm={use_batch_norm}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] == 0: # Check for Batch, Dim and non-empty feature dim
            logger.warning(f"ProjectionHead input has unexpected shape: {x.shape}. Expected [Batch, InDim > 0]. Returning input or raising error.")
            if x.shape[1] == 0 and x.ndim == 2: # Empty feature dim, but correct num dims
                 # This might happen if previous layer produced empty features.
                 # Need to return something of expected output shape if possible, or error.
                 # Returning zeros of out_dim to avoid crashing loss function if batch size > 0.
                 return torch.zeros(x.shape[0], self.head[-1].out_features, device=x.device) # head[-1] is the last Linear layer
            return x # Fallback, might cause issues later
        return self.head(x)