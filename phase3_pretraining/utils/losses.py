# phase3_pretraining/utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# TEMPERATURE will be imported from config in the main script
# from config import TEMPERATURE # Avoid direct import here if passed as arg

logger = logging.getLogger(__name__)

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        logger.info(f"InfoNCELoss initialized with temperature: {self.temperature}")

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features1 (torch.Tensor): Projections from the first augmented view [Batch, ProjDim].
            features2 (torch.Tensor): Projections from the second augmented view [Batch, ProjDim].
        Returns:
            torch.Tensor: Scalar InfoNCE loss.
        """
        if not torch.is_tensor(features1) or not torch.is_tensor(features2):
            raise TypeError("Inputs to InfoNCELoss must be PyTorch tensors.")
        if features1.shape != features2.shape:
            raise ValueError("Input feature tensors must have the same shape.")
        if features1.ndim != 2:
            raise ValueError("Input features should be 2D tensors (Batch, Dim).")

        # Normalize features
        z_i = F.normalize(features1, p=2, dim=1)
        z_j = F.normalize(features2, p=2, dim=1)

        batch_size = z_i.shape[0]

        # Similarity matrix (cosine similarity)
        # sim_matrix[i, j] is the similarity between z_i[i] and z_j[j]
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature # Shape: (Batch, Batch)

        # Create labels: positive pairs are (i, i)
        labels = torch.arange(batch_size, device=z_i.device, dtype=torch.long)

        # Loss calculation:
        # For each z_i[k], the positive pair is z_j[k].
        # CrossEntropyLoss(sim_matrix_row_k, label_k)
        # Equivalent to calculating loss for (z_i @ z_j.T) and (z_j @ z_i.T) separately and averaging.
        
        loss_ij = F.cross_entropy(sim_matrix, labels) # Loss for z_i predicting z_j
        loss_ji = F.cross_entropy(sim_matrix.T, labels) # Loss for z_j predicting z_i
        
        loss = (loss_ij + loss_ji) / 2.0

        if not torch.isfinite(loss):
            logger.warning(f"InfoNCELoss: Non-finite loss detected: {loss.item()}. Sim matrix min/max: {sim_matrix.min()}/{sim_matrix.max()}")
            # Potentially add more debugging here if NaNs occur (e.g., check temperature, feature norms)

        return loss