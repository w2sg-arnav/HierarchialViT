# phase3_pretraining/utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07): # Default temperature
        super().__init__()
        self.temperature = temperature
        logger.info(f"InfoNCELoss initialized with temperature: {self.temperature}")

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        if not (torch.is_tensor(features1) and torch.is_tensor(features2)):
            raise TypeError("Inputs must be PyTorch tensors.")
        if features1.shape != features2.shape:
            raise ValueError(f"Input feature tensors must have the same shape. Got {features1.shape} and {features2.shape}")
        if features1.ndim != 2:
            raise ValueError(f"Input features should be 2D tensors (Batch, Dim). Got {features1.ndim} dims.")
        if features1.shape[0] == 0 : # Handle empty batch
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
            logger.warning(f"InfoNCELoss: Non-finite loss: {loss.item()}. Temp: {self.temperature}. SimMat min/max: {sim_matrix.min().item()}/{sim_matrix.max().item()}")
        return loss