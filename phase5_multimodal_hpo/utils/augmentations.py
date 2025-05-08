# phase5_multimodal_hpo/utils/augmentations.py
import torch
import torchvision.transforms.v2 as T_v2 # Use v2
import logging
from typing import Tuple # Import Tuple

logger = logging.getLogger(__name__)

class FinetuneAugmentation:
    """ 
    Augmentations suitable for fine-tuning supervised models during Phase 5. 
    Applies augmentations primarily to the RGB modality.
    """
    def __init__(self, img_size: Tuple[int, int]):
        self.img_size = img_size # Target size for reference/consistency
        
        # Define the sequence of augmentations
        # Adjust parameters based on experimentation or HPO results
        self.transform = T_v2.Compose([
            T_v2.RandomHorizontalFlip(p=0.5),
            T_v2.RandomVerticalFlip(p=0.5),
            T_v2.RandomRotation(degrees=20), 
            T_v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), 
            T_v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            T_v2.RandomApply([ 
                T_v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)) # Kernel must be odd
            ], p=0.2),
            # Note: Normalization should be part of the dataset's base transform usually
        ])
        logger.info(f"FinetuneAugmentation initialized for img_size {img_size}.")

    def __call__(self, rgb_batch: torch.Tensor) -> torch.Tensor:
        """ 
        Apply augmentations to a batch of RGB image tensors. 
        
        Args:
            rgb_batch (torch.Tensor): Batch of RGB tensors [B, C, H, W].
            
        Returns:
            torch.Tensor: Augmented batch of RGB tensors.
        """
        # logger.debug(f"Applying fine-tune augmentations to RGB batch shape {rgb_batch.shape}")
        # torchvision v2 transforms generally support batch inputs directly.
        return self.transform(rgb_batch)