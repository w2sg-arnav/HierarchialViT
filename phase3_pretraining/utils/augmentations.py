# phase3_pretraining/utils/augmentations.py
from typing import Optional, Union, Tuple
import torch
import torchvision.transforms.v2 as T_v2 # Use transforms.v2
import logging

logger = logging.getLogger(__name__)

class SimCLRAugmentation:
    """
    SimCLR style augmentations using torchvision.transforms.v2.
    Applies two random views to an input image tensor.
    Expects input tensor to be in [0, 1] range if not normalized.
    """
    def __init__(self, img_size: tuple, 
                 s: float = 1.0, # Strength of color jitter
                 p_grayscale: float = 0.2,
                 p_gaussian_blur: float = 0.5 
                 ):
        self.img_size = img_size
        
        # Color jitter parameters (s is strength)
        # For brightness, contrast, saturation, jitter by up to s*100% from default.
        # For hue, jitter by up to 0.5*s radians (or 0.5*s*180/pi degrees).
        # Values from SimCLR paper: brightness=0.8s, contrast=0.8s, saturation=0.8s, hue=0.2s.
        # If s=1, brightness_factor = 0.8, contrast_factor = 0.8, etc.
        # torchvision brightness: [max(0, 1-factor), 1+factor]
        # torchvision hue: [-factor, factor] where factor <=0.5
        
        self.transform = T_v2.Compose([
            T_v2.RandomResizedCrop(size=img_size, scale=(0.2, 1.0), antialias=True), # SimCLR uses 0.08-1.0
            T_v2.RandomHorizontalFlip(p=0.5),
            T_v2.RandomApply([
                T_v2.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)
            ], p=0.8),
            T_v2.RandomGrayscale(p=p_grayscale),
            T_v2.RandomApply([
                T_v2.GaussianBlur(kernel_size=img_size[0]//10*2+1 | 1, sigma=(0.1, 2.0)) # kernel size must be odd
            ], p=p_gaussian_blur),
            # Normalization should be done AFTER all augmentations if it's standard ImageNet norm
            # If model is pretrained with specific norm, apply it here.
            # For self-supervised, often no normalization or a dataset-specific one is used.
            # T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        logger.info(f"SimCLRAugmentation initialized for img_size {img_size}.")

    def __call__(self, x_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_batch (torch.Tensor): Batch of images, shape [B, C, H, W].
                                    Expected to be on the correct device.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Two augmented views of the batch.
        """
        # logger.debug(f"Augmenting batch of shape {x_batch.shape} on device {x_batch.device}")
        view1 = self.transform(x_batch)
        view2 = self.transform(x_batch)
        return view1, view2

from typing import Tuple # Add for Tuple type hint