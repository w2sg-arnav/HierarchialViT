# hvit/data/augmentations.py
"""Data augmentation pipelines for training and inference.

This module provides:
- SimCLRAugmentation: Contrastive learning augmentations for SSL pre-training
- Disease-specific augmentations for fine-tuning
- Factory functions for creating standard transform pipelines
"""

from typing import Tuple, Optional
import random

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T_v2
import torchvision.transforms.functional as TF
from PIL import Image
import logging

logger = logging.getLogger(__name__)


# ===================================================================
# Pre-training Augmentations (SimCLR)
# ===================================================================

class SimCLRAugmentation:
    """SimCLR augmentation pipeline for self-supervised pre-training.
    
    Generates two randomly augmented views of each input image for
    contrastive learning.
    
    Args:
        img_size: Target image size as (height, width).
        s: Color jitter strength multiplier.
        p_grayscale: Probability of grayscale conversion.
        p_gaussian_blur: Probability of Gaussian blur.
        rrc_scale_min: Minimum scale for random resized crop.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int],
        s: float = 1.0,
        p_grayscale: float = 0.2,
        p_gaussian_blur: float = 0.5,
        rrc_scale_min: float = 0.08
    ) -> None:
        self.img_size = img_size

        logger.info(
            f"SimCLRAugmentation: img_size={img_size}, s={s}, "
            f"p_gray={p_grayscale}, p_blur={p_gaussian_blur}, rrc_min_scale={rrc_scale_min}"
        )

        blur_kernel_val = max(3, (self.img_size[0] // 20) * 2 + 1)

        self.transform = T_v2.Compose([
            T_v2.RandomResizedCrop(size=img_size, scale=(rrc_scale_min, 1.0), antialias=True),
            T_v2.RandomHorizontalFlip(p=0.5),
            T_v2.RandomApply([
                T_v2.ColorJitter(
                    brightness=0.8 * s,
                    contrast=0.8 * s,
                    saturation=0.8 * s,
                    hue=0.2 * s
                )
            ], p=0.8),
            T_v2.RandomGrayscale(p=p_grayscale),
            T_v2.RandomApply([
                T_v2.GaussianBlur(kernel_size=blur_kernel_val, sigma=(0.1, 2.0))
            ], p=p_gaussian_blur),
        ])

    def __call__(self, x_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate two augmented views.
        
        Args:
            x_batch: Input tensor batch.
        
        Returns:
            Tuple of two augmented view tensors.
        """
        view1 = self.transform(x_batch)
        view2 = self.transform(x_batch)
        return view1, view2


# ===================================================================
# Fine-tuning Augmentations (Disease-specific)
# ===================================================================

class EnhancedColorJitter(nn.Module):
    """Enhanced color jitter with disease symptom simulation.
    
    Applies standard color jitter plus simulates disease-like symptoms
    such as chlorosis (yellowing) and necrosis (browning/dying tissue).
    
    Args:
        brightness: Brightness jitter factor.
        contrast: Contrast jitter factor.
        saturation: Saturation jitter factor.
        hue: Hue jitter factor.
    
    Note:
        Expects a float32 Tensor with values in [0.0, 1.0].
    """
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1
    ) -> None:
        super().__init__()
        self.base_jitter = T_v2.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        self.chlorosis_prob = 0.2
        self.necrosis_prob = 0.15

    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        if img_batch.dtype != torch.float32:
            raise TypeError(
                f"EnhancedColorJitter expects a float32 tensor, but got {img_batch.dtype}"
            )

        is_batch = img_batch.ndim == 4
        if not is_batch:
            img_batch = img_batch.unsqueeze(0)

        img_batch = self.base_jitter(img_batch)

        for i in range(img_batch.shape[0]):
            img = img_batch[i]
            if random.random() < self.chlorosis_prob:
                img = self._simulate_chlorosis(img)
            if random.random() < self.necrosis_prob:
                img = self._simulate_necrosis(img)
            img_batch[i] = img

        return img_batch if is_batch else img_batch.squeeze(0)

    def _simulate_chlorosis(self, img: torch.Tensor) -> torch.Tensor:
        """Simulate chlorosis (yellowing) effect."""
        r, g, b = img[0], img[1], img[2]
        new_g = g * random.uniform(0.7, 0.9)
        new_r = torch.clamp(
            r * random.uniform(1.0, 1.1) + (g - new_g) * 0.5, 0, 1
        )
        return torch.stack([
            new_r,
            torch.clamp(new_g, 0, 1),
            b * random.uniform(0.8, 1.0)
        ], dim=0)

    def _simulate_necrosis(self, img: torch.Tensor) -> torch.Tensor:
        """Simulate necrosis (browning) effect."""
        brown_factor = random.uniform(0.1, 0.3)
        desat_factor = random.uniform(0.6, 0.9)
        gray = TF.rgb_to_grayscale(img.unsqueeze(0), num_output_channels=3).squeeze(0)
        r, g, b = img[0], img[1], img[2]
        new_r = torch.clamp(r * (1 + brown_factor * 0.5) + g * brown_factor * 0.2, 0, 1)
        new_g = torch.clamp(g * (1 - brown_factor * 0.3), 0, 1)
        new_b = torch.clamp(b * (1 - brown_factor * 0.6), 0, 1)
        img_mod = torch.stack([new_r, new_g, new_b], dim=0)
        return torch.clamp(img_mod * desat_factor + gray * (1 - desat_factor), 0, 1)


class LightingVariation(nn.Module):
    """Simulates various lighting conditions.
    
    Applies random over/underexposure and uneven lighting gradients.
    """
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        variation_type = random.choice(['overexposure', 'underexposure', 'uneven_h', 'uneven_v'])
        
        if variation_type == 'overexposure':
            factor = random.uniform(1.1, 1.4)
            img_mod = torch.clamp(img * factor, 0, 1)
        elif variation_type == 'underexposure':
            factor = random.uniform(0.6, 0.9)
            img_mod = img * factor
        else:
            h, w = img.shape[-2:]
            if variation_type == 'uneven_h':
                gradient = torch.linspace(
                    random.uniform(0.7, 0.9),
                    random.uniform(1.1, 1.3),
                    w,
                    device=img.device
                ).view(1, 1, -1)
            else:  # uneven_v
                gradient = torch.linspace(
                    random.uniform(0.7, 0.9),
                    random.uniform(1.1, 1.3),
                    h,
                    device=img.device
                ).view(1, -1, 1)
            img_mod = torch.clamp(img * gradient, 0, 1)
        
        return img_mod


class GaussianNoise(nn.Module):
    """Adds Gaussian noise to tensors.
    
    Args:
        std: Standard deviation of the noise.
    """
    
    def __init__(self, std: float = 0.02) -> None:
        super().__init__()
        self.std = std

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(img) * self.std
        return torch.clamp(img + noise, 0, 1)


# ===================================================================
# Factory Functions
# ===================================================================

def get_train_transforms(
    img_size: Tuple[int, int],
    severity: str = 'moderate'
) -> T_v2.Compose:
    """Create training transform pipeline with disease-specific augmentations.
    
    Args:
        img_size: Target image size as (height, width).
        severity: Augmentation severity ('mild', 'moderate', or 'strong').
    
    Returns:
        Composed transform pipeline.
    """
    severity_configs = {
        'mild': {'rot': 15, 'scale': (0.8, 1.2), 'blur_p': 0.2, 'erase_p': 0.1, 'light_p': 0.4},
        'moderate': {'rot': 25, 'scale': (0.7, 1.3), 'blur_p': 0.25, 'erase_p': 0.15, 'light_p': 0.5},
        'strong': {'rot': 35, 'scale': (0.6, 1.4), 'blur_p': 0.3, 'erase_p': 0.2, 'light_p': 0.6}
    }
    params = severity_configs.get(severity, severity_configs['moderate'])
    
    logger.info(f"Creating train transforms with severity '{severity}'")
    
    return T_v2.Compose([
        T_v2.RandomResizedCrop(
            size=img_size,
            scale=params['scale'],
            ratio=(0.75, 1.33),
            interpolation=T_v2.InterpolationMode.BICUBIC,
            antialias=True
        ),
        T_v2.RandomHorizontalFlip(p=0.5),
        T_v2.RandomVerticalFlip(p=0.5),
        T_v2.RandomRotation(degrees=params['rot']),
        T_v2.ToImage(),
        T_v2.ToDtype(torch.float32, scale=True),
        EnhancedColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.15),
        T_v2.RandomApply([LightingVariation()], p=params['light_p']),
        T_v2.RandomApply([
            T_v2.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 1.5))
        ], p=params['blur_p']),
        T_v2.RandomApply([GaussianNoise(std=0.02)], p=0.3),
        T_v2.RandomErasing(
            p=params['erase_p'],
            scale=(0.02, 0.12),
            ratio=(0.3, 3.0),
            value=0
        ),
        T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(img_size: Tuple[int, int]) -> T_v2.Compose:
    """Create validation/inference transform pipeline.
    
    Args:
        img_size: Target image size as (height, width).
    
    Returns:
        Composed transform pipeline.
    """
    logger.info(f"Creating validation transforms for size {img_size}")
    
    return T_v2.Compose([
        T_v2.Resize(img_size, interpolation=T_v2.InterpolationMode.BICUBIC, antialias=True),
        T_v2.CenterCrop(img_size),
        T_v2.ToImage(),
        T_v2.ToDtype(torch.float32, scale=True),
        T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


__all__ = [
    # Pre-training
    "SimCLRAugmentation",
    # Fine-tuning components
    "EnhancedColorJitter",
    "LightingVariation",
    "GaussianNoise",
    # Factory functions
    "get_train_transforms",
    "get_val_transforms",
]
