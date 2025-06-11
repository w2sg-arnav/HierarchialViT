# phase4_finetuning/utils/augmentations.py

from typing import Tuple, Dict, List
import torch
import torchvision.transforms.v2 as T_v2
import torchvision.transforms.functional as TF
from PIL import Image
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)


# ===================================================================
# == Custom Augmentation Modules
# ===================================================================

class EnhancedColorJitter(torch.nn.Module):
    """
    Applies standard color jitter plus simulates disease-like symptoms
    such as chlorosis (yellowing) and necrosis (browning/dying tissue).
    Expects a float32 Tensor with values in [0.0, 1.0].
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        super().__init__()
        self.base_jitter = T_v2.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.chlorosis_prob = 0.2
        self.necrosis_prob = 0.15

    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        if img_batch.dtype != torch.float32:
            raise TypeError(f"EnhancedColorJitter expects a float32 tensor, but got {img_batch.dtype}")

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
        r, g, b = img[0], img[1], img[2]
        new_g = g * random.uniform(0.7, 0.9)
        new_r = torch.clamp(r * random.uniform(1.0, 1.1) + (g - new_g) * 0.5, 0, 1)
        return torch.stack([new_r, torch.clamp(new_g, 0, 1), b * random.uniform(0.8, 1.0)], dim=0)

    def _simulate_necrosis(self, img: torch.Tensor) -> torch.Tensor:
        brown_factor = random.uniform(0.1, 0.3)
        desat_factor = random.uniform(0.6, 0.9)
        gray = TF.rgb_to_grayscale(img.unsqueeze(0), num_output_channels=3).squeeze(0)
        r, g, b = img[0], img[1], img[2]
        new_r = torch.clamp(r * (1 + brown_factor * 0.5) + g * brown_factor * 0.2, 0, 1)
        new_g = torch.clamp(g * (1 - brown_factor * 0.3), 0, 1)
        new_b = torch.clamp(b * (1 - brown_factor * 0.6), 0, 1)
        img_mod = torch.stack([new_r, new_g, new_b], dim=0)
        return torch.clamp(img_mod * desat_factor + gray * (1 - desat_factor), 0, 1)


class LightingVariation(torch.nn.Module):
    """ Simulates over/underexposure and uneven lighting gradients on Tensors. """
    def __init__(self):
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
                gradient = torch.linspace(random.uniform(0.7, 0.9), random.uniform(1.1, 1.3), w, device=img.device).view(1, 1, -1)
            else: # uneven_v
                gradient = torch.linspace(random.uniform(0.7, 0.9), random.uniform(1.1, 1.3), h, device=img.device).view(1, -1, 1)
            img_mod = torch.clamp(img * gradient, 0, 1)
        return img_mod


class GaussianNoise(torch.nn.Module):
    """ Adds Gaussian noise to a Tensor. """
    def __init__(self, std=0.02):
        super().__init__()
        self.std = std
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(img) * self.std
        return torch.clamp(img + noise, 0, 1)


# ===================================================================
# == Main Augmentation Class and Factory
# ===================================================================

class CottonLeafDiseaseAugmentation:
    """
    Builds a complete augmentation pipeline for training.
    It takes a PIL image and returns a transformed, normalized tensor.
    """
    def __init__(self, img_size: Tuple[int, int], severity: str = 'moderate'):
        self.img_size = img_size
        self.severity = severity
        severity_configs = {
            'mild':     {'rot': 15, 'scale': (0.8, 1.2), 'blur_p': 0.2, 'erase_p': 0.1, 'light_p': 0.4},
            'moderate': {'rot': 25, 'scale': (0.7, 1.3), 'blur_p': 0.25,'erase_p': 0.15,'light_p': 0.5},
            'strong':   {'rot': 35, 'scale': (0.6, 1.4), 'blur_p': 0.3, 'erase_p': 0.2, 'light_p': 0.6}
        }
        params = severity_configs.get(severity, severity_configs['moderate'])
        self.transform_pipeline = T_v2.Compose([
            T_v2.RandomResizedCrop(size=img_size, scale=params['scale'], ratio=(0.75, 1.33), interpolation=T_v2.InterpolationMode.BICUBIC, antialias=True),
            T_v2.RandomHorizontalFlip(p=0.5),
            T_v2.RandomVerticalFlip(p=0.5),
            T_v2.RandomRotation(degrees=params['rot']),
            T_v2.ToImage(),
            T_v2.ToDtype(torch.float32, scale=True),
            EnhancedColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.15),
            T_v2.RandomApply([LightingVariation()], p=params['light_p']),
            T_v2.RandomApply([T_v2.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 1.5))], p=params['blur_p']),
            T_v2.RandomApply([GaussianNoise(std=0.02)], p=0.3),
            T_v2.RandomErasing(p=params['erase_p'], scale=(0.02, 0.12), ratio=(0.3, 3.0), value=0),
            T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info(f"CottonLeafDiseaseAugmentation initialized with severity '{severity}'.")

    def __call__(self, pil_image: Image.Image) -> torch.Tensor:
        return self.transform_pipeline(pil_image)


def create_cotton_leaf_augmentation(
    strategy: str,
    img_size: Tuple[int, int],
    **kwargs
) -> T_v2.Compose:
    """
    Factory function to create the appropriate augmentation pipeline.
    """
    logger.info(f"Creating augmentation strategy: '{strategy}' for image size {img_size}")

    if strategy == 'minimal':
        return T_v2.Compose([
            T_v2.Resize(img_size, interpolation=T_v2.InterpolationMode.BICUBIC, antialias=True),
            T_v2.CenterCrop(img_size),
            T_v2.ToImage(),
            T_v2.ToDtype(torch.float32, scale=True),
            T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif strategy == 'cotton_disease' or strategy == 'aggressive_medical':
        severity = kwargs.get('severity', 'moderate')
        if strategy == 'aggressive_medical':
            severity_map = {'mild': 'mild', 'moderate': 'moderate', 'high': 'strong'}
            severity = severity_map.get(kwargs.get('severity', 'high'), 'strong')
        
        return CottonLeafDiseaseAugmentation(
            img_size=img_size,
            severity=severity
        )
        # ^^^ The extra parenthesis was here. It has been removed. ^^^

    else:
        logger.error(f"Strategy '{strategy}' is unknown. Please use 'minimal' or 'cotton_disease'.")
        raise ValueError(f"Unknown augmentation strategy: {strategy}")