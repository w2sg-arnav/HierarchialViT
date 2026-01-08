# phase3_pretraining/dataset.py
"""Dataset module for phase3 SimCLR pretraining.

This module provides a specialized dataset class with built-in transforms
for SimCLR self-supervised pretraining.
"""

import logging
from typing import Tuple, Optional, List

import numpy as np
import torch
import torchvision.transforms.v2 as T_v2
from PIL import Image, UnidentifiedImageError

from shared.dataset import BaseSARCLD2024Dataset, SARCLD_CLASSES

# Import config dictionary for default values
try:
    from .config import config as phase3_default_config
except ImportError:
    print("PANIC (dataset.py): Could not import phase3_default_config. Using defaults.")
    phase3_default_config = {
        "seed": 42,
        "hvt_params_for_backbone": {"spectral_channels": 0},
        "pretrain_img_size": (224, 224),
        "original_dataset_name": "Original Dataset",
        "augmented_dataset_name": "Augmented Dataset",
    }

logger = logging.getLogger(__name__)


class SARCLD2024PretrainDataset(BaseSARCLD2024Dataset):
    """Dataset for SimCLR pretraining with built-in base transforms.
    
    This extends the base dataset with:
    - Built-in resize and tensor conversion transforms
    - Optional normalization for linear probing
    - Support for spectral channels (placeholder for future use)
    
    Args:
        root_dir: Path to the dataset root directory.
        img_size: Target image size as (height, width) tuple.
        split: Dataset split - 'train', 'val', 'validation', or 'test'.
        train_split_ratio: Fraction of data to use for training.
        normalize_for_model: If True, apply ImageNet normalization (for probing).
        use_spectral: Whether to use spectral channels (placeholder).
        spectral_channels: Number of spectral channels (placeholder).
        original_dataset_name: Name of the original dataset folder.
        augmented_dataset_name: Name of the augmented dataset folder.
        random_seed: Random seed for reproducible splits.
    """
    
    def __init__(
        self,
        root_dir: str,
        img_size: Tuple[int, int],
        split: str = "train",
        train_split_ratio: float = 0.8,
        normalize_for_model: bool = False,
        use_spectral: bool = False,
        spectral_channels: int = 0,
        original_dataset_name: str = phase3_default_config["original_dataset_name"],
        augmented_dataset_name: str = phase3_default_config["augmented_dataset_name"],
        random_seed: int = phase3_default_config["seed"],
    ):
        super().__init__(
            root_dir=root_dir,
            img_size=img_size,
            split=split,
            train_split_ratio=train_split_ratio,
            original_dataset_name=original_dataset_name,
            augmented_dataset_name=augmented_dataset_name,
            random_seed=random_seed,
        )
        
        self.normalize = normalize_for_model
        self.use_spectral = use_spectral and (spectral_channels > 0)
        self.spectral_channels = spectral_channels if self.use_spectral else 0
        
        # Build base transform pipeline
        rgb_transforms_list = [
            T_v2.Resize(self.img_size, interpolation=T_v2.InterpolationMode.BICUBIC, antialias=True),
            T_v2.ToImage(),
            T_v2.ToDtype(torch.float32, scale=True),
        ]
        
        if self.normalize:
            rgb_transforms_list.append(
                T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        
        self.base_transform_rgb = T_v2.Compose(rgb_transforms_list)
        
        logger.info(
            f"SARCLD2024PretrainDataset: split='{self.split}', "
            f"img_size={self.img_size}, normalize={self.normalize}"
        )
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample with built-in transforms.
        
        Args:
            idx: Index into the current split.
            
        Returns:
            Tuple of (image_tensor, label_tensor).
        """
        actual_idx = self.current_indices[idx]
        img_path = self.image_paths[actual_idx]
        label = self.labels[actual_idx]
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        try:
            img = self._load_image(img_path)
            rgb_tensor = self.base_transform_rgb(img)
            return rgb_tensor, label_tensor
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}. Returning placeholder.")
            dummy_rgb = torch.zeros((3, *self.img_size), dtype=torch.float32)
            return dummy_rgb, torch.tensor(-1, dtype=torch.long)


# Backward compatibility alias
SARCLD2024Dataset = SARCLD2024PretrainDataset


__all__ = [
    "SARCLD2024PretrainDataset",
    "SARCLD2024Dataset",
    "SARCLD_CLASSES",
]