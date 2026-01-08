# shared/dataset.py
"""Shared dataset utilities for SAR-CLD-2024 cotton leaf disease dataset.

This module provides the base dataset class used by both pretraining and finetuning phases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional, List, Callable

import numpy as np
from numpy.typing import NDArray
import torch
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Standard class names for SAR-CLD-2024 dataset
SARCLD_CLASSES = [
    "Bacterial Blight",
    "Curl Virus",
    "Healthy Leaf",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation",
]


class BaseSARCLD2024Dataset(Dataset[Tuple[torch.Tensor, int]]):
    """Base dataset class for SAR-CLD-2024 cotton leaf disease dataset.
    
    This provides common functionality for loading, splitting, and accessing
    images from the SAR-CLD-2024 dataset structure.
    
    Args:
        root_dir: Path to the dataset root directory.
        img_size: Target image size as (height, width) tuple.
        split: Dataset split - 'train', 'val', 'validation', or 'test'.
        train_split_ratio: Fraction of data to use for training (default 0.8).
        original_dataset_name: Name of the original dataset folder.
        augmented_dataset_name: Name of the augmented dataset folder.
        random_seed: Random seed for reproducible splits.
    """
    
    def __init__(
        self,
        root_dir: str,
        img_size: Tuple[int, int],
        split: str,
        train_split_ratio: float = 0.8,
        original_dataset_name: str = "Original Dataset",
        augmented_dataset_name: str = "Augmented Dataset",
        random_seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.img_size = tuple(img_size)
        self.split = split.lower()
        self.train_split_ratio = train_split_ratio
        self.random_seed = random_seed
        
        # Standard class configuration
        self.classes = SARCLD_CLASSES.copy()
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Image storage
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        
        logger.info(f"Initializing BaseSARCLD2024Dataset: split='{self.split}', img_size={self.img_size}")
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")
        
        # Scan dataset directories
        self._load_image_paths(original_dataset_name, augmented_dataset_name)
        
        if not self.image_paths:
            raise ValueError(f"No valid images found in {self.root_dir}")
        
        logger.info(f"Found {len(self.image_paths)} total images.")
        
        # Create train/val split
        self._create_split()
        
        # Cache for class weights
        self._class_weights: Optional[torch.Tensor] = None
    
    def _load_image_paths(
        self,
        original_dataset_name: str,
        augmented_dataset_name: str,
    ) -> None:
        """Load all image paths from dataset directories."""
        for dataset_name in [original_dataset_name, augmented_dataset_name]:
            if not dataset_name:
                continue
            
            dataset_path = self.root_dir / dataset_name
            if not dataset_path.is_dir():
                logger.debug(f"Dataset folder not found, skipping: {dataset_path}")
                continue
            
            logger.debug(f"Scanning dataset folder: {dataset_path}")
            
            for class_name in self.classes:
                class_folder = dataset_path / class_name
                if not class_folder.is_dir():
                    continue
                
                for item in class_folder.iterdir():
                    if item.is_file() and item.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        self.image_paths.append(str(item))
                        self.labels.append(self.class_to_idx[class_name])
    
    def _create_split(self) -> None:
        """Create train/validation split based on random seed."""
        indices = np.arange(len(self.image_paths))
        
        # Reproducible shuffle
        rng = np.random.default_rng(self.random_seed)
        rng.shuffle(indices)
        
        split_idx = int(len(indices) * self.train_split_ratio)
        
        if self.split == "train":
            self.current_indices = indices[:split_idx]
        elif self.split in ["val", "validation", "test"]:
            self.current_indices = indices[split_idx:]
        else:
            raise ValueError(f"Invalid split name: '{self.split}'. "
                           f"Expected 'train', 'val', 'validation', or 'test'.")
        
        self.current_split_labels = np.array(self.labels)[self.current_indices]
        logger.info(f"Split '{self.split}' contains {len(self.current_indices)} samples.")
    
    def __len__(self) -> int:
        return len(self.current_indices)
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return self.classes.copy()
    
    def get_targets(self) -> NDArray[np.int64]:
        """Get labels for current split."""
        return self.current_split_labels
    
    def get_class_weights(self) -> Optional[torch.Tensor]:
        """Compute inverse frequency class weights for handling imbalanced data."""
        if self._class_weights is None:
            if len(self.current_split_labels) == 0:
                return None
            
            class_counts = Counter(self.current_split_labels)
            weights = torch.ones(self.num_classes, dtype=torch.float32)
            
            for i in range(self.num_classes):
                count = class_counts.get(i, 0)
                if count > 0:
                    weights[i] = len(self.current_split_labels) / (self.num_classes * count)
            
            self._class_weights = weights
            logger.info(f"Computed class weights: {weights.numpy().round(3)}")
        
        return self._class_weights
    
    def _load_image(self, path: str) -> Image.Image:
        """Load an image from path, converting to RGB."""
        try:
            return Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            logger.error(f"Failed to load image {path}: {e}")
            raise


class SARCLD2024Dataset(BaseSARCLD2024Dataset):
    """Standard SAR-CLD-2024 dataset with configurable transforms.
    
    This is the recommended dataset class for finetuning workflows.
    
    Args:
        root_dir: Path to the dataset root directory.
        img_size: Target image size as (height, width) tuple.
        split: Dataset split - 'train', 'val', 'validation', or 'test'.
        transform: Optional transform to apply to images.
        train_split_ratio: Fraction of data to use for training.
        original_dataset_name: Name of the original dataset folder.
        augmented_dataset_name: Name of the augmented dataset folder.
        random_seed: Random seed for reproducible splits.
    """
    
    def __init__(
        self,
        root_dir: str,
        img_size: Tuple[int, int],
        split: str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        train_split_ratio: float = 0.8,
        original_dataset_name: str = "Original Dataset",
        augmented_dataset_name: str = "Augmented Dataset",
        random_seed: int = 42,
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
        self.transform: Optional[Callable[[Image.Image], torch.Tensor]] = transform
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample.
        
        Args:
            idx: Index into the current split.
            
        Returns:
            Tuple of (image_tensor, label).
        """
        actual_idx: int = self.current_indices[idx]
        img_path: str = self.image_paths[actual_idx]
        label: int = self.labels[actual_idx]
        
        try:
            img = self._load_image(img_path)
            
            if self.transform is not None:
                img_tensor: torch.Tensor = self.transform(img)
                return img_tensor, label
            
            # If no transform, we need to convert PIL Image to tensor
            # This shouldn't happen in normal usage as transforms are expected
            raise ValueError("No transform provided - cannot return raw PIL Image as Tensor")
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}. Returning dummy.")
            dummy_tensor = torch.zeros((3, *self.img_size), dtype=torch.float32)
            return dummy_tensor, -1


__all__ = [
    "SARCLD_CLASSES",
    "BaseSARCLD2024Dataset",
    "SARCLD2024Dataset",
]
