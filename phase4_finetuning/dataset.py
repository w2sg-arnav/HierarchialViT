# phase4_finetuning/dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import numpy as np
import logging
from collections import Counter
from pathlib import Path
from typing import Tuple, Optional, List, Callable

logger = logging.getLogger(__name__)

class SARCLD2024Dataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 img_size: tuple,
                 split: str,
                 transform: Optional[Callable],
                 train_split_ratio: float,
                 original_dataset_name: str,
                 augmented_dataset_name: str,
                 random_seed: int):

        self.root_dir = Path(root_dir)
        self.img_size = tuple(img_size)
        self.split = split.lower()
        self.transform = transform
        self.train_split_ratio = train_split_ratio
        self.random_seed = random_seed
        self.classes = ["Bacterial Blight", "Curl Virus", "Healthy Leaf", "Herbicide Growth Damage", "Leaf Hopper Jassids", "Leaf Redding", "Leaf Variegation"]
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.image_paths: List[str] = []
        self.labels: List[int] = []

        logger.info(f"Initializing Dataset for split '{self.split}'...")
        if not self.root_dir.exists(): raise FileNotFoundError(f"Dataset root missing: {self.root_dir}")

        for dataset_name in [original_dataset_name, augmented_dataset_name]:
            if not dataset_name: continue
            dataset_path = self.root_dir / dataset_name
            if not dataset_path.is_dir(): continue
            for class_name in self.classes:
                class_folder = dataset_path / class_name
                if not class_folder.is_dir(): continue
                for item in class_folder.iterdir():
                    if item.is_file() and item.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        self.image_paths.append(str(item))
                        self.labels.append(self.class_to_idx[class_name])
        
        if not self.image_paths: raise ValueError("No valid images found.")
        logger.info(f"Found {len(self.image_paths)} total image paths.")
        
        indices = np.arange(len(self.image_paths))
        np.random.seed(self.random_seed); np.random.shuffle(indices)
        split_idx = int(len(indices) * self.train_split_ratio)

        if self.split == "train": self.current_indices = indices[:split_idx]
        elif self.split in ["val", "validation", "test"]: self.current_indices = indices[split_idx:]
        else: raise ValueError(f"Invalid split name '{self.split}'")
        
        self.current_split_labels = np.array(self.labels)[self.current_indices]
        logger.info(f"Split '{self.split}' size: {len(self.current_indices)} samples.")
        self.class_weights_computed = None

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        actual_idx = self.current_indices[idx]
        img_path = self.image_paths[actual_idx]
        label = self.labels[actual_idx]
        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            logger.error(f"Error loading/transforming image {img_path}: {e}. Returning dummy.")
            dummy_tensor = torch.zeros((3, *self.img_size), dtype=torch.float32)
            return dummy_tensor, -1

    def get_class_names(self) -> List[str]: return self.classes
    def get_targets(self) -> np.ndarray: return self.current_split_labels
    
    def get_class_weights(self) -> Optional[torch.Tensor]:
        if self.class_weights_computed is None:
            if len(self.current_split_labels) == 0: return None
            class_counts = Counter(self.current_split_labels)
            weights = torch.ones(self.num_classes, dtype=torch.float)
            for i in range(self.num_classes):
                count = class_counts.get(i, 0)
                if count > 0: weights[i] = len(self.current_split_labels) / (self.num_classes * count)
            self.class_weights_computed = weights
            logger.info(f"Computed class weights for split '{self.split}': {self.class_weights_computed.numpy().round(3)}")
        return self.class_weights_computed