# phase4_finetuning/dataset.py
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T_v2
from PIL import Image, UnidentifiedImageError
import numpy as np
import logging
from collections import Counter
from pathlib import Path # Use pathlib
from typing import Tuple, Optional, List

# Import the main config dictionary for default values
try:
    from phase4_finetuning.config import config as global_finetune_config
except ImportError:
    print("PANIC (phase4_dataset.py): Could not import global_finetune_config. Hardcoding critical defaults.")
    global_finetune_config = { # Minimal fallback
        "seed": 42,
        "original_dataset_name": "Original Dataset",
        "augmented_dataset_name": "Augmented Dataset",
        "num_classes": 7,
    }

logger = logging.getLogger(__name__)

class SARCLD2024Dataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 img_size: tuple,
                 split: str = "train",
                 train_split_ratio: float = 0.8,
                 normalize_for_model: bool = True,
                 # Use global config for these if not overridden by direct args (though usually they are from main.py's config)
                 original_dataset_name: str = global_finetune_config.get('original_dataset_name',"Original Dataset"),
                 augmented_dataset_name: str = global_finetune_config.get('augmented_dataset_name', "Augmented Dataset"),
                 random_seed: int = global_finetune_config.get('seed', 42)
                 ):
        self.root_dir = Path(root_dir) # Use pathlib
        self.img_size = img_size
        self.split = split.lower()
        self.train_split_ratio = train_split_ratio
        self.normalize = normalize_for_model

        self.classes = [ # Consider getting this from config too if it can change
            "Bacterial Blight", "Curl Virus", "Healthy Leaf",
            "Herbicide Growth Damage", "Leaf Hopper Jassids",
            "Leaf Redding", "Leaf Variegation"
        ]
        self.num_classes = len(self.classes) # Or from global_finetune_config['num_classes']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.image_paths: List[str] = []
        self.labels: List[int] = []

        logger.info(f"[DATASET INIT - {self.split}] Root: {self.root_dir}, ImgSize: {self.img_size}")
        if not self.root_dir.exists():
            logger.error(f"FATAL [DATASET INIT - {self.split}] Dataset root missing: {self.root_dir}")
            raise FileNotFoundError(f"Dataset root missing: {self.root_dir}")

        items_scanned = 0
        for dataset_type_name in [original_dataset_name, augmented_dataset_name]:
            dataset_path = self.root_dir / dataset_type_name
            if not dataset_path.is_dir():
                logger.debug(f"[DATASET INIT - {self.split}] Path not found, skipping: {dataset_path}")
                continue
            logger.info(f"[DATASET INIT - {self.split}] Scanning: {dataset_path}")
            for class_name in self.classes:
                class_folder_path = dataset_path / class_name
                if not class_folder_path.is_dir(): continue
                label = self.class_to_idx[class_name]
                valid_extensions = (".jpg", ".jpeg", ".png"); n_skipped = 0
                try:
                    for item in class_folder_path.iterdir():
                        items_scanned +=1
                        if item.is_file() and item.suffix.lower() in valid_extensions:
                            try:
                                # Basic check for file readability before adding to list
                                # Defer full Image.open() to __getitem__ to avoid loading all at init
                                if os.path.getsize(item) > 0: # Simple check for empty file
                                    self.image_paths.append(str(item))
                                    self.labels.append(label)
                                else:
                                    logger.warning(f"Skipping empty file: {item}")
                                    n_skipped +=1
                            except OSError: # handles getsize error on broken symlinks etc.
                                logger.warning(f"Skipping file due to OSError (e.g. broken symlink): {item}")
                                n_skipped +=1
                except OSError as os_err: logger.error(f"OS error scanning folder {class_folder_path}: {os_err}")
                if n_skipped > 0: logger.warning(f"Skipped {n_skipped} files (empty/OS error) in {class_folder_path}")

        if not self.image_paths:
            logger.error(f"FATAL [DATASET INIT - {self.split}] No valid image paths found in {self.root_dir}.")
            raise ValueError(f"No valid images found. Please check dataset structure in {self.root_dir}.")

        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        logger.info(f"[DATASET INIT - {self.split}] Total valid image paths collected: {len(self.image_paths)} from {items_scanned} items scanned.")

        # Splitting logic
        indices = np.arange(len(self.image_paths))
        np.random.seed(random_seed) # Use consistent seed for splitting
        np.random.shuffle(indices)
        split_idx = int(len(indices) * self.train_split_ratio)

        if self.split == "train": self.current_indices = indices[:split_idx]
        elif self.split in ["val", "validation", "test"]: self.current_indices = indices[split_idx:]
        else: logger.error(f"Invalid split name '{self.split}'."); raise ValueError(f"Invalid split name '{self.split}'")
        
        self.current_split_labels = self.labels[self.current_indices]
        logger.info(f"[DATASET INIT - {self.split}] Dataset split size: {len(self.current_indices)} samples.")
        self.class_weights_computed = None # Initialize

        # Base transforms (applied in __getitem__)
        transforms_list = [
            T_v2.ToImage(), # PIL to Tensor (scales to [0,1])
            T_v2.ToDtype(torch.float32, scale=False), # Ensure float32, scale=False as ToImage did it
            T_v2.Resize(self.img_size, interpolation=T_v2.InterpolationMode.BICUBIC, antialias=True),
        ]
        if self.normalize:
             transforms_list.append(T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.base_transform = T_v2.Compose(transforms_list)
        logger.info(f"[DATASET INIT - {self.split}] Base RGB Transforms: {self.base_transform}")


    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        actual_idx = self.current_indices[idx]
        img_path = self.image_paths[actual_idx]
        label = self.labels[actual_idx]
        
        try:
            img = Image.open(img_path).convert("RGB")
            # img.load() # Force loading, good for catching errors early if not too slow
        except Exception as e:
            logger.error(f"Error loading image {img_path} at index {idx} (actual index {actual_idx}): {e}. Returning dummy.")
            # Fallback to a dummy tensor to avoid crashing the DataLoader
            dummy_tensor = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            if self.normalize: # Apply normalization to dummy if it's part of pipeline
                normalizer = T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                dummy_tensor = normalizer(dummy_tensor)
            return dummy_tensor, torch.tensor(-1, dtype=torch.long) # Dummy error label

        # Apply base transformations (Resize, ToTensor, Normalize)
        # Augmentations (like RandomFlip, ColorJitter) are applied in the Trainer
        try:
            rgb_tensor = self.base_transform(img)
        except Exception as e_transform:
            logger.error(f"Error transforming image {img_path}: {e_transform}. Returning dummy.")
            dummy_tensor = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            if self.normalize:
                normalizer = T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                dummy_tensor = normalizer(dummy_tensor)
            return dummy_tensor, torch.tensor(-1, dtype=torch.long)

        return rgb_tensor, torch.tensor(label, dtype=torch.long)

    def get_class_names(self) -> List[str]:
        return self.classes

    def get_class_weights(self) -> Optional[torch.Tensor]:
        if self.class_weights_computed is None:
            if len(self.current_split_labels) == 0:
                logger.warning(f"Cannot compute class weights: split '{self.split}' has no labels.")
                return None
            class_counts = Counter(self.current_split_labels)
            num_classes_in_dataset = self.num_classes # Use the defined number of classes

            weights = torch.ones(num_classes_in_dataset, dtype=torch.float) # Default to 1.0
            for i in range(num_classes_in_dataset):
                count = class_counts.get(i, 0)
                if count > 0:
                    # Inverse frequency weighting
                    weights[i] = len(self.current_split_labels) / (num_classes_in_dataset * count)
                else:
                    logger.warning(f"Class '{self.classes[i] if i < len(self.classes) else i}' not found in split '{self.split}'. Weight remains 1.0 (or handle differently).")
            self.class_weights_computed = weights / weights.sum() # Normalize weights to sum to 1
            logger.info(f"Computed class weights for split '{self.split}': {self.class_weights_computed.numpy().round(3)}")
        return self.class_weights_computed