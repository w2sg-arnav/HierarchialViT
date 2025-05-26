# phase3_pretraining/dataset.py
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T_v2
from PIL import Image, UnidentifiedImageError
import numpy as np
import logging
from collections import Counter
from pathlib import Path
from typing import Tuple, Optional, List

# Import config dictionary for default values
try:
    from .config import config as phase3_default_config # Relative import
except ImportError:
    # This fallback should ideally not be hit if the package structure is correct.
    print("PANIC (dataset.py): Could not import phase3_default_config. Hardcoding dataset defaults.")
    phase3_default_config = {
        "seed": 42, "hvt_spectral_channels": 0, "pretrain_img_size": (224,224),
        "original_dataset_name": "Original Dataset", "augmented_dataset_name": "Augmented Dataset"
    }

logger = logging.getLogger(__name__)

class SARCLD2024Dataset(Dataset):
    def __init__(self,
                 root_dir: str,                # From config['data_root']
                 img_size: tuple,              # From config['pretrain_img_size']
                 split: str = "train",
                 train_split_ratio: float = 0.8, # From config['train_split_ratio']
                 normalize_for_model: bool = False, # True for probe, False for SimCLR pretrain
                 use_spectral: bool = False,   # Should be False for SimCLR RGB pre-training
                 spectral_channels: int = phase3_default_config['hvt_params_for_backbone']['spectral_channels'],
                 original_dataset_name: str = phase3_default_config['original_dataset_name'],
                 augmented_dataset_name: str = phase3_default_config['augmented_dataset_name'],
                 random_seed: int = phase3_default_config['seed']
                 ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.split = split.lower()
        self.train_split_ratio = train_split_ratio
        self.normalize = normalize_for_model
        self.use_spectral = use_spectral and (spectral_channels > 0) # Only use if channels > 0
        self.spectral_channels = spectral_channels if self.use_spectral else 0

        # Standardized class list for SAR-CLD-2024
        self.classes = ["Bacterial Blight", "Curl Virus", "Healthy Leaf", "Herbicide Growth Damage", "Leaf Hopper Jassids", "Leaf Redding", "Leaf Variegation"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.image_tuples: List[Tuple[str, Optional[str], int]] = [] # rgb_path, spectral_path, label

        logger.info(f"Initializing SARCLD2024Dataset: split='{self.split}', img_size={self.img_size}, use_spectral={self.use_spectral}")
        if not self.root_dir.exists():
            logger.error(f"Dataset root directory not found: {self.root_dir}")
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")

        for dataset_type_folder_name in [original_dataset_name, augmented_dataset_name]:
            dataset_path = self.root_dir / dataset_type_folder_name
            if not dataset_path.is_dir():
                logger.debug(f"Sub-dataset path not found, skipping: {dataset_path}")
                continue
            logger.debug(f"Scanning sub-dataset: {dataset_path}")
            for class_name in self.classes:
                class_folder_path = dataset_path / class_name
                if not class_folder_path.is_dir(): continue
                for item in class_folder_path.iterdir():
                    if item.is_file() and item.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        rgb_path = str(item)
                        # For SimCLR pre-training, we don't need spectral path finding
                        spectral_path = None # Set to None for RGB-only SimCLR pre-training
                        label = self.class_to_idx[class_name]
                        self.image_tuples.append((rgb_path, spectral_path, label))
        
        if not self.image_tuples:
            logger.error(f"No images found in specified dataset paths under {self.root_dir}.")
            raise ValueError("No images found. Check dataset paths and structure.")
        logger.info(f"Found {len(self.image_tuples)} total image entries.")

        # Shuffle and split indices
        all_labels_for_split = np.array([label for _, _, label in self.image_tuples])
        indices = np.arange(len(self.image_tuples))
        np.random.seed(random_seed); np.random.shuffle(indices) # Shuffle once based on seed

        split_idx = int(len(indices) * self.train_split_ratio)
        if self.split == "train": self.current_indices = indices[:split_idx]
        elif self.split in ["val", "validation", "test"]: self.current_indices = indices[split_idx:]
        else: self.current_indices = indices # Use full dataset if split name is unrecognized
        
        self.current_split_labels = all_labels_for_split[self.current_indices]
        logger.info(f"Dataset split '{self.split}' size: {len(self.current_indices)} samples.")
        if len(self.current_indices) == 0 and len(self.image_tuples) > 0:
            logger.warning(f"Split '{self.split}' is empty. Check train_split_ratio or dataset content.")

        # Base RGB transforms (resize, to tensor, to float [0,1])
        # Normalization is applied conditionally for probing. Augmentations for SimCLR are applied in trainer.
        rgb_transforms_list = [
            T_v2.Resize(self.img_size, interpolation=T_v2.InterpolationMode.BICUBIC, antialias=True),
            T_v2.ToImage(), T_v2.ToDtype(torch.float32, scale=True)
        ]
        if self.normalize: # For linear probe
             rgb_transforms_list.append(T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.base_transform_rgb = T_v2.Compose(rgb_transforms_list)
        self.class_weights_computed = None


    def __len__(self) -> int:
        return len(self.current_indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]: # Returns (rgb_tensor, label_tensor)
        actual_idx = self.current_indices[idx]
        rgb_path, _, label = self.image_tuples[actual_idx]
        label_tensor = torch.tensor(label, dtype=torch.long)

        try:
            img_rgb_pil = Image.open(rgb_path).convert("RGB")
            img_rgb_pil.load() # Ensure image data is loaded
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            logger.error(f"Error loading image {rgb_path} at index {idx} (actual {actual_idx}): {e}. Returning placeholder.")
            dummy_rgb = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return dummy_rgb, torch.tensor(-1, dtype=torch.long) # Error label

        try:
            rgb_tensor = self.base_transform_rgb(img_rgb_pil)
        except Exception as e_transform:
            logger.error(f"Error transforming image {rgb_path}: {e_transform}. Returning placeholder.")
            dummy_rgb = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return dummy_rgb, torch.tensor(-1, dtype=torch.long) # Error label

        return rgb_tensor, label_tensor

    def get_class_names(self) -> List[str]: return self.classes

    def get_class_weights(self) -> Optional[torch.Tensor]:
        if self.class_weights_computed is None:
            if len(self.current_split_labels) == 0: return None
            class_counts = Counter(self.current_split_labels)
            total_samples = len(self.current_split_labels)
            weights = torch.zeros(len(self.classes), dtype=torch.float)
            for i_cls in range(len(self.classes)):
                count = class_counts.get(i_cls, 0)
                weights[i_cls] = total_samples / (len(self.classes) * count) if count > 0 else 0.0
            self.class_weights_computed = weights
            logger.info(f"Computed class weights for split '{self.split}': {self.class_weights_computed.numpy().round(2)}")
        return self.class_weights_computed