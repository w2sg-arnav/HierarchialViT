# phase4_finetuning/dataset.py
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T_v2
from PIL import Image, UnidentifiedImageError
import numpy as np
import logging
from collections import Counter
from pathlib import Path
from typing import Tuple, Optional, List, Callable, Any # Added Callable, Any

# Try to import the main config to access default values if needed
try:
    from phase4_finetuning.config import config as global_finetune_config_defaults
except ImportError:
    print("Warning (phase4_dataset.py): Could not import global_finetune_config_defaults. Using hardcoded dataset defaults.")
    global_finetune_config_defaults = {
        "seed": 42, "original_dataset_name": "Original Dataset",
        "augmented_dataset_name": "Augmented Dataset", "num_classes": 7,
        # "normalize_data": True # This is no longer used here
    }

logger = logging.getLogger(__name__)

class SARCLD2024Dataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 img_size: tuple,
                 split: str = "train",
                 transform: Optional[Callable] = None, # MODIFIED: Added transform
                 train_split_ratio: float = 0.8,
                 # normalize_for_model: bool = True, # REMOVED: Normalization handled by transform
                 original_dataset_name: Optional[str] = None,
                 augmented_dataset_name: Optional[str] = None,
                 random_seed: Optional[int] = None):

        self.root_dir = Path(root_dir)
        self.img_size = tuple(img_size)
        self.split = split.lower()
        self.ext_transform = transform # MODIFIED: Store external transform
        self.train_split_ratio = train_split_ratio
        # self.normalize = normalize_for_model # REMOVED

        self.original_dataset_name = original_dataset_name if original_dataset_name is not None else global_finetune_config_defaults.get('original_dataset_name')
        self.augmented_dataset_name = augmented_dataset_name if augmented_dataset_name is not None else global_finetune_config_defaults.get('augmented_dataset_name')
        self.random_seed = random_seed if random_seed is not None else global_finetune_config_defaults.get('seed')

        self.classes = ["Bacterial Blight", "Curl Virus", "Healthy Leaf", "Herbicide Growth Damage", "Leaf Hopper Jassids", "Leaf Redding", "Leaf Variegation"]
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.image_paths: List[str] = []
        self.labels: List[int] = []

        logger.info(f"[DATASET INIT - {self.split}] Root: {self.root_dir}, ImgSize: {self.img_size}") # Removed Normalize from log
        if not self.root_dir.exists():
            logger.error(f"FATAL [DATASET INIT - {self.split}] Dataset root missing: {self.root_dir}")
            raise FileNotFoundError(f"Dataset root missing: {self.root_dir}")

        items_scanned_count = 0
        dataset_names_to_scan = [
            self.original_dataset_name,
            self.augmented_dataset_name
        ]

        for dataset_type_name in dataset_names_to_scan:
            if not dataset_type_name:
                logger.debug(f"[DATASET INIT - {self.split}] Dataset name '{dataset_type_name}' is None/empty, skipping.")
                continue
            dataset_path = self.root_dir / dataset_type_name
            if not dataset_path.is_dir():
                logger.debug(f"[DATASET INIT - {self.split}] Path not found or not a dir, skipping: {dataset_path}")
                continue

            logger.info(f"[DATASET INIT - {self.split}] Scanning: {dataset_path}")
            for class_name in self.classes:
                class_folder_path = dataset_path / class_name
                if not class_folder_path.is_dir(): continue

                valid_extensions = (".jpg", ".jpeg", ".png"); n_skipped_in_folder = 0
                try:
                    for item in class_folder_path.iterdir():
                        items_scanned_count += 1
                        if item.is_file() and item.suffix.lower() in valid_extensions:
                            try:
                                if os.path.getsize(item) > 0:
                                    self.image_paths.append(str(item))
                                    self.labels.append(self.class_to_idx[class_name])
                                else:
                                    logger.warning(f"Skipping empty file: {item}")
                                    n_skipped_in_folder += 1
                            except OSError:
                                logger.warning(f"Skipping file due to OSError (e.g., broken symlink): {item}")
                                n_skipped_in_folder +=1
                except OSError as os_err:
                    logger.error(f"OS error scanning folder {class_folder_path}: {os_err}")
                if n_skipped_in_folder > 0:
                    logger.warning(f"Skipped {n_skipped_in_folder} files in {class_folder_path}")

        if not self.image_paths:
            logger.error(f"FATAL [DATASET INIT - {self.split}] No valid image paths found in {self.root_dir} using specified dataset names.")
            raise ValueError(f"No valid images found. Please check dataset structure and names in {self.root_dir}.")

        self.image_paths_np = np.array(self.image_paths)
        self.labels_np = np.array(self.labels)
        logger.info(f"[DATASET INIT - {self.split}] Total valid image paths collected: {len(self.image_paths_np)} from ~{items_scanned_count} items considered.")

        indices = np.arange(len(self.image_paths_np))
        if self.random_seed is not None: np.random.seed(self.random_seed)
        np.random.shuffle(indices)
        split_idx = int(len(indices) * self.train_split_ratio)

        if self.split == "train": self.current_indices = indices[:split_idx]
        elif self.split in ["val", "validation", "test"]: self.current_indices = indices[split_idx:]
        else: logger.error(f"Invalid split name '{self.split}'."); raise ValueError(f"Invalid split name '{self.split}'")

        self.current_split_labels = self.labels_np[self.current_indices]
        logger.info(f"[DATASET INIT - {self.split}] Dataset split size: {len(self.current_indices)} samples.")
        self.class_weights_computed = None

        # REMOVED: self.base_transform definition
        # logger.info(f"[DATASET INIT - {self.split}] Base RGB Transforms: {self.base_transform}") # REMOVED

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= idx < len(self.current_indices)):
            logger.error(f"Index {idx} out of bounds for current_indices (len {len(self.current_indices)}) in split '{self.split}'.")
            raise IndexError(f"Index {idx} out of bounds.")

        actual_idx_in_full_list = self.current_indices[idx]
        img_path = self.image_paths_np[actual_idx_in_full_list]
        label = self.labels_np[actual_idx_in_full_list]
        
        processed_tensor: torch.Tensor

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path} (idx {idx}, actual_idx {actual_idx_in_full_list}): {e}. Returning dummy.")
            dummy_tensor = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return dummy_tensor, torch.tensor(-1, dtype=torch.long)

        if self.ext_transform:
            try:
                processed_tensor = self.ext_transform(img)
            except Exception as e_transform:
                logger.error(f"Error applying external transform to image {img_path}: {e_transform}. Returning dummy.", exc_info=True)
                dummy_tensor = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
                return dummy_tensor, torch.tensor(-1, dtype=torch.long)
        else:
            # Fallback if no transform is provided (e.g., for dummy dataset if __getitem__ is called)
            logger.warning(f"No external transform provided for image {img_path}. Applying minimal fallback (ToImage, ToDtype, Resize).")
            fallback_transform = T_v2.Compose([
                T_v2.ToImage(),
                T_v2.ToDtype(torch.float32, scale=True),
                T_v2.Resize(self.img_size, interpolation=T_v2.InterpolationMode.BICUBIC, antialias=True)
            ])
            try:
                processed_tensor = fallback_transform(img)
            except Exception as e_fallback_transform:
                logger.error(f"Error in fallback transform for image {img_path}: {e_fallback_transform}. Returning dummy.", exc_info=True)
                dummy_tensor = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
                return dummy_tensor, torch.tensor(-1, dtype=torch.long)
        
        return processed_tensor, torch.tensor(label, dtype=torch.long)

    def get_class_names(self) -> List[str]:
        return self.classes

    def get_targets(self) -> np.ndarray: # Added to ensure it exists for weighted sampler/loss
        return self.current_split_labels

    def get_class_weights(self) -> Optional[torch.Tensor]:
        if self.class_weights_computed is None:
            if len(self.current_split_labels) == 0:
                logger.warning(f"Cannot compute class weights: split '{self.split}' has no labels.")
                return None
            class_counts = Counter(self.current_split_labels)

            weights = torch.ones(self.num_classes, dtype=torch.float)
            for i in range(self.num_classes):
                count = class_counts.get(i, 0)
                if count > 0:
                    weights[i] = len(self.current_split_labels) / (self.num_classes * count)
                else:
                    weights[i] = 1.0 
                    logger.debug(f"Class '{self.classes[i]}' (idx {i}) not found in split '{self.split}' for weight calculation. Weight set to 1.0.")
            self.class_weights_computed = weights
            logger.info(f"Computed class weights (inv_freq for loss) for split '{self.split}': {self.class_weights_computed.numpy().round(3)}")
        return self.class_weights_computed