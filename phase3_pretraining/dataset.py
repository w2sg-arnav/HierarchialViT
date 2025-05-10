# phase3_pretraining/dataset.py
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T_v2
from PIL import Image, UnidentifiedImageError
import numpy as np
import logging # Keep logging for errors
from collections import Counter
from pathlib import Path
from typing import Tuple, Optional, List

# Import config dictionary
try:
    from phase3_pretraining.config import config as default_config
    config_RANDOM_SEED_default = default_config.get('seed', 42)
    config_SPECTRAL_CHANNELS_default = default_config.get('hvt_spectral_channels', 1)
    config_IMG_SIZE_default = default_config.get('pretrain_img_size', (224, 224))
except ImportError:
    print("PANIC (dataset.py): Could not import default_config. Hardcoding dataset defaults.")
    default_config = {}
    config_RANDOM_SEED_default = 42
    config_SPECTRAL_CHANNELS_default = 1
    config_IMG_SIZE_default = (224,224)

logger = logging.getLogger(__name__)

# Counter for __getitem__ calls for initial verbose logging
GETITEM_CALL_COUNT = 0
MAX_VERBOSE_GETITEM_CALLS = 5 # Log details for the first N calls

class SARCLD2024Dataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 img_size: tuple,
                 split: str = "train",
                 train_split_ratio: float = 0.8,
                 normalize_for_model: bool = False,
                 use_spectral: bool = False,
                 spectral_channels: int = config_SPECTRAL_CHANNELS_default,
                 original_dataset_name: str = "Original Dataset",
                 augmented_dataset_name: str = "Augmented Dataset",
                 random_seed: int = config_RANDOM_SEED_default
                 ):
        global GETITEM_CALL_COUNT # Ensure we modify the global counter
        GETITEM_CALL_COUNT = 0 # Reset for new dataset instance

        original_dataset_name = default_config.get('original_dataset_name', original_dataset_name)
        augmented_dataset_name = default_config.get('augmented_dataset_name', augmented_dataset_name)

        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.split = split.lower()
        self.train_split_ratio = train_split_ratio
        self.normalize = normalize_for_model
        self.use_spectral = use_spectral
        self.spectral_channels = spectral_channels

        self.classes = ["Bacterial Blight", "Curl Virus", "Healthy Leaf", "Herbicide Growth Damage", "Leaf Hopper Jassids", "Leaf Redding", "Leaf Variegation"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.image_tuples = []

        logger.info(f"[DATASET INIT - {self.split}] Root: {self.root_dir}, ImgSize: {self.img_size}")
        if not self.root_dir.exists():
            logger.critical(f"[DATASET INIT - {self.split}] Dataset root missing: {self.root_dir}")
            raise FileNotFoundError(f"Dataset root missing: {self.root_dir}")

        files_scanned_count = 0
        for dataset_type_name in [original_dataset_name, augmented_dataset_name]:
            dataset_path = self.root_dir / dataset_type_name
            if not dataset_path.is_dir():
                logger.warning(f"[DATASET INIT - {self.split}] Path not found, skipping: {dataset_path}")
                continue
            logger.info(f"[DATASET INIT - {self.split}] Scanning: {dataset_path}")
            for class_name in self.classes:
                class_folder_path = dataset_path / class_name
                if not class_folder_path.is_dir(): continue
                label = self.class_to_idx[class_name]
                valid_extensions = (".jpg", ".jpeg", ".png"); n_skipped = 0
                try:
                    for item in class_folder_path.iterdir():
                        files_scanned_count += 1
                        if item.is_file() and item.suffix.lower() in valid_extensions:
                            rgb_path = str(item); spectral_path_str = None
                            self.image_tuples.append((rgb_path, spectral_path_str, label))
                        # No need for Image.open().verify() here, it's slow. __getitem__ handles errors.
                except OSError as os_err: logger.error(f"[DATASET INIT - {self.split}] OS error scanning {class_folder_path}: {os_err}")
                if n_skipped > 0: logger.warning(f"[DATASET INIT - {self.split}] Skipped {n_skipped} initially unreadable images in {class_folder_path}")
        
        if not self.image_tuples:
            logger.critical(f"[DATASET INIT - {self.split}] No valid images found in {self.root_dir} after scanning {files_scanned_count} items.")
            raise ValueError(f"No valid images found in {self.root_dir}.")
        logger.info(f"[DATASET INIT - {self.split}] Total valid image tuples initially found: {len(self.image_tuples)} from {files_scanned_count} items scanned.")
        
        self.all_labels = np.array([label for _, _, label in self.image_tuples])
        indices = np.arange(len(self.image_tuples)); np.random.seed(random_seed); np.random.shuffle(indices)
        split_idx = int(len(indices) * self.train_split_ratio)

        if self.split == "train": self.current_indices = indices[:split_idx]
        elif self.split in ["val", "validation", "test"]: self.current_indices = indices[split_idx:]
        else: self.current_indices = indices # full dataset if split is not recognized
        
        logger.info(f"[DATASET INIT - {self.split}] Dataset split size: {len(self.current_indices)} samples.")
        if len(self.current_indices) == 0 and len(self.image_tuples) > 0:
            logger.warning(f"[DATASET INIT - {self.split}] Current split is empty, but total images found. Check split logic or ratio.")


        rgb_transforms_list = [
            T_v2.Resize(self.img_size, interpolation=T_v2.InterpolationMode.BICUBIC, antialias=True),
            T_v2.ToImage(), T_v2.ToDtype(torch.float32, scale=True) # scale=True for [0,1]
        ]
        if self.normalize:
             # ImageNet stats, ensure images are [0,1] before this
             rgb_transforms_list.append(T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.base_transform_rgb = T_v2.Compose(rgb_transforms_list)
        logger.debug(f"[DATASET INIT - {self.split}] RGB Transforms: {self.base_transform_rgb}")
        self.class_weights_computed = None # For get_class_weights


    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        global GETITEM_CALL_COUNT, MAX_VERBOSE_GETITEM_CALLS
        GETITEM_CALL_COUNT += 1
        verbose = GETITEM_CALL_COUNT <= MAX_VERBOSE_GETITEM_CALLS

        if verbose: logger.debug(f"[DATASET GETITEM {idx} - {self.split}] Called (Count: {GETITEM_CALL_COUNT}).")
        
        if idx >= len(self.current_indices):
            logger.error(f"[DATASET GETITEM {idx} - {self.split}] Index out of bounds (len: {len(self.current_indices)}). Returning DUMMY.")
            dummy_rgb = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return dummy_rgb, torch.tensor(-1, dtype=torch.long)

        actual_idx = self.current_indices[idx]
        if verbose: logger.debug(f"[DATASET GETITEM {idx} - {self.split}] Actual index from current_indices: {actual_idx}")
        
        rgb_path, _, label = self.image_tuples[actual_idx]
        if verbose: logger.debug(f"[DATASET GETITEM {idx} - {self.split}] RGB path: {rgb_path}, Label: {label}")
        
        label_tensor = torch.tensor(label, dtype=torch.long)

        try:
            if verbose: logger.debug(f"[DATASET GETITEM {idx} - {self.split}] Attempting Image.open: {rgb_path}")
            img_rgb_pil = Image.open(rgb_path)
            img_rgb_pil.load() # Force loading the image data
            if verbose: logger.debug(f"[DATASET GETITEM {idx} - {self.split}] Image loaded. Mode: {img_rgb_pil.mode}, Size: {img_rgb_pil.size}")
            
            if img_rgb_pil.mode != 'RGB':
                if verbose: logger.debug(f"[DATASET GETITEM {idx} - {self.split}] Converting image from {img_rgb_pil.mode} to RGB.")
                img_rgb_pil = img_rgb_pil.convert("RGB")

        except FileNotFoundError:
            logger.error(f"[DATASET GETITEM {idx} - {self.split}] FileNotFoundError: {rgb_path}. Returning DUMMY.")
            dummy_rgb = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return dummy_rgb, torch.tensor(-1, dtype=torch.long)
        except UnidentifiedImageError:
            logger.error(f"[DATASET GETITEM {idx} - {self.split}] UnidentifiedImageError: {rgb_path}. Returning DUMMY.")
            dummy_rgb = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return dummy_rgb, torch.tensor(-1, dtype=torch.long)
        except Exception as e:
            logger.error(f"[DATASET GETITEM {idx} - {self.split}] Generic error loading {rgb_path}: {e}. Returning DUMMY.", exc_info=verbose) # only log exc_info if verbose
            dummy_rgb = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return dummy_rgb, torch.tensor(-1, dtype=torch.long)

        try:
            if verbose: logger.debug(f"[DATASET GETITEM {idx} - {self.split}] Applying base_transform_rgb to image of size {img_rgb_pil.size}...")
            rgb_tensor = self.base_transform_rgb(img_rgb_pil)
            if verbose: logger.debug(f"[DATASET GETITEM {idx} - {self.split}] Transform successful. Tensor shape: {rgb_tensor.shape}")
        except Exception as e_transform:
            logger.error(f"[DATASET GETITEM {idx} - {self.split}] Transform error for {rgb_path}: {e_transform}. Returning DUMMY.", exc_info=verbose)
            dummy_rgb = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return dummy_rgb, torch.tensor(-1, dtype=torch.long)

        if verbose: logger.debug(f"[DATASET GETITEM {idx} - {self.split}] SUCCESS.")
        return rgb_tensor, label_tensor

    def get_class_names(self) -> List[str]:
        return self.classes

    def get_class_weights(self) -> Optional[torch.Tensor]:
        # Ensure current_split_labels is populated correctly
        if not hasattr(self, 'current_indices') or len(self.current_indices) == 0:
             logger.warning(f"Cannot compute weights: split '{self.split}' has no indices or not initialized properly.")
             return None
        
        # Populate current_split_labels if not already done
        if not hasattr(self, 'current_split_labels') or len(self.current_split_labels) != len(self.current_indices):
            self.current_split_labels = self.all_labels[self.current_indices]

        if self.class_weights_computed is None:
            class_counts = Counter(self.current_split_labels)
            total_samples = len(self.current_split_labels)
            
            if not class_counts or total_samples == 0:
                logger.warning(f"Cannot compute weights: split '{self.split}' is effectively empty or labels not found.")
                return None

            weights = torch.zeros(len(self.classes), dtype=torch.float)
            for i in range(len(self.classes)):
                count = class_counts.get(i, 0)
                if count > 0:
                    weights[i] = total_samples / (len(self.classes) * count)
                else:
                    # logger.debug(f"Class {self.classes[i]} (idx {i}) has 0 samples in split '{self.split}'. Weight set to 0.")
                    weights[i] = 0.0 # Or a very small number if 0 causes issues later, but 0 is standard.
            self.class_weights_computed = weights
            logger.info(f"Computed class weights for split '{self.split}': {self.class_weights_computed.numpy().round(2)}")
        return self.class_weights_computed

    def get_current_split_labels_as_tensor(self) -> Optional[torch.Tensor]:
        if not hasattr(self, 'current_indices') or len(self.current_indices) == 0:
            logger.warning("No indices available for the current split to get labels.")
            return None
        if not hasattr(self, 'current_split_labels') or len(self.current_split_labels) != len(self.current_indices):
            self.current_split_labels = self.all_labels[self.current_indices]
            
        if len(self.current_split_labels) > 0:
            return torch.from_numpy(self.current_split_labels)
        else:
            logger.warning("No labels available for the current split (current_split_labels is empty).")
            return None