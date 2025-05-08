# phase5_multimodal_hpo/dataset.py
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T_v2 
from PIL import Image, UnidentifiedImageError
import numpy as np
import logging
from collections import Counter
from pathlib import Path
from typing import Tuple, Optional 

# Import necessary config values, ensure relative import works if config.py is sibling
try:
    from .config import config as default_config # Use config dictionary
    config_RANDOM_SEED = default_config['seed']
    config_SPECTRAL_CHANNELS = default_config['hvt_spectral_channels']
    config_IMG_SIZE = default_config['img_size'] 
except ImportError:
    # Fallback if run standalone or config import fails
    print("Warning (dataset.py): Could not import config, using fallback defaults.")
    config_RANDOM_SEED = 42
    config_SPECTRAL_CHANNELS = 1
    config_IMG_SIZE = (384, 384) 

logger = logging.getLogger(__name__)

class SARCLD2024Dataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 img_size: tuple, 
                 split: str = "train", 
                 train_split_ratio: float = 0.8, 
                 normalize_for_model: bool = True, 
                 use_spectral: bool = True, 
                 spectral_channels: int = config_SPECTRAL_CHANNELS, 
                 original_dataset_name: str = "Original Dataset",
                 augmented_dataset_name: str = "Augmented Dataset",
                 random_seed: int = config_RANDOM_SEED 
                 ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.split = split.lower()
        self.train_split_ratio = train_split_ratio
        self.normalize = normalize_for_model
        self.use_spectral = use_spectral
        self.spectral_channels = spectral_channels
        # REMOVED self.use_augmentations - augmentations are external now

        self.classes = [
            "Bacterial Blight", "Curl Virus", "Healthy Leaf",
            "Herbicide Growth Damage", "Leaf Hopper Jassids",
            "Leaf Redding", "Leaf Variegation"
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.image_tuples = [] 

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory does not exist: {self.root_dir}")
        logger.info(f"Initializing SARCLD2024Dataset from: {self.root_dir} for split: {self.split}")

        # --- Scan for images --- (Keep scanning logic as is)
        for dataset_type_name in [original_dataset_name, augmented_dataset_name]:
            dataset_path = self.root_dir / dataset_type_name
            if not dataset_path.is_dir(): continue
            logger.info(f"Scanning dataset type: {dataset_type_name}")
            for class_name in self.classes:
                class_folder_path = dataset_path / class_name
                if not class_folder_path.is_dir(): continue
                label = self.class_to_idx[class_name]
                valid_extensions = (".jpg", ".jpeg", ".png")
                try:
                    for item in class_folder_path.iterdir():
                        if item.is_file() and item.suffix.lower() in valid_extensions:
                            rgb_path = str(item)
                            spectral_path = item.with_name(f"{item.stem}_spectral.npy")
                            spectral_path_str = str(spectral_path) if spectral_path.exists() else None
                            try:
                                with Image.open(rgb_path) as img_test: img_test.verify() 
                                self.image_tuples.append((rgb_path, spectral_path_str, label))
                            except (UnidentifiedImageError, IOError, SyntaxError) as img_err:
                                logger.warning(f"Skipping unreadable/corrupt image: {rgb_path} ({img_err})")
                except OSError as os_err: logger.error(f"OS error scanning {class_folder_path}: {os_err}")
        
        if not self.image_tuples: raise ValueError(f"No valid images found in {self.root_dir}.")
        logger.info(f"Total valid image tuples found: {len(self.image_tuples)}")
        self.all_labels = np.array([label for _, _, label in self.image_tuples])
        # (Keep logging class counts)

        # --- Split indices --- (Keep splitting logic as is)
        indices = np.arange(len(self.image_tuples))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        split_idx = int(len(indices) * self.train_split_ratio)
        if self.split == "train": self.current_indices = indices[:split_idx]
        elif self.split in ["val", "validation", "test"]: self.current_indices = indices[split_idx:]
        else: raise ValueError(f"Invalid split name '{self.split}'")
        self.current_split_labels = self.all_labels[self.current_indices]
        logger.info(f"Dataset split '{self.split}' size: {len(self.current_indices)} samples.")

        # --- Base Transforms (Applied in __getitem__) ---
        rgb_transforms = [
            T_v2.Resize(self.img_size, interpolation=T_v2.InterpolationMode.BICUBIC, antialias=True),
            T_v2.ToImage(), 
            T_v2.ToDtype(torch.float32, scale=True)
        ]
        spectral_transforms = [ # Assuming input is HWC numpy or similar
             T_v2.ToImage(), # Convert numpy HWC to CHW Tensor
             T_v2.Resize(self.img_size, interpolation=T_v2.InterpolationMode.BILINEAR, antialias=True),
             T_v2.ToDtype(torch.float32, scale=False) # Keep original scale if not [0, 255] uint8
        ]
        # Add normalization ONLY if requested (usually done externally AFTER augmentations for training)
        if self.normalize:
             rgb_transforms.append(T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
             # Potentially normalize spectral data too if needed (e.g., per-channel Z-score)
             # spectral_transforms.append(T_v2.Normalize(mean=[SPECTRAL_MEAN], std=[SPECTRAL_STD]))
             
        self.base_transform_rgb = T_v2.Compose(rgb_transforms)
        self.base_transform_spectral = T_v2.Compose(spectral_transforms)
        
        self.class_weights_computed = None # Cache for class weights

    def _simulate_spectral(self) -> torch.Tensor:
        """ Simulates single-channel spectral data (like NDVI) - simplified return zeros. """
        # Returning zeros as placeholder - simulation logic can be complex.
        # Ensure correct shape [C, H, W]
        logger.debug("Simulating spectral data as zeros.")
        return torch.zeros((self.spectral_channels, self.img_size[0], self.img_size[1]), dtype=torch.float32)

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        actual_idx = self.current_indices[idx]
        rgb_path, spectral_path_str, label = self.image_tuples[actual_idx]
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        try:
            img_rgb_pil = Image.open(rgb_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading RGB image {rgb_path}: {e}")
            dummy_rgb = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return dummy_rgb, None, torch.tensor(-1, dtype=torch.long) 

        # Apply base RGB transform (Resize, ToTensor, Normalize if enabled)
        rgb_tensor = self.base_transform_rgb(img_rgb_pil)

        spectral_tensor: Optional[torch.Tensor] = None
        if self.use_spectral:
            spectral_data = None
            if spectral_path_str:
                try:
                    spectral_data = np.load(spectral_path_str).astype(np.float32)
                except Exception as e:
                    logger.warning(f"Error loading spectral file {spectral_path_str}: {e}. Simulating.")
                    spectral_tensor = self._simulate_spectral() 

            if spectral_data is not None: # Loaded successfully
                 try:
                    # Handle shape: make sure it's [C, H, W] before transform
                    if spectral_data.ndim == 2: spectral_data = spectral_data[np.newaxis, ...]
                    elif spectral_data.ndim == 3 and spectral_data.shape[-1] == self.spectral_channels: spectral_data = np.transpose(spectral_data, (2, 0, 1))
                    
                    if spectral_data.shape[0] != self.spectral_channels:
                         logger.warning(f"Loaded spectral data {spectral_path_str} has {spectral_data.shape[0]} channels, expected {self.spectral_channels}. Fixing/Padding.")
                         # Fix channels: take first N or pad with zeros
                         if spectral_data.shape[0] > self.spectral_channels:
                              spectral_data = spectral_data[:self.spectral_channels, :, :]
                         else: # Pad
                              pad_width = ((0, self.spectral_channels - spectral_data.shape[0]), (0,0), (0,0))
                              spectral_data = np.pad(spectral_data, pad_width, mode='constant', constant_values=0)
                    
                    # Apply base spectral transform (ToTensor, Resize)
                    spectral_tensor = self.base_transform_spectral(spectral_data) 
                 except Exception as e:
                      logger.error(f"Error processing loaded spectral data {spectral_path_str}: {e}. Simulating.", exc_info=True)
                      spectral_tensor = self._simulate_spectral()
            elif spectral_tensor is None: # No file and simulation didn't run yet (e.g., loading failed before simulation)
                 spectral_tensor = self._simulate_spectral()
        
        # *** NO Augmentations applied here - done externally ***

        return rgb_tensor, spectral_tensor, label_tensor

    def get_class_names(self):
        return self.classes
    
    def get_class_weights(self):
        # (Keep this method as is from previous version)
        if self.class_weights_computed is None:
            class_counts = Counter(self.current_split_labels) 
            total_samples = len(self.current_split_labels)
            if not class_counts or total_samples == 0: return None
            weights = torch.zeros(len(self.classes), dtype=torch.float)
            for i in range(len(self.classes)):
                count = class_counts.get(i, 0) 
                if count > 0: weights[i] = total_samples / (len(self.classes) * count) 
                else: weights[i] = 1.0 
            self.class_weights_computed = weights
            logger.info(f"Computed class weights for split '{self.split}': {self.class_weights_computed.numpy().round(2)}")
        return self.class_weights_computed