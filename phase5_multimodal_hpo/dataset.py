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
from typing import Tuple, Optional # Add type hints

# Import RANDOM_SEED, SPECTRAL_CHANNELS from config
# Use relative import assuming config.py is in the same dir level or accessible via path
try:
    from .config import RANDOM_SEED as config_RANDOM_SEED
    from .config import HVT_SPECTRAL_CHANNELS as config_SPECTRAL_CHANNELS
    from .config import IMG_SIZE as config_IMG_SIZE # Needed for spectral simulation fallback
except ImportError:
    config_RANDOM_SEED = 42
    config_SPECTRAL_CHANNELS = 1
    config_IMG_SIZE = (224, 224) # Fallback default

logger = logging.getLogger(__name__)

class SARCLD2024Dataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 img_size: tuple, 
                 split: str = "train", 
                 train_split_ratio: float = 0.8, 
                 normalize_for_model: bool = True, 
                 use_spectral: bool = True, # Flag to enable spectral loading
                 spectral_channels: int = config_SPECTRAL_CHANNELS, # Get expected channels
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

        self.classes = [
            "Bacterial Blight", "Curl Virus", "Healthy Leaf",
            "Herbicide Growth Damage", "Leaf Hopper Jassids",
            "Leaf Redding", "Leaf Variegation"
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.image_tuples = [] # Store tuples: (rgb_path, spectral_path_or_None, label)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory does not exist: {self.root_dir}")
        logger.info(f"Initializing SARCLD2024Dataset from: {self.root_dir} for split: {self.split}")

        # --- Scan for images and potentially linked spectral files ---
        for dataset_type_name in [original_dataset_name, augmented_dataset_name]:
            dataset_path = self.root_dir / dataset_type_name
            if not dataset_path.is_dir():
                logger.warning(f"Dataset path does not exist or is not a directory, skipping: {dataset_path}")
                continue
            
            logger.info(f"Scanning dataset type: {dataset_type_name}")
            for class_name in self.classes:
                class_folder_path = dataset_path / class_name
                if not class_folder_path.is_dir():
                    logger.warning(f"Class folder does not exist, skipping: {class_folder_path}")
                    continue
                
                label = self.class_to_idx[class_name]
                valid_extensions = (".jpg", ".jpeg", ".png")
                try:
                    for item in class_folder_path.iterdir():
                        if item.is_file() and item.suffix.lower() in valid_extensions:
                            rgb_path = str(item)
                            # Construct expected spectral path (.npy)
                            spectral_path = item.with_name(f"{item.stem}_spectral.npy")
                            spectral_path_str = str(spectral_path) if spectral_path.exists() else None
                            
                            # Basic check if file is readable as image before adding
                            try:
                                with Image.open(rgb_path) as img_test:
                                    img_test.verify() 
                                self.image_tuples.append((rgb_path, spectral_path_str, label))
                            except (UnidentifiedImageError, IOError, SyntaxError) as img_err:
                                logger.warning(f"Skipping unreadable/corrupt image: {rgb_path} ({img_err})")
                except OSError as os_err:
                     logger.error(f"OS error scanning folder {class_folder_path}: {os_err}")

        if not self.image_tuples:
            raise ValueError(f"No valid image tuples found. Check dataset structure in {self.root_dir}.")

        logger.info(f"Total valid image tuples found: {len(self.image_tuples)}")
        
        # Store labels separately for splitting/weighting logic
        self.all_labels = np.array([label for _, _, label in self.image_tuples])
        class_counts = Counter(self.all_labels)
        self.class_weights_computed = None 
        for idx, count in sorted(class_counts.items()):
            logger.info(f"  Class '{self.classes[idx]}': {count} samples")

        # --- Split indices ---
        indices = np.arange(len(self.image_tuples))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        split_idx = int(len(indices) * self.train_split_ratio)
        if self.split == "train":
            self.current_indices = indices[:split_idx]
        elif self.split in ["val", "validation", "test"]:
            self.current_indices = indices[split_idx:]
        else:
            raise ValueError(f"Invalid split name '{self.split}'")
        
        self.current_split_labels = self.all_labels[self.current_indices]
        logger.info(f"Dataset split '{self.split}' size: {len(self.current_indices)} samples.")

        # --- Base Transforms ---
        transforms_list = [
            T_v2.Resize(self.img_size, interpolation=T_v2.InterpolationMode.BICUBIC, antialias=True),
            T_v2.ToImage(), 
            T_v2.ToDtype(torch.float32, scale=True)
        ]
        if self.normalize:
             transforms_list.append(T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.base_transform_rgb = T_v2.Compose(transforms_list)
        
        # Base transform for spectral (only resize usually needed before model-specific processing)
        # Assuming spectral data is single channel float after loading
        self.base_transform_spectral = T_v2.Compose([
            T_v2.Resize(self.img_size, interpolation=T_v2.InterpolationMode.BILINEAR, antialias=True)
        ])


    def _simulate_spectral(self, rgb_pil_img: Image.Image) -> torch.Tensor:
        """ Simulates single-channel spectral data (like NDVI) from RGB. """
        try:
            # Ensure resize happens before simulation if needed, or simulate then resize tensor
            # Let's simulate on original size then resize tensor
            rgb_array = np.array(rgb_pil_img.convert('RGB')) / 255.0
            # Simple NDVI proxy (adjust weights as needed)
            nir_proxy = 0.2 * rgb_array[:, :, 0] + 0.5 * rgb_array[:, :, 1] + 0.3 * rgb_array[:, :, 2] 
            red_band = rgb_array[:, :, 0]
            numerator = nir_proxy - red_band
            denominator = nir_proxy + red_band + 1e-8 
            sim_data = np.clip(numerator / denominator, -1.0, 1.0)
            # Ensure shape is [C, H, W] = [1, H, W]
            sim_tensor = torch.tensor(sim_data, dtype=torch.float32).unsqueeze(0) 
            # Resize using spectral transform
            resized_sim = self.base_transform_spectral(sim_tensor)
            # Ensure correct number of channels after resize (should be 1)
            if resized_sim.shape[0] != 1:
                 logger.warning(f"Simulated spectral has {resized_sim.shape[0]} channels after resize, expected 1. Taking first.")
                 resized_sim = resized_sim[0:1, :, :]
                 
            # Pad/crop if spectral channels expected > 1? No, assume model handles 1 channel input.
            if self.spectral_channels != 1:
                 logger.warning(f"Simulating 1 spectral channel, but model expects {self.spectral_channels}. Using simulation.")
            
            return resized_sim

        except Exception as e:
            logger.error(f"Error simulating spectral data: {e}. Returning zero tensor.", exc_info=True)
            return torch.zeros((1, self.img_size[0], self.img_size[1]), dtype=torch.float32)

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        actual_idx = self.current_indices[idx]
        rgb_path, spectral_path_str, label = self.image_tuples[actual_idx]
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        try:
            img = Image.open(rgb_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {rgb_path}: {e}")
            dummy_rgb = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            # Return None for spectral, main script/collate should handle this if it occurs
            return dummy_rgb, None, torch.tensor(-1, dtype=torch.long) 

        # Process RGB
        rgb_tensor = self.base_transform_rgb(img)

        # Process Spectral (if enabled)
        spectral_tensor: Optional[torch.Tensor] = None
        if self.use_spectral:
            if spectral_path_str:
                try:
                    spectral_data = np.load(spectral_path_str).astype(np.float32)
                    # Handle shape: assume [H, W] or [C, H, W] or [H, W, C]
                    if spectral_data.ndim == 2: # H, W -> 1, H, W
                        spectral_data = spectral_data[np.newaxis, ...]
                    elif spectral_data.ndim == 3 and spectral_data.shape[-1] == self.spectral_channels: # H, W, C -> C, H, W
                        spectral_data = np.transpose(spectral_data, (2, 0, 1))
                    
                    # Check channel dimension
                    if spectral_data.shape[0] != self.spectral_channels:
                         logger.warning(f"Loaded spectral data {spectral_path_str} has {spectral_data.shape[0]} channels, "
                                        f"expected {self.spectral_channels}. Attempting to use first {self.spectral_channels}.")
                         spectral_data = spectral_data[:self.spectral_channels, :, :]
                         # Pad with zeros if loaded has fewer channels than expected? Risky.
                         if spectral_data.shape[0] < self.spectral_channels:
                              padding_shape = (self.spectral_channels - spectral_data.shape[0], spectral_data.shape[1], spectral_data.shape[2])
                              spectral_data = np.concatenate([spectral_data, np.zeros(padding_shape, dtype=spectral_data.dtype)], axis=0)
                              logger.warning(f"Padded spectral data to {self.spectral_channels} channels.")


                    spectral_tensor_raw = torch.from_numpy(spectral_data)
                    spectral_tensor = self.base_transform_spectral(spectral_tensor_raw)
                except Exception as e:
                    logger.warning(f"Error loading spectral file {spectral_path_str}: {e}. Simulating.")
                    spectral_tensor = self._simulate_spectral(img) # Simulate if loading fails
            else:
                # Simulate if no spectral file exists
                spectral_tensor = self._simulate_spectral(img)
        
        # If use_spectral is False, spectral_tensor remains None
        # Collate function might need adjustment if spectral_tensor can sometimes be None when use_spectral is True
        # For simplicity now, assume simulate always provides a tensor if use_spectral is True

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

