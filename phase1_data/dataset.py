# dataset.py
import os
from pathlib import Path
import torch
from PIL import Image, UnidentifiedImageError
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T # For T.Resize specifically, rest can be v2
import torchvision.transforms.v2 as T_v2 # For main transforms
from typing import Tuple, Optional, Dict, Any
import logging

from config import DEFAULT_STAGE_MAP, SPECTRAL_SIZE, IMAGE_SIZE
from progression import DiseaseProgressionSimulator

logger = logging.getLogger(__name__)

class CottonLeafDataset(Dataset):
    """Custom dataset for cotton leaf disease detection with support for original and augmented datasets."""

    def __init__(self, root_dir: str, transform: Optional[T_v2.Compose] = None,
                 stage_map: Optional[Dict[int, str]] = None, apply_progression: bool = False,
                 use_spectral: bool = True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.stage_map = stage_map or DEFAULT_STAGE_MAP
        self.apply_progression = apply_progression
        self.progression_simulator = DiseaseProgressionSimulator() if apply_progression else None
        self.use_spectral = use_spectral

        if not self.root_dir.exists():
            logger.error(f"Dataset root directory not found: {self.root_dir}")
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")

        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        if not self.classes:
            logger.warning(f"No class subdirectories found in {self.root_dir}.")

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        self.images = self._load_images()
        if not self.images:
            logger.warning(f"No images loaded from {self.root_dir}. Check dataset structure and image formats.")

    def _load_images(self) -> list:
        images = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.is_dir():
                continue
            label = self.class_to_idx[cls_name]
            for img_file in cls_dir.iterdir():
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                img_path = img_file
                try:
                    # Minimal check, PIL will raise error on open if severely corrupt
                    # with Image.open(img_path) as img:
                    #     img.verify() # Basic check, but can be slow. Open itself is a good check.
                    spectral_path_str = str(img_path.with_name(f"{img_path.stem}_spectral.npy"))
                    has_spectral = Path(spectral_path_str).exists()
                    images.append((str(img_path), spectral_path_str if has_spectral else None, label))
                except (IOError, SyntaxError, UnidentifiedImageError) as e: # Catch more PIL errors
                    logger.warning(f"Skipping corrupt or unreadable image: {img_path} - {e}")
        logger.info(f"Found {len(images)} potential image entries from {self.root_dir} across {len(self.classes)} classes.")
        return images

    def _simulate_ndvi(self, rgb_pil_img: Image.Image) -> torch.Tensor:
        try:
            rgb_array = np.array(rgb_pil_img.convert('RGB')) / 255.0
            # Adjusted proxy to prevent all-zero NIR if R is high and G,B are low (common in green leaves)
            nir_proxy = 0.2 * rgb_array[:, :, 0] + 0.5 * rgb_array[:, :, 1] + 0.3 * rgb_array[:, :, 2] # Emphasize Green for NIR proxy
            red_band = rgb_array[:, :, 0]
            
            numerator = nir_proxy - red_band
            denominator = nir_proxy + red_band + 1e-8 
            ndvi = numerator / denominator
            ndvi = np.clip(ndvi, -1.0, 1.0)
            ndvi_tensor = torch.tensor(ndvi, dtype=torch.float32).unsqueeze(0)
            
            # Use torchvision.transforms.Resize for tensor
            resize_transform = T.Resize(SPECTRAL_SIZE, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
            resized_ndvi = resize_transform(ndvi_tensor)
            # Ensure it has 1 channel, H, W
            if resized_ndvi.ndim == 2: # Should not happen if unsqueeze(0) worked
                 resized_ndvi = resized_ndvi.unsqueeze(0)
            return resized_ndvi
        except Exception as e:
            logger.error(f"Error simulating NDVI: {e}. Returning zero tensor.", exc_info=True)
            return torch.zeros((1, SPECTRAL_SIZE[0], SPECTRAL_SIZE[1]), dtype=torch.float32)


    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, str]:
        img_path_str, spectral_path_str, label = self.images[idx]
        
        try:
            rgb_pil_img = Image.open(img_path_str).convert('RGB')
        except (UnidentifiedImageError, FileNotFoundError, IOError) as e:
            logger.error(f"Critical error opening image {img_path_str} at index {idx}: {e}. Returning placeholder for this item.")
            placeholder_rgb = torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=torch.float32)
            # Return None for spectral, collate_fn will handle it
            return placeholder_rgb, None, -1, "error_loading_image"

        stage = self.stage_map.get(label, 'unknown')

        if self.apply_progression and self.progression_simulator:
            rgb_pil_img = self.progression_simulator.apply(rgb_pil_img.copy(), stage)

        spectral_tensor: Optional[torch.Tensor] = None
        if self.use_spectral:
            if spectral_path_str:
                try:
                    spectral_data = np.load(spectral_path_str)
                    if spectral_data.ndim == 2:
                        spectral_data = spectral_data[np.newaxis, ...]
                    elif spectral_data.ndim == 3 and spectral_data.shape[0] != 1 and spectral_data.shape[-1] == 1: # HWC to CHW
                        spectral_data = np.transpose(spectral_data, (2,0,1))
                    
                    spectral_tensor_raw = torch.tensor(spectral_data, dtype=torch.float32)
                    resize_spectral = T.Resize(SPECTRAL_SIZE, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
                    spectral_tensor = resize_spectral(spectral_tensor_raw)
                except Exception as e:
                    logger.warning(f"Could not load/process spectral data {spectral_path_str}: {e}. Simulating NDVI.")
                    spectral_tensor = self._simulate_ndvi(rgb_pil_img.copy()) # Pass a copy
            else:
                spectral_tensor = self._simulate_ndvi(rgb_pil_img.copy()) # Pass a copy
        
        # else: spectral_tensor remains None if use_spectral is False

        if self.transform:
            rgb_tensor = self.transform(rgb_pil_img)
        else:
            rgb_tensor = T_v2.Compose([T_v2.ToImage(), T_v2.ToDtype(torch.float32, scale=True)])(rgb_pil_img)


        # Final check for spectral_tensor shape if it's not None
        if spectral_tensor is not None:
            if spectral_tensor.ndim == 2: # Should be [C, H, W]
                spectral_tensor = spectral_tensor.unsqueeze(0)
            if spectral_tensor.shape[1:] != SPECTRAL_SIZE: # If resize somehow failed or was skipped
                 logger.warning(f"Resizing spectral tensor {spectral_tensor.shape} to {SPECTRAL_SIZE} post-hoc.")
                 resize_spectral = T.Resize(SPECTRAL_SIZE, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
                 spectral_tensor = resize_spectral(spectral_tensor)


        return rgb_tensor, spectral_tensor, label, stage