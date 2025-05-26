# dataset.py
import os
from pathlib import Path
import torch
from PIL import Image, UnidentifiedImageError
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T_old # For T.Resize in _simulate_ndvi if needed
import torchvision.transforms.v2 as T_v2 # For main transforms
from typing import Tuple, Optional, Dict, List, Any
import logging

from config import DEFAULT_STAGE_MAP, IMAGE_SIZE_SPECTRAL, IMAGE_SIZE_RGB
from progression import DiseaseProgressionSimulator

logger = logging.getLogger(__name__)

class CottonLeafDataset(Dataset):
    """Custom dataset for cotton leaf disease detection."""

    def __init__(self, root_dir: str,
                 transform_rgb: Optional[T_v2.Compose] = None,
                 transform_spectral: Optional[T_v2.Compose] = None, # If spectral needs different transforms
                 stage_map: Optional[Dict[int, str]] = None,
                 apply_progression: bool = False,
                 use_spectral: bool = False,
                 class_to_idx: Optional[Dict[str, int]] = None, # Pass this if pre-defined
                 classes: Optional[List[str]] = None): # Pass this if pre-defined

        self.root_dir = Path(root_dir)
        self.transform_rgb = transform_rgb
        self.transform_spectral = transform_spectral # Currently unused, assuming spectral processed to tensor then resized
        self.stage_map = stage_map or DEFAULT_STAGE_MAP
        self.apply_progression = apply_progression
        self.progression_simulator = DiseaseProgressionSimulator() if apply_progression else None
        self.use_spectral = use_spectral

        if not self.root_dir.exists():
            logger.error(f"Dataset root directory not found: {self.root_dir}")
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")

        if classes and class_to_idx:
            self.classes = classes
            self.class_to_idx = class_to_idx
        else:
            self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            if not self.classes:
                logger.warning(f"No class subdirectories found in {self.root_dir}. Dataset will be empty.")
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.num_classes = len(self.classes)

        self.images_metadata = self._load_images_metadata()
        if not self.images_metadata:
            logger.warning(f"No images loaded from {self.root_dir}. Check dataset structure and image formats.")

    def _load_images_metadata(self) -> list:
        images_meta = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.is_dir():
                continue
            label = self.class_to_idx[cls_name]
            for img_file in cls_dir.iterdir():
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                img_path_str = str(img_file)
                spectral_path_str = str(img_file.with_name(f"{img_file.stem}_spectral.npy"))
                has_spectral = Path(spectral_path_str).exists()
                
                images_meta.append({
                    "rgb_path": img_path_str,
                    "spectral_path": spectral_path_str if has_spectral else None,
                    "label": label,
                    "class_name": cls_name
                })
        logger.info(f"Found {len(images_meta)} potential image entries from {self.root_dir} across {len(self.classes)} classes.")
        return images_meta

    def _simulate_ndvi(self, rgb_pil_img: Image.Image) -> torch.Tensor:
        try:
            rgb_array = np.array(rgb_pil_img.convert('RGB')) / 255.0
            nir_proxy = 0.2 * rgb_array[:, :, 0] + 0.5 * rgb_array[:, :, 1] + 0.3 * rgb_array[:, :, 2]
            red_band = rgb_array[:, :, 0]
            
            numerator = nir_proxy - red_band
            denominator = nir_proxy + red_band + 1e-8 
            ndvi = numerator / denominator
            ndvi = np.clip(ndvi, -1.0, 1.0)
            ndvi_tensor = torch.tensor(ndvi, dtype=torch.float32).unsqueeze(0) # [1, H, W]
            
            resize_transform = T_old.Resize(IMAGE_SIZE_SPECTRAL, interpolation=T_old.InterpolationMode.BILINEAR, antialias=True)
            resized_ndvi = resize_transform(ndvi_tensor)
            return resized_ndvi
        except Exception as e:
            logger.error(f"Error simulating NDVI: {e}. Returning zero tensor.", exc_info=True)
            return torch.zeros((1, IMAGE_SIZE_SPECTRAL[0], IMAGE_SIZE_SPECTRAL[1]), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.images_metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_meta = self.images_metadata[idx]
        img_path_str = item_meta["rgb_path"]
        spectral_path_str = item_meta["spectral_path"]
        label = item_meta["label"]
        
        output_dict: Dict[str, Any] = {
            "rgb_image": None, "spectral_image": None, "label": label, "stage": "unknown", "id": idx
        }

        try:
            rgb_pil_img = Image.open(img_path_str).convert('RGB')
        except (UnidentifiedImageError, FileNotFoundError, IOError) as e:
            logger.error(f"Critical error opening RGB image {img_path_str} at index {idx}: {e}. Placeholder will be used by collate_fn if needed.")
            output_dict["label"] = -1 # Indicate error
            output_dict["stage"] = "error_loading_rgb"
            # Return default sized tensors for RGB, spectral will be None
            output_dict["rgb_image"] = torch.zeros((3, IMAGE_SIZE_RGB[0], IMAGE_SIZE_RGB[1]), dtype=torch.float32)
            return output_dict # Spectral remains None

        # Determine stage for progression simulation
        # The `stage_map` uses class index. If you want to map by class name, adjust `stage_map` or logic.
        stage_for_sim = self.stage_map.get(label, 'unknown')
        output_dict["stage"] = stage_for_sim

        if self.apply_progression and self.progression_simulator:
            rgb_pil_img_prog = self.progression_simulator.apply(rgb_pil_img.copy(), stage_for_sim)
        else:
            rgb_pil_img_prog = rgb_pil_img

        # Process RGB
        if self.transform_rgb:
            rgb_tensor = self.transform_rgb(rgb_pil_img_prog)
        else: # Default minimal transform
            rgb_tensor = T_v2.Compose([
                T_v2.ToImage(),
                T_v2.ToDtype(torch.float32, scale=True) # Scales to [0,1]
            ])(rgb_pil_img_prog)
        output_dict["rgb_image"] = rgb_tensor

        # Process Spectral
        spectral_tensor: Optional[torch.Tensor] = None
        if self.use_spectral:
            if spectral_path_str:
                try:
                    spectral_data = np.load(spectral_path_str) # Expects [C, H, W] or [H, W]
                    if spectral_data.ndim == 2: # H, W
                        spectral_data = spectral_data[np.newaxis, ...] # 1, H, W
                    elif spectral_data.ndim == 3 and spectral_data.shape[0] != 1 and spectral_data.shape[-1] == 1: # H, W, 1
                         spectral_data = np.transpose(spectral_data, (2,0,1)) # 1, H, W
                    
                    spectral_tensor_raw = torch.tensor(spectral_data, dtype=torch.float32)
                    # Ensure C, H, W format before resize
                    if spectral_tensor_raw.ndim == 2: spectral_tensor_raw = spectral_tensor_raw.unsqueeze(0)

                    resize_spectral = T_old.Resize(IMAGE_SIZE_SPECTRAL, interpolation=T_old.InterpolationMode.BILINEAR, antialias=True)
                    spectral_tensor = resize_spectral(spectral_tensor_raw)

                except Exception as e:
                    logger.warning(f"Could not load/process spectral data {spectral_path_str}: {e}. Simulating NDVI.")
                    spectral_tensor = self._simulate_ndvi(rgb_pil_img.copy()) # Use original RGB for NDVI
            else:
                logger.debug(f"No spectral path for {img_path_str}, simulating NDVI.")
                spectral_tensor = self._simulate_ndvi(rgb_pil_img.copy()) # Use original RGB for NDVI
        
        if spectral_tensor is not None:
            # Ensure correct channel dim and size for spectral tensor
            if spectral_tensor.ndim == 2: spectral_tensor = spectral_tensor.unsqueeze(0)
            if spectral_tensor.shape[0] == 0 : # If somehow it became 0 channels
                logger.warning(f"Spectral tensor has 0 channels for {img_path_str}. Simulating NDVI.")
                spectral_tensor = self._simulate_ndvi(rgb_pil_img.copy())
            elif spectral_tensor.shape[1:] != IMAGE_SIZE_SPECTRAL:
                 logger.warning(f"Resizing spectral tensor {spectral_tensor.shape} to target spectral size post-hoc for {img_path_str}.")
                 resize_spectral = T_old.Resize(IMAGE_SIZE_SPECTRAL, interpolation=T_old.InterpolationMode.BILINEAR, antialias=True)
                 spectral_tensor = resize_spectral(spectral_tensor)
            output_dict["spectral_image"] = spectral_tensor
            
        return output_dict