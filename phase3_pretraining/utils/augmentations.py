# phase3_pretraining/utils/augmentations.py
from typing import Optional, Tuple
import torch
import torchvision.transforms.v2 as T_v2
import logging

try:
    from ..config import config as phase3_cfg # Relative import
except ImportError:
    phase3_cfg = {} # Fallback if config cannot be imported

logger = logging.getLogger(__name__)

class SimCLRAugmentation:
    def __init__(self, img_size: tuple,
                 s: Optional[float] = None,
                 p_grayscale: Optional[float] = None,
                 p_gaussian_blur: Optional[float] = None,
                 rrc_scale_min: Optional[float] = None): # Added rrc_scale_min argument
        self.img_size = img_size

        # Use parameters from config if available, else use provided args or defaults
        _s = s if s is not None else phase3_cfg.get("simclr_s", 1.0)
        _p_grayscale = p_grayscale if p_grayscale is not None else phase3_cfg.get("simclr_p_grayscale", 0.2)
        _p_gaussian_blur = p_gaussian_blur if p_gaussian_blur is not None else phase3_cfg.get("simclr_p_gaussian_blur", 0.5)
        _rrc_scale_min = rrc_scale_min if rrc_scale_min is not None else phase3_cfg.get("simclr_rrc_scale_min", 0.08) # Explicitly use from config or arg

        logger.info(f"SimCLRAugmentation: img_size={img_size}, s={_s}, p_gray={_p_grayscale}, p_blur={_p_gaussian_blur}, rrc_min_scale={_rrc_scale_min}")

        blur_kernel_val = max(3, (self.img_size[0] // 20) * 2 + 1)

        self.transform = T_v2.Compose([
            T_v2.RandomResizedCrop(size=img_size, scale=(_rrc_scale_min, 1.0), antialias=True), # Uses the resolved _rrc_scale_min
            T_v2.RandomHorizontalFlip(p=0.5),
            T_v2.RandomApply([
                T_v2.ColorJitter(brightness=0.8*_s, contrast=0.8*_s, saturation=0.8*_s, hue=0.2*_s)
            ], p=0.8),
            T_v2.RandomGrayscale(p=_p_grayscale),
            T_v2.RandomApply([
                T_v2.GaussianBlur(kernel_size=blur_kernel_val, sigma=(0.1, 2.0))
            ], p=_p_gaussian_blur),
        ])

    def __call__(self, x_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        view1 = self.transform(x_batch)
        view2 = self.transform(x_batch)
        return view1, view2