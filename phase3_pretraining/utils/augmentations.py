# phase3_pretraining/utils/augmentations.py
from typing import Optional, Union, Tuple
import torch
import torchvision.transforms.v2 as T_v2 # Use transforms.v2
import logging

# Attempt to import config to make augmentations configurable, with fallbacks
try:
    from ..config import config as phase3_cfg
except ImportError:
    phase3_cfg = {} # Fallback to an empty dict if config cannot be imported

logger = logging.getLogger(__name__)

class SimCLRAugmentation:
    """
    SimCLR style augmentations using torchvision.transforms.v2.
    Applies two random views to an input image tensor.
    Expects input tensor to be in [0, 1] range if not normalized.
    """
    def __init__(self, img_size: tuple,
                 s: Optional[float] = None, # Strength of color jitter
                 p_grayscale: Optional[float] = None,
                 p_gaussian_blur: Optional[float] = None,
                 rrc_scale_min: Optional[float] = None
                 ):
        self.img_size = img_size

        # Get parameters from config if available, else use provided args or defaults
        _s = s if s is not None else phase3_cfg.get("simclr_s", 1.0)
        _p_grayscale = p_grayscale if p_grayscale is not None else phase3_cfg.get("simclr_p_grayscale", 0.2)
        _p_gaussian_blur = p_gaussian_blur if p_gaussian_blur is not None else phase3_cfg.get("simclr_p_gaussian_blur", 0.5)
        # Use the more aggressive RRC scale from config if available, else default to SimCLR paper's 0.08
        _rrc_scale_min = rrc_scale_min if rrc_scale_min is not None else phase3_cfg.get("simclr_rrc_scale_min", 0.08)

        logger.info(
            f"SimCLRAugmentation initialized for img_size {img_size} with effective params: "
            f"s={_s}, p_grayscale={_p_grayscale}, p_gaussian_blur={_p_gaussian_blur}, rrc_scale_min={_rrc_scale_min}"
        )

        # Kernel size for GaussianBlur should be odd and positive
        # A common practice is to make it proportional to image size, e.g., 10% of height/width
        # Ensure it's an odd integer: (size // 10) | 1 might not be robust if size//10 is 0.
        # Let's use a fixed reasonable kernel size or make it more robust
        # For 224px, kernel_size=23 is common. For 448px, kernel_size could be larger.
        # Here, we ensure it's at least 3 and odd.
        blur_kernel_val = max(3, (self.img_size[0] // 20) * 2 + 1) # e.g. 448//20 = 22 -> 45. 224//20 = 11 -> 23.

        self.transform = T_v2.Compose([
            T_v2.RandomResizedCrop(size=img_size, scale=(_rrc_scale_min, 1.0), antialias=True), # Using configured/SimCLR default scale
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
        """
        Args:
            x_batch (torch.Tensor): Batch of images, shape [B, C, H, W].
                                    Expected to be on the correct device.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Two augmented views of the batch.
        """
        view1 = self.transform(x_batch)
        view2 = self.transform(x_batch)
        return view1, view2