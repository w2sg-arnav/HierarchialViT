# phase4_finetuning/utils/augmentations.py
import torch
import torchvision.transforms.v2 as T_v2
import logging
import numpy as np

logger = logging.getLogger(__name__)

class StableEnhancedFinetuneAugmentation:
    """ Stable enhanced augmentations for fine-tuning supervised models.
        Designed to prevent NaN losses while maintaining training robustness.
        Applies conservative augmentations with proper value clamping.
    """
    def __init__(self, img_size: tuple, severity: str = 'mild'):
        self.img_size = img_size
        self.severity = severity

        # Define severity levels
        severity_configs = {
            'mild': {
                'rotation_degrees': 10,
                'color_jitter_params': (0.1, 0.1, 0.1, 0.05), # brightness, contrast, saturation, hue
                'translate': (0.05, 0.05),
                'scale': (0.95, 1.05),
                'shear': 5,
                'perspective_prob': 0.15,
                'perspective_scale': 0.1,
                'blur_prob': 0.1,
                'blur_kernel': (3,3), # Ensure odd kernel
                'blur_sigma': (0.1, 0.8), # Slightly wider sigma for mild
                'erase_prob': 0.05,
                'erase_scale': (0.01, 0.05)
            },
            'moderate': {
                'rotation_degrees': 15,
                'color_jitter_params': (0.15, 0.15, 0.15, 0.08),
                'translate': (0.08, 0.08),
                'scale': (0.9, 1.1),
                'shear': 8,
                'perspective_prob': 0.2,
                'perspective_scale': 0.15,
                'blur_prob': 0.15,
                'blur_kernel': (3,3),
                'blur_sigma': (0.1, 1.0), # Wider sigma for moderate
                'erase_prob': 0.08,
                'erase_scale': (0.02, 0.08)
            },
            'strong': {
                'rotation_degrees': 20,
                'color_jitter_params': (0.2, 0.2, 0.2, 0.1),
                'translate': (0.1, 0.1),
                'scale': (0.85, 1.15),
                'shear': 10,
                'perspective_prob': 0.25,
                'perspective_scale': 0.2,
                'blur_prob': 0.2,
                'blur_kernel': (3,5), # Allow slightly larger kernel too
                'blur_sigma': (0.1, 1.5),
                'erase_prob': 0.1,
                'erase_scale': (0.02, 0.1)
            }
        }

        config = severity_configs.get(severity, severity_configs['mild'])
        
        # Ensure blur kernel is tuple of odd integers
        def _ensure_odd_tuple(val):
            if isinstance(val, int):
                return (val if val % 2 != 0 else val + 1,) * 2
            if isinstance(val, (list, tuple)):
                return tuple(v if v % 2 != 0 else v + 1 for v in val)
            return (3,3) # Default

        blur_kernel_size = _ensure_odd_tuple(config['blur_kernel'])


        # Build augmentation pipeline with stability checks
        transforms_list = [
            T_v2.RandomHorizontalFlip(p=0.5),
            T_v2.RandomVerticalFlip(p=0.3), # Keep this moderate, can be dataset specific
            T_v2.RandomRotation(
                degrees=config['rotation_degrees'],
                interpolation=T_v2.InterpolationMode.BILINEAR,
                fill=0.0 
            ),
            StableColorJitter( 
                brightness=config['color_jitter_params'][0],
                contrast=config['color_jitter_params'][1],
                saturation=config['color_jitter_params'][2],
                hue=config['color_jitter_params'][3]
            ),
            T_v2.RandomAffine(
                degrees=config['rotation_degrees'] // 2, 
                translate=config['translate'],
                scale=config['scale'],
                shear=config['shear'],
                interpolation=T_v2.InterpolationMode.BILINEAR,
                fill=0.0
            ),
            T_v2.RandomPerspective(
                distortion_scale=config['perspective_scale'],
                p=config['perspective_prob'],
                interpolation=T_v2.InterpolationMode.BILINEAR,
                fill=0.0
            ),
            T_v2.RandomApply([
                T_v2.GaussianBlur(
                    kernel_size=blur_kernel_size,
                    sigma=config['blur_sigma']
                )
            ], p=config['blur_prob']),
            StableRandomErasing( 
                p=config['erase_prob'],
                scale=config['erase_scale'],
                ratio=(0.5, 2.0) 
            ),
            T_v2.ToDtype(torch.float32, scale=False), # Ensure float32 for model
            # Normalization should happen in dataset or right before model if not part of aug pipeline
            # T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Example
        ]

        self.transform = T_v2.Compose(transforms_list)

        logger.info(f"StableEnhancedFinetuneAugmentation initialized with '{severity}' severity for stability.")
        logger.debug(f"Augmentation pipeline: {self.transform}")

    def __call__(self, x_batch: torch.Tensor) -> torch.Tensor:
        """ Apply stable augmentations to a batch of image tensors.
            Assumes input x_batch is already normalized if normalization is used.
        """
        # Ensure input is float32, as some transforms expect it.
        # If input can be uint8 [0,255], it should be converted to float [0,1] or [0,255] first.
        # Assuming x_batch is already float (e.g. [0,1] or normalized [-X, X])
        
        # Check if input is already float. If not, convert.
        # This check is important if dataset yields uint8 tensors.
        if not x_batch.is_floating_point():
            # logger.debug("Input batch is not floating point. Converting to float32 and scaling to [0,1] if max > 1.")
            x_batch_float = x_batch.to(torch.float32)
            if x_batch_float.max() > 1.0 + 1e-5: # Heuristic for [0, 255] range
                 x_batch_float = x_batch_float / 255.0
            x_batch = x_batch_float
        
        augmented = self.transform(x_batch)

        # Clamping values to prevent extreme outliers.
        # This range should be appropriate for ImageNet-normalized data.
        # If data is [0,1], clamp to [0,1].
        # Assuming normalized data for this clamp range.
        # If your data is [0,1] before normalization, clamping to [0,1] after augmentations (before norm) is safer.
        # If data is already normalized, this clamp is a safeguard.
        min_val, max_val = -3.5, 3.5 # For typical ImageNet normalized data
        # Example: if data was [0,1] and then normalized, this clamp is post-normalization.
        # If augmentations are applied on [0,1] data *before* normalization, clamp to [0,1].

        augmented = torch.clamp(augmented, min=min_val, max=max_val)


        if not torch.isfinite(augmented).all():
            logger.warning("Non-finite values detected after augmentation and clamping. Returning original batch.")
            return x_batch.clone() # Return a clone of the original pre-augmentation batch

        return augmented


class StableColorJitter(torch.nn.Module):
    """ Stable version of ColorJitter.
        Applies brightness, contrast, saturation, and hue adjustments.
    """
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05):
        super().__init__()
        # torchvision ColorJitter expects factors for brightness, contrast, saturation
        # and a range [-hue, hue] for hue.
        self.brightness_factor = brightness if isinstance(brightness, (tuple, list)) else (max(0, 1 - brightness), 1 + brightness)
        self.contrast_factor = contrast if isinstance(contrast, (tuple, list)) else (max(0, 1 - contrast), 1 + contrast)
        self.saturation_factor = saturation if isinstance(saturation, (tuple, list)) else (max(0, 1 - saturation), 1 + saturation)
        self.hue_factor = hue if isinstance(hue, (tuple, list)) else (-hue, hue)
        
        self.prob_apply_all = 0.8 
        self.jitter_transform = T_v2.ColorJitter(
            brightness=self.brightness_factor,
            contrast=self.contrast_factor,
            saturation=self.saturation_factor,
            hue=self.hue_factor
        )
        # T_v2.ColorJitter applies them in a random order by default if multiple are non-zero.
        # The previous implementation built a list and shuffled; T_v2.ColorJitter handles this.

    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) > self.prob_apply_all:
            return img_batch
        
        # Apply the single T_v2.ColorJitter transform
        img_processed = self.jitter_transform(img_batch)
        
        return img_processed


class StableRandomErasing(torch.nn.Module):
    """ Stable version of RandomErasing with proper fill values for normalized data. """
    def __init__(self, p=0.05, scale=(0.01, 0.05), ratio=(0.5, 2.0), value_mode='zero'):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value_mode = value_mode 

        if self.value_mode == 'zero':
            self.fill_value = 0.0
        elif self.value_mode == 'noise':
            # T_v2.RandomErasing's 'random' value samples from U(0,1) by default,
            # which might not be ideal for already normalized data around zero.
            # Using 0.0 is often safer for normalized data.
            self.fill_value = 0.0 # Default to zero for normalized data if 'noise' specified
            logger.debug("StableRandomErasing: 'noise' mode selected, using fill_value=0.0 for stability with normalized data.")
        elif isinstance(self.value_mode, (int, float)):
            self.fill_value = float(self.value_mode)
        else: # Potentially 'random' string for T_v2
            self.fill_value = self.value_mode


        self.eraser = T_v2.RandomErasing(p=self.p, scale=self.scale, ratio=self.ratio, value=self.fill_value, inplace=False)

    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        return self.eraser(img_batch)


class MinimalStableAugmentation:
    """ Minimal augmentation strategy for maximum stability during fine-tuning. """
    def __init__(self, img_size: tuple):
        self.img_size = img_size
        self.transform = T_v2.Compose([
            T_v2.RandomHorizontalFlip(p=0.5),
            T_v2.RandomRotation(degrees=5, fill=0.0, interpolation=T_v2.InterpolationMode.BILINEAR),
            T_v2.ToDtype(torch.float32, scale=False),
        ])
        logger.info("MinimalStableAugmentation initialized - maximum stability mode.")

    def __call__(self, x_batch: torch.Tensor) -> torch.Tensor:
        if not x_batch.is_floating_point():
            x_batch_float = x_batch.to(torch.float32)
            if x_batch_float.max() > 1.0 + 1e-5: # Heuristic for [0, 255] range
                 x_batch_float = x_batch_float / 255.0
            x_batch = x_batch_float

        augmented = self.transform(x_batch)
        augmented = torch.clamp(augmented, min=-3.5, max=3.5) # Clamp for normalized data
        if not torch.isfinite(augmented).all():
            logger.warning("Non-finite values detected after MINIMAL augmentation. Returning original batch.")
            return x_batch.clone()
        return augmented

# For backward compatibility
class EnhancedFinetuneAugmentation(StableEnhancedFinetuneAugmentation):
    def __init__(self, img_size: tuple):
        logger.warning("Legacy 'EnhancedFinetuneAugmentation' called. Redirecting to 'StableEnhancedFinetuneAugmentation' with 'moderate' severity.")
        super().__init__(img_size, severity='moderate')


# Factory function
def create_augmentation(strategy: str = 'stable_enhanced', img_size: tuple = (448, 448), **kwargs):
    """ Factory function to create augmentation strategies.
    Args:
        strategy: 'stable_enhanced', 'minimal', or 'enhanced' (legacy)
        img_size: Image size tuple
        **kwargs: Additional arguments (e.g., severity for stable_enhanced)
    """
    logger.info(f"Creating augmentation strategy: '{strategy}' with img_size: {img_size}, kwargs: {kwargs}")
    if strategy == 'stable_enhanced':
        severity = kwargs.get('severity', 'mild')
        return StableEnhancedFinetuneAugmentation(img_size, severity=severity)
    elif strategy == 'minimal':
        return MinimalStableAugmentation(img_size)
    elif strategy == 'enhanced': 
        return EnhancedFinetuneAugmentation(img_size)
    else:
        logger.warning(f"Unknown augmentation strategy: {strategy}. Defaulting to no augmentations (None).")
        return None