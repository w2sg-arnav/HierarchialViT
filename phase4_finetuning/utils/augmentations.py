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
                'erase_prob': 0.1,
                'erase_scale': (0.02, 0.1)
            }
        }

        config = severity_configs.get(severity, severity_configs['mild'])

        # Build augmentation pipeline with stability checks
        transforms_list = [
            T_v2.RandomHorizontalFlip(p=0.5),
            T_v2.RandomVerticalFlip(p=0.3),
            T_v2.RandomRotation(
                degrees=config['rotation_degrees'],
                interpolation=T_v2.InterpolationMode.BILINEAR,
                fill=0.0 # Fill with 0.0, assumes normalized data mean is close to 0
            ),
            StableColorJitter( # Custom stable color jitter
                brightness=config['color_jitter_params'][0],
                contrast=config['color_jitter_params'][1],
                saturation=config['color_jitter_params'][2],
                hue=config['color_jitter_params'][3]
            ),
            T_v2.RandomAffine(
                degrees=config['rotation_degrees'] // 2, # Lesser rotation for affine
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
                    kernel_size=(3, 3), # Smaller kernel for less blur
                    sigma=(0.1, 0.5)    # Tighter sigma range
                )
            ], p=config['blur_prob']),
            StableRandomErasing( # Custom stable random erasing
                p=config['erase_prob'],
                scale=config['erase_scale'],
                ratio=(0.5, 2.0) # standard aspect ratio
            ),
        ]

        self.transform = T_v2.Compose(transforms_list)

        logger.info(f"StableEnhancedFinetuneAugmentation initialized with '{severity}' severity for stability.")
        logger.debug(f"Augmentation pipeline: {self.transform}")

    def __call__(self, x_batch: torch.Tensor) -> torch.Tensor:
        """ Apply stable augmentations to a batch of normalized image tensors. """
        augmented = self.transform(x_batch)

        # Clamp values to prevent extreme outliers that could cause NaNs
        # For ImageNet normalized data, most values are within [-2.5, 2.5] roughly.
        # Clamping to [-3, 3] or [-4, 4] can be a safeguard.
        augmented = torch.clamp(augmented, min=-3.5, max=3.5)

        if not torch.isfinite(augmented).all():
            logger.warning("Non-finite values detected after augmentation and clamping. Returning original batch.")
            # Ensure the original batch is returned with the same device and dtype
            return x_batch.clone()

        return augmented


class StableColorJitter(torch.nn.Module):
    """ Stable version of ColorJitter.
        Applies brightness, contrast, saturation, and hue adjustments.
        Designed to be gentler on normalized data.
    """
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05):
        super().__init__()
        self.brightness_factor_range = (max(0, 1 - brightness), 1 + brightness)
        self.contrast_factor_range = (max(0, 1 - contrast), 1 + contrast)
        self.saturation_factor_range = (max(0, 1 - saturation), 1 + saturation)
        self.hue_factor_range = (-hue, hue) # hue is additive shift in HSV space

        self.prob_apply_all = 0.8 # Probability to apply the jittering block

    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) > self.prob_apply_all:
            return img_batch

        # Create a transform list for jittering
        jitter_transforms = []
        if self.brightness_factor_range[0] < self.brightness_factor_range[1] : # if brightness is non-zero
             jitter_transforms.append(T_v2.ColorJitter(brightness=self.brightness_factor_range, contrast=0, saturation=0, hue=0))
        if self.contrast_factor_range[0] < self.contrast_factor_range[1]:
             jitter_transforms.append(T_v2.ColorJitter(brightness=0, contrast=self.contrast_factor_range, saturation=0, hue=0))
        if self.saturation_factor_range[0] < self.saturation_factor_range[1]:
             jitter_transforms.append(T_v2.ColorJitter(brightness=0, contrast=0, saturation=self.saturation_factor_range, hue=0))
        if self.hue_factor_range[0] < self.hue_factor_range[1]:
             jitter_transforms.append(T_v2.ColorJitter(brightness=0, contrast=0, saturation=0, hue=self.hue_factor_range))

        if not jitter_transforms:
            return img_batch

        # Apply transforms in a random order
        order = torch.randperm(len(jitter_transforms)).tolist()
        img_processed = img_batch
        for i in order:
            img_processed = jitter_transforms[i](img_processed)
        
        # No need to clamp here, main augmentation class will clamp.
        return img_processed


class StableRandomErasing(torch.nn.Module):
    """ Stable version of RandomErasing with proper fill values for normalized data. """
    def __init__(self, p=0.05, scale=(0.01, 0.05), ratio=(0.5, 2.0), value_mode='zero'):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value_mode = value_mode # 'zero', 'noise', or specific float value

    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        # T_v2.RandomErasing handles batch internally
        if self.value_mode == 'zero':
            fill_value = 0.0
        elif self.value_mode == 'noise':
            # For noise, T_v2.RandomErasing 'random' value might be better if it's adapted for tensor input range.
            # Or, implement custom noise per patch. For simplicity, let's use 0 for now or allow 'random'
            fill_value = 'random' # T_v2 'random' typically samples from image itself or U(0,1)
                                # For normalized data, 0 is a safer bet if 'random' isn't ideal.
                                # Let's stick to T_v2's RandomErasing and use its value param
            fill_value = 0.0 # Overriding to 0 for more stability on normalized data
        else:
            fill_value = float(self.value_mode)

        eraser = T_v2.RandomErasing(p=self.p, scale=self.scale, ratio=self.ratio, value=fill_value, inplace=False)
        return eraser(img_batch)


class MinimalStableAugmentation:
    """ Minimal augmentation strategy for maximum stability during fine-tuning. """
    def __init__(self, img_size: tuple):
        self.img_size = img_size
        self.transform = T_v2.Compose([
            T_v2.RandomHorizontalFlip(p=0.5),
            # T_v2.RandomVerticalFlip(p=0.3), # Vertical flip might be too much for some datasets
            T_v2.RandomRotation(degrees=5, fill=0.0, interpolation=T_v2.InterpolationMode.BILINEAR),
        ])
        logger.info("MinimalStableAugmentation initialized - maximum stability mode.")

    def __call__(self, x_batch: torch.Tensor) -> torch.Tensor:
        augmented = self.transform(x_batch)
        # Even minimal augmentations might benefit from a light clamp on normalized data.
        augmented = torch.clamp(augmented, min=-3.5, max=3.5)
        if not torch.isfinite(augmented).all():
            logger.warning("Non-finite values detected after MINIMAL augmentation. Returning original batch.")
            return x_batch.clone()
        return augmented

# For backward compatibility with the previous 'EnhancedFinetuneAugmentation' name if used directly
class EnhancedFinetuneAugmentation(StableEnhancedFinetuneAugmentation):
    def __init__(self, img_size: tuple):
        logger.warning("Legacy 'EnhancedFinetuneAugmentation' called. Redirecting to 'StableEnhancedFinetuneAugmentation' with 'moderate' severity.")
        super().__init__(img_size, severity='moderate')


# Factory function for easy switching
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
    elif strategy == 'enhanced': # Legacy call
        return EnhancedFinetuneAugmentation(img_size) # Redirects to StableEnhanced with moderate
    else:
        logger.warning(f"Unknown augmentation strategy: {strategy}. Defaulting to no augmentations (None).")
        return None