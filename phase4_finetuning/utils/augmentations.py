# Enhanced augmentations specifically for cotton leaf disease classification
import torch
import torchvision.transforms.v2 as T_v2
import torchvision.transforms.functional as TF
import logging
import numpy as np
import random
from typing import Tuple, List

logger = logging.getLogger(__name__)

class CottonLeafDiseaseAugmentation:
    """
    Specialized augmentations for cotton leaf disease classification.
    Designed to simulate realistic variations in agricultural imaging conditions.
    """
    def __init__(self, img_size: tuple, severity: str = 'moderate', 
                 use_mixup: bool = True, use_cutmix: bool = True):
        self.img_size = img_size
        self.severity = severity
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        
        # Cotton leaf specific severity configs
        severity_configs = {
            'mild': {
                'rotation_degrees': 15,  # Leaves can be at various angles
                'color_jitter_params': (0.2, 0.2, 0.2, 0.1),  # More aggressive for leaf color variations
                'translate': (0.1, 0.1),
                'scale': (0.8, 1.2),  # Account for different leaf sizes
                'shear': 8,
                'perspective_prob': 0.3,  # Leaves often photographed at angles
                'perspective_scale': 0.2,
                'blur_prob': 0.2,  # Camera blur is common in field conditions
                'blur_kernel': (3, 7),
                'blur_sigma': (0.1, 1.2),
                'erase_prob': 0.1,
                'erase_scale': (0.02, 0.08),
                'lighting_prob': 0.4,  # Field lighting variations
                'shadow_prob': 0.3,
                'noise_prob': 0.2
            },
            'moderate': {
                'rotation_degrees': 25,
                'color_jitter_params': (0.3, 0.3, 0.25, 0.15),
                'translate': (0.15, 0.15),
                'scale': (0.7, 1.3),
                'shear': 12,
                'perspective_prob': 0.4,
                'perspective_scale': 0.25,
                'blur_prob': 0.25,
                'blur_kernel': (3, 9),
                'blur_sigma': (0.1, 1.5),
                'erase_prob': 0.15,
                'erase_scale': (0.02, 0.12),
                'lighting_prob': 0.5,
                'shadow_prob': 0.4,
                'noise_prob': 0.3
            },
            'strong': {
                'rotation_degrees': 35,
                'color_jitter_params': (0.4, 0.4, 0.3, 0.2),
                'translate': (0.2, 0.2),
                'scale': (0.6, 1.4),
                'shear': 15,
                'perspective_prob': 0.5,
                'perspective_scale': 0.3,
                'blur_prob': 0.3,
                'blur_kernel': (3, 11),
                'blur_sigma': (0.1, 2.0),
                'erase_prob': 0.2,
                'erase_scale': (0.02, 0.15),
                'lighting_prob': 0.6,
                'shadow_prob': 0.5,
                'noise_prob': 0.4
            }
        }
        
        config = severity_configs.get(severity, severity_configs['moderate'])
        
        # Build enhanced augmentation pipeline
        transforms_list = [
            T_v2.RandomHorizontalFlip(p=0.5),
            T_v2.RandomVerticalFlip(p=0.5),  # Leaves can be oriented any way
            
            # Geometric transformations
            T_v2.RandomRotation(
                degrees=config['rotation_degrees'],
                interpolation=T_v2.InterpolationMode.BILINEAR,
                fill=0.0
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
            
            # Enhanced color augmentations for leaf diseases
            EnhancedColorJitter(
                brightness=config['color_jitter_params'][0],
                contrast=config['color_jitter_params'][1],
                saturation=config['color_jitter_params'][2],
                hue=config['color_jitter_params'][3]
            ),
            
            # Agricultural-specific augmentations
            T_v2.RandomApply([
                LightingVariation()
            ], p=config['lighting_prob']),
            
            T_v2.RandomApply([
                SyntheticShadow()
            ], p=config['shadow_prob']),
            
            T_v2.RandomApply([
                T_v2.GaussianBlur(
                    kernel_size=self._ensure_odd_tuple(config['blur_kernel']),
                    sigma=config['blur_sigma']
                )
            ], p=config['blur_prob']),
            
            T_v2.RandomApply([
                GaussianNoise(std=0.02)
            ], p=config['noise_prob']),
            
            StableRandomErasing(
                p=config['erase_prob'],
                scale=config['erase_scale'],
                ratio=(0.3, 3.0)  # More varied ratios for leaf shapes
            ),
            
            T_v2.ToDtype(torch.float32, scale=False),
        ]
        
        self.geometric_transform = T_v2.Compose(transforms_list)
        
        # MixUp and CutMix parameters
        self.mixup_alpha = 0.2 if use_mixup else 0.0
        self.cutmix_alpha = 1.0 if use_cutmix else 0.0
        self.mix_prob = 0.3  # Probability of applying mixing augmentations
        
        logger.info(f"CottonLeafDiseaseAugmentation initialized with '{severity}' severity")
    
    def _ensure_odd_tuple(self, val):
        if isinstance(val, int):
            return (val if val % 2 != 0 else val + 1,) * 2
        if isinstance(val, (list, tuple)):
            return tuple(v if v % 2 != 0 else v + 1 for v in val)
        return (3, 3)
    
    def __call__(self, x_batch: torch.Tensor, targets: torch.Tensor = None) -> tuple:
        """
        Apply augmentations including mixing augmentations if targets provided.
        """
        # Ensure float32
        if not x_batch.is_floating_point():
            x_batch_float = x_batch.to(torch.float32)
            if x_batch_float.max() > 1.0 + 1e-5:
                x_batch_float = x_batch_float / 255.0
            x_batch = x_batch_float
        
        # Apply geometric and color augmentations
        augmented = self.geometric_transform(x_batch)
        
        # Apply mixing augmentations if targets provided
        if targets is not None and random.random() < self.mix_prob:
            if random.random() < 0.5 and self.mixup_alpha > 0:
                augmented, targets = self._mixup(augmented, targets)
            elif self.cutmix_alpha > 0:
                augmented, targets = self._cutmix(augmented, targets)
        
        # Stability checks
        augmented = torch.clamp(augmented, min=-3.5, max=3.5)
        
        if not torch.isfinite(augmented).all():
            logger.warning("Non-finite values detected. Returning original batch.")
            if targets is not None:
                return x_batch.clone(), targets
            return x_batch.clone()
        
        if targets is not None:
            return augmented, targets
        return augmented
    
    def _mixup(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        """Apply MixUp augmentation"""
        batch_size = x.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        
        y_a, y_b = y, y[index]
        return mixed_x, (y_a, y_b, lam)
    
    def _cutmix(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        """Apply CutMix augmentation"""
        batch_size = x.size(0)
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        index = torch.randperm(batch_size).to(x.device)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to match exact area ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        
        y_a, y_b = y, y[index]
        return x, (y_a, y_b, lam)
    
    def _rand_bbox(self, size, lam):
        """Generate random bounding box for CutMix"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


class EnhancedColorJitter(torch.nn.Module):
    """Enhanced color jitter with disease-specific variations"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
        # Additional disease-specific color variations
        self.chlorosis_prob = 0.2  # Yellowing simulation
        self.necrosis_prob = 0.15  # Browning simulation
        
    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        # Standard color jitter
        if random.random() < 0.8:
            img_batch = T_v2.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue
            )(img_batch)
        
        # Disease-specific color variations
        if random.random() < self.chlorosis_prob:
            img_batch = self._simulate_chlorosis(img_batch)
        
        if random.random() < self.necrosis_prob:
            img_batch = self._simulate_necrosis(img_batch)
        
        return img_batch
    
    def _simulate_chlorosis(self, img: torch.Tensor) -> torch.Tensor:
        """Simulate yellowing of leaves (chlorosis)"""
        # Increase yellow channel, reduce green slightly
        img_hsv = torch.zeros_like(img)
        for i in range(img.shape[0]):
            img_pil = TF.to_pil_image(img[i])
            img_hsv_pil = img_pil.convert('HSV')
            img_hsv[i] = TF.to_tensor(img_hsv_pil)
        
        # Shift hue slightly towards yellow
        img_hsv[:, 0] = torch.clamp(img_hsv[:, 0] + random.uniform(-0.05, 0.05), 0, 1)
        img_hsv[:, 1] = torch.clamp(img_hsv[:, 1] * random.uniform(0.8, 1.2), 0, 1)
        
        # Convert back to RGB
        for i in range(img.shape[0]):
            img_hsv_pil = TF.to_pil_image(img_hsv[i])
            img_rgb_pil = img_hsv_pil.convert('RGB')
            img[i] = TF.to_tensor(img_rgb_pil)
        
        return img
    
    def _simulate_necrosis(self, img: torch.Tensor) -> torch.Tensor:
        """Simulate browning of leaves (necrosis)"""
        # Reduce saturation and shift towards brown
        brown_factor = random.uniform(0.1, 0.3)
        img[:, 0] = torch.clamp(img[:, 0] * (1 + brown_factor), 0, 1)  # Increase red
        img[:, 1] = torch.clamp(img[:, 1] * (1 - brown_factor * 0.5), 0, 1)  # Slight green reduction
        img[:, 2] = torch.clamp(img[:, 2] * (1 - brown_factor), 0, 1)  # Reduce blue
        return img


class LightingVariation(torch.nn.Module):
    """Simulate various lighting conditions in agricultural settings"""
    def __init__(self):
        super().__init__()
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        variation_type = random.choice(['overexposure', 'underexposure', 'uneven'])
        
        if variation_type == 'overexposure':
            # Simulate bright sunlight
            factor = random.uniform(1.1, 1.4)
            img = torch.clamp(img * factor, 0, 1)
        
        elif variation_type == 'underexposure':
            # Simulate shadow or cloudy conditions
            factor = random.uniform(0.6, 0.9)
            img = img * factor
        
        elif variation_type == 'uneven':
            # Simulate uneven lighting (gradient)
            h, w = img.shape[-2:]
            gradient = torch.linspace(0.7, 1.3, w).view(1, 1, 1, -1).expand_as(img)
            if random.random() < 0.5:  # Vertical gradient
                gradient = gradient.transpose(-1, -2)
            img = torch.clamp(img * gradient, 0, 1)
        
        return img


class SyntheticShadow(torch.nn.Module):
    """Add synthetic shadows to simulate field conditions"""
    def __init__(self):
        super().__init__()
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        b, c, h, w = img.shape
        
        for i in range(b):
            # Create random shadow shape
            shadow_type = random.choice(['ellipse', 'rectangle', 'irregular'])
            
            if shadow_type == 'ellipse':
                mask = self._create_ellipse_mask(h, w)
            elif shadow_type == 'rectangle':
                mask = self._create_rectangle_mask(h, w)
            else:
                mask = self._create_irregular_mask(h, w)
            
            # Apply shadow with random intensity
            shadow_intensity = random.uniform(0.3, 0.7)
            mask = mask * shadow_intensity
            
            img[i] = img[i] * (1 - mask)
        
        return img
    
    def _create_ellipse_mask(self, h: int, w: int) -> torch.Tensor:
        """Create elliptical shadow mask"""
        mask = torch.zeros(h, w)
        center_x, center_y = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
        a, b = random.randint(w//8, w//3), random.randint(h//8, h//3)
        
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        ellipse = ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2
        mask[ellipse <= 1] = 1
        
        return mask.unsqueeze(0)
    
    def _create_rectangle_mask(self, h: int, w: int) -> torch.Tensor:
        """Create rectangular shadow mask"""
        mask = torch.zeros(h, w)
        x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
        x2, y2 = random.randint(w//2, w), random.randint(h//2, h)
        mask[y1:y2, x1:x2] = 1
        
        return mask.unsqueeze(0)
    
    def _create_irregular_mask(self, h: int, w: int) -> torch.Tensor:
        """Create irregular shadow mask"""
        mask = torch.zeros(h, w)
        # Create multiple overlapping circles for irregular shape
        num_circles = random.randint(3, 6)
        
        for _ in range(num_circles):
            center_x, center_y = random.randint(0, w), random.randint(0, h)
            radius = random.randint(min(h, w)//8, min(h, w)//4)
            
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            circle = (x - center_x) ** 2 + (y - center_y) ** 2
            mask[circle <= radius ** 2] = 1
        
        return mask.unsqueeze(0)


class GaussianNoise(torch.nn.Module):
    """Add Gaussian noise to simulate sensor noise"""
    def __init__(self, std=0.02):
        super().__init__()
        self.std = std
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(img) * self.std
        return torch.clamp(img + noise, 0, 1)


class StableRandomErasing(torch.nn.Module):
    """Enhanced random erasing for agricultural images"""
    def __init__(self, p=0.1, scale=(0.02, 0.12), ratio=(0.3, 3.0)):
        super().__init__()
        self.eraser = T_v2.RandomErasing(
            p=p, scale=scale, ratio=ratio, 
            value=0.0, inplace=False
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.eraser(img)


# Factory function for cotton leaf disease augmentations
def create_cotton_leaf_augmentation(
    strategy: str = 'cotton_disease', 
    img_size: tuple = (448, 448), 
    **kwargs
):
    """
    Factory function for cotton leaf disease augmentations.
    
    Args:
        strategy: 'cotton_disease', 'stable_enhanced', 'minimal'
        img_size: Image size tuple
        **kwargs: Additional arguments
    """
    logger.info(f"Creating augmentation strategy: '{strategy}' for cotton leaf disease")
    
    if strategy == 'cotton_disease':
        severity = kwargs.get('severity', 'moderate')
        use_mixup = kwargs.get('use_mixup', True)
        use_cutmix = kwargs.get('use_cutmix', True)
        return CottonLeafDiseaseAugmentation(
            img_size, severity=severity, 
            use_mixup=use_mixup, use_cutmix=use_cutmix
        )
    else:
        # Fall back to original augmentations
        from .augmentations import create_augmentation
        return create_augmentation(strategy, img_size, **kwargs)