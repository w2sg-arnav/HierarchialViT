# transforms.py
import torch
import torchvision.transforms.v2 as T_v2 # Using v2 for modern features
from config import IMAGE_SIZE_RGB

# Normalization constants for ImageNet pre-trained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_rgb_transforms(train: bool = True, image_size: tuple = IMAGE_SIZE_RGB) -> T_v2.Compose:
    """
    Return augmentation pipeline for RGB images.
    Uses torchvision.transforms.v2.
    Args:
        train (bool): If True, returns training augmentations, else validation/test augmentations.
        image_size (tuple): Target image size (height, width).
    """
    if train:
        # Comprehensive training augmentations
        return T_v2.Compose([
            T_v2.Resize(image_size, interpolation=T_v2.InterpolationMode.BILINEAR, antialias=True),
            T_v2.RandomHorizontalFlip(p=0.5),
            T_v2.RandomVerticalFlip(p=0.5),
            T_v2.RandomRotation(degrees=30),
            T_v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5, -5, 5)),
            T_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T_v2.RandomApply([T_v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
            T_v2.ToImage(),  # Converts PIL to tensor, does not scale
            T_v2.ToDtype(torch.float32, scale=True), # Scales to [0,1] if input is uint8
            T_v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Minimal augmentations for validation/testing
        return T_v2.Compose([
            T_v2.Resize(image_size, interpolation=T_v2.InterpolationMode.BILINEAR, antialias=True),
            T_v2.ToImage(),
            T_v2.ToDtype(torch.float32, scale=True),
            T_v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

def denormalize_image_tensor(tensor_image: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """Denormalizes a tensor image for visualization."""
    if tensor_image is None:
        return None
    # Ensure mean and std are tensors and on the same device as the image
    mean_tensor = torch.tensor(mean, device=tensor_image.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=tensor_image.device).view(-1, 1, 1)
    
    denorm_image = tensor_image.clone() # Clone to avoid modifying the original tensor
    denorm_image.mul_(std_tensor).add_(mean_tensor) # In-place multiplication and addition on the clone
    return torch.clamp(denorm_image, 0, 1)

# Note: Spectral transforms might be different, e.g., no color jitter.
# For now, spectral data is assumed to be processed directly to tensor and resized.
# If more complex spectral augmentations are needed, a get_spectral_transforms function can be added.