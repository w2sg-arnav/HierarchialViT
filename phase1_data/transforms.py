# transforms.py
import torch
import torchvision.transforms.v2 as T # Using v2 for modern features like antialias in Resize
from config import IMAGE_SIZE

# Normalization constants for ImageNet pre-trained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(train: bool = True, image_size: tuple = IMAGE_SIZE) -> T.Compose:
    """
    Return augmentation pipeline for training or evaluation.
    Uses torchvision.transforms.v2.
    Args:
        train (bool): If True, returns training augmentations, else validation/test augmentations.
        image_size (tuple): Target image size (height, width).
    """
    if train:
        # Comprehensive training augmentations
        return T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=20),
            T.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=(-10, 10, -10, 10)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
            # Convert to image tensor; ToDtype scales to [0,1] if input is uint8
            T.ToImage(), 
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Minimal augmentations for validation/testing
        return T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

def denormalize_image(tensor_image: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """Denormalizes a tensor image."""
    if tensor_image is None:
        return None
    mean_tensor = torch.tensor(mean, device=tensor_image.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=tensor_image.device).view(-1, 1, 1)
    
    denorm_image = tensor_image * std_tensor + mean_tensor
    return torch.clamp(denorm_image, 0, 1)