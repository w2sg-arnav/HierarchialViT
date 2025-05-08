# phase4_finetuning/utils/augmentations.py
import torch
import torchvision.transforms.v2 as T_v2 # Use v2
import logging

logger = logging.getLogger(__name__)

class FinetuneAugmentation:
    """ Augmentations suitable for fine-tuning supervised models. """
    def __init__(self, img_size: tuple):
        self.img_size = img_size # img_size is mainly for reference here, resize happens in dataset
        
        # Common fine-tuning augmentations
        self.transform = T_v2.Compose([
            T_v2.RandomHorizontalFlip(p=0.5),
            T_v2.RandomVerticalFlip(p=0.5),
            T_v2.RandomRotation(degrees=20), 
            T_v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), 
            T_v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            T_v2.RandomApply([ 
                T_v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)) 
            ], p=0.2),
        ])
        logger.info(f"FinetuneAugmentation initialized.")

    def __call__(self, x_batch: torch.Tensor) -> torch.Tensor:
        """ Apply augmentations to a batch of image tensors. """
        # Assumes input x_batch is already tensorized and potentially normalized
        return self.transform(x_batch)