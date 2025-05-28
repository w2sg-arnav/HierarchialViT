# phase4_finetuning/utils/augmentations.py
import torch
import torchvision.transforms.v2 as T_v2 
import logging

logger = logging.getLogger(__name__)

class FinetuneAugmentation:
    """ Augmentations suitable for fine-tuning supervised models. """
    def __init__(self, img_size: tuple):
        self.img_size = img_size 
        
        self.transform = T_v2.Compose([
            T_v2.RandomHorizontalFlip(p=0.5),
            T_v2.RandomVerticalFlip(p=0.3), 
            T_v2.RandomRotation(degrees=30), 
            T_v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.2), 
            T_v2.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=5), 
            T_v2.RandomApply([T_v2.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))], p=0.3), 
            T_v2.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
            T_v2.RandomSolarize(threshold=0.5, p=0.1), # Corrected threshold for float [0,1] images
        ])
        logger.info(f"FinetuneAugmentation initialized with enhanced transforms.")
        logger.debug(f"Augmentation pipeline: {self.transform}")


    def __call__(self, x_batch: torch.Tensor) -> torch.Tensor:
        """ Apply augmentations to a batch of image tensors. """
        # Assumes input x_batch is a float tensor in [0,1] range, as normalization is now off in dataset
        return self.transform(x_batch)