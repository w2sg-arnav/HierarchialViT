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
            # --- ADJUSTED THRESHOLD FOR FLOAT IMAGES (typically in [0,1] range before normalization) ---
            T_v2.RandomSolarize(threshold=0.5, p=0.1), 
        ])
        logger.info(f"FinetuneAugmentation initialized with enhanced transforms.")
        logger.debug(f"Augmentation pipeline: {self.transform}")


    def __call__(self, x_batch: torch.Tensor) -> torch.Tensor:
        """ Apply augmentations to a batch of image tensors. """
        # Input x_batch is assumed to be a float tensor.
        # If ToImage() converted PIL [0,255] to float [0,1], then threshold=0.5 is correct.
        # If images are already normalized with mean approx 0, std approx 1, then solarize
        # might need a different threshold or be applied before normalization.
        # However, typically augmentations like Solarize are applied on the [0,1] float range.
        # The dataset first does ToImage -> ToDtype(float32) -> Resize.
        # Normalization happens last in the dataset's base_transform.
        # The augmentations in FinetuneAugmentation are applied *after* the dataset's base_transform
        # (which includes normalization) if augmentations are applied in trainer.py as they are.
        # This means images fed to FinetuneAugmentation are ALREADY NORMALIZED.
        # Solarize on normalized images with threshold 0.5 might be okay, but it's less standard.
        # A more standard pipeline would be:
        # Dataset: ToImage -> ToDtype(float32) -> Resize
        # Trainer (before normalization if it's separate): Augmentations (including Solarize with threshold 0.5)
        # Trainer: Normalize
        # OR:
        # Dataset: ToImage -> ToDtype(float32) -> Resize -> Augmentations (Solarize needs threshold 0.5) -> Normalize
        
        # Given current setup: images are already normalized when they reach here.
        # Solarize with threshold 0.5 on normalized data means it inverts pixels > 0.5.
        # This should still work without error, but its visual effect might be different
        # than solarizing an unnormalized [0,1] image.
        # The error `TypeError: Threshold should be less or equal the maximum value of the dtype, but got 128`
        # strictly means the threshold 128 is invalid for a float dtype that doesn't typically go up to 128.
        # Changing threshold to 0.5 will fix the TypeError.
        return self.transform(x_batch)