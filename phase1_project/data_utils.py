# data_utils.py
import logging
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split

from dataset import CottonLeafDataset
from transforms import get_rgb_transforms
from config import (
    IMAGE_SIZE_RGB, IMAGE_SIZE_SPECTRAL, BATCH_SIZE, NUM_WORKERS,
    DEFAULT_STAGE_MAP, VALIDATION_SPLIT_RATIO, TEST_SPLIT_RATIO
)

logger = logging.getLogger(__name__)

def custom_collate_fn_for_training(batch: List[Dict[str, Any]]) \
        -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Collate function for training. Handles Optional spectral_tensors.
    Filters out items where rgb_image is None (due to loading error).
    Returns: rgb_batch, spectral_batch (can be None), label_batch
    """
    valid_batch = [item for item in batch if item["rgb_image"] is not None and item["label"] != -1]

    if not valid_batch: # All items in batch failed to load
        # This case should be rare with drop_last=True and a reasonable batch size.
        # If it happens, we might return empty tensors or raise an error.
        # For now, let's assume it won't completely empty a batch if dataset is large enough.
        # If BATCH_SIZE is 1 and that one fails, this will be an issue.
        # The DataLoader should ideally skip such problematic indices if identified beforehand.
        logger.warning("Collate function received an entirely empty or error-filled batch.")
        # Depending on strictness, you might raise an error or return dummy tensors.
        # For now, returning empty tensors and hoping the training loop handles it gracefully (e.g., skips batch).
        # This requires the training loop to check batch size.
        return torch.empty(0), None, torch.empty(0, dtype=torch.long)


    rgb_images = torch.stack([item["rgb_image"] for item in valid_batch])
    labels = torch.tensor([item["label"] for item in valid_batch], dtype=torch.long)
    
    batched_spectral: Optional[torch.Tensor] = None
    has_any_spectral_data = any(item["spectral_image"] is not None for item in valid_batch)

    if has_any_spectral_data:
        spectral_images_processed = []
        ref_spectral_shape = None
        # Find a reference shape from the first available spectral image
        for item in valid_batch:
            if item["spectral_image"] is not None:
                ref_spectral_shape = item["spectral_image"].shape
                break
        
        if ref_spectral_shape is None: # Should not happen if has_any_spectral_data is True
             # Fallback if logic error, though one spectral image should exist
            ref_spectral_shape = (1, IMAGE_SIZE_SPECTRAL[0], IMAGE_SIZE_SPECTRAL[1]) # C, H, W

        for item in valid_batch:
            if item["spectral_image"] is not None:
                spectral_images_processed.append(item["spectral_image"])
            else:
                spectral_images_processed.append(torch.zeros(ref_spectral_shape, dtype=torch.float32))
        try:
            batched_spectral = torch.stack(spectral_images_processed)
        except RuntimeError as e:
            logger.error(f"RuntimeError stacking spectral images: {e}. Shapes:")
            for i, s_img in enumerate(spectral_images_processed):
                logger.error(f"  Item {i} shape: {s_img.shape if s_img is not None else 'None'}")
            batched_spectral = None # Fallback
            logger.error("Fallback: Batched spectral set to None due to stacking error.")
            
    return rgb_images, batched_spectral, labels


def get_dataloaders(
    dataset_root: str,
    apply_progression: bool = False,
    use_spectral: bool = False,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_split_ratio: float = VALIDATION_SPLIT_RATIO,
    test_split_ratio: float = TEST_SPLIT_RATIO,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], int, List[str]]:
    """
    Creates and returns train, validation, and optional test DataLoaders.
    Returns: train_loader, val_loader, test_loader (or None), num_classes, class_names
    """
    logger.info(f"Creating DataLoaders for dataset at: {dataset_root}")
    logger.info(f"Apply progression: {apply_progression}, Use spectral: {use_spectral}")
    logger.info(f"Val split: {val_split_ratio}, Test split: {test_split_ratio}")

    transform_train_rgb = get_rgb_transforms(train=True, image_size=IMAGE_SIZE_RGB)
    transform_val_rgb = get_rgb_transforms(train=False, image_size=IMAGE_SIZE_RGB)
    # Spectral transforms can be added here if needed, for now assumed minimal processing in dataset

    # Instantiate the full dataset first to get classes and class_to_idx
    full_dataset_check = CottonLeafDataset(
        root_dir=dataset_root,
        transform_rgb=None, # Not needed yet, just for metadata
        apply_progression=False,
        use_spectral=False
    )
    class_names = full_dataset_check.classes
    class_to_idx = full_dataset_check.class_to_idx
    num_classes = len(class_names)
    logger.info(f"Discovered {num_classes} classes: {class_names}")
    if num_classes == 0:
        raise ValueError("No classes found in the dataset. Please check the dataset directory structure.")

    # Now create datasets with actual transforms
    dataset_train = CottonLeafDataset(
        root_dir=dataset_root,
        transform_rgb=transform_train_rgb,
        stage_map=DEFAULT_STAGE_MAP,
        apply_progression=apply_progression,
        use_spectral=use_spectral,
        class_to_idx=class_to_idx, # Pass shared mapping
        classes=class_names
    )
    dataset_val = CottonLeafDataset(
        root_dir=dataset_root, # Base dataset for val/test uses same root
        transform_rgb=transform_val_rgb,
        stage_map=DEFAULT_STAGE_MAP,
        apply_progression=False, # Typically no progression sim for val/test
        use_spectral=use_spectral,
        class_to_idx=class_to_idx,
        classes=class_names
    )
    
    # Splitting logic
    dataset_size = len(dataset_train) # Use train dataset instance as reference for size
    indices = list(range(dataset_size))
    labels_for_split = [dataset_train.images_metadata[i]['label'] for i in indices] # Get all labels for stratification

    train_indices, val_indices = [], []
    test_indices = [] # Keep test_indices separate

    if test_split_ratio > 0:
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_split_ratio,
            stratify=[labels_for_split[i] for i in indices], # Stratify on all available labels
            random_state=seed
        )
        # Stratify again for train/val from the train_val_indices pool
        # Adjust val_split_ratio to be relative to the train_val set
        adjusted_val_split_ratio = val_split_ratio / (1.0 - test_split_ratio) if (1.0 - test_split_ratio) > 0 else 0
        if adjusted_val_split_ratio > 0 and adjusted_val_split_ratio < 1:
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=adjusted_val_split_ratio,
                stratify=[labels_for_split[i] for i in train_val_indices],
                random_state=seed
            )
        else: # If adjusted_val_split is 0 or >= 1 (e.g. test_split_ratio was too high)
            train_indices = train_val_indices # All remaining go to train, no validation set
            val_indices = []
    elif val_split_ratio > 0 : # No test split, only train/val
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_split_ratio,
            stratify=labels_for_split,
            random_state=seed
        )
    else: # No test or val split, all data for training
        train_indices = indices
        val_indices = []

    if not train_indices:
        raise ValueError("Training set is empty after splitting. Check split ratios and dataset size.")

    train_subset = Subset(dataset_train, train_indices)
    val_subset = Subset(dataset_val, val_indices) if val_indices else None # dataset_val uses val_transforms
    
    logger.info(f"Dataset split: Train: {len(train_subset)} samples")
    if val_subset:
        logger.info(f"Validation: {len(val_subset)} samples")
    else:
        logger.info("Validation: 0 samples (val_split_ratio might be 0 or too small)")
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn_for_training,
        pin_memory=True,
        drop_last=True # Good for training stability if last batch is small
    )
    
    val_loader = None
    if val_subset and len(val_subset) > 0:
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=custom_collate_fn_for_training,
            pin_memory=True
        )

    test_loader = None
    if test_indices:
        # Test dataset should use val_transforms (no heavy augmentation)
        dataset_test = CottonLeafDataset(
            root_dir=dataset_root,
            transform_rgb=transform_val_rgb,
            stage_map=DEFAULT_STAGE_MAP,
            apply_progression=False,
            use_spectral=use_spectral,
            class_to_idx=class_to_idx,
            classes=class_names
        )
        test_subset = Subset(dataset_test, test_indices)
        logger.info(f"Test: {len(test_subset)} samples")
        if len(test_subset) > 0:
            test_loader = DataLoader(
                test_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=custom_collate_fn_for_training,
                pin_memory=True
            )
    else:
        logger.info("Test: 0 samples (test_split_ratio is 0)")

    return train_loader, val_loader, test_loader, num_classes, class_names