# phase3_pretraining/dataset.py
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T_v2 # Use v2 for consistency
from PIL import Image, UnidentifiedImageError
import numpy as np
import logging
from collections import Counter

# Config imports will be handled by the main script that uses this dataset
# from .config import DATASET_BASE_PATH, ORIGINAL_DATASET_NAME, AUGMENTED_DATASET_NAME, TRAIN_SPLIT_RATIO

logger = logging.getLogger(__name__)

class SARCLD2024Dataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 img_size: tuple, 
                 split: str = "train", 
                 train_split_ratio: float = 0.8, 
                 use_augmentations: bool = False, # For pretraining, augmentations are usually separate
                 normalize_for_model: bool = False, # For pretraining, often no normalization before SimCLRAug
                 original_dataset_name: str = "Original Dataset",
                 augmented_dataset_name: str = "Augmented Dataset"):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split.lower()
        self.train_split_ratio = train_split_ratio
        self.use_augmentations = use_augmentations # Not used if SimCLRAug is applied later
        self.normalize_for_model = normalize_for_model
        self.original_dataset_name = original_dataset_name
        self.augmented_dataset_name = augmented_dataset_name


        self.classes = [ # As per SAR-CLD-2024 paper/common cotton diseases
            "Bacterial Blight", "Curl Virus", "Healthy Leaf",
            "Herbicide Growth Damage", "Leaf Hopper Jassids",
            "Leaf Redding", "Leaf Variegation"
        ] # This should match NUM_CLASSES in config
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.image_paths = []
        self.labels = [] # Labels are loaded but might not be used for self-supervised pretraining

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory does not exist: {root_dir}")
        logger.info(f"Initializing SARCLD2024Dataset from: {root_dir} for split: {self.split}")

        for dataset_type_name in [self.original_dataset_name, self.augmented_dataset_name]:
            dataset_path = os.path.join(root_dir, dataset_type_name)
            if not os.path.exists(dataset_path):
                logger.warning(f"Dataset path does not exist, skipping: {dataset_path}")
                continue
            
            logger.info(f"Scanning dataset type: {dataset_type_name}")
            for class_name in self.classes:
                class_folder_path = os.path.join(dataset_path, class_name)
                if not os.path.exists(class_folder_path):
                    logger.warning(f"Class folder does not exist, skipping: {class_folder_path}")
                    continue
                
                for img_name in os.listdir(class_folder_path):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_full_path = os.path.join(class_folder_path, img_name)
                        self.image_paths.append(img_full_path)
                        self.labels.append(self.class_to_idx[class_name])
        
        if not self.image_paths:
            raise ValueError(f"No images found. Please check dataset structure in {root_dir} "
                             f"and expected subfolders: {self.classes} within "
                             f"{self.original_dataset_name} and/or {self.augmented_dataset_name}.")

        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        
        logger.info(f"Total images found across all types: {len(self.image_paths)}")
        class_counts = Counter(self.labels)
        for idx, count in sorted(class_counts.items()):
            logger.info(f"  Class '{self.classes[idx]}': {count} samples")

        # Create stratified train/validation split if needed (primarily for linear probing)
        # For pre-training, often the entire dataset (or a large chunk of it) is used as "train".
        indices = np.arange(len(self.image_paths))
        # TODO: Implement stratified shuffle split if precise splitting is needed
        # For now, simple shuffle and split
        np.random.seed(config_RANDOM_SEED if 'config_RANDOM_SEED' in globals() else 42) # Use seed from config if available
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * self.train_split_ratio)
        if self.split == "train":
            self.current_indices = indices[:split_idx]
        elif self.split == "val" or self.split == "validation" or self.split == "test": # Use "val" or "test" for the remainder
            self.current_indices = indices[split_idx:]
        else: # Use all data if split is not 'train' or 'val'/'test'
            logger.warning(f"Unknown split '{self.split}', using all data. Expected 'train', 'val', or 'test'.")
            self.current_indices = indices
            
        logger.info(f"Dataset split '{self.split}' size: {len(self.current_indices)} samples.")

        # Basic transforms: Resize and ToTensor. Augmentations are applied separately for SimCLR.
        # Normalization is also often separate or part of the model's input processing.
        transforms_list = [
            T_v2.Resize(self.img_size, antialias=True),
            T_v2.ToImage(), # Converts PIL to tensor
            T_v2.ToDtype(torch.float32, scale=True) # Scales to [0,1]
        ]
        if self.normalize_for_model: # E.g. if feeding to a model expecting ImageNet normalization
             transforms_list.append(T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        self.base_transform = T_v2.Compose(transforms_list)

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        actual_idx = self.current_indices[idx]
        img_path = self.image_paths[actual_idx]
        label = self.labels[actual_idx] # Label is returned for potential linear probing later
        
        try:
            img = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            logger.error(f"Cannot identify image file {img_path}. Skipping.")
            # Return a placeholder or skip. For DataLoader, best to handle this by filtering list.
            # Here, we might return a dummy image of the right type.
            # This should ideally be caught during initial dataset scan and problematic images excluded.
            dummy_tensor = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return dummy_tensor, torch.tensor(-1, dtype=torch.long) # Dummy label for error

        rgb_tensor = self.base_transform(img)
        # For self-supervised pretraining (SimCLR), only the image tensor (rgb_tensor) is needed.
        # The two augmented views will be created from this by SimCLRAugmentation.
        return rgb_tensor, torch.tensor(label, dtype=torch.long)

# Import RANDOM_SEED from config for reproducibility in dataset split
from phase3_pretraining.config import RANDOM_SEED as config_RANDOM_SEED