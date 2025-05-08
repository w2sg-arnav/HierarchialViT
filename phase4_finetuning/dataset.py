# phase4_finetuning/dataset.py
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T_v2 # Use v2 transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
import logging
from collections import Counter

# Import RANDOM_SEED from the consolidated config if needed for split consistency
# This relies on the project root being in sys.path when this module is loaded.
try:
    from phase4_finetuning.config import config as default_config
    config_RANDOM_SEED = default_config['seed']
except ImportError:
    config_RANDOM_SEED = 42 # Fallback seed

logger = logging.getLogger(__name__)

class SARCLD2024Dataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 img_size: tuple, 
                 split: str = "train", 
                 train_split_ratio: float = 0.8, 
                 normalize_for_model: bool = True, # Default True for fine-tuning
                 original_dataset_name: str = "Original Dataset",
                 augmented_dataset_name: str = "Augmented Dataset",
                 random_seed: int = config_RANDOM_SEED # Use seed from config/default
                 ):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split.lower()
        self.train_split_ratio = train_split_ratio
        self.normalize = normalize_for_model
        # Note: self.use_augmentations is removed, augmentations handled externally

        self.classes = [
            "Bacterial Blight", "Curl Virus", "Healthy Leaf",
            "Herbicide Growth Damage", "Leaf Hopper Jassids",
            "Leaf Redding", "Leaf Variegation"
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory does not exist: {root_dir}")
        logger.info(f"Initializing SARCLD2024Dataset from: {root_dir} for split: {self.split}")

        for dataset_type_name in [original_dataset_name, augmented_dataset_name]:
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
                
                valid_extensions = (".jpg", ".jpeg", ".png")
                try:
                    for img_name in os.listdir(class_folder_path):
                        if img_name.lower().endswith(valid_extensions):
                            img_full_path = os.path.join(class_folder_path, img_name)
                            try:
                                with Image.open(img_full_path) as img_test:
                                    img_test.verify() 
                                self.image_paths.append(img_full_path)
                                self.labels.append(self.class_to_idx[class_name])
                            except (UnidentifiedImageError, IOError, SyntaxError) as img_err:
                                logger.warning(f"Skipping unreadable/corrupt image: {img_full_path} ({img_err})")
                except OSError as os_err:
                     logger.error(f"OS error scanning folder {class_folder_path}: {os_err}")

        
        if not self.image_paths:
            raise ValueError(f"No valid images found. Please check dataset structure in {root_dir}.")

        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        
        logger.info(f"Total valid images found: {len(self.image_paths)}")
        class_counts = Counter(self.labels)
        self.class_weights_computed = None 
        for idx, count in sorted(class_counts.items()):
            logger.info(f"  Class '{self.classes[idx]}': {count} samples")

        indices = np.arange(len(self.image_paths))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * self.train_split_ratio)
        if self.split == "train":
            self.current_indices = indices[:split_idx]
        elif self.split in ["val", "validation", "test"]:
            self.current_indices = indices[split_idx:]
        else:
            logger.error(f"Invalid split name '{self.split}'. Use 'train', 'val', or 'test'.")
            raise ValueError(f"Invalid split name '{self.split}'")
            
        # Store labels for the current split (useful for weighted sampler)
        self.current_split_labels = self.labels[self.current_indices]
        logger.info(f"Dataset split '{self.split}' size: {len(self.current_indices)} samples.")

        # Base transforms
        transforms_list = [
            T_v2.Resize(self.img_size, interpolation=T_v2.InterpolationMode.BICUBIC, antialias=True),
            T_v2.ToImage(), 
            T_v2.ToDtype(torch.float32, scale=True)
        ]
        if self.normalize:
             transforms_list.append(T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        self.base_transform = T_v2.Compose(transforms_list)

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        actual_idx = self.current_indices[idx]
        img_path = self.image_paths[actual_idx]
        label = self.labels[actual_idx] 
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path} at index {idx} (actual index {actual_idx}): {e}")
            dummy_tensor = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return dummy_tensor, torch.tensor(-1, dtype=torch.long) 

        rgb_tensor = self.base_transform(img)
        
        # *** REMOVED internal augmentation application ***
        # Augmentations should be applied externally (e.g., in the trainer)

        return rgb_tensor, torch.tensor(label, dtype=torch.long)

    def get_class_names(self):
        return self.classes
    
    def get_class_weights(self):
        if self.class_weights_computed is None:
            class_counts = Counter(self.current_split_labels) # Use labels from the current split
            total_samples = len(self.current_split_labels)
            
            if not class_counts or total_samples == 0:
                logger.warning(f"Cannot compute class weights: split '{self.split}' has no samples or class counts.")
                return None

            weights = torch.zeros(len(self.classes), dtype=torch.float)
            for i in range(len(self.classes)):
                count = class_counts.get(i, 0) 
                if count > 0:
                    weights[i] = total_samples / (len(self.classes) * count) 
                else:
                    weights[i] = 1.0 
                    logger.warning(f"Class {self.classes[i]} not found in split '{self.split}'. Assigning weight 1.0.")

            self.class_weights_computed = weights
            logger.info(f"Computed class weights for split '{self.split}': {self.class_weights_computed.numpy().round(2)}")
        
        return self.class_weights_computed