# phase5_analysis_and_ablation/analyze_best_model.py

import torch
import yaml
import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm

# --- Path Setup & Imports ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path: sys.path.insert(0, _project_root)

from phase4_finetuning.dataset import SARCLD2024Dataset
from phase4_finetuning.utils.augmentations import create_cotton_leaf_augmentation
from phase2_model.models.hvt import create_disease_aware_hvt

# --- Configuration ---
# IMPORTANT: Update these paths to point to your BEST run
CONFIG_PATH = "/teamspace/studios/this_studio/cvpr25/phase5_analysis_and_ablation/temp_configs/03_ablation_no_advanced_augs.yaml"
CHECKPOINT_PATH = "phase4_finetuning/logs_finetune/03_ablation_no_advanced_augs_20250613-163050/checkpoints/best_model.pth"
OUTPUT_DIR = os.path.join(_current_dir, "analysis_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze():
    print("======== Starting Analysis of Best Fine-tuned Model ========")
    
    # --- Load Config and Setup ---
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = cfg['device']
    img_size = tuple(cfg['data']['img_size'])
    
    # --- FIX #1: Pass the correct dataset folder names from the config ---
    # This first call is just to get the class names, so transform can be None.
    val_dataset_for_names = SARCLD2024Dataset(
        root_dir=cfg['data']['root_dir'], img_size=img_size, split="val", transform=None,
        train_split_ratio=cfg['data']['train_split_ratio'],
        original_dataset_name=cfg['data']['original_dataset_name'], # <-- Use value from config
        augmented_dataset_name=cfg['data']['augmented_dataset_name'],# <-- Use value from config
        random_seed=cfg['seed']
    )
    class_names = val_dataset_for_names.get_class_names()
    num_classes = len(class_names)
    print(f"Analyzing for {num_classes} classes: {class_names}")

    # --- Create Model ---
    print("Loading model architecture...")
    model = create_disease_aware_hvt(
        current_img_size=img_size,
        num_classes=num_classes,
        model_params_dict=cfg['model']['hvt_params']
    )
    
    # --- Load Best Weights ---
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Best checkpoint not found at {CHECKPOINT_PATH}")
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
    
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
        
    model.to(device)
    model.eval()

    # --- Create Validation Dataloader with minimal augmentations ---
    val_augs = create_cotton_leaf_augmentation(strategy='minimal', img_size=img_size)
    
    # --- FIX #2: Pass the correct dataset folder names here as well ---
    val_dataset = SARCLD2024Dataset(
        root_dir=cfg['data']['root_dir'], split="val", transform=val_augs,
        img_size=img_size,
        train_split_ratio=cfg['data']['train_split_ratio'],
        original_dataset_name=cfg['data']['original_dataset_name'], # <-- Use value from config
        augmented_dataset_name=cfg['data']['augmented_dataset_name'],# <-- Use value from config
        random_seed=cfg['seed']
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size']*2, shuffle=False, num_workers=4)

    # --- Run Inference ---
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Running validation inference"):
            if -1 in labels: continue # Skip error samples
            images = images.to(device)
            outputs = model(images) # Your fine-tuned model's forward should return logits
            main_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            preds = torch.argmax(main_logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # --- Generate and Plot Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix for HVT-Leaf (Best Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    analyze()