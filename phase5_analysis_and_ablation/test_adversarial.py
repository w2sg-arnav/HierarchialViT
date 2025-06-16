# phase5_analysis_and_ablation/test_adversarial.py

import torch
import torch.nn as nn
import yaml
import os
import sys
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchattacks

# --- Path Setup ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
    
from phase4_finetuning.dataset import SARCLD2024Dataset
from phase4_finetuning.utils.augmentations import create_cotton_leaf_augmentation
from phase2_model.models.hvt import create_disease_aware_hvt
from phase4_finetuning.utils.metrics import compute_metrics

# --- Configuration ---
# IMPORTANT: Update these paths to your BEST run
CONFIG_PATH = "/teamspace/studios/this_studio/cvpr25/phase5_analysis_and_ablation/temp_configs/03_ablation_no_advanced_augs.yaml"
CHECKPOINT_PATH = "/teamspace/studios/this_studio/cvpr25/phase4_finetuning/logs_finetune/03_ablation_no_advanced_augs_20250613-163050/checkpoints/best_model.pth"
OUTPUT_DIR = os.path.join(_current_dir, "analysis_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate_adversarial(model, loader, attack, device):
    """Evaluates the model's performance on adversarial examples."""
    model.eval()
    all_preds, all_labels = [], []
    
    for images, labels in tqdm(loader, desc=f"Evaluating FGSM Attack"):
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        adv_images = attack(images, labels)
        
        with torch.no_grad():
            outputs = model(adv_images)
            preds = torch.argmax(outputs, dim=1)
        
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        
    preds_cat = torch.cat(all_preds).numpy()
    labels_cat = torch.cat(all_labels).numpy()
    
    metrics = compute_metrics(preds_cat, labels_cat, num_classes=7)
    return metrics['f1_macro'], metrics['accuracy']

def main():
    print("======== Starting Adversarial Robustness Analysis (FGSM) ========")
    # --- Load Config and Best Model ---
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint file not found at {CHECKPOINT_PATH}")

    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    device = cfg['device']
    img_size = tuple(cfg['data']['img_size'])
    
    # Load the HVT-Leaf model architecture
    model = create_disease_aware_hvt(
        current_img_size=img_size,
        num_classes=7,
        model_params_dict=cfg['model']['hvt_params']
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu')['model_state_dict'])
    model.to(device).eval() # Ensure model is in eval mode

    # --- Setup Dataloader (with minimal transforms for clean evaluation) ---
    val_transform = create_cotton_leaf_augmentation(strategy='minimal', img_size=img_size)
    val_dataset = SARCLD2024Dataset(
        root_dir=cfg['data']['root_dir'],
        split="val",
        transform=val_transform,
        img_size=img_size,
        train_split_ratio=cfg['data']['train_split_ratio'],
        original_dataset_name=cfg['data']['original_dataset_name'],
        augmented_dataset_name=cfg['data']['augmented_dataset_name'],
        random_seed=cfg['seed']
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=4)

    # --- Define the FGSM Attack ---
    # Epsilon is the magnitude of the perturbation. 8/255 is a standard value.
    epsilon = 8/255
    attack = torchattacks.FGSM(model, eps=epsilon)
    print(f"FGSM Attack Initialized with Epsilon = {epsilon:.4f}")

    # --- Run Evaluation ---
    adversarial_f1, adversarial_acc = evaluate_adversarial(model, val_loader, attack, device)
    
    # --- Display Results ---
    # For comparison, get the clean accuracy from your logs (or re-calculate it)
    clean_f1 = 96.67  # From your Exp. 03
    clean_acc = 96.61 # From your Exp. 03

    results = {
        "Evaluation Type": ["Clean (Original)", "Adversarial (FGSM)"],
        "Macro F1 (%)": [clean_f1, adversarial_f1 * 100],
        "Accuracy (%)": [clean_acc, adversarial_acc * 100]
    }
    df = pd.DataFrame(results)
    
    print("\n\n--- Adversarial Robustness Results ---")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()