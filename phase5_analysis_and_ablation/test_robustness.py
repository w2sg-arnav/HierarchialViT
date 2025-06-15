# phase5_analysis_and_ablation/test_robustness.py

import torch
import torch.nn as nn
import yaml
import os
import sys
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.v2 as T_v2

# --- Path Setup ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
    
from phase4_finetuning.dataset import SARCLD2024Dataset
from phase2_model.models.hvt import create_disease_aware_hvt
from phase4_finetuning.utils.metrics import compute_metrics

# --- Configuration ---
# IMPORTANT: Update these paths to your BEST run
CONFIG_PATH = "/teamspace/studios/this_studio/cvpr25/phase5_analysis_and_ablation/temp_configs/03_ablation_no_advanced_augs.yaml"
CHECKPOINT_PATH = "/teamspace/studios/this_studio/cvpr25/phase4_finetuning/logs_finetune/03_ablation_no_advanced_augs_20250613-163050/checkpoints/best_model.pth"

# --- START OF FIX: Custom, version-independent transform modules ---
class AddGaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std

class Clamp(nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor, self.min_val, self.max_val)
# --- END OF FIX ---


def evaluate_on_corruption(model, transform, cfg, device, desc="Evaluating..."):
    """Evaluates the model on a dataloader with a specific corruption transform."""
    img_size = tuple(cfg['data']['img_size'])
    dataset = SARCLD2024Dataset(
        root_dir=cfg['data']['root_dir'],
        split="val",
        transform=transform,
        img_size=img_size,
        train_split_ratio=cfg['data']['train_split_ratio'],
        original_dataset_name=cfg['data']['original_dataset_name'],
        augmented_dataset_name=cfg['data']['augmented_dataset_name'],
        random_seed=cfg['seed']
    )
    loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'] * 2, shuffle=False, num_workers=4)
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            if -1 in labels: continue
            images = images.to(device)
            outputs = model(images) 
            all_preds.append(torch.argmax(outputs, dim=1).cpu())
            all_labels.append(labels)
    
    if not all_preds:
        print("Warning: No valid batches were processed during evaluation.")
        return 0.0

    preds_cat = torch.cat(all_preds).numpy()
    labels_cat = torch.cat(all_labels).numpy()
    
    metrics = compute_metrics(preds_cat, labels_cat, num_classes=7)
    return metrics['f1_macro']

def main():
    print("======== Starting Robustness Analysis ========")
    # --- Load Config and Best Model ---
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}. Please update the path.")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint file not found at {CHECKPOINT_PATH}. Please update the path.")

    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    device = cfg['device']
    img_size = tuple(cfg['data']['img_size'])
    
    model = create_disease_aware_hvt(
        current_img_size=img_size,
        num_classes=7,
        model_params_dict=cfg['model']['hvt_params']
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu')['model_state_dict'])
    model.to(device)

    # --- Define Corruptions using correct and universal T_v2 API ---
    base_transform = T_v2.Compose([
        T_v2.Resize(img_size, antialias=True),
        T_v2.ToImage(),
        T_v2.ToDtype(torch.float32, scale=True)
    ])
    
    normalize = T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Using our custom, version-safe modules
    corruptions = {
        "Clean": T_v2.Compose([base_transform, normalize]),
        "Gaussian Noise (σ=0.1)": T_v2.Compose([base_transform, AddGaussianNoise(std=0.1), Clamp(0.0, 1.0), normalize]),
        "Gaussian Blur (k=11, σ=5)": T_v2.Compose([base_transform, T_v2.GaussianBlur(kernel_size=11, sigma=5.0), normalize]),
        "Occlusion (20%)": T_v2.Compose([base_transform, T_v2.RandomErasing(p=1.0, scale=(0.2, 0.2), ratio=(1.0, 1.0)), normalize])
    }

    # --- Run Evaluation ---
    results = {}
    for name, transform in corruptions.items():
        print(f"\nTesting robustness against: {name}")
        f1_score = evaluate_on_corruption(model, transform, cfg, device, desc=f"Eval [{name}]")
        results[name] = f1_score * 100
    
    # --- Display Results ---
    df = pd.DataFrame(list(results.items()), columns=['Corruption Type', 'F1 Macro (%)'])
    print("\n\n--- Robustness Analysis Results ---")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()