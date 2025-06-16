# phase5_analysis_and_ablation/test_on_plantvillage.py
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import os
import sys
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Path Setup
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path: sys.path.insert(0, _project_root)

from phase2_model.models.hvt import create_disease_aware_hvt
from phase4_finetuning.utils.augmentations import create_cotton_leaf_augmentation

# --- CONFIGURATION ---
PLANT_VILLAGE_PATH = "/path/to/your/plantvillage/color" # IMPORTANT: UPDATE THIS PATH
HVT_LEAF_SSL_CHECKPOINT = "phase3_pretraining/pretrain_checkpoints_hvt_xl/hvt_xl_simclr_t4_resumed_best_probe.pth"
HVT_CONFIG = { # A snapshot of your HVT-XL config
    "patch_size": 14, "embed_dim_rgb": 192, "spectral_channels": 0,
    "depths": [3, 6, 24, 3], "num_heads": [6, 12, 24, 48], "mlp_ratio": 4.0,
    "qkv_bias": True, "model_drop_rate": 0.1, "drop_path_rate": 0.2, "norm_layer_name": "LayerNorm",
    "use_dfca": False, "use_gradient_checkpointing": True, "enable_consistency_loss_heads": False,
}
IMG_SIZE = (224, 224) # Use a standard size like 224 for PlantVillage
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda"

def run_finetuning(model, model_name, train_loader, val_loader, num_classes):
    print(f"\n--- Fine-tuning {model_name} on PlantVillage ---")
    
    # Adapt the head for the new number of classes
    if hasattr(model, 'head'): # Timm ViT
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif hasattr(model, 'classifier_head'): # Our HVT
        model.classifier_head = nn.Linear(model.classifier_head.in_features, num_classes)

    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch} | {model_name} Val Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    # --- Dataloaders for PlantVillage ---
    train_transform = create_cotton_leaf_augmentation(strategy='cotton_disease', img_size=IMG_SIZE, severity='moderate')
    val_transform = create_cotton_leaf_augmentation(strategy='minimal', img_size=IMG_SIZE)
    
    train_dataset = ImageFolder(root=os.path.join(PLANT_VILLAGE_PATH, 'train'), transform=train_transform)
    val_dataset = ImageFolder(root=os.path.join(PLANT_VILLAGE_PATH, 'val'), transform=val_transform)
    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes in PlantVillage dataset.")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- Experiment 1: ViT-Base Baseline ---
    vit_model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
    vit_acc = run_finetuning(vit_model, "ViT-Base", train_loader, val_loader, num_classes)
    
    # --- Experiment 2: Our HVT-Leaf with SSL ---
    hvt_model = create_disease_aware_hvt(current_img_size=IMG_SIZE, num_classes=num_classes, model_params_dict=HVT_CONFIG)
    print("Loading SSL weights into HVT-Leaf...")
    ckpt = torch.load(HVT_LEAF_SSL_CHECKPOINT, map_location='cpu')
    hvt_model.load_state_dict(ckpt['model_backbone_state_dict'], strict=False)
    hvt_acc = run_finetuning(hvt_model, "HVT-Leaf (SSL)", train_loader, val_loader, num_classes)

    # --- Final Results ---
    print("\n--- PlantVillage Generalization Results ---")
    print(f"ViT-Base (ImageNet Pre-trained) Final Accuracy: {vit_acc:.2f}%")
    print(f"HVT-Leaf (Our SSL Pre-trained) Final Accuracy: {hvt_acc:.2f}%")

if __name__ == "__main__":
    main()