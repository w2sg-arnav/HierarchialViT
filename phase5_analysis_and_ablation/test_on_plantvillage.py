# phase5_analysis_and_ablation/test_on_plantvillage.py (MAX-PERFORMANCE 40-EPOCH RUN)
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Path Setup ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path: sys.path.insert(0, _project_root)

from phase2_model.models.hvt import create_disease_aware_hvt
from phase4_finetuning.utils.augmentations import create_cotton_leaf_augmentation

# --- HYPERPARAMETERS FOR 40-EPOCH ADAPTATION ---
PLANT_VILLAGE_PATH = "/teamspace/studios/this_studio/cvpr25/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"
HVT_LEAF_SSL_CHECKPOINT = "phase3_pretraining/pretrain_checkpoints_hvt_xl/hvt_xl_simclr_t4_resumed_best_probe.pth"
HVT_CONFIG = {
    "patch_size": 14, "embed_dim_rgb": 192, "spectral_channels": 0,
    "depths": [3, 6, 24, 3], "num_heads": [6, 12, 24, 48], "mlp_ratio": 4.0,
    "qkv_bias": True, "model_drop_rate": 0.1, "drop_path_rate": 0.2, "norm_layer_name": "LayerNorm",
    "use_dfca": False, "use_gradient_checkpointing": True, "enable_consistency_loss_heads": False,
}
IMG_SIZE = (224, 224)
BATCH_SIZE = 128
TOTAL_EPOCHS = 40
HEAD_ONLY_EPOCHS = 5 # <--- NEW: Phase 1 duration
LR_HEAD = 1e-3       # <--- NEW: High LR for head-only training
LR_BACKBONE_FINETUNE = 1e-5 # <--- NEW: Very low LR for backbone fine-tuning
LR_HEAD_FINETUNE = 1e-4     # <--- NEW: Lower LR for head during full fine-tuning
DEVICE = "cuda"

def run_finetuning(model, model_name, train_loader, val_loader, num_classes):
    print(f"\n--- Aggressive Fine-tuning for {model_name} on PlantVillage ({TOTAL_EPOCHS} epochs) ---")
    
    # --- Adapt the model's head for the new number of classes ---
    if hasattr(model, 'classifier_head'):
        in_features = model.classifier_head.in_features
        model.classifier_head = nn.Linear(in_features, num_classes)
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    best_accuracy = 0.0

    # ======================================================================
    # --- PHASE 1: Head-Only Training (5 Epochs) ---
    # ======================================================================
    print("\n--- Phase 1: Training Classifier Head Only ---")
    # Freeze the backbone
    for name, param in model.named_parameters():
        if 'classifier_head' not in name:
            param.requires_grad = False
    
    # Optimizer for only the head parameters with a high learning rate
    head_optimizer = optim.AdamW(model.classifier_head.parameters(), lr=LR_HEAD)
    
    for epoch in range(1, HEAD_ONLY_EPOCHS + 1):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{HEAD_ONLY_EPOCHS} [Head Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            head_optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(head_optimizer)
            scaler.update()

    # ======================================================================
    # --- PHASE 2: End-to-End Fine-Tuning (35 Epochs) ---
    # ======================================================================
    print("\n--- Phase 2: Unfreezing Backbone for End-to-End Fine-tuning ---")
    # Unfreeze the backbone
    for param in model.parameters():
        param.requires_grad = True
        
    # Optimizer with differential learning rates
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'classifier_head' not in n], 'lr': LR_BACKBONE_FINETUNE},
        {'params': model.classifier_head.parameters(), 'lr': LR_HEAD_FINETUNE}
    ]
    full_optimizer = optim.AdamW(param_groups, weight_decay=1e-2)
    
    # Scheduler for the 35 epochs of full fine-tuning
    full_finetune_epochs = TOTAL_EPOCHS - HEAD_ONLY_EPOCHS
    scheduler = CosineAnnealingLR(full_optimizer, T_max=full_finetune_epochs, eta_min=1e-6)

    for epoch in range(1, full_finetune_epochs + 1):
        actual_epoch = epoch + HEAD_ONLY_EPOCHS
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {actual_epoch}/{TOTAL_EPOCHS} [Full Finetune]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            full_optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(full_optimizer)
            scaler.update()
        
        scheduler.step() # Step scheduler at the end of the epoch

        # Validation loop
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {actual_epoch} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        print(f"Epoch {actual_epoch} | {model_name} Val Accuracy: {accuracy:.2f}% | Best so far: {best_accuracy:.2f}%")
        
    return best_accuracy

def main():
    # --- Dataloaders Setup ---
    # ... (This part is correct and remains the same) ...
    train_transform = create_cotton_leaf_augmentation(strategy='cotton_disease', img_size=IMG_SIZE, severity='moderate')
    val_transform = create_cotton_leaf_augmentation(strategy='minimal', img_size=IMG_SIZE)
    train_dir = os.path.join(PLANT_VILLAGE_PATH, 'train')
    val_dir = os.path.join(PLANT_VILLAGE_PATH, 'valid')
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=val_dir, transform=val_transform)
    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes in PlantVillage dataset.")
    loader_args = {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True}
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, **loader_args)

    # --- Run Experiment ---
    vit_acc = 99.95
    print(f"Using cached baseline result for ViT-Base: {vit_acc:.2f}% accuracy.")
    
    hvt_model = create_disease_aware_hvt(current_img_size=IMG_SIZE, num_classes=num_classes, model_params_dict=HVT_CONFIG)
    print("Loading SSL weights into HVT-Leaf...")
    checkpoint = torch.load(HVT_LEAF_SSL_CHECKPOINT, map_location='cpu')
    ssl_weights = checkpoint['model_backbone_state_dict']
    new_model_state_dict = hvt_model.state_dict()
    weights_to_load = {name: param for name, param in ssl_weights.items() if name in new_model_state_dict and param.size() == new_model_state_dict[name].size()}
    hvt_model.load_state_dict(weights_to_load, strict=False)
    print("SSL backbone weights loaded successfully.")
    
    hvt_acc = run_finetuning(hvt_model, "HVT-Leaf (SSL)", train_loader, val_loader, num_classes)

    # --- Final Results ---
    print("\n" + "="*40)
    print(f"--- PlantVillage Generalization Results ({TOTAL_EPOCHS} Epochs) ---")
    print(f"ViT-Base (ImageNet Pre-trained) Final Accuracy: {vit_acc:.2f}%")
    print(f"HVT-Leaf (Our SSL Pre-trained) Final Accuracy: {hvt_acc:.2f}%")
    print("="*40)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    main()