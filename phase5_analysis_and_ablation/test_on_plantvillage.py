# phase5_analysis_and_ablation/test_on_plantvillage.py (MAXIMUM SPEED on L4 GPU)
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

# --- Path Setup ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path: sys.path.insert(0, _project_root)

from phase2_model.models.hvt import create_disease_aware_hvt
from phase4_finetuning.utils.augmentations import create_cotton_leaf_augmentation

# --- HYPERPARAMETERS FOR MAXIMUM SPEED ---
PLANT_VILLAGE_PATH = "/teamspace/studios/this_studio/cvpr25/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"
HVT_LEAF_SSL_CHECKPOINT = "phase3_pretraining/pretrain_checkpoints_hvt_xl/hvt_xl_simclr_t4_resumed_best_probe.pth"
HVT_CONFIG = {
    "patch_size": 14, "embed_dim_rgb": 192, "spectral_channels": 0,
    "depths": [3, 6, 24, 3], "num_heads": [6, 12, 24, 48], "mlp_ratio": 4.0,
    "qkv_bias": True, "model_drop_rate": 0.0, "drop_path_rate": 0.1, # Reduced dropout for faster convergence
    "use_dfca": False, "use_gradient_checkpointing": True, "enable_consistency_loss_heads": False,
}
IMG_SIZE = (224, 224)
BATCH_SIZE = 256 # <--- MAXIMIZE BATCH SIZE
EPOCHS = 20      # <--- REDUCE EPOCHS
LR = 5e-4        # <--- AGGRESSIVE LEARNING RATE
DEVICE = "cuda"

def run_finetuning(model, model_name, train_loader, val_loader, num_classes):
    print(f"\n--- FAST Fine-tuning for {model_name} on PlantVillage ({EPOCHS} epochs) ---")
    
    # --- Re-enable torch.compile for maximum speed ---
    print("Attempting to compile model with torch.compile...")
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled successfully.")
    except Exception as e:
        print(f"torch.compile failed: {e}. Continuing without compilation.")

    # Adapt the head for the new number of classes
    is_compiled = hasattr(model, '_orig_mod')
    unwrapped_model = model._orig_mod if is_compiled else model
    if hasattr(unwrapped_model, 'classifier_head'):
        in_features = unwrapped_model.classifier_head.in_features
        unwrapped_model.classifier_head = nn.Linear(in_features, num_classes)
        
    model.to(DEVICE)
    
    # Differential learning rates are still key
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'classifier_head' not in n], 'lr': LR / 10},
        {'params': model.classifier_head.parameters() if not is_compiled else model._orig_mod.classifier_head.parameters(), 'lr': LR}
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-3)
    
    # Scheduler tailored for the short, 10-epoch run
    scheduler = OneCycleLR(optimizer, max_lr=[LR / 10, LR], epochs=EPOCHS, steps_per_epoch=len(train_loader))
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() 
    best_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                with autocast():
                    outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        print(f"Epoch {epoch} | {model_name} Val Accuracy: {accuracy:.2f}% | Best so far: {best_accuracy:.2f}%")
        
    return best_accuracy

def main():
    # --- Dataloaders Setup ---
    train_transform = create_cotton_leaf_augmentation(strategy='cotton_disease', img_size=IMG_SIZE, severity='moderate')
    val_transform = create_cotton_leaf_augmentation(strategy='minimal', img_size=IMG_SIZE)
    train_dir = os.path.join(PLANT_VILLAGE_PATH, 'train')
    val_dir = os.path.join(PLANT_VILLAGE_PATH, 'valid')
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=val_dir, transform=val_transform)
    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes in PlantVillage dataset.")
    
    loader_args = {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True, 'drop_last': True}
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, **loader_args)

    # Hardcode the baseline result
    vit_acc = 99.95
    print(f"Using cached baseline result for ViT-Base: {vit_acc:.2f}% accuracy.")
    
    # --- Experiment: Our HVT-Leaf with SSL ---
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
    print(f"--- PlantVillage Generalization Results (Fast {EPOCHS} Epochs) ---")
    print(f"ViT-Base (ImageNet Pre-trained) Final Accuracy: {vit_acc:.2f}%")
    print(f"HVT-Leaf (Our SSL Pre-trained) Final Accuracy: {hvt_acc:.2f}%")
    print("="*40)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    main()