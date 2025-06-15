# phase5_analysis_and_ablation/visualize_attention.py

import torch
import torch.nn as nn
import yaml
import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# --- Path Setup ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
    
from phase2_model.models.hvt import create_disease_aware_hvt
from phase4_finetuning.dataset import SARCLD2024Dataset

# --- Configuration ---
CONFIG_PATH = "/teamspace/studios/this_studio/cvpr25/phase5_analysis_and_ablation/temp_configs/03_ablation_no_advanced_augs.yaml"
CHECKPOINT_PATH = "/teamspace/studios/this_studio/cvpr25/phase4_finetuning/logs_finetune/03_ablation_no_advanced_augs_20250613-163050/checkpoints/best_model.pth"
OUTPUT_DIR = os.path.join(_current_dir, "analysis_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def attention_rollout(attention_maps):
    """
    Computes attention rollout across all layers of an HVT's last stage.
    """
    # The first attention map has shape [1, num_tokens, num_tokens]
    num_tokens = attention_maps[0].shape[-1]
    rollout = torch.eye(num_tokens, device=attention_maps[0].device)
    
    for attn_map in attention_maps:
        # Squeeze the batch dimension and move to correct device
        attn_map_sq = attn_map.squeeze(0).to(rollout.device)
        
        # Add residual connection influence
        attn_map_with_residual = 0.5 * attn_map_sq + 0.5 * torch.eye(num_tokens, device=rollout.device)
        
        # Chain the matrix multiplication
        rollout = torch.matmul(attn_map_with_residual, rollout)
        
    # Return the attention from the [CLS] token (index 0) to all patch tokens (index 1 onwards)
    return rollout[0, 1:] # FIX: Correct slicing for a 2D matrix

def visualize_single_image(model, device, image_path, cfg):
    """Processes a single image and generates a smooth attention heatmap."""
    img_size = tuple(cfg['data']['img_size'])
    pil_img = Image.open(image_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # --- Hooking logic to capture attention maps ---
    attention_maps_last_stage = []
    hooks = []
    
    # Custom forward function to be monkey-patched onto attention modules
    def new_forward_factory(attn_module):
        original_forward = attn_module.forward
        def new_forward(x):
            B, N, C = x.shape
            qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * attn_module.scale
            attn = attn.softmax(dim=-1)
            attn_module.attn_map = attn # Save the attention map
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = attn_module.proj(x)
            x = attn_module.proj_drop(x)
            return x
        return original_forward, new_forward

    def hook_fn(module, input, output):
        if hasattr(module, 'attn_map'):
            attention_maps_last_stage.append(module.attn_map.mean(dim=1).detach())

    last_stage = model.rgb_stages[-1]
    original_forwards = {}
    for i, block in enumerate(last_stage.blocks):
        original_forwards[i], new_fwd = new_forward_factory(block.attn)
        block.attn.forward = new_fwd
        hooks.append(block.attn.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(input_tensor)

    for i, block in enumerate(last_stage.blocks):
        if i in original_forwards:
            block.attn.forward = original_forwards[i]
    for hook in hooks:
        hook.remove()

    if not attention_maps_last_stage:
        return pil_img, None 

    rollout_map = attention_rollout(attention_maps_last_stage)
    
    # --- Grid size calculation and padding ---
    hvt_params = cfg['model']['hvt_params']
    patch_size = hvt_params['patch_size']
    num_downsamples = len(hvt_params['depths']) - 1
    final_grid_h = img_size[0] // (patch_size * (2**num_downsamples))
    final_grid_w = img_size[1] // (patch_size * (2**num_downsamples))
    expected_tokens = final_grid_h * final_grid_w
    
    if rollout_map.shape[0] < expected_tokens:
        pad_size = expected_tokens - rollout_map.shape[0]
        # FIX: Ensure padding is 1D to match the 1D rollout_map
        padding = torch.zeros(pad_size, device=rollout_map.device)
        rollout_map = torch.cat((rollout_map, padding), dim=0)

    mask = rollout_map.reshape(final_grid_h, final_grid_w).cpu().numpy()
    
    # --- Heatmap generation ---
    mask = cv2.resize(mask, (pil_img.width, pil_img.height), interpolation=cv2.INTER_CUBIC)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    pil_img_np = np.array(pil_img)
    superimposed_img = np.uint8(heatmap * 0.4 + pil_img_np * 0.6)
    
    return pil_img, Image.fromarray(superimposed_img)

def main():
    print("======== Starting Attention Visualization ========")
    # ... (Loading config and model remains the same) ...
    if not os.path.exists(CHECKPOINT_PATH): raise FileNotFoundError(f"Checkpoint file not found at {CHECKPOINT_PATH}")
    with open(CONFIG_PATH, 'r') as f: cfg = yaml.safe_load(f)
    device = cfg['device']
    img_size = tuple(cfg['data']['img_size'])
    model = create_disease_aware_hvt(current_img_size=img_size, num_classes=7, model_params_dict=cfg['model']['hvt_params'])
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu')['model_state_dict'])
    model.to(device).eval()

    val_dataset = SARCLD2024Dataset(
        root_dir=cfg['data']['root_dir'], split="val", transform=None, img_size=img_size,
        train_split_ratio=cfg['data']['train_split_ratio'],
        original_dataset_name=cfg['data']['original_dataset_name'],
        augmented_dataset_name=cfg['data']['augmented_dataset_name'],
        random_seed=cfg['seed']
    )
    
    image_paths_to_viz = {}
    classes_to_find = ["Bacterial Blight", "Curl Virus", "Leaf Redding", "Healthy Leaf"]
    for i in range(len(val_dataset)):
        actual_idx = val_dataset.current_indices[i]
        img_path = val_dataset.image_paths[actual_idx]
        label_idx = val_dataset.labels[actual_idx]
        class_name = val_dataset.classes[label_idx]
        if class_name in classes_to_find and class_name not in image_paths_to_viz:
            image_paths_to_viz[class_name] = img_path
        if len(image_paths_to_viz) == len(classes_to_find): break
            
    num_images = len(image_paths_to_viz)
    fig, axes = plt.subplots(2, num_images, figsize=(5 * num_images, 11))
    if num_images == 1: axes = axes.reshape(2, 1)
    
    for i, (class_name, img_path) in enumerate(image_paths_to_viz.items()):
        print(f"Visualizing attention for class: {class_name}")
        pil_img, attention_overlay = visualize_single_image(model, device, img_path, cfg)
        
        axes[0, i].imshow(pil_img)
        axes[0, i].set_title(f"Original: {class_name}")
        axes[0, i].axis('off')
        
        if attention_overlay:
            axes[1, i].imshow(attention_overlay)
        axes[1, i].set_title("Attention Overlay")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "attention_rollout_visualization.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Attention visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()