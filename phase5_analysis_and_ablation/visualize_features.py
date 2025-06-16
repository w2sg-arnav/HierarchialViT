# phase5_analysis_and_ablation/visualize_features.py

import torch
import torch.nn as nn
import yaml
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import inspect  # To inspect function arguments

# --- Path Setup ---
# Ensures the script can find modules from other project phases
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import necessary components from other project phases
from phase4_finetuning.dataset import SARCLD2024Dataset
from phase4_finetuning.utils.augmentations import create_cotton_leaf_augmentation
from phase2_model.models.hvt import create_disease_aware_hvt

def extract_features(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts features from the layer just before the final classifier using a forward hook.
    
    Args:
        model: The PyTorch model to evaluate.
        loader: The DataLoader providing validation data.
        device: The device to run inference on ('cuda' or 'cpu').

    Returns:
        A tuple containing a numpy array of features and a numpy array of labels.
    """
    model.to(device).eval()
    features_list = []
    labels_list = []
    
    # This dictionary will be populated by the hook
    features_capture = {}
    
    def get_features_hook(module, input_data, output_data):
        # The input to the nn.Linear layer is a tuple; the features are the first element.
        features_capture['feats'] = input_data[0].detach()

    # Register a forward hook on the final classification layer.
    # This is a robust way to capture the embeddings before the final projection.
    hook_handle = model.classifier_head.register_forward_hook(get_features_hook)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features", leave=False):
            # Run a forward pass to trigger the hook. The output is not needed here.
            _ = model(images.to(device))
            
            # Append captured features and labels to our lists
            features_list.append(features_capture['feats'].cpu().numpy())
            labels_list.append(labels.numpy())
            
    hook_handle.remove() # Crucial: always remove hooks after use to prevent memory leaks.
    
    return np.concatenate(features_list), np.concatenate(labels_list)

def main():
    """
    Main function to load models, extract features, run t-SNE, and plot the results.
    """
    print("--- Starting Feature Space Visualization ---")
    
    # --- Configuration ---
    # Define paths to the configuration and checkpoint files for both models
    
    # 1. Best Model (SSL-Pretrained)
    cfg_path_ssl = "/teamspace/studios/this_studio/cvpr25/phase5_analysis_and_ablation/temp_configs/03_ablation_no_advanced_augs.yaml"
    ckpt_path_ssl = "/teamspace/studios/this_studio/cvpr25/phase4_finetuning/logs_finetune/03_ablation_no_advanced_augs_20250613-163050/checkpoints/best_model.pth"
    
    # 2. Baseline Model (Trained from Scratch)
    cfg_path_scratch = "/teamspace/studios/this_studio/cvpr25/phase5_analysis_and_ablation/temp_configs/02_ablation_no_ssl.yaml"
    ckpt_path_scratch = "/teamspace/studios/this_studio/cvpr25/phase4_finetuning/logs_finetune/02_ablation_no_ssl_20250613-091018/checkpoints/best_model.pth"
    
    with open(cfg_path_ssl, 'r') as f:
        cfg_ssl = yaml.safe_load(f)
    with open(cfg_path_scratch, 'r') as f:
        cfg_scratch = yaml.safe_load(f)

    device = cfg_ssl.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    img_size = tuple(cfg_ssl['data']['img_size'])
    num_classes = 7 

    # --- Model Loading ---
    print("Loading models...")
    # Load SSL-Pretrained Model
    model_ssl = create_disease_aware_hvt(current_img_size=img_size, num_classes=num_classes, model_params_dict=cfg_ssl['model']['hvt_params'])
    model_ssl.load_state_dict(torch.load(ckpt_path_ssl, map_location='cpu')['model_state_dict'])

    # Load From-Scratch Model
    model_scratch = create_disease_aware_hvt(current_img_size=img_size, num_classes=num_classes, model_params_dict=cfg_scratch['model']['hvt_params'])
    model_scratch.load_state_dict(torch.load(ckpt_path_scratch, map_location='cpu')['model_state_dict'])

    # --- Dataloader Preparation ---
    print("Preparing validation dataloader...")
    val_transform = create_cotton_leaf_augmentation(strategy='minimal', img_size=img_size)
    
    # Get the names of the arguments that the SARCLD2024Dataset constructor accepts
    dataset_constructor_params = inspect.signature(SARCLD2024Dataset.__init__).parameters.keys()
    
    # Prepare the arguments from the config file
    dataset_args_from_config = cfg_ssl['data'].copy()
    dataset_args_from_config['random_seed'] = cfg_ssl['seed']
    
    # Filter the config arguments to only include those accepted by the constructor
    valid_dataset_args = {
        key: dataset_args_from_config[key] 
        for key in dataset_args_from_config 
        if key in dataset_constructor_params
    }

    # Instantiate the dataset with only the valid arguments
    val_dataset = SARCLD2024Dataset(
        split="val",
        transform=val_transform,
        **valid_dataset_args
    )
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # --- Feature Extraction ---
    features_ssl, labels_ssl = extract_features(model_ssl, val_loader, device)
    features_scratch, labels_scratch = extract_features(model_scratch, val_loader, device)

    # --- t-SNE Dimensionality Reduction ---
    print("Running t-SNE... (this may take a moment)")
    # Using PCA for initial reduction is a standard practice for better t-SNE performance and stability
    pca = PCA(n_components=50, random_state=42)
    # Using modern, stable t-SNE parameters
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    
    features_ssl_tsne = tsne.fit_transform(pca.fit_transform(features_ssl))
    features_scratch_tsne = tsne.fit_transform(pca.fit_transform(features_scratch))

    # --- Plotting for Publication ---
    print("Generating plot...")
    
    OUTPUT_DIR = "phase5_analysis_and_ablation/analysis_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    cmap = plt.get_cmap('tab10', num_classes)
    class_names = val_dataset.get_class_names()

    # Plot for From-Scratch model
    axes[0].scatter(features_scratch_tsne[:, 0], features_scratch_tsne[:, 1], c=labels_scratch, cmap=cmap, alpha=0.7, s=12)
    axes[0].set_title("Feature Space: HVT-Leaf (Trained from Scratch)", fontsize=20, pad=15)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plot for SSL-Pretrained model
    scatter = axes[1].scatter(features_ssl_tsne[:, 0], features_ssl_tsne[:, 1], c=labels_ssl, cmap=cmap, alpha=0.7, s=12)
    axes[1].set_title("Feature Space: HVT-Leaf (SSL Pre-trained)", fontsize=20, pad=15)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # Create a single, shared legend for the entire figure for consistency
    legend_handles, _ = scatter.legend_elements(num=len(class_names))
    fig.legend(legend_handles, class_names, title="Disease Classes", loc="center right", fontsize=14, title_fontsize=16, borderpad=1)
    
    # Adjust layout to prevent the legend from overlapping the plots
    plt.tight_layout(rect=[0, 0, 0.88, 1]) 
    
    save_path = os.path.join(OUTPUT_DIR, "tsne_feature_space_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE plot saved successfully to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()