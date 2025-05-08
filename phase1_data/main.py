# main.py
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import os
import logging
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

from dataset import CottonLeafDataset
from transforms import get_transforms, denormalize_image
from progression import DiseaseProgressionSimulator
from config import (
    ORIGINAL_DATASET_ROOT, AUGMENTED_DATASET_ROOT, IMAGE_SIZE, SPECTRAL_SIZE,
    BATCH_SIZE, NUM_WORKERS, DEFAULT_STAGE_MAP
)

logger = logging.getLogger(__name__)

# Create a directory for saving visualizations
VISUALIZATION_DIR = "visualizations"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# --- Custom Collate Function ---
def custom_collate_fn(batch: List[Tuple[torch.Tensor, Optional[torch.Tensor], int, str]]) \
        -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, List[str]]:
    """
    Collate function that handles Optional[torch.Tensor] for spectral_tensors.
    If an item has spectral_tensor as None, a zero tensor of expected shape is used for batching.
    If all spectral_tensors in the batch are None (e.g., use_spectral=False), 
    the batched spectral_tensor will be None.
    """
    rgb_images = []
    spectral_images_processed = []
    labels = []
    stages = []

    # Determine if any item in the batch has actual spectral data
    # And find a reference shape for placeholder if needed
    has_any_spectral_data = False
    ref_spectral_shape = None

    for item in batch:
        rgb_images.append(item[0])
        labels.append(item[2])
        stages.append(item[3])
        
        current_spectral = item[1]
        if current_spectral is not None:
            spectral_images_processed.append(current_spectral)
            has_any_spectral_data = True
            if ref_spectral_shape is None:
                 # Assuming [C, H, W]
                ref_spectral_shape = current_spectral.shape
        else:
            # Append None for now, will be replaced by placeholder if mixed batch
            spectral_images_processed.append(None)

    batched_rgb = torch.stack(rgb_images, 0)
    batched_labels = torch.tensor(labels, dtype=torch.long)
    
    batched_spectral: Optional[torch.Tensor] = None
    if has_any_spectral_data:
        # At least one item has spectral data. Replace Nones with placeholders.
        if ref_spectral_shape is None: # Should not happen if has_any_spectral_data is True
            logger.warning("Reference spectral shape not found even though spectral data present. Using default.")
            ref_spectral_shape = (1, SPECTRAL_SIZE[0], SPECTRAL_SIZE[1]) # Default placeholder shape C,H,W

        final_spectral_list = []
        for s_img in spectral_images_processed:
            if s_img is not None:
                final_spectral_list.append(s_img)
            else:
                # Create a zero tensor as a placeholder
                # logger.debug(f"Creating placeholder for None spectral tensor with shape: {ref_spectral_shape}")
                final_spectral_list.append(torch.zeros(ref_spectral_shape, dtype=torch.float32))
        
        try:
            batched_spectral = torch.stack(final_spectral_list, 0)
        except RuntimeError as e:
            logger.error(f"RuntimeError during stacking spectral images: {e}. Shapes were:")
            for i, s_img in enumerate(final_spectral_list):
                logger.error(f"  Item {i} shape: {s_img.shape if s_img is not None else 'None'}")
            # If stacking fails, perhaps return None or raise error
            batched_spectral = None # Fallback
            logger.error("Fallback: Batched spectral set to None due to stacking error.")

    # If not has_any_spectral_data, all spectral_images_processed items are None,
    # so batched_spectral remains None, which is the desired behavior for use_spectral=False.

    return batched_rgb, batched_spectral, batched_labels, stages


# --- Visualization Functions ---
def plot_dataset_sample(rgb_tensor: torch.Tensor,
                        spectral_tensor: Optional[torch.Tensor],
                        label: int,
                        stage: str,
                        class_names: List[str],
                        filename: str = "dataset_sample.png"):
    num_plots = 1 + (1 if spectral_tensor is not None else 0)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1: # Ensure axes is always iterable
        axes = [axes] 
    
    title_class_name = "Unknown Class"
    if 0 <= label < len(class_names): # Check if label is valid index
        title_class_name = class_names[label]
    elif label == -1 and stage == "error_loading_image":
        title_class_name = "Error Image"
    
    fig.suptitle(f"Class: {title_class_name} (Label: {label}), Stage: {stage}", fontsize=14)

    rgb_display = denormalize_image(rgb_tensor.cpu())
    axes[0].imshow(rgb_display.permute(1, 2, 0).numpy())
    axes[0].set_title("Processed RGB Image")
    axes[0].axis('off')

    if spectral_tensor is not None:
        spectral_display_np = spectral_tensor.cpu().numpy()
        # Handle single vs multi-channel spectral for display
        if spectral_display_np.shape[0] == 1: # Single channel (e.g., NDVI)
            spectral_img_to_show = spectral_display_np.squeeze()
            cmap = 'RdYlGn'
        elif spectral_display_np.shape[0] >= 3: # Multi-channel, show first 3 as RGB (heuristic)
            spectral_img_to_show = np.transpose(spectral_display_np[:3], (1, 2, 0))
            # Normalize for display if it's not already in [0,1] range for RGB-like viz
            min_vals = spectral_img_to_show.min(axis=(0,1), keepdims=True)
            max_vals = spectral_img_to_show.max(axis=(0,1), keepdims=True)
            if not np.allclose(max_vals - min_vals, 0): # Avoid division by zero
                spectral_img_to_show = (spectral_img_to_show - min_vals) / (max_vals - min_vals)
            else:
                spectral_img_to_show = np.zeros_like(spectral_img_to_show) # Or some other placeholder display
            spectral_img_to_show = np.clip(spectral_img_to_show, 0, 1)
            cmap = None # Use default RGB rendering
        else: # Other multi-channel cases (e.g. 2 channels), show first
            spectral_img_to_show = spectral_display_np[0]
            cmap = 'viridis'
            
        im = axes[1].imshow(spectral_img_to_show, cmap=cmap)
        axes[1].set_title("Processed Spectral")
        axes[1].axis('off')
        if cmap: # Add colorbar only if cmap is used
             fig.colorbar(im, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

    safe_filename = filename.replace("/", "_").replace(" ", "_")
    save_path = os.path.join(VISUALIZATION_DIR, safe_filename)
    plt.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved dataset sample visualization to {save_path}")


def save_progression_visualization(original_pil_img: Image.Image,
                                   progressed_pil_imgs: Dict[str, Image.Image],
                                   base_filename: str):
    num_images = 1 + len(progressed_pil_imgs)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images == 1: axes = [axes] # Ensure iterable

    axes[0].imshow(original_pil_img)
    axes[0].set_title("Original")
    axes[0].axis('off')

    for i, (stage, p_img) in enumerate(progressed_pil_imgs.items()):
        axes[i + 1].imshow(p_img)
        axes[i + 1].set_title(f"Sim. Stage: {stage}")
        axes[i + 1].axis('off')

    safe_filename = base_filename.replace("/", "_").replace(" ", "_")
    save_path = os.path.join(VISUALIZATION_DIR, f"{safe_filename}_progression_stages.png")
    plt.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved progression visualization to {save_path}")

# --- Testing Functions ---
def demonstrate_progression_simulation(dataset_root: str, sample_idx: int = 0):
    logger.info(f"Demonstrating Disease Progression Simulator using an image from {dataset_root}")
    temp_dataset = CottonLeafDataset(root_dir=dataset_root, transform=None, apply_progression=False, use_spectral=False)
    if not temp_dataset.images:
        logger.error("No images found in temp_dataset for progression demo. Exiting.")
        return
    if sample_idx >= len(temp_dataset):
        logger.warning(f"Sample index {sample_idx} out of bounds. Using index 0.")
        sample_idx = 0

    img_path_str, _, _ = temp_dataset.images[sample_idx]
    try:
        original_pil = Image.open(img_path_str).convert('RGB').resize(IMAGE_SIZE) # Resize for consistency
    except Exception as e:
        logger.error(f"Could not open image {img_path_str} for progression demo: {e}")
        return

    simulator = DiseaseProgressionSimulator()
    progressed_stages: Dict[str, Image.Image] = {}
    for stage_key in simulator.stage_effects.keys(): # Iterate known effects
        progressed_stages[stage_key] = simulator.apply(original_pil.copy(), stage_key)

    base_filename = Path(img_path_str).stem
    save_progression_visualization(original_pil, progressed_stages, base_filename)


def test_dataset_loading(root_dir: str, dataset_name: str, apply_progression: bool = False, use_spectral: bool = True):
    logger.info(f"Testing {dataset_name} from: {root_dir} (Progression: {apply_progression}, Spectral: {use_spectral})")
    current_transforms = get_transforms(train=True, image_size=IMAGE_SIZE)
    dataset = CottonLeafDataset(
        root_dir=root_dir,
        transform=current_transforms,
        stage_map=DEFAULT_STAGE_MAP,
        apply_progression=apply_progression,
        use_spectral=use_spectral
    )

    if len(dataset) == 0:
        logger.warning(f"Dataset '{dataset_name}' is empty. Skipping test.")
        return

    # Use NUM_WORKERS = 0 for easier debugging if issues persist
    effective_num_workers = NUM_WORKERS 
    # effective_num_workers = 0 # Uncomment for debugging DataLoader issues

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=effective_num_workers, collate_fn=custom_collate_fn, drop_last=True)
    try:
        rgb_batch, spectral_batch, labels_batch, stages_batch = next(iter(dataloader))
        
        logger.info(f"Sample Batch from {dataset_name}:")
        logger.info(f"  RGB tensor shape: {rgb_batch.shape}")
        if spectral_batch is not None:
            logger.info(f"  Spectral tensor shape: {spectral_batch.shape}")
        else:
            logger.info(f"  Spectral tensor: None (use_spectral might be False or all items had spectral errors)")
        logger.info(f"  Labels (first 4): {labels_batch[:4]}")
        logger.info(f"  Stages (first 4): {stages_batch[:4]}")

        if not dataset.classes and labels_batch[0].item() != -1 : # if classes list is empty but we got valid labels
             logger.warning("Dataset.classes is empty, cannot map label to class name for visualization.")
             class_names_for_plot = [f"Label {i}" for i in range(max(labels_batch.tolist())+1)] # Placeholder names
        elif not dataset.classes and labels_batch[0].item() == -1:
             class_names_for_plot = ["Error/Unknown"]
        else:
            class_names_for_plot = dataset.classes

        plot_dataset_sample(
            rgb_tensor=rgb_batch[0],
            spectral_tensor=spectral_batch[0] if spectral_batch is not None else None,
            label=labels_batch[0].item(),
            stage=stages_batch[0],
            class_names=class_names_for_plot,
            filename=f"{dataset_name}_sample_0.png"
        )
        logger.info(f"Successfully tested and visualized sample from {dataset_name}.")

    except StopIteration:
        logger.error(f"DataLoader for {dataset_name} is empty despite dataset having {len(dataset)} items.")
    except Exception as e:
        logger.error(f"Error during batch fetching or processing for {dataset_name}: {e}", exc_info=True)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Starting data loading and progression demonstration...")

    demonstrate_progression_simulation(ORIGINAL_DATASET_ROOT, sample_idx=0)

    test_dataset_loading(
        root_dir=ORIGINAL_DATASET_ROOT,
        dataset_name="Original_Dataset_With_Progression_And_Spectral",
        apply_progression=True,
        use_spectral=True
    )
    
    test_dataset_loading(
        root_dir=ORIGINAL_DATASET_ROOT,
        dataset_name="Original_Dataset_No_Progression_No_Spectral",
        apply_progression=False,
        use_spectral=False # This will test custom_collate_fn with spectral_tensor=None
    )

    test_dataset_loading(
        root_dir=AUGMENTED_DATASET_ROOT,
        dataset_name="Augmented_Dataset_No_Progression_With_Spectral",
        apply_progression=False,
        use_spectral=True
    )
    logger.info("Finished all tests.")

if __name__ == "__main__":
    main()