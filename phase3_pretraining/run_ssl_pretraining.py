# phase3_pretraining/pretrain.py
import torch
import os
import sys
import logging
import numpy as np
from torch.utils.data import DataLoader
import time
from datetime import datetime # For unique run timestamp

# --- Path Setup (More Robust) ---
# This script is intended to be run as the main entry point for pre-training.
# It assumes it's located at phase3_pretraining/pretrain.py
# The PROJECT_ROOT_PATH should be correctly set in config.py for dataset/checkpoint paths.

# --- Module Imports (using relative imports for modules within this package) ---
try:
    from .config import config # Relative import for config.py
    # Setup logging as early as possible, after config is loaded (config defines paths)
    from .utils.logging_setup import setup_logging
    _run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _abs_log_dir = os.path.join(config["PACKAGE_ROOT_PATH"], config["log_dir_name"])
    setup_logging(
        log_dir_abs_path=_abs_log_dir,
        log_file_name=config.get("log_file_pretrain", f"pretrain_run_{_run_timestamp}.log"),
        log_level_file=logging.DEBUG,
        log_level_console=logging.INFO,
        run_timestamp=_run_timestamp # Pass timestamp for unique log file per run
    )
    logger = logging.getLogger(__name__) # Get logger for this script

    from .utils.augmentations import SimCLRAugmentation
    from .utils.losses import InfoNCELoss
    from .dataset import SARCLD2024Dataset
    from .models.hvt_wrapper import HVTForPretraining
    from .pretrain.trainer import Pretrainer
except ImportError as e:
    # Basic print for critical import errors if logging isn't even set up
    print(f"CRITICAL IMPORT ERROR in pretrain.py: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
# --- End Module Imports ---


def apply_pytorch_optimizations(run_cfg: dict):
    """Applies PyTorch performance settings from the config."""
    if run_cfg['device'] == 'cuda' and torch.cuda.is_available():
        if run_cfg.get("cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True
            logger.info("torch.backends.cudnn.benchmark = True")
        
        matmul_precision = run_cfg.get("matmul_precision")
        if matmul_precision and hasattr(torch, 'set_float32_matmul_precision'):
            try:
                torch.set_float32_matmul_precision(matmul_precision)
                logger.info(f"torch.set_float32_matmul_precision('{matmul_precision}')")
            except Exception as e_matmul:
                logger.warning(f"Failed to set matmul_precision '{matmul_precision}': {e_matmul}")
        elif matmul_precision:
            logger.warning(f"Matmul_precision '{matmul_precision}' configured, but torch.set_float32_matmul_precision not available in this PyTorch version.")
    else:
        logger.info("CUDA not available or not selected. Skipping CUDA-specific PyTorch optimizations.")


def main_pretrain_script():
    logger.info("======== Starting Phase 3: HVT Self-Supervised Pre-training ========")
    logger.info(f"Full run configuration: {config}") # Log the entire config for reproducibility

    # --- Seed and PyTorch Optimizations ---
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if config['device'] == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    apply_pytorch_optimizations(config)
    logger.info(f"Global random seed set to: {config['seed']}")
    logger.info(f"Using device: {config['device']}")
    if config['device'] == 'cuda' and torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}, PyTorch CUDA Version: {torch.version.cuda}")

    # --- Datasets and DataLoaders ---
    img_size_pretrain = tuple(config['pretrain_img_size'])
    logger.info(f"Target image size for pre-training: {img_size_pretrain}")

    dataset_common_args = {
        "root_dir": config['data_root'],
        "img_size": img_size_pretrain, # Pre-training uses this size
        "train_split_ratio": config['train_split_ratio'],
        "original_dataset_name": config['original_dataset_name'],
        "augmented_dataset_name": config['augmented_dataset_name'],
        "random_seed": config['seed'],
        "spectral_channels": config['hvt_params_for_backbone']['spectral_channels']
    }
    try:
        # For SimCLR pre-training, `normalize_for_model` is False. Images are [0,1].
        # `use_spectral` is False because SimCLR here is on RGB.
        pretrain_dataset = SARCLD2024Dataset(**dataset_common_args, split="train", normalize_for_model=False, use_spectral=False)
        # For linear probe, `normalize_for_model` is True.
        probe_train_dataset = SARCLD2024Dataset(**dataset_common_args, split="train", normalize_for_model=True, use_spectral=False)
        probe_val_dataset = SARCLD2024Dataset(**dataset_common_args, split="val", normalize_for_model=True, use_spectral=False)
    except Exception as e_data:
        logger.error(f"Fatal error initializing datasets: {e_data}", exc_info=True); sys.exit(1)

    if len(pretrain_dataset) == 0: logger.error("Pretrain dataset is empty. Exiting."); sys.exit(1)
    logger.info(f"Pretrain dataset size: {len(pretrain_dataset)}")
    logger.info(f"Probe train dataset: {len(probe_train_dataset)}, Probe val dataset: {len(probe_val_dataset)}")

    num_w = config.get('num_workers', 0)
    prefetch_f = config.get('prefetch_factor', 2) if num_w > 0 else None
    persistent_w = (config['device'] == 'cuda' and num_w > 0 and torch.cuda.is_available())
    loader_args = {"num_workers": num_w, "pin_memory": (config['device'] == 'cuda'), "persistent_workers": persistent_w}
    if prefetch_f is not None: loader_args["prefetch_factor"] = prefetch_f
    if num_w == 0: loader_args["persistent_workers"] = False; loader_args.pop("prefetch_factor", None)

    pretrain_loader = DataLoader(pretrain_dataset, batch_size=config['pretrain_batch_size'], shuffle=True, drop_last=True, **loader_args)
    probe_train_loader = DataLoader(probe_train_dataset, batch_size=config['probe_batch_size'], shuffle=True, drop_last=False, **loader_args)
    probe_val_loader = DataLoader(probe_val_dataset, batch_size=config['probe_batch_size'], shuffle=False, drop_last=False, **loader_args)
    logger.info(f"Pretrain DataLoader: {len(pretrain_loader)} batches of size {config['pretrain_batch_size']}. Num workers: {num_w}")

    # --- Model, Augmentations, Loss ---
    logger.info("Initializing HVTForPretraining model wrapper...")
    # HVT wrapper takes img_size for backbone instantiation and num_classes for its internal head (if used for finetuning)
    model = HVTForPretraining(img_size=img_size_pretrain, num_classes_for_probe=config['num_classes']).to(config['device'])

    if config.get("enable_torch_compile", False) and hasattr(torch, 'compile'):
        logger.info(f"Attempting torch.compile(mode='{config.get('torch_compile_mode')}')...")
        try: model = torch.compile(model, mode=config.get('torch_compile_mode')); logger.info("Model compiled successfully.")
        except Exception as e_compile: logger.warning(f"torch.compile() failed: {e_compile}. Proceeding without compilation.", exc_info=True)
    
    augment_pipeline = SimCLRAugmentation(img_size=img_size_pretrain, s=config.get("simclr_s"),
                                        p_grayscale=config.get("simclr_p_grayscale"),
                                        p_gaussian_blur=config.get("simclr_p_gaussian_blur"),
                                        rrc_scale_min=config.get("simclr_rrc_scale_min"))
    criterion = InfoNCELoss(temperature=config['temperature'])

    # --- Trainer ---
    logger.info("Initializing Pretrainer instance...")
    pretrainer = Pretrainer(model=model, augmentations=augment_pipeline, loss_fn=criterion, device=config['device'],
                            train_loader_for_probe=probe_train_loader, val_loader_for_probe=probe_val_loader)

    # --- Training Loop ---
    pretrain_total_epochs = config['pretrain_epochs']
    eval_every = config['evaluate_every_n_epochs']
    save_every = config.get('save_every_n_epochs', 20)
    logger.info(f"Starting SimCLR pre-training for {pretrain_total_epochs} epochs.")
    
    best_probe_accuracy = -1.0
    last_completed_epoch = 0
    try:
        batches_per_epoch = len(pretrain_loader)
        if batches_per_epoch == 0: raise ValueError("Pretrain DataLoader is empty.")

        for epoch in range(1, pretrain_total_epochs + 1):
            epoch_start_time = time.time()
            avg_loss = pretrainer.train_one_epoch(pretrain_loader, epoch, pretrain_total_epochs, batches_per_epoch)
            epoch_duration = time.time() - epoch_start_time
            samples_per_sec = (batches_per_epoch * config['pretrain_batch_size']) / epoch_duration if epoch_duration > 0 else 0
            
            logger.info(f"SSL Epoch {epoch}/{pretrain_total_epochs} | Duration: {epoch_duration:.2f}s | Samples/sec: {samples_per_sec:.2f} | Avg Loss: {avg_loss:.4f} | LR: {pretrainer.optimizer.param_groups[0]['lr']:.2e}")

            current_probe_acc = -1.0
            if epoch % eval_every == 0 or epoch == pretrain_total_epochs:
                current_probe_acc = pretrainer.evaluate_linear_probe(current_ssl_epoch=epoch)
                if current_probe_acc > best_probe_accuracy:
                    best_probe_accuracy = current_probe_acc
                    logger.info(f"New best probe accuracy: {best_probe_accuracy:.2f}% (SSL Epoch {epoch}). Saving best model.")
                    pretrainer.save_checkpoint(epoch=f"{epoch}_bestprobe", best_metric=best_probe_accuracy,
                                               file_name_override=f"{config.get('model_arch_name_for_ckpt','hvt_simclr')}_best_probe.pth")
            
            if epoch % save_every == 0 or epoch == pretrain_total_epochs:
                 metric_for_regular_save = current_probe_acc if current_probe_acc != -1.0 else best_probe_accuracy
                 pretrainer.save_checkpoint(epoch=epoch, best_metric=metric_for_regular_save)
            
            last_completed_epoch = epoch
            if torch.cuda.is_available(): logger.debug(f"CUDA Mem E{epoch}: Alloc {torch.cuda.memory_allocated(0)/1024**2:.1f}MB, MaxAlloc {torch.cuda.max_memory_allocated(0)/1024**2:.1f}MB")

    except KeyboardInterrupt: logger.warning(f"Pre-training interrupted by user after {last_completed_epoch} completed epochs.")
    except Exception as e_train: logger.error(f"Fatal error during pre-training loop at epoch {last_completed_epoch+1}: {e_train}", exc_info=True); sys.exit(1)
    finally:
        logger.info(f"Pre-training finished or was interrupted. Last completed epoch: {last_completed_epoch}.")
        final_save_name = f"{config.get('model_arch_name_for_ckpt','hvt_simclr')}_final_epoch{last_completed_epoch}.pth"
        pretrainer.save_checkpoint(epoch=last_completed_epoch if last_completed_epoch > 0 else 'interrupted_e0',
                                    best_metric=best_probe_accuracy, file_name_override=final_save_name)
        logger.info(f"Final pre-trained model checkpoint saved. Best probe accuracy during run: {best_probe_accuracy:.2f}%")

if __name__ == "__main__":
    # Ensure PROJECT_ROOT_PATH is correctly set in config.py before running.
    # This script is intended to be run from the project root (e.g., cvpr25/) as:
    # python -m phase3_pretraining.pretrain
    # Or, if you add phase3_pretraining to PYTHONPATH, you might run it directly.
    main_pretrain_script()