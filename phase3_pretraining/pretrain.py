# phase3_pretraining/pretrain.py
import torch
import os
import sys
import logging
import numpy as np
from torch.utils.data import DataLoader
import time
# import math # Not directly used here, but trainer might need it

# --- Path Setup ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(script_dir) # Should be phase3_pretraining
    # project_root = os.path.dirname(package_root) # Should be parent of phase3_pretraining
    # For imports, ensure package_root is enough if structure is phase3_pretraining/../
    # sys.path is usually for finding top-level packages.
    # If 'phase3_pretraining' is a package itself, and you run scripts from within it
    # or from its parent, Python's import mechanism should handle it.
    # Let's assume 'phase3_pretraining' is in PYTHONPATH or you run from its parent.
    if package_root not in sys.path:
        sys.path.insert(0, package_root) # Add phase3_pretraining to path
    # if project_root not in sys.path: # Adding project root is also common
    #    sys.path.insert(0, project_root)

except Exception as e:
    print(f"ERROR during path setup in pretrain.py: {e}"); sys.exit(1)

# --- Project Imports ---
try:
    from phase3_pretraining.config import config
    from phase3_pretraining.utils.logging_setup import setup_logging # MODIFIED
    from phase3_pretraining.utils.augmentations import SimCLRAugmentation
    from phase3_pretraining.utils.losses import InfoNCELoss
    from phase3_pretraining.dataset import SARCLD2024Dataset
    from phase3_pretraining.models.hvt_wrapper import HVTForPretraining
    from phase3_pretraining.pretrain.trainer import Pretrainer
except ImportError as e:
    # Try to provide more context on ImportError
    import traceback
    print(f"CRITICAL ERROR: Failed to import necessary modules for pretrain.py: {e}")
    print("Traceback:\n", traceback.format_exc())
    print("Current sys.path:", sys.path)
    print("Current working directory:", os.getcwd())
    sys.exit(1)

# --- Setup Logging ---
log_file_name = config.get("log_file_pretrain", "default_pretrain.log")
log_dir_from_config = config.get("log_dir", "logs") # relative to package_root
# Construct absolute log_dir_path based on package_root
log_dir_path = os.path.join(package_root, log_dir_from_config)
os.makedirs(log_dir_path, exist_ok=True)

# Configure logging: DEBUG to file, INFO to console
# Ensure this is called only once for the root logger
if not logging.getLogger().hasHandlers(): # Check if root logger already has handlers
    setup_logging(
        log_file_name=log_file_name,
        log_dir=log_dir_path,
        log_level_file=logging.DEBUG,    # Detailed logs to file
        log_level_console=logging.INFO   # Minimal logs to console
    )
    # print(f"DEBUG (pretrain.py): Root logger configured. Log level File: DEBUG, Console: INFO. File: {os.path.join(log_dir_path, log_file_name)}")
# else:
    # print(f"DEBUG (pretrain.py): Root logger seems already configured.")

logger = logging.getLogger(__name__) # Get logger for the current module

# --- Apply Seed and PyTorch Optimizations ---
# (Keep this section as is)
RANDOM_SEED = config['seed']
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
logger.info(f"Global random seed set to: {RANDOM_SEED}") # Will appear on console
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    if config.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
        logger.info("torch.backends.cudnn.benchmark = True") # Console
    matmul_precision = config.get("matmul_precision")
    if matmul_precision and hasattr(torch, 'set_float32_matmul_precision'):
        try:
            torch.set_float32_matmul_precision(matmul_precision)
            logger.info(f"torch.set_float32_matmul_precision('{matmul_precision}')") # Console
        except Exception as e:
            logger.warning(f"Failed to set matmul_precision '{matmul_precision}': {e}") # Console
    elif matmul_precision:
        logger.warning(f"Matmul_precision '{matmul_precision}' configured, but torch.set_float32_matmul_precision not available.")
else:
    logger.warning("CUDA not available. PyTorch optimizations for CUDA will not be applied.") # Console


def main_pretrain():
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        logger.warning("CUDA specified but not available. Switched to CPU.") # Console
    logger.info(f"Using device: {device}") # Console
    if device == 'cuda':
        logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}); CUDA Version: {torch.version.cuda}") # Console

    img_size = tuple(config['pretrain_img_size'])
    logger.info(f"Selected image size for pre-training: {img_size}") # Console

    logger.info("Initializing datasets...") # Console
    dataset_kwargs = {
        "root_dir": config['data_root'], "img_size": img_size,
        "train_split_ratio": config['train_split_ratio'],
        "original_dataset_name": config['original_dataset_name'],
        "augmented_dataset_name": config['augmented_dataset_name'],
        "random_seed": config['seed'], "use_spectral": False,
        "spectral_channels": config['hvt_spectral_channels']
    }
    try:
        # Dataset __init__ logs at INFO/DEBUG level.
        # INFO messages from dataset __init__ (like "Scanning...") will appear on console.
        pretrain_dataset = SARCLD2024Dataset(**dataset_kwargs, split="train", normalize_for_model=False)
        
        # --- Manual __getitem__(0) test ---
        if len(pretrain_dataset) > 0:
            logger.debug("Manually testing pretrain_dataset[0]...") # DEBUG - Not on console
            try:
                test_rgb, test_label = pretrain_dataset[0] # __getitem__ will log verbosely for first few calls (DEBUG)
                logger.debug(f"Manual pretrain_dataset[0] SUCCESS. RGB shape: {test_rgb.shape}, Label: {test_label}") # DEBUG
            except Exception as e_getitem_test:
                logger.error(f"Manual pretrain_dataset[0] FAILED: {e_getitem_test}", exc_info=True) # ERROR - Console
                sys.exit("Exiting due to manual __getitem__(0) test failure.")
        else:
            logger.error("Pretrain dataset is empty after initialization. Cannot proceed.") # ERROR - Console
            sys.exit("Exiting due to empty pretrain dataset.")
        # --- End manual __getitem__(0) test ---

        probe_train_dataset = SARCLD2024Dataset(**dataset_kwargs, split="train", normalize_for_model=True)
        probe_val_dataset = SARCLD2024Dataset(**dataset_kwargs, split="val", normalize_for_model=True)
    except Exception as e:
        logger.error(f"Failed to initialize datasets: {e}", exc_info=True); sys.exit(1) # ERROR - Console

    logger.info(f"Pretrain dataset size: {len(pretrain_dataset)}") # Console
    logger.info(f"Probe train dataset size: {len(probe_train_dataset)}, Probe val dataset size: {len(probe_val_dataset)}") # Console

    num_workers_val = config.get('num_workers', 0)
    # The warning about num_workers != 0 will still appear on console if triggered.
    if num_workers_val != 0:
        logger.warning(f"NUM_WORKERS IS NOT 0 (it's {num_workers_val}). For current debugging, consider setting 'num_workers': 0 in config.py for simpler debugging trace.")

    prefetch_factor_val = config.get('prefetch_factor', 2) if num_workers_val > 0 else None
    batch_size_pretrain = config['pretrain_batch_size']
    batch_size_probe = config.get('probe_batch_size', batch_size_pretrain)
    use_persistent_workers = (device == 'cuda' and num_workers_val > 0 and torch.cuda.is_available())

    common_loader_args = {"num_workers": num_workers_val, "pin_memory": (device == 'cuda'),
                          "persistent_workers": use_persistent_workers}
    if prefetch_factor_val is not None: # Only add if num_workers > 0
        common_loader_args["prefetch_factor"] = prefetch_factor_val
    if num_workers_val == 0: # Explicitly set to False for num_workers = 0
        common_loader_args["persistent_workers"] = False
        common_loader_args.pop("prefetch_factor", None)


    try:
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size_pretrain, shuffle=True, drop_last=True, **common_loader_args)
        probe_train_loader = DataLoader(probe_train_dataset, batch_size=batch_size_probe, shuffle=True, drop_last=False, **common_loader_args)
        probe_val_loader = DataLoader(probe_val_dataset, batch_size=batch_size_probe, shuffle=False, drop_last=False, **common_loader_args)
    except Exception as e:
        logger.error(f"Failed to create DataLoaders: {e}", exc_info=True); sys.exit(1) # ERROR - Console
    
    logger.info(f"Pretrain DataLoader: {len(pretrain_loader)} batches of size {batch_size_pretrain}. num_workers={num_workers_val}, prefetch={prefetch_factor_val if num_workers_val > 0 else 'N/A'}, persistent={common_loader_args['persistent_workers']}") # Console

    logger.info("Initializing HVTForPretraining model...") # Console
    # HVTWrapper __init__ logs at INFO level - these will appear on console
    model_to_train = HVTForPretraining(img_size=img_size).to(device)

    if config.get("enable_torch_compile", False) and hasattr(torch, 'compile'):
        compile_mode = config.get("torch_compile_mode", "reduce-overhead")
        logger.info(f"Attempting to compile model with torch.compile(mode='{compile_mode}')...") # Console
        try:
            model_to_train = torch.compile(model_to_train, mode=compile_mode)
            logger.info("Model compiled.") # Console
        except Exception as e:
            logger.warning(f"torch.compile() failed: {e}. No compilation.", exc_info=False) # exc_info=False for console
            logger.debug("torch.compile() failure details:", exc_info=True) # Full details to file
    else:
        logger.info("torch.compile() not enabled or not available.") # Console

    augmentations = SimCLRAugmentation(img_size=img_size, s=config.get("simclr_s",1.0), p_grayscale=config.get("simclr_p_grayscale",0.2), p_gaussian_blur=config.get("simclr_p_gaussian_blur",0.5))
    loss_fn = InfoNCELoss(temperature=config['temperature'])
    
    logger.info("Initializing Pretrainer...") # Console
    # Pretrainer __init__ logs at INFO level - these will appear on console
    try:
        pretrainer_instance = Pretrainer(model=model_to_train, augmentations=augmentations, loss_fn=loss_fn, device=device,
                                         train_loader_for_probe=probe_train_loader, val_loader_for_probe=probe_val_loader,
                                         h100_optim_config=config)
    except Exception as e:
        logger.error(f"Failed to initialize Pretrainer: {e}", exc_info=True); sys.exit(1) # ERROR - Console

    pretrain_epochs = config['pretrain_epochs']
    evaluate_every_n_epochs = config['evaluate_every_n_epochs']
    save_every_n_epochs = config.get('save_every_n_epochs', 20)
    logger.info(f"Starting pre-training for {pretrain_epochs} epochs.") # Console
    start_epoch = 1
    best_probe_acc = -1.0
    last_completed_epoch = 0

    try:
        batches_per_epoch_pretrain = len(pretrain_loader)
    except TypeError: # Handles cases where DataLoader has no __len__ (e.g. IterableDataset without explicit length)
        logger.warning("Could not get len(pretrain_loader) directly. Will iterate without a fixed batch count per epoch for tqdm.")
        batches_per_epoch_pretrain = None # Or estimate if possible, or handle in trainer
    except Exception as e:
        logger.error(f"Could not get len(pretrain_loader): {e}", exc_info=True); sys.exit(1)

    if batches_per_epoch_pretrain == 0 and len(pretrain_dataset) > 0:
        logger.error("Pretrain DataLoader empty but dataset not. Check DataLoader batch_size and drop_last settings, or dataset integrity."); sys.exit(1)
    elif len(pretrain_dataset) == 0:
        logger.error("Pretrain dataset empty. Cannot start training."); sys.exit(1)

    try:
        for epoch in range(start_epoch, pretrain_epochs + 1):
            last_completed_epoch = epoch -1 # In case of interruption within the epoch
            epoch_start_time = time.time()
            
            # train_one_epoch logs its own INFO messages for start/end/avg_loss, which will go to console.
            # tqdm progress bar will also be on console.
            # Detailed per-batch logs from train_one_epoch are DEBUG, so only in file.
            avg_loss = pretrainer_instance.train_one_epoch(pretrain_loader, epoch, pretrain_epochs, batches_per_epoch=batches_per_epoch_pretrain)
            
            epoch_duration = time.time() - epoch_start_time
            samples_per_sec = (batches_per_epoch_pretrain * batch_size_pretrain) / epoch_duration if epoch_duration > 0 and batches_per_epoch_pretrain is not None else 0
            
            # This is the main epoch summary for console
            logger.info(
                f"SSL Epoch {epoch}/{pretrain_epochs} COMPLETED. "
                f"Duration: {epoch_duration:.2f}s. "
                f"Samples/sec: {samples_per_sec:.2f}. "
                f"Avg Loss: {avg_loss:.4f}. "
                f"LR: {pretrainer_instance.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            probe_accuracy = -1.0 # Reset for this epoch's potential evaluation
            if epoch % evaluate_every_n_epochs == 0 or epoch == pretrain_epochs:
                if len(probe_val_loader.dataset) > 0 and len(probe_train_loader.dataset) > 0:
                    probe_start_time = time.time()
                    # evaluate_linear_probe logs its own INFO messages, which will go to console.
                    probe_accuracy = pretrainer_instance.evaluate_linear_probe(current_epoch=epoch)
                    probe_duration = time.time() - probe_start_time
                    logger.info(f"Linear Probe (after SSL E{epoch}) duration: {probe_duration:.2f}s. Accuracy: {probe_accuracy:.2f}%") # Console
                    if probe_accuracy > best_probe_acc:
                        best_probe_acc = probe_accuracy
                        logger.info(f"New best probe acc: {best_probe_acc:.2f}% (SSL E{epoch}). Saving best model checkpoint.") # Console
                        pretrainer_instance.save_checkpoint(epoch=f"{epoch}_bestprobe", best_metric=best_probe_acc, file_name=f"{config.get('model_name','hvt').lower()}_pretrain_best_probe.pth")
                else:
                    logger.warning(f"Skipping probe evaluation at SSL E{epoch}: one or both probe datasets are empty.") # Console
            
            if epoch % save_every_n_epochs == 0 or epoch == pretrain_epochs:
                 # save_checkpoint logs INFO messages which will go to console
                 pretrainer_instance.save_checkpoint(epoch=epoch, best_metric=probe_accuracy if probe_accuracy != -1.0 else best_probe_acc) # Save current or best known probe acc
            
            last_completed_epoch = epoch # Mark epoch as fully completed

            if torch.cuda.is_available():
                # CUDA memory log is DEBUG - only to file
                logger.debug(
                    f"CUDA Mem E{epoch}: "
                    f"Alloc: {torch.cuda.memory_allocated(0)/1024**2:.1f}MB "
                    f"Reserved: {torch.cuda.memory_reserved(0)/1024**2:.1f}MB "
                    f"MaxAlloc: {torch.cuda.max_memory_allocated(0)/1024**2:.1f}MB "
                    f"MaxReserved: {torch.cuda.max_memory_reserved(0)/1024**2:.1f}MB"
                )

    except KeyboardInterrupt:
        logger.warning(f"Pre-training interrupted by user at epoch {last_completed_epoch+1}.") # Console
    except Exception as e:
        logger.exception(f"Critical error during pre-training at epoch {last_completed_epoch+1}: {e}") # Console + Full Traceback to file
        sys.exit(1)
    finally:
        logger.info(f"Pre-training finished or interrupted after {last_completed_epoch} completed epochs.") # Console
        if 'pretrainer_instance' in locals() and pretrainer_instance is not None:
            final_save_name = config.get("pretrain_final_checkpoint_name", f"{config.get('model_name','hvt').lower()}_pretrain_epoch_final.pth")
            # save_checkpoint logs INFO messages which will go to console
            pretrainer_instance.save_checkpoint(epoch=last_completed_epoch if last_completed_epoch > 0 else 'final_interrupted', best_metric=best_probe_acc, file_name=final_save_name)
            logger.info(f"Final pre-trained model saved. Best probe accuracy during run: {best_probe_acc:.2f}%") # Console

if __name__ == "__main__":
    main_pretrain()