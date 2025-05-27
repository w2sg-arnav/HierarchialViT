# phase3_pretraining/run_ssl_pretraining.py
import torch
import os
import sys
import logging
import numpy as np
from torch.utils.data import DataLoader
import time
from datetime import datetime
import re # For parsing epoch from string

# --- Explicit Path Setup for Sibling Package 'phase2_model' ---
try:
    _current_script_path = os.path.abspath(__file__)
    _phase3_root_path = os.path.dirname(_current_script_path)
    _project_root_path_for_imports = os.path.dirname(_phase3_root_path)
    if _project_root_path_for_imports not in sys.path:
        sys.path.insert(0, _project_root_path_for_imports)
except NameError: pass

# --- Module Imports ---
try:
    from .config import config
    from .utils.logging_setup import setup_logging
    _run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _abs_log_dir = os.path.join(config["PACKAGE_ROOT_PATH"], config["log_dir_name"])
    _log_file_base = os.path.splitext(config.get("log_file_pretrain", "pretrain_run.log"))[0]
    _final_log_file_name = f"{_log_file_base}_{_run_timestamp}.log"

    setup_logging(
        log_dir_abs_path=_abs_log_dir, log_file_name=_final_log_file_name,
        log_level_file=logging.DEBUG, log_level_console=logging.INFO,
        run_timestamp=_run_timestamp
    )
    logger = logging.getLogger(__name__)

    from .utils.augmentations import SimCLRAugmentation
    from .utils.losses import InfoNCELoss
    from .dataset import SARCLD2024Dataset
    from .models.hvt_wrapper import HVTForPretraining
    from .pretrain.trainer import Pretrainer
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR in run_ssl_pretraining.py: {e}", file=sys.stderr)
    import traceback; traceback.print_exc(); sys.exit(1)

def apply_pytorch_optimizations(run_cfg: dict):
    # ... (same as before) ...
    if run_cfg['device'] == 'cuda' and torch.cuda.is_available():
        if run_cfg.get("cudnn_benchmark", False): torch.backends.cudnn.benchmark = True; logger.info("torch.backends.cudnn.benchmark = True")
        matmul_precision = run_cfg.get("matmul_precision")
        if matmul_precision and hasattr(torch, 'set_float32_matmul_precision'):
            try: torch.set_float32_matmul_precision(matmul_precision); logger.info(f"torch.set_float32_matmul_precision('{matmul_precision}')")
            except Exception as e: logger.warning(f"Failed to set matmul_precision '{matmul_precision}': {e}")
    else: logger.info("CUDA not available/selected. Skipping CUDA-specific opts.")


def main_pretrain_script():
    logger.info(f"======== Starting Phase 3: HVT Self-Supervised Pre-training (Run ID: {_run_timestamp}) ========")
    logger.info(f"Full run configuration snapshot: {config}")

    torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    if config['device'] == 'cuda' and torch.cuda.is_available(): torch.cuda.manual_seed_all(config['seed'])
    apply_pytorch_optimizations(config)
    logger.info(f"Global random seed: {config['seed']}. Device: {config['device']}.")
    if config['device'] == 'cuda' and torch.cuda.is_available(): logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    img_size_pretrain = tuple(config['pretrain_img_size'])
    dataset_common_args = { # ... (same as before) ...
        "root_dir": config['data_root'], "img_size": img_size_pretrain,
        "train_split_ratio": config['train_split_ratio'], "original_dataset_name": config['original_dataset_name'],
        "augmented_dataset_name": config['augmented_dataset_name'], "random_seed": config['seed'],
        "spectral_channels": config['hvt_params_for_backbone']['spectral_channels']
    }
    try: # ... (dataset loading same as before) ...
        pretrain_dataset = SARCLD2024Dataset(**dataset_common_args, split="train", normalize_for_model=False, use_spectral=False)
        probe_train_dataset = SARCLD2024Dataset(**dataset_common_args, split="train", normalize_for_model=True, use_spectral=False)
        probe_val_dataset = SARCLD2024Dataset(**dataset_common_args, split="val", normalize_for_model=True, use_spectral=False)
    except Exception as e_data: logger.error(f"Dataset init error: {e_data}", exc_info=True); sys.exit(1)
    if len(pretrain_dataset) == 0: logger.error("Pretrain dataset empty. Exiting."); sys.exit(1)
    logger.info(f"Dataset sizes: Pretrain={len(pretrain_dataset)}, ProbeTrain={len(probe_train_dataset)}, ProbeVal={len(probe_val_dataset)}")

    num_w = config.get('num_workers', 0); prefetch_f = config.get('prefetch_factor', 2) if num_w > 0 and config['device'] == 'cuda' else None
    persistent_w = (config['device'] == 'cuda' and num_w > 0 and torch.cuda.is_available())
    loader_args = {"num_workers": num_w, "pin_memory": (config['device']=='cuda'), "persistent_workers": persistent_w}
    if prefetch_f: loader_args["prefetch_factor"] = prefetch_f
    if num_w == 0: loader_args["persistent_workers"] = False; loader_args.pop("prefetch_factor", None)

    pretrain_loader = DataLoader(pretrain_dataset, batch_size=config['pretrain_batch_size'], shuffle=True, drop_last=True, **loader_args)
    probe_train_loader = DataLoader(probe_train_dataset, batch_size=config['probe_batch_size'], shuffle=True, drop_last=False, **loader_args)
    probe_val_loader = DataLoader(probe_val_dataset, batch_size=config['probe_batch_size'], shuffle=False, drop_last=False, **loader_args)
    logger.info(f"Pretrain DataLoader: {len(pretrain_loader)} batches, BS={config['pretrain_batch_size']}, Workers={num_w}")

    model = HVTForPretraining(img_size=img_size_pretrain, num_classes_for_probe=config['num_classes']).to(config['device'])
    augment_pipeline = SimCLRAugmentation(img_size=img_size_pretrain, s=config.get("simclr_s"), p_grayscale=config.get("simclr_p_grayscale"), p_gaussian_blur=config.get("simclr_p_gaussian_blur"), rrc_scale_min=config.get("simclr_rrc_scale_min"))
    criterion = InfoNCELoss(temperature=config['temperature'])
    pretrainer = Pretrainer(model=model, augmentations=augment_pipeline, loss_fn=criterion, device=config['device'], train_loader_for_probe=probe_train_loader, val_loader_for_probe=probe_val_loader)

    start_epoch = 1; best_probe_accuracy = -1.0
    numerically_last_completed_epoch = 0 # 0-based, used for scheduler re-initialization

    resume_path = config.get("resume_checkpoint_path")
    if resume_path and resume_path.strip() != "" and os.path.exists(resume_path):
        logger.info(f"Attempting to resume training from checkpoint: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=config['device'])
            
            if 'model_backbone_state_dict' in checkpoint and 'projection_head_state_dict' in checkpoint:
                pretrainer.model.backbone.load_state_dict(checkpoint['model_backbone_state_dict'])
                pretrainer.model.projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
                logger.info("Loaded backbone and projection head state dicts.")
            elif 'model_state_dict' in checkpoint and checkpoint.get('is_wrapper_fallback', False):
                 pretrainer.model.load_state_dict(checkpoint['model_state_dict']); logger.info("Loaded full wrapper state (fallback).")
            else: logger.warning("Checkpoint model state dict keys not found. Weights not loaded.")

            if 'optimizer_state_dict' in checkpoint: pretrainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict']); logger.info("Optimizer state loaded.")
            else: logger.warning("Optimizer state_dict not found in checkpoint.")
            if 'scaler_state_dict' in checkpoint and pretrainer.scaler.is_enabled(): pretrainer.scaler.load_state_dict(checkpoint['scaler_state_dict']); logger.info("GradScaler state loaded.")
            elif 'scaler_state_dict' in checkpoint: logger.info("Scaler state found but current run has scaler disabled.")
            else: logger.warning("GradScaler state_dict not found in checkpoint.")
            
            raw_loaded_epoch = checkpoint.get('epoch')
            if isinstance(raw_loaded_epoch, str): # Handle "20_bestprobe" etc.
                match = re.search(r'\d+', raw_loaded_epoch)
                if match: numerically_last_completed_epoch = int(match.group(0))
                else: logger.warning(f"Could not parse num epoch from '{raw_loaded_epoch}'. Assuming 0."); numerically_last_completed_epoch = 0
            elif isinstance(raw_loaded_epoch, int):
                numerically_last_completed_epoch = raw_loaded_epoch
            else: # Default if 'epoch' key is missing or invalid type
                logger.warning(f"Invalid or missing 'epoch' key in checkpoint. Type: {type(raw_loaded_epoch)}. Assuming 0 completed epochs."); numerically_last_completed_epoch = 0

            start_epoch = numerically_last_completed_epoch + 1
            best_probe_accuracy = checkpoint.get('best_probe_metric', -1.0)
            
            if 'scheduler_state_dict' in checkpoint: logger.info("Scheduler state_dict found. Pretrainer will use 'numerically_last_completed_epoch' for re-initialization.")
            logger.info(f"Resuming from completed SSL epoch {numerically_last_completed_epoch}. Will start training at epoch {start_epoch}. Best probe: {best_probe_accuracy:.2f}%")

        except Exception as e_resume:
            logger.error(f"Error loading checkpoint from {resume_path}: {e_resume}. Starting fresh.", exc_info=True)
            start_epoch = 1; best_probe_accuracy = -1.0; numerically_last_completed_epoch = 0
    else:
        if resume_path and resume_path.strip() != "": logger.warning(f"Resume checkpoint path '{resume_path}' not found. Starting fresh.")
        else: logger.info("No resume checkpoint. Starting fresh.")
        numerically_last_completed_epoch = 0 # Ensure it's 0 for fresh run scheduler init

    if config.get("enable_torch_compile", False) and hasattr(torch, 'compile'):
        logger.info(f"Attempting torch.compile(model, mode='{config.get('torch_compile_mode')}')...");
        try: pretrainer.model = torch.compile(pretrainer.model, mode=config.get('torch_compile_mode')); logger.info("Model compiled.")
        except Exception as e_compile: logger.warning(f"torch.compile() failed: {e_compile}. No compilation.", exc_info=True)

    pretrain_total_epochs = config['pretrain_epochs']
    eval_every = config['evaluate_every_n_epochs']; save_every = config.get('save_every_n_epochs', 20)
    logger.info(f"Starting/Resuming SimCLR pre-training from epoch {start_epoch} up to {pretrain_total_epochs} total epochs.")
    
    completed_epochs_this_run = 0
    try:
        batches_per_epoch = len(pretrain_loader);
        if batches_per_epoch == 0: raise ValueError("Pretrain DataLoader empty.")

        for epoch_1_based in range(start_epoch, pretrain_total_epochs + 1):
            epoch_start_time = time.time()
            avg_loss = pretrainer.train_one_epoch(
                pretrain_loader, epoch_1_based, pretrain_total_epochs, batches_per_epoch,
                numerically_last_completed_epoch_0_based=numerically_last_completed_epoch # Pass 0-based last completed
            )
            numerically_last_completed_epoch = epoch_1_based # Update for next potential scheduler init on interruption
            
            epoch_duration = time.time() - epoch_start_time; samples_per_sec = (batches_per_epoch * config['pretrain_batch_size'] * config['accumulation_steps']) / epoch_duration if epoch_duration > 0 else 0
            logger.info(f"SSL Epoch {epoch_1_based}/{pretrain_total_epochs} | Duration: {epoch_duration:.2f}s | Eff. Samples/sec: {samples_per_sec:.2f} | Avg Loss: {avg_loss:.4f} | LR: {pretrainer.optimizer.param_groups[0]['lr']:.2e}")

            current_probe_acc_this_eval = -1.0
            if epoch_1_based % eval_every == 0 or epoch_1_based == pretrain_total_epochs:
                current_probe_acc_this_eval = pretrainer.evaluate_linear_probe(current_ssl_epoch=epoch_1_based)
                if current_probe_acc_this_eval > best_probe_accuracy:
                    best_probe_accuracy = current_probe_acc_this_eval
                    logger.info(f"New best probe: {best_probe_accuracy:.2f}% (SSL E{epoch_1_based}). Saving best model.")
                    pretrainer.save_checkpoint(numerical_epoch_completed=epoch_1_based, best_metric=best_probe_accuracy, is_best_probe_save=True)
            
            if epoch_1_based % save_every == 0 or epoch_1_based == pretrain_total_epochs:
                 # Save regular epoch checkpoint, using current_probe_acc_this_eval if just evaluated, else best_probe_accuracy
                 metric_for_save = current_probe_acc_this_eval if current_probe_acc_this_eval != -1.0 else best_probe_accuracy
                 pretrainer.save_checkpoint(numerical_epoch_completed=epoch_1_based, best_metric=metric_for_save, is_best_probe_save=False)
            
            completed_epochs_this_run += 1
            if torch.cuda.is_available(): logger.debug(f"CUDA Mem E{epoch_1_based} END: Alloc {torch.cuda.memory_allocated(0)/1024**2:.1f}MB, MaxAlloc {torch.cuda.max_memory_allocated(0)/1024**2:.1f}MB")

    except KeyboardInterrupt: logger.warning(f"Pre-training interrupted. Epochs this run: {completed_epochs_this_run}. Total considered completed: {start_epoch + completed_epochs_this_run -1}.")
    except Exception as e_train: logger.error(f"Fatal error during pre-training at SSL epoch {start_epoch + completed_epochs_this_run}: {e_train}", exc_info=True); sys.exit(1)
    finally:
        final_epoch_abs = start_epoch + completed_epochs_this_run -1 if completed_epochs_this_run > 0 else start_epoch -1
        if completed_epochs_this_run == 0 and start_epoch == 1 : final_epoch_abs = 0

        logger.info(f"Pre-training finished/interrupted. Absolute epochs completed: {final_epoch_abs}.")
        if 'pretrainer' in locals() and pretrainer is not None:
            final_save_name = f"{config.get('model_arch_name_for_ckpt','hvt_simclr')}_final_epoch{final_epoch_abs}.pth"
            pretrainer.save_checkpoint(numerical_epoch_completed=final_epoch_abs if final_epoch_abs >=0 else 'interrupted_before_first_epoch',
                                        best_metric=best_probe_accuracy, file_name_override=final_save_name)
            logger.info(f"Final model checkpoint saved. Best probe accuracy during run: {best_probe_accuracy:.2f}%")

if __name__ == "__main__":
    main_pretrain_script()