# phase4_finetuning/main.py
from collections import OrderedDict
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR, LinearLR
import argparse
import yaml
import torch.nn.functional as F
import math
import sys
from typing import Tuple, Optional, Dict, Any, List
import traceback
from datetime import datetime

logger: Optional[logging.Logger] = None

try:
    from .config import config as base_finetune_config
    from .dataset import SARCLD2024Dataset
    from .utils.augmentations import FinetuneAugmentation
    from .utils.logging_setup import setup_logging
    from .finetune.trainer import Finetuner
    from phase2_model.models.hvt import DiseaseAwareHVT, create_disease_aware_hvt
except ImportError as e_imp:
    print(f"CRITICAL IMPORT ERROR in phase4_main.py: {e_imp}.", file=sys.stderr); traceback.print_exc(); sys.exit(1)

# --- Helper Functions (get_cosine_schedule_with_warmup_step, load_config_from_yaml_or_default, parse_arguments, set_global_seed, _interpolate_positional_embedding, load_and_prepare_hvt_model) ---
# These functions remain IDENTICAL to the previous complete main.py I provided.
# For brevity, I'm not repeating them here. Ensure they are present.
# Key: load_and_prepare_hvt_model should correctly re-initialize the head.

def get_cosine_schedule_with_warmup_step(optimizer: torch.optim.Optimizer, num_warmup_steps: int,
                                    num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
    _local_logger_sched = logging.getLogger(f"{__name__}.get_cosine_schedule_with_warmup_step")
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps: return float(current_step) / float(max(1, num_warmup_steps))
        eff_training_steps = num_training_steps - num_warmup_steps
        if eff_training_steps <=0: return 0.0
        progress = float(current_step - num_warmup_steps) / float(max(1, eff_training_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def load_config_from_yaml_or_default(yaml_config_path: Optional[str] = None) -> Dict[str, Any]:
    config_to_use = base_finetune_config.copy()
    if yaml_config_path and os.path.exists(yaml_config_path):
        try:
            with open(yaml_config_path, 'r') as f: yaml_override_config = yaml.safe_load(f)
            if yaml_override_config and isinstance(yaml_override_config, dict):
                config_to_use.update(yaml_override_config)
                temp_logger = logging.getLogger(f"{__name__}.config_loader")
                if not temp_logger.hasHandlers(): temp_logger.addHandler(logging.StreamHandler(sys.stdout))
                temp_logger.info(f"Loaded and applied config overrides from YAML: {yaml_config_path}")
        except Exception as e_yaml: print(f"WARNING (config_loader): Could not load/parse YAML {yaml_config_path}: {e_yaml}. Using defaults.", file=sys.stderr)
    elif yaml_config_path: print(f"WARNING (config_loader): Specified YAML config {yaml_config_path} not found. Using defaults.", file=sys.stderr)
    return config_to_use

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning script for HVT models on Cotton Leaf Disease.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file (optional).")
    return parser.parse_args()

def set_global_seed(seed_value: int):
    torch.manual_seed(seed_value); np.random.seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)
    if logger: logger.info(f"Global random seed set to: {seed_value}")

def _interpolate_positional_embedding(
    checkpoint_pos_embed: torch.Tensor, model_pos_embed_param: nn.Parameter,
    hvt_patch_size: int, target_img_size_for_finetune: Tuple[int, int],
    ssl_img_size_for_pos_embed: Tuple[int, int]
    ):
    _local_logger_pe = logging.getLogger(f"{__name__}._interpolate_pe")
    if checkpoint_pos_embed.ndim == 2: checkpoint_pos_embed = checkpoint_pos_embed.unsqueeze(0)
    N_ckpt, C_ckpt = checkpoint_pos_embed.shape[1], checkpoint_pos_embed.shape[2]
    N_model, C_model = model_pos_embed_param.shape[1], model_pos_embed_param.shape[2]

    if C_ckpt != C_model: _local_logger_pe.error(f"PE C-dim mismatch: Ckpt {C_ckpt} vs Model {C_model}. No interp."); return model_pos_embed_param.data
    if N_ckpt == N_model: _local_logger_pe.debug("PE N-dim match. Using ckpt PE directly."); return checkpoint_pos_embed.view_as(model_pos_embed_param.data)

    H0_patches = ssl_img_size_for_pos_embed[0] // hvt_patch_size
    W0_patches = ssl_img_size_for_pos_embed[1] // hvt_patch_size
    if H0_patches * W0_patches != N_ckpt:
        if math.isqrt(N_ckpt)**2 == N_ckpt: H0_patches = W0_patches = math.isqrt(N_ckpt)
        else: _local_logger_pe.error(f"PE Interp: Cannot infer source grid (H0,W0) for N_ckpt={N_ckpt}. SSL img_size {ssl_img_size_for_pos_embed}. No interp."); return model_pos_embed_param.data

    Ht_patches = target_img_size_for_finetune[0] // hvt_patch_size
    Wt_patches = target_img_size_for_finetune[1] // hvt_patch_size
    if Ht_patches * Wt_patches != N_model: _local_logger_pe.error(f"PE Interp: Target grid {Ht_patches}x{Wt_patches} != N_model {N_model}. No interp."); return model_pos_embed_param.data

    _local_logger_pe.info(f"PE Interpolating: {N_ckpt}({H0_patches}x{W0_patches}) to {N_model}({Ht_patches}x{Wt_patches}) patches.")
    try:
        pe_to_interp = checkpoint_pos_embed.reshape(1, H0_patches, W0_patches, C_ckpt).permute(0, 3, 1, 2)
        pe_interpolated = F.interpolate(pe_to_interp, size=(Ht_patches, Wt_patches), mode='bicubic', align_corners=False)
        pe_interpolated = pe_interpolated.permute(0, 2, 3, 1).flatten(1, 2)
        if pe_interpolated.shape == model_pos_embed_param.shape: return pe_interpolated
        else: _local_logger_pe.error(f"PE Interp: Final shape mismatch ({pe_interpolated.shape} vs {model_pos_embed_param.shape})."); return model_pos_embed_param.data
    except Exception as e: _local_logger_pe.error(f"PE Interp Error: {e}", exc_info=True); return model_pos_embed_param.data

def load_and_prepare_hvt_model(hvt_model_instance: DiseaseAwareHVT, cfg: Dict[str, Any], device: str) -> DiseaseAwareHVT:
    _local_logger_load = logging.getLogger(f"{__name__}.load_and_prepare_hvt_model")
    hvt_model_instance.cpu()

    if cfg.get("load_pretrained_backbone", False):
        ssl_ckpt_path_from_cfg = cfg.get("pretrained_checkpoint_path")
        resolved_ssl_ckpt_path = ssl_ckpt_path_from_cfg
        if resolved_ssl_ckpt_path and not os.path.isabs(resolved_ssl_ckpt_path) and cfg.get("PROJECT_ROOT_PATH"):
             resolved_ssl_ckpt_path = os.path.join(cfg["PROJECT_ROOT_PATH"], ssl_ckpt_path_from_cfg)
        
        if resolved_ssl_ckpt_path and os.path.exists(resolved_ssl_ckpt_path):
            _local_logger_load.info(f"Loading SSL backbone weights from: {resolved_ssl_ckpt_path}")
            checkpoint = torch.load(resolved_ssl_ckpt_path, map_location='cpu')
            ssl_backbone_sd = checkpoint.get('model_backbone_state_dict')

            if ssl_backbone_sd:
                current_model_sd = hvt_model_instance.state_dict()
                new_sd_for_backbone = OrderedDict()
                ssl_run_cfg_snapshot = checkpoint.get('run_config_snapshot', {})
                ssl_img_size_val = ssl_run_cfg_snapshot.get('pretrain_img_size', cfg.get("ssl_pretrain_img_size_fallback", cfg["img_size"]))
                ssl_img_size_tuple = tuple(ssl_img_size_val)
                if ssl_img_size_tuple == tuple(cfg["img_size"]): _local_logger_load.info(f"SSL and Finetune img_size match: {ssl_img_size_tuple}. No PE interpolation if patch counts same.")
                else: _local_logger_load.info(f"SSL img_size: {ssl_img_size_tuple}, Finetune img_size: {cfg['img_size']}. PE interpolation may occur.")

                loaded_count = 0; pe_interp_count = 0; head_skip_count = 0
                for k_ckpt, v_ckpt in ssl_backbone_sd.items():
                    if k_ckpt not in current_model_sd: continue
                    if k_ckpt.startswith("classifier_head."): head_skip_count += 1; continue
                    is_pe = k_ckpt in ["rgb_pos_embed", "spectral_pos_embed"]
                    target_pe_p = getattr(hvt_model_instance, k_ckpt, None) if is_pe else None

                    if is_pe and target_pe_p is not None and v_ckpt.shape != target_pe_p.shape:
                        interp_pe_val = _interpolate_positional_embedding(v_ckpt, target_pe_p, cfg['hvt_params_for_model_init']['patch_size'], tuple(cfg["img_size"]), ssl_img_size_tuple)
                        if interp_pe_val.shape == target_pe_p.shape: new_sd_for_backbone[k_ckpt] = interp_pe_val; pe_interp_count += 1
                        else: _local_logger_load.warning(f"Skipping PE {k_ckpt} due to interp fail/shape mismatch.")
                    elif v_ckpt.shape == current_model_sd[k_ckpt].shape: new_sd_for_backbone[k_ckpt] = v_ckpt; loaded_count += 1
                    else: _local_logger_load.warning(f"Shape mismatch for {k_ckpt}: Ckpt {v_ckpt.shape} vs Model {current_model_sd[k_ckpt].shape}. Skipping.")

                msg = hvt_model_instance.load_state_dict(new_sd_for_backbone, strict=False)
                _local_logger_load.info(f"SSL Backbone weights loaded: {loaded_count} direct, {pe_interp_count} PE interp, {head_skip_count} head skipped.")
                if msg.missing_keys: _local_logger_load.warning(f"Missing keys in backbone load: {msg.missing_keys}")
                if msg.unexpected_keys: _local_logger_load.warning(f"Unexpected keys in backbone load: {msg.unexpected_keys}")

                if hasattr(hvt_model_instance, 'classifier_head') and isinstance(hvt_model_instance.classifier_head, nn.Linear):
                    in_feats = hvt_model_instance.classifier_head.in_features
                    hvt_model_instance.classifier_head = nn.Linear(in_feats, cfg["num_classes"])
                    _local_logger_load.info(f"Re-initialized HVT classifier_head for {cfg['num_classes']} classes (in_features={in_feats}).")
                else: _local_logger_load.error("Could not find 'classifier_head.in_features' to re-init. THIS IS LIKELY AN ERROR.")
            else: _local_logger_load.error(f"'model_backbone_state_dict' not in ckpt {resolved_ssl_ckpt_path}. Backbone random.")
        elif ssl_ckpt_path_from_cfg: _local_logger_load.warning(f"SSL ckpt path '{resolved_ssl_ckpt_path}' not found. Backbone random.")
        else: _local_logger_load.info("No SSL checkpoint path provided. Backbone random.")
    else: _local_logger_load.info("load_pretrained_backbone is False. Backbone random.")
    return hvt_model_instance.to(device)


def main_execution_logic():
    global logger
    args = parse_arguments()
    cfg = load_config_from_yaml_or_default(args.config)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    abs_pkg_root = cfg.get("PACKAGE_ROOT_PATH", os.path.dirname(os.path.abspath(__file__)))
    abs_log_dir = os.path.join(abs_pkg_root, cfg.get("log_dir", "logs_finetune_phase4"))
    log_file_base = os.path.splitext(cfg.get("log_file_finetune", "finetune_run.log"))[0]
    final_log_filename = f"{log_file_base}_{run_ts}.log"

    root_logger_obj = logging.getLogger()
    if root_logger_obj.hasHandlers():
        for handler in root_logger_obj.handlers[:]: root_logger_obj.removeHandler(handler); handler.close()
    setup_logging(log_file_name=final_log_filename, log_dir=abs_log_dir, log_level=logging.DEBUG, logger_name=None, run_timestamp=run_ts)
    logger = logging.getLogger(__name__)

    logger.info(f"======== Starting Phase 4: HVT Fine-tuning (Run ID: {run_ts}) ========")
    logger.info(f"Full run configuration: {cfg}")
    set_global_seed(cfg["seed"])
    device = cfg["device"]
    if device == 'cuda' and not torch.cuda.is_available(): device = 'cpu'; cfg['device'] = 'cpu'; logger.warning("CUDA specified but unavailable. Using CPU.")
    logger.info(f"Using device: {device}")
    if device == 'cuda': logger.info(f"GPU: {torch.cuda.get_device_name(0)}; CUDA Ver: {torch.version.cuda}")
    if cfg.get("cudnn_benchmark", True) and device == 'cuda': torch.backends.cudnn.benchmark = True
    if cfg.get("matmul_precision") and hasattr(torch,'set_float32_matmul_precision'):
        try: torch.set_float32_matmul_precision(cfg["matmul_precision"])
        except Exception as e: logger.warning(f"Failed to set matmul_precision: {e}")

    dataset_args = {"root_dir": cfg["data_root"], "img_size": tuple(cfg["img_size"]), "train_split_ratio": cfg["train_split_ratio"], "normalize_for_model": cfg["normalize_data"], "original_dataset_name": cfg["original_dataset_name"], "augmented_dataset_name": cfg.get("augmented_dataset_name", None), "random_seed": cfg["seed"]}
    train_dataset = SARCLD2024Dataset(**dataset_args, split="train"); val_dataset = SARCLD2024Dataset(**dataset_args, split="val")
    class_names = train_dataset.get_class_names()
    sampler = None
    if cfg.get("use_weighted_sampler", False):
        class_weights = train_dataset.get_class_weights();
        if class_weights is not None and hasattr(train_dataset, 'current_split_labels') and len(train_dataset.current_split_labels)>0:
            sample_weights = torch.zeros(len(train_dataset.current_split_labels));
            for i, label_idx in enumerate(train_dataset.current_split_labels): sample_weights[i]=class_weights[label_idx]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),replacement=True); logger.info("Using WeightedRandomSampler.")
    loader_args = {"num_workers": cfg.get('num_workers',4), "pin_memory": (device=='cuda'), "persistent_workers": (device=='cuda' and cfg.get('num_workers',4)>0)}
    if cfg.get('num_workers',4) > 0 and cfg.get('prefetch_factor') is not None : loader_args["prefetch_factor"] = cfg['prefetch_factor']
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], sampler=sampler, shuffle=(sampler is None), drop_last=True, **loader_args)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, drop_last=False, **loader_args)
    logger.info(f"Dataloaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    hvt_model_instance = create_disease_aware_hvt(current_img_size=tuple(cfg["img_size"]), num_classes=cfg["num_classes"], model_params_dict=cfg['hvt_params_for_model_init'])
    model = load_and_prepare_hvt_model(hvt_model_instance, cfg, device)
    total_params = sum(p.numel() for p in model.parameters()); trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model ready. Total params: {total_params:,}, Trainable params: {trainable_params:,}")

    if cfg.get("enable_torch_compile", False) and hasattr(torch, 'compile'):
        compile_mode = cfg.get("torch_compile_mode", "reduce-overhead"); logger.info(f"Attempting torch.compile (mode='{compile_mode}')...")
        try: model = torch.compile(model, mode=compile_mode); logger.info("Model compiled.")
        except Exception as e: logger.warning(f"torch.compile() failed: {e}. No compilation.", exc_info=True)
        
    augmentations_pipeline = FinetuneAugmentation(tuple(cfg["img_size"])) if cfg.get("augmentations_enabled", True) else None
    loss_cls_weights = train_dataset.get_class_weights().to(device) if cfg.get("use_weighted_loss", False) and not cfg.get("use_weighted_sampler", False) and train_dataset.get_class_weights() is not None else None
    criterion = nn.CrossEntropyLoss(weight=loss_cls_weights, label_smoothing=cfg.get("loss_label_smoothing", 0.0))

    # --- Optimizer Setup ---
    freeze_backbone_epochs_cfg = cfg.get("freeze_backbone_epochs", 0)
    head_module_name = "classifier_head"
    
    base_lr_cfg = cfg["learning_rate"] # This is the base LR, used for backbone if/when unfrozen
    head_lr_cfg = base_lr_cfg * cfg.get("head_lr_multiplier", 1.0)
    backbone_unfreeze_lr = base_lr_cfg * cfg.get("unfreeze_backbone_lr_factor", 1.0) # LR for backbone after unfreezing
    wd_cfg = cfg.get("weight_decay", 0.01)
    optim_common_kwargs = cfg.get("optimizer_params", {})

    param_groups = []
    # Always add head parameters, they are trainable from the start
    head_params = [p for n, p in model.named_parameters() if n.startswith(head_module_name + ".")]
    if head_params:
        for p in head_params: p.requires_grad = True # Ensure head is trainable
        param_groups.append({'params': head_params, 'lr': head_lr_cfg, 'name': 'head', 'weight_decay': wd_cfg})
    else:
        logger.warning("No parameters found for the head module. Check model structure and head_module_name.")

    # Handle backbone parameters based on freezing strategy
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith(head_module_name + ".")]
    if backbone_params:
        initial_backbone_lr = 0.0 if freeze_backbone_epochs_cfg > 0 else backbone_unfreeze_lr # Start at 0 if frozen
        for p in backbone_params:
            p.requires_grad = (freeze_backbone_epochs_cfg == 0) # Only requires_grad if not freezing
        param_groups.append({'params': backbone_params, 'lr': initial_backbone_lr, 'name': 'backbone', 'weight_decay': wd_cfg})
    
    if not param_groups or all(len(pg['params'])==0 for pg in param_groups) : logger.critical("No parameters for optimizer after group setup!"); sys.exit(1)
    
    optim_name_cfg = cfg.get("optimizer", "AdamW").lower()
    if optim_name_cfg == "adamw": optimizer = torch.optim.AdamW(param_groups, **optim_common_kwargs) # LR taken from groups
    elif optim_name_cfg == "sgd": optim_common_kwargs.setdefault('momentum', 0.9); optimizer = torch.optim.SGD(param_groups, **optim_common_kwargs)
    else: logger.warning(f"Unsupported optimizer. Default AdamW."); optimizer = torch.optim.AdamW(param_groups, **optim_common_kwargs)
    logger.info(f"Optimizer: {optimizer.__class__.__name__}. Initial Group LRs: {[pg['lr'] for pg in optimizer.param_groups]}")

    # --- Scheduler Setup ---
    scheduler = None; lr_scheduler_on_batch_flag = False
    sched_cfg_name = cfg.get("scheduler", "None").lower(); warmup_ep_cfg = cfg.get("warmup_epochs", 0)
    total_epochs_for_sched = cfg["epochs"]; eff_accum_steps_sched = max(1, cfg.get("accumulation_steps", 1))
    steps_per_epoch_for_sched = len(train_loader) // eff_accum_steps_sched
    if steps_per_epoch_for_sched == 0 and len(train_loader.dataset) > 0: steps_per_epoch_for_sched = 1
    
    if sched_cfg_name != "none" and steps_per_epoch_for_sched > 0 :
        if sched_cfg_name == "warmupcosine":
            total_sched_steps = total_epochs_for_sched * steps_per_epoch_for_sched; warmup_sched_steps = warmup_ep_cfg * steps_per_epoch_for_sched
            if warmup_sched_steps >= total_sched_steps and total_sched_steps > 0: warmup_sched_steps = max(1, int(0.1 * total_sched_steps))
            if total_sched_steps > 0 : scheduler = get_cosine_schedule_with_warmup_step(optimizer, warmup_sched_steps, total_sched_steps, num_cycles=0.5); lr_scheduler_on_batch_flag = True
            if scheduler: logger.info(f"Scheduler: WarmupCosine (per-step). WU Steps: {warmup_sched_steps}, Total Steps: {total_sched_steps}")
        elif sched_cfg_name == "cosineannealinglr":
            epochs_cos = total_epochs_for_sched - (warmup_ep_cfg if cfg.get("scheduler_warmup_type","sequential")=="sequential" and warmup_ep_cfg > 0 else 0)
            main_cos_sched = CosineAnnealingLR(optimizer, T_max=max(1,epochs_cos), eta_min=cfg.get("eta_min_lr",1e-7))
            if warmup_ep_cfg > 0 and cfg.get("scheduler_warmup_type", "sequential") == "sequential":
                warmup_sched = LinearLR(optimizer, start_factor=cfg.get("warmup_lr_init_factor",0.01), total_iters=warmup_ep_cfg)
                scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, main_cos_sched], milestones=[warmup_ep_cfg])
                logger.info(f"Scheduler: Sequential (Linear WU {warmup_ep_cfg}E then Cosine {max(1,epochs_cos)}E).")
            else: scheduler = main_cos_sched; logger.info(f"Scheduler: CosineAnnealingLR (per-epoch). T_max={max(1,epochs_cos)}E.")
    
    scaler_obj = GradScaler(enabled=(cfg.get("amp_enabled", True) and device == 'cuda'))
    finetuner_obj = Finetuner(model=model, optimizer=optimizer, criterion=criterion, device=device,
                              scaler=scaler_obj, scheduler=scheduler, lr_scheduler_on_batch=lr_scheduler_on_batch_flag,
                              accumulation_steps=eff_accum_steps_sched, clip_grad_norm=cfg.get("clip_grad_norm"),
                              augmentations=augmentations_pipeline, num_classes=cfg["num_classes"])

    # --- Training Loop ---
    best_val_metric_val = 0.0 if cfg.get("metric_to_monitor_early_stopping", "f1_macro") != "val_loss" else float('inf')
    metric_to_watch = cfg.get("metric_to_monitor_early_stopping", "f1_macro")
    patience_val = cfg.get("early_stopping_patience", float('inf'))
    patience_count = 0; last_completed_ep = 0
    abs_ckpt_save_dir = os.path.join(abs_log_dir, cfg.get("checkpoint_save_dir_name", "checkpoints"))
    os.makedirs(abs_ckpt_save_dir, exist_ok=True)

    logger.info(f"Starting fine-tuning: {cfg['epochs']} epochs. Monitor: '{metric_to_watch}', Patience: {patience_val}.")
    try:
        for epoch_1_based in range(1, cfg["epochs"] + 1):
            last_completed_ep = epoch_1_based -1
            
            if freeze_backbone_epochs_cfg > 0 and epoch_1_based == freeze_backbone_epochs_cfg + 1:
                logger.info(f"Epoch {epoch_1_based}: Unfreezing HVT backbone layers.")
                model_to_thaw = model._orig_mod if hasattr(model, '_orig_mod') and isinstance(model._orig_mod, nn.Module) else model
                
                unfrozen_this_step_count = 0
                for name, param in model_to_thaw.named_parameters():
                    if not name.startswith(head_module_name + "."): # Unfreeze backbone parameters
                        if not param.requires_grad:
                            param.requires_grad = True
                            unfrozen_this_step_count += 1
                
                # Update LR for the backbone group in the existing optimizer
                updated_lr_for_backbone = False
                for pg in optimizer.param_groups:
                    if pg.get('name') == 'backbone':
                        pg['lr'] = base_lr * cfg.get("unfreeze_backbone_lr_factor", 1.0)
                        logger.info(f"Optimizer group 'backbone' LR set to {pg['lr']:.2e}.")
                        updated_lr_for_backbone = True
                        break
                if not updated_lr_for_backbone and backbone_params_list : # Should not happen if backbone group was added with LR 0
                     logger.error("Could not find 'backbone' param group to update LR after unfreezing. This is an issue.")

                current_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"{unfrozen_this_step_count} backbone params newly set to requires_grad=True. Total trainable params now: {current_trainable_params:,}")


            finetuner_obj.train_one_epoch(train_loader, epoch_1_based, cfg["epochs"])
            
            current_val_metric_for_decision = None
            if epoch_1_based % cfg.get("evaluate_every_n_epochs", 1) == 0 or epoch_1_based == cfg["epochs"]:
                avg_val_loss, val_metrics_dict = finetuner_obj.validate_one_epoch(val_loader, class_names=class_names)
                current_val_metric_for_decision = val_metrics_dict.get(metric_to_watch)

                if isinstance(scheduler, ReduceLROnPlateau) and current_val_metric_for_decision is not None: scheduler.step(current_val_metric_for_decision)
                elif scheduler and not finetuner_obj.lr_scheduler_on_batch: scheduler.step()

                if current_val_metric_for_decision is not None:
                    is_better = (current_val_metric_for_decision > best_val_metric_val) if metric_to_watch != "val_loss" else (current_val_metric_for_decision < best_val_metric_val)
                    if is_better:
                        best_val_metric_val = current_val_metric_for_decision
                        if cfg.get("best_model_filename"): finetuner_obj.save_model_checkpoint(os.path.join(abs_ckpt_save_dir, cfg["best_model_filename"]))
                        logger.info(f"E{epoch_1_based}: New best! Val {metric_to_watch}: {best_val_metric_val:.4f}")
                        patience_count = 0
                    else: patience_count += 1; logger.info(f"E{epoch_1_based}: Val {metric_to_watch} ({current_val_metric_for_decision:.4f}) not better than {best_val_metric_val:.4f}. Patience: {patience_count}/{patience_val}")
            if patience_count >= patience_val: logger.info(f"Early stopping at E{epoch_1_based}."); break
            last_completed_ep = epoch_1_based
            if device == 'cuda': logger.debug(f"CUDA Mem E{epoch_1_based} End: Alloc {torch.cuda.memory_allocated(0)/1024**2:.1f}MB")

    except KeyboardInterrupt: logger.warning(f"Fine-tuning interrupted. Last completed epoch: {last_completed_ep}.")
    except Exception as e_fatal: logger.critical(f"Finetuning error at E{last_completed_ep + 1}: {e_fatal}", exc_info=True); sys.exit(1)
    finally:
        logger.info(f"Fine-tuning ended. Last completed epoch: {last_completed_ep}.")
        if 'finetuner_obj' in locals() and finetuner_obj is not None and cfg.get("final_model_filename"):
            finetuner_obj.save_model_checkpoint(os.path.join(abs_ckpt_save_dir, cfg["final_model_filename"]))
        logger.info(f"Fine-tuning summary: Best validation '{metric_to_watch}': {best_val_metric_val:.4f}")

if __name__ == "__main__":
    try:
        main_execution_logic()
    except SystemExit as se:
        final_logger_exit = logging.getLogger(__name__) if logger and logger.hasHandlers() else logging.getLogger("main_fallback_exit")
        if not final_logger_exit.hasHandlers() and not (final_logger_exit.parent and final_logger_exit.parent.hasHandlers()): logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        if se.code != 0 and se.code is not None: final_logger_exit.info(f"Application exited with status code {se.code}.")
        sys.exit(se.code if se.code is not None else 0)
    except Exception as e_unhandled:
        final_logger_unhandled = logging.getLogger(__name__) if logger and logger.hasHandlers() else logging.getLogger("main_fallback_unhandled")
        if not final_logger_unhandled.hasHandlers() and not (final_logger_unhandled.parent and final_logger_unhandled.parent.hasHandlers()): logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
        final_logger_unhandled.critical(f"Unhandled CRITICAL exception in __main__ execution: {e_unhandled}", exc_info=True)
        sys.exit(1)