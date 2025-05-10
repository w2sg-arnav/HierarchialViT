# phase5_multimodal_hpo/main.py
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, LinearLR, SequentialLR
import argparse
import yaml
import torch.nn.functional as F 
import math 
from collections import OrderedDict 
import time 

# --- Path Setup ---
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
if project_root not in sys.path: sys.path.insert(0, project_root); print(f"DEBUG: Added project root: {project_root}")

# --- Project Imports ---
from phase5_multimodal_hpo.config import config as default_config 
from phase5_multimodal_hpo.dataset import SARCLD2024Dataset   
from phase5_multimodal_hpo.utils.logging_setup import setup_logging
from phase5_multimodal_hpo.finetune.trainer import Finetuner 
from phase2_model.models.hvt import DiseaseAwareHVT 

# --- Helper Functions --- 
def load_config_yaml(config_path=None): 
    # (Keep unchanged)
    config = default_config.copy(); 
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f: yaml_config = yaml.safe_load(f)
            if yaml_config: config.update(yaml_config)
            print(f"Loaded configuration from YAML: {config_path}")
        except Exception as e: print(f"Warning: Could not load/parse YAML {config_path}. Error: {e}. Using defaults.")
    else: print("No config file path provided or file not found. Using default/base config.")
    return config

def parse_args(): 
    # (Keep unchanged)
    parser = argparse.ArgumentParser(description="Fine-tuning script (Phase 5)"); # ... (rest as before) ...
    return parser.parse_args()

def set_seed(seed): 
    # (Keep unchanged)
    torch.manual_seed(seed); np.random.seed(seed); 
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _interpolate_pos_embed(chkpt_embed: torch.Tensor, model_embed: nn.Parameter, patch_size: int, target_img_size_from_config: tuple) -> torch.Tensor: 
    # (Keep unchanged)
    logger = logging.getLogger(__name__) # ... (rest as before) ...
    return pos_embed_interp

def load_pretrained_backbone(model: nn.Module, checkpoint_path: str, config: dict):
    """ Loads weights, handles pos embed interpolation and head mismatch. """
    logger = logging.getLogger(__name__)
    if not os.path.isabs(checkpoint_path):
        # Resolve path relative to project root more reliably
        base_path = project_root if project_root else os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.abspath(os.path.join(base_path, checkpoint_path))
        logger.info(f"Resolved relative checkpoint path to: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        # --- MODIFIED: Raise error if load_pretrained is True ---
        if config.get("load_pretrained", False): # Check the flag from config
             logger.error(f"CRITICAL: Pretrained checkpoint NOT FOUND at {checkpoint_path} but load_pretrained is True.")
             raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
        else:
             logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}. Training from scratch as configured.")
             return model
        # -------------------------------------------------------

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu'); logger.info(f"Loaded checkpoint: {checkpoint_path}")
        current_model_dict = model.state_dict(); new_state_dict = OrderedDict() 
        
        for k, v in checkpoint.items():
            if k not in current_model_dict: continue
            if k in ["rgb_pos_embed", "spectral_pos_embed"]:
                model_param = current_model_dict.get(k)
                if model_param is not None and v.shape != model_param.shape:
                    interpolated_embed = _interpolate_pos_embed(v, model_param, config['hvt_patch_size'], config['img_size'])
                    if interpolated_embed.shape == model_param.shape: new_state_dict[k] = interpolated_embed
                    else: logger.warning(f"Interpolation failed for {k}. Skipping.")
                elif model_param is not None: new_state_dict[k] = v 
                continue 
            if k.startswith("head."): continue
            model_param = current_model_dict.get(k)
            if model_param is not None and v.shape == model_param.shape: new_state_dict[k] = v
            else: model_shape_str = model_param.shape if model_param is not None else 'MISSING'; logger.warning(f"Skipping '{k}': shape mismatch ckpt {v.shape} vs model {model_shape_str}")
        
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False); logger.info("Attempted loading weights.")
        truly_missing = [k for k in missing_keys if not k.startswith(('head.', 'head_norm.'))]; expected_missing = [k for k in missing_keys if k.startswith(('head.', 'head_norm.'))]
        if expected_missing: logger.info(f"  Missing head keys (expected): {expected_missing}")
        if truly_missing: logger.warning(f"  Weights MISSING for other keys: {truly_missing}")
        if unexpected_keys: logger.error(f"  Error: UNEXPECTED keys in state_dict: {unexpected_keys}")
        
        logger.info("Successfully processed pre-trained backbone weights loading.")
        return model
    except Exception as e: 
        logger.error(f"Unexpected error loading checkpoint {checkpoint_path}: {e}", exc_info=True)
        if config.get("load_pretrained", False):
             logger.error("CRITICAL: Failed to load mandatory pre-trained weights. Exiting.")
             raise e # Re-raise the error if loading was mandatory
        else:
             logger.warning("Could not load pre-trained weights. Training from scratch.")
             return model


def get_layer_wise_params(model: nn.Module, learning_rate: float, layer_decay_rate: float, weight_decay: float):
    # (Keep LLRD logic unchanged)
    param_groups = []; no_decay_list = ['bias', 'Norm.bias', 'Norm.weight']; assigned_params = set()
    logger = logging.getLogger(__name__)
    try:
        num_block_layers = 0; module_iterator = []
        if hasattr(model, 'rgb_stages') and model.rgb_stages is not None: module_iterator = model.rgb_stages
        elif hasattr(model, 'stages'): module_iterator = model.stages
        elif hasattr(model, 'blocks'): module_iterator = model.blocks; num_block_layers = len(module_iterator); logger.warning("LLRD: Using flat 'blocks'.")
        if module_iterator and not num_block_layers: num_block_layers = sum(len(stage.blocks) for stage in module_iterator if hasattr(stage, 'blocks')) 
        if num_block_layers == 0: raise AttributeError("Could not find compatible block structure for LLRD.")
        logger.info(f"Applying LLRD with rate {layer_decay_rate} over {num_block_layers} block layers.")
        base_lr = learning_rate; base_wd = weight_decay; lr_scale_embed = layer_decay_rate ** (num_block_layers + 1)
        group_emb = {'params': [], 'weight_decay': base_wd, 'lr': base_lr * lr_scale_embed, 'layer': 'embed'}
        group_emb_no_decay = {'params': [], 'weight_decay': 0.0, 'lr': base_lr * lr_scale_embed, 'layer': 'embed_nd'}
        for name, param in model.named_parameters():
             if name.startswith(('patch_embed', 'pos_embed', 'cls_token')): target_group = group_emb_no_decay if any(nd in name for nd in no_decay_list) else group_emb; target_group['params'].append(param); assigned_params.add(param)
        if group_emb['params']: param_groups.append(group_emb); 
        if group_emb_no_decay['params']: param_groups.append(group_emb_no_decay)
        current_abs_depth = 0
        for i_stage, stage in enumerate(module_iterator):
             if hasattr(stage, 'blocks'): 
                 num_blocks_in_stage = len(stage.blocks)
                 for i_block in range(num_blocks_in_stage):
                      layer_depth = current_abs_depth + i_block; lr_scale = layer_decay_rate ** (num_block_layers - layer_depth)
                      group_decay = {'params': [], 'weight_decay': base_wd, 'lr': base_lr * lr_scale, 'layer': f's{i_stage}b{i_block}'}; group_no_decay = {'params': [], 'weight_decay': 0.0, 'lr': base_lr * lr_scale, 'layer': f's{i_stage}b{i_block}_nd'}
                      block_rgb = stage.blocks[i_block] 
                      for name, param in block_rgb.named_parameters():
                          if param not in assigned_params: target_group = group_no_decay if any(nd in name for nd in no_decay_list) else group_decay; target_group['params'].append(param); assigned_params.add(param)
                      if hasattr(model, 'spectral_stages') and model.spectral_stages is not None and i_stage < len(model.spectral_stages):
                           stage_spec = model.spectral_stages[i_stage]
                           if hasattr(stage_spec, 'blocks') and i_block < len(stage_spec.blocks):
                                block_spec = stage_spec.blocks[i_block]
                                for name, param in block_spec.named_parameters():
                                     if param not in assigned_params: target_group = group_no_decay if any(nd in name for nd in no_decay_list) else group_decay; target_group['params'].append(param); assigned_params.add(param)
                      if group_decay['params']: param_groups.append(group_decay); 
                      if group_no_decay['params']: param_groups.append(group_no_decay)
                 current_abs_depth += num_blocks_in_stage
                 if hasattr(stage, 'downsample_layer') and stage.downsample_layer is not None:
                      lr_scale = layer_decay_rate ** (num_block_layers - (current_abs_depth - 1 if current_abs_depth > 0 else 0) )
                      group_down_decay = {'params': [], 'weight_decay': base_wd, 'lr': base_lr * lr_scale, 'layer': f's{i_stage}ds'}; group_down_no_decay = {'params': [], 'weight_decay': 0.0, 'lr': base_lr * lr_scale, 'layer': f's{i_stage}ds_nd'}
                      for name, param in stage.downsample_layer.named_parameters():
                           if param not in assigned_params: target_group = group_down_no_decay if any(nd in name for nd in no_decay_list) else group_down_decay; target_group['params'].append(param); assigned_params.add(param)
                      if hasattr(model, 'spectral_stages') and model.spectral_stages is not None and i_stage < len(model.spectral_stages) and hasattr(model.spectral_stages[i_stage], 'downsample_layer') and model.spectral_stages[i_stage].downsample_layer is not None:
                          for name, param in model.spectral_stages[i_stage].downsample_layer.named_parameters():
                              if param not in assigned_params: target_group = group_down_no_decay if any(nd in name for nd in no_decay_list) else group_down_decay; target_group['params'].append(param); assigned_params.add(param)
                      if group_down_decay['params']: param_groups.append(group_down_decay)
                      if group_down_no_decay['params']: param_groups.append(group_down_no_decay)
             elif isinstance(stage, nn.Module): # Flat list of blocks
                  layer_depth = current_abs_depth; lr_scale = layer_decay_rate ** (num_block_layers - layer_depth)
                  group_decay = {'params': [], 'weight_decay': base_wd, 'lr': base_lr * lr_scale, 'layer': f'b{layer_depth}'}; group_no_decay = {'params': [], 'weight_decay': 0.0, 'lr': base_lr * lr_scale, 'layer': f'b{layer_depth}_nd'}
                  for name, param in stage.named_parameters():
                      if param not in assigned_params: target_group = group_no_decay if any(nd in name for nd in no_decay_list) else group_decay; target_group['params'].append(param); assigned_params.add(param)
                  if group_decay['params']: param_groups.append(group_decay); 
                  if group_no_decay['params']: param_groups.append(group_no_decay)
                  current_abs_depth += 1
        group_final_decay = {'params': [], 'weight_decay': base_wd, 'lr': base_lr, 'layer': 'head'}; group_final_no_decay = {'params': [], 'weight_decay': 0.0, 'lr': base_lr, 'layer': 'head_nd'}
        for name, param in model.named_parameters(): 
             if param.requires_grad and param not in assigned_params: target_group = group_final_no_decay if any(nd in name for nd in no_decay_list) else group_final_decay; target_group['params'].append(param); assigned_params.add(param)
        if group_final_decay['params']: param_groups.append(group_final_decay)
        if group_final_no_decay['params']: param_groups.append(group_final_no_decay)
        return param_groups
    except AttributeError as e:
         logger.error(f"LLRD Error: Model structure incompatible ({e}). Falling back.", exc_info=True); no_decay_list = ['bias', 'LayerNorm', 'norm']; decay = {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay_list)], 'weight_decay': weight_decay}; no_decay = {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay_list)], 'weight_decay': 0.0}; return [group for group in [decay, no_decay] if group['params']] 

# --- Main Training Function ---
def run_training_session(config: dict) -> float:
    """ Runs a full training and validation loop, returns best validation metric. """
    
    logger = logging.getLogger(__name__) 
    set_seed(config["seed"])
    device = config["device"]
    
    # --- Datasets & DataLoaders ---
    logger.info("Setting up datasets...")
    train_dataset = SARCLD2024Dataset(root_dir=config["data_root"], img_size=config["img_size"], split="train", train_split_ratio=config["train_split_ratio"], normalize_for_model=config["normalize_data"], use_spectral=(config["hvt_spectral_channels"] > 0), spectral_channels=config["hvt_spectral_channels"], random_seed=config["seed"])
    val_dataset = SARCLD2024Dataset(root_dir=config["data_root"], img_size=config["img_size"], split="val", train_split_ratio=config["train_split_ratio"], normalize_for_model=config["normalize_data"], use_spectral=(config["hvt_spectral_channels"] > 0), spectral_channels=config["hvt_spectral_channels"], random_seed=config["seed"])
    sampler = None; class_weights = None
    if config.get("use_weighted_sampler", False):
        class_weights = train_dataset.get_class_weights() 
        if class_weights is not None: logger.info("Using WeightedRandomSampler."); train_labels = train_dataset.current_split_labels; sample_weights = torch.zeros(len(train_labels)); 
        for i in range(config["num_classes"]):
            if i < len(class_weights): sample_weights[train_labels == i] = class_weights[i]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        else: logger.warning("Disabling WeightedRandomSampler.")
    num_workers = config.get("num_workers", 4)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler, shuffle=(sampler is None), num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    class_names = train_dataset.get_class_names()
    logger.info(f"Loaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}. Workers: {num_workers}")

    # --- Model ---
    logger.info(f"Initializing model: {config['model_name']}")
    model = DiseaseAwareHVT(img_size=config["img_size"], patch_size=config["hvt_patch_size"], num_classes=config["num_classes"], embed_dim_rgb=config["hvt_embed_dim_rgb"], embed_dim_spectral=config["hvt_embed_dim_spectral"], spectral_channels=config["hvt_spectral_channels"], depths=config["hvt_depths"], num_heads=config["hvt_num_heads"], mlp_ratio=config["hvt_mlp_ratio"], qkv_bias=config["hvt_qkv_bias"], drop_rate=config["hvt_model_drop_rate"], attn_drop_rate=config["hvt_attn_drop_rate"], drop_path_rate=config["hvt_drop_path_rate"], use_dfca=config["hvt_use_dfca"],)
    
    # Load weights BEFORE moving model to device for potentially faster loading
    if config["load_pretrained"]: 
        model = load_pretrained_backbone(model, config["pretrained_checkpoint_path"], config)
    
    model = model.to(device) # Move model to device

    # --- Freeze Backbone Logic --- NO LONGER FREEZING in this config
    freeze_epochs = config.get("freeze_backbone_epochs", 0)
    if freeze_epochs > 0:
         logger.error("Freeze backbone epochs > 0, but freezing logic is disabled in this script version. Set freeze_backbone_epochs: 0 in config or re-enable logic.")
         # Ensure all params are trainable if freeze is disabled
         for param in model.parameters(): param.requires_grad = True 
    
    # --- Trainer Components ---
    loss_weights = class_weights.to(device) if class_weights is not None and not config.get("use_weighted_sampler", False) else None
    if loss_weights is not None: logger.info("Using weighted CrossEntropyLoss.")
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=config["loss_label_smoothing"])
    
    # --- Optimizer Setup ---
    initial_lr = config["learning_rate"] # Start with the main LR
    current_wd = config["weight_decay"] 
    
    if config.get("use_llrd", False) and 'llrd_rate' in config: 
        optimizer_grouped_parameters = get_layer_wise_params(model, initial_lr, config["llrd_rate"], current_wd)
    else: 
         logger.info("Using standard optimizer groups.")
         no_decay_list = ['bias', 'LayerNorm', 'norm']; decay_params = []; no_decay_params = []
         for n, p in model.named_parameters():
             if p.requires_grad:
                 if any(nd in n for nd in no_decay_list): no_decay_params.append(p)
                 else: decay_params.append(p)
         optimizer_grouped_parameters = [{'params': decay_params, 'weight_decay': current_wd}, {'params': no_decay_params, 'weight_decay': 0.0}]
         for group in optimizer_grouped_parameters: group['lr'] = initial_lr 

    if config["optimizer"].lower() == "adamw": optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=initial_lr, **config.get("optimizer_params", {}))
    else: optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=initial_lr, **config.get("optimizer_params", {}))
        
    # --- Schedulers ---
    scheduler = None; lr_reducer = None
    if config.get("scheduler"):
        if config["warmup_epochs"] > 0: warmup = LinearLR(optimizer, start_factor=config.get("warmup_lr_init_factor", 0.1), total_iters=config["warmup_epochs"])
        else: warmup = None
        sched_type = config["scheduler"].lower(); main_sched = None
        if sched_type == "cosineannealingwarmrestarts": main_sched = CosineAnnealingWarmRestarts(optimizer, T_0=config["cosine_t_0"], T_mult=config["cosine_t_mult"], eta_min=config["eta_min"]); logger.info("Using CosineAnnealingWarmRestarts.")
        elif sched_type == "reducelronplateau": lr_reducer = ReduceLROnPlateau(optimizer, mode='max', factor=config["reducelr_factor"], patience=config["reducelr_patience"], verbose=False); logger.info("Using ReduceLROnPlateau.")
        else: logger.warning(f"Unsupported scheduler: {config['scheduler']}.")
        schedulers_to_combine = [s for s in [warmup, main_sched] if s is not None]
        if len(schedulers_to_combine) > 1: scheduler = SequentialLR(optimizer, schedulers=schedulers_to_combine, milestones=[config["warmup_epochs"]]); logger.info("Combined Warmup+Scheduler.")
        elif len(schedulers_to_combine) == 1: scheduler = schedulers_to_combine[0]
        elif lr_reducer is None: logger.info("No valid scheduler configured.")

    scaler = GradScaler(enabled=config["amp_enabled"])
    
    # --- Initialize Trainer ---
    trainer = Finetuner(
        model=model, optimizer=optimizer, criterion=criterion, device=device, scaler=scaler,
        scheduler=scheduler, accumulation_steps=config["accumulation_steps"],
        clip_grad_norm=config["clip_grad_norm"], 
        num_classes=config["num_classes"],
        mixup_alpha=config.get("mixup_alpha", 0.0), # Using 0.0 from config now
        cutmix_alpha=config.get("cutmix_alpha", 0.0), # Using 0.0 from config now
        mixup_cutmix_prob=config.get("mixup_cutmix_prob", 0.0), # Using 0.0 from config now
        img_size=config["img_size"], 
        tta_enabled=config.get("tta_enabled", False),
        tta_augmentations=config.get("tta_augmentations", ['hflip']),
        augmentations_enabled=config["augmentations_enabled"]
    )

    # --- Training Loop ---
    best_val_metric = -1.0 
    metric_to_monitor = config.get("metric_to_monitor", 'f1_weighted') 
    patience_counter = 0
    early_stopping_patience = config.get("early_stopping_patience", float('inf'))
    logger.info(f"Starting training loop for {config['epochs']} epochs... Monitoring: {metric_to_monitor}") 
    
    os.makedirs(os.path.dirname(config["best_model_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["final_model_path"]), exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        start_time = time.time(); avg_train_loss = trainer.train_one_epoch(train_loader, epoch, config["epochs"]); epoch_train_time = time.time() - start_time
        start_time = time.time(); avg_val_loss, val_metrics = trainer.validate_one_epoch(val_loader, class_names=class_names); epoch_val_time = time.time() - start_time
        current_val_metric = val_metrics.get(metric_to_monitor, 0.0)
        
        if lr_reducer: lr_reducer.step(current_val_metric) 
        
        logger.info(f"Epoch {epoch}/{config['epochs']} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_metrics.get('accuracy', 0.0):.4f} | Val F1w: {val_metrics.get('f1_weighted', 0.0):.4f} | Train Time: {epoch_train_time:.2f}s | Val Time: {epoch_val_time:.2f}s")
        
        if current_val_metric > best_val_metric: 
            best_val_metric = current_val_metric; trainer.save_model_checkpoint(config["best_model_path"], epoch=epoch, best_metric=best_val_metric); logger.info(f"Epoch {epoch}: New best val {metric_to_monitor}: {best_val_metric:.4f}"); patience_counter = 0
        else: 
            patience_counter += 1; logger.info(f"Epoch {epoch}: Val {metric_to_monitor} ({current_val_metric:.4f}) no improve. Patience: {patience_counter}/{early_stopping_patience}")
        
        if patience_counter >= early_stopping_patience: logger.info(f"Early stopping triggered."); break
            
        # Step schedulers (excluding ReduceLROnPlateau)
        if trainer.scheduler and not lr_reducer: 
             trainer.scheduler.step()

    logger.info(f"Training finished. Best validation {metric_to_monitor}: {best_val_metric:.4f}")
    trainer.save_model_checkpoint(config["final_model_path"], epoch=config['epochs'], best_metric=best_val_metric) # Save final model state
    return best_val_metric 

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    config = load_config_yaml(args.config)
    log_file = config.get("log_file_finetune", "finetune_adv.log"); log_dir = config.get("log_dir", "logs") 
    setup_logging(log_file_name=log_file, log_dir=log_dir, log_level=logging.INFO, logger_name=None) 
    logger = logging.getLogger(__name__)
    
    if config.get("hpo_enabled", False): logger.warning("HPO enabled. Run hpo.py for optimization.")
    else:
        try:
            # Override config with command-line args
            override_keys = ['learning_rate', 'weight_decay', 'loss_label_smoothing', 'epochs', 'batch_size', 'llrd_rate', 'mixup_alpha', 'cutmix_alpha']
            cli_args = vars(args); HPO_PARAMS_FOUND = False
            for key in override_keys:
                 config_key_map = {'llrd': 'llrd_rate', 'mixup': 'mixup_alpha', 'cutmix': 'cutmix_alpha'}
                 config_key = config_key_map.get(key, key) 
                 cli_value = cli_args.get(key) 
                 if cli_value is not None: config[config_key] = cli_value; logger.info(f"Overriding {config_key}: {cli_value}"); HPO_PARAMS_FOUND=True
            if HPO_PARAMS_FOUND: logger.info("Overrode config with CLI parameters.")
            
            import time # Ensure time is imported
            run_training_session(config)
        except Exception as e: logger.exception(f"An critical error occurred: {e}"); sys.exit(1)