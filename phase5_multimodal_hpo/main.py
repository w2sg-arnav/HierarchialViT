from typing import Optional, List, Dict # Added Dict
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau,
    LinearLR, SequentialLR
)
import argparse
import yaml # PyYAML
import time
from collections import OrderedDict
import math
import sys
import random

# --- Path Setup ---
# Ensure the project root is in sys.path to allow imports like phase5_multimodal_hpo.config
current_dir = os.path.dirname(os.path.abspath(__file__)) # phase5_multimodal_hpo
project_root = os.path.dirname(current_dir) # Parent of phase5_multimodal_hpo
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"DEBUG: Added project root to sys.path: {project_root}")


# --- Project Imports ---
from phase5_multimodal_hpo.config import config as default_config
from phase5_multimodal_hpo.dataset import SARCLD2024Dataset
from phase5_multimodal_hpo.utils.logging_setup import setup_logging
from phase5_multimodal_hpo.finetune.trainer import Finetuner
from phase5_multimodal_hpo.models.hvt_xl import DiseaseAwareHVT_XL

# --- LAMB Optimizer Implementation (Simplified from timm) ---
class Lamb(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, adam=False): # Removed layer_decay from __init__ as it's handled in param groups
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay) # Removed layer_decay
        super(Lamb, self).__init__(params, defaults)
        self.adam = adam

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse: raise RuntimeError('Lamb does not support sparse gradients.')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Group's lr is already LLRD adjusted if applicable
                current_lr = group['lr'] 

                # AdamW style weight decay
                if group['weight_decay'] != 0:
                     p.data.add_(p.data, alpha=-group['weight_decay'] * current_lr)

                adam_step = exp_avg / (exp_avg_sq.sqrt().add_(group['eps']))
                weight_norm = torch.norm(p.data)
                adam_norm = torch.norm(adam_step)
                trust_ratio = 1.0 if weight_norm == 0 or adam_norm == 0 else weight_norm / adam_norm
                state['trust_ratio'] = trust_ratio
                
                update_direction = adam_step
                if not self.adam: update_direction = trust_ratio * adam_step
                p.data.add_(update_direction, alpha=-current_lr)
        return loss

# --- Helper Functions ---
def load_config_yaml(config_path=None, base_config_dict=None):
    config_data = base_config_dict.copy() if base_config_dict else {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f: yaml_config = yaml.safe_load(f)
            if yaml_config: config_data.update(yaml_config)
            print(f"Loaded and merged configuration from YAML: {config_path}")
        except Exception as e: print(f"Warn: Could not load/parse YAML {config_path}. Error: {e}. Using base.")
    else: print("No external config file or file not found. Using base config.")
    return config_data

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning script for DiseaseAwareHVT_XL (Phase 5)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--wd", type=float, default=None, help="Override weight_decay")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--img_size_h", type=int, default=None, help="Override image height")
    parser.add_argument("--img_size_w", type=int, default=None, help="Override image width")
    return parser.parse_args()

def set_seed(seed_val):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

def _interpolate_pos_embed_single(chkpt_embed, model_embed, embed_dim, patch_size, target_img_size_tuple, stream_name):
    logger_pe = logging.getLogger(f"{__name__}.pos_embed") # Specific logger
    N_src, C_src = chkpt_embed.shape[1], chkpt_embed.shape[2]
    N_tgt, C_tgt = model_embed.shape[1], model_embed.shape[2]

    if N_src == N_tgt and C_src == C_tgt:
        logger_pe.info(f"Pos embed for {stream_name} matches. No interpolation needed.")
        return chkpt_embed
    if C_src != C_tgt:
        logger_pe.warning(f"Pos embed dim mismatch for {stream_name}: ckpt {C_src} vs model {C_tgt}. Using model's default.")
        return model_embed.data

    logger_pe.info(f"Interpolating {stream_name} pos embed from {N_src} to {N_tgt} patches.")
    
    gs_old = 0
    if math.isqrt(N_src)**2 == N_src: gs_old = math.isqrt(N_src)
    else: # Heuristics for common ViT patch numbers
        if N_src == 197 and patch_size == 16: gs_old=14 # 224/16, +1 CLS (remove CLS for grid)
        elif N_src == 196 and patch_size == 16: gs_old=14
        elif N_src == (256//patch_size)**2 : gs_old = 256//patch_size
        elif N_src == (384//patch_size)**2 : gs_old = 384//patch_size
        elif N_src == (512//patch_size)**2 : gs_old = 512//patch_size
        else: logger_pe.error(f"Cannot infer source grid {N_src} for {stream_name}."); return model_embed.data
        # This HVT_XL doesn't have a CLS token in pos_embed, so N_src should be square.

    gs_new_h = target_img_size_tuple[0] // patch_size
    gs_new_w = target_img_size_tuple[1] // patch_size
    
    if N_tgt != gs_new_h * gs_new_w:
         logger_pe.error(f"Target N_tgt {N_tgt} != calc new grid {gs_new_h}x{gs_new_w} for {stream_name}."); return model_embed.data

    pos_embed_2d = chkpt_embed.permute(0, 2, 1).reshape(1, embed_dim, gs_old, gs_old)
    interpolated_embed_2d = F.interpolate(pos_embed_2d, size=(gs_new_h, gs_new_w), mode='bicubic', align_corners=False)
    interpolated_final = interpolated_embed_2d.reshape(1, embed_dim, N_tgt).permute(0, 2, 1)
    
    if interpolated_final.shape == model_embed.shape:
        logger_pe.info(f"Pos embed for {stream_name} interpolated to {interpolated_final.shape}.")
        return interpolated_final
    else:
        logger_pe.error(f"Interpolated shape {interpolated_final.shape} mismatch {stream_name} vs model {model_embed.shape}."); return model_embed.data

def load_pretrained_weights(model: nn.Module, checkpoint_path: str, config_dict: dict):
    logger_lpw = logging.getLogger(f"{__name__}.load_weights") # Specific logger
    if not os.path.exists(checkpoint_path):
        logger_lpw.warning(f"Pretrained ckpt not found: {checkpoint_path}. Training from scratch/as init.")
        return model
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
        logger_lpw.info(f"Loaded pretrained checkpoint: {checkpoint_path}")

        current_model_dict = model.state_dict()
        new_state_dict = OrderedDict()
        
        # Interpolate positional embeddings first if necessary
        chkpt_rgb_pe = state_dict.get('rgb_pos_embed')
        model_rgb_pe_param = getattr(model, 'rgb_pos_embed', None)
        
        chkpt_spec_pe = state_dict.get('spectral_pos_embed')
        model_spec_pe_param = getattr(model, 'spectral_pos_embed', None)

        interp_rgb_pe, interp_spec_pe = None, None

        if chkpt_rgb_pe is not None and model_rgb_pe_param is not None:
            interp_rgb_pe = _interpolate_pos_embed_single(
                chkpt_rgb_pe, model_rgb_pe_param, config_dict['hvt_embed_dim_rgb'],
                config_dict['hvt_patch_size'], tuple(config_dict['img_size']), "RGB"
            )
        if chkpt_spec_pe is not None and model_spec_pe_param is not None:
            interp_spec_pe = _interpolate_pos_embed_single(
                chkpt_spec_pe, model_spec_pe_param, config_dict['hvt_embed_dim_spectral'],
                config_dict['hvt_patch_size'], tuple(config_dict['img_size']), "Spectral"
            )

        for k_ckpt, v_ckpt in state_dict.items():
            k_model = k_ckpt.replace("module.", "")
            if k_model not in current_model_dict: continue
            if k_model == "rgb_pos_embed" and interp_rgb_pe is not None: new_state_dict[k_model] = interp_rgb_pe; continue
            if k_model == "spectral_pos_embed" and interp_spec_pe is not None: new_state_dict[k_model] = interp_spec_pe; continue
            if k_model.startswith("head.") or k_model.startswith("head_norm."): continue
            if v_ckpt.shape == current_model_dict[k_model].shape: new_state_dict[k_model] = v_ckpt
            else: logger_lpw.warning(f"Skipping '{k_model}': shape mismatch ckpt {v_ckpt.shape} vs model {current_model_dict[k_model].shape}")
        
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        logger_lpw.info("Pretrained weights loading summary:")
        if [k for k in missing_keys if not k.startswith(('head.','head_norm.'))]: logger_lpw.warning(f"  Weights truly MISSING: {[k for k in missing_keys if not k.startswith(('head.','head_norm.'))]}")
        if unexpected_keys: logger_lpw.error(f"  UNEXPECTED keys (not loaded): {unexpected_keys}")
        logger_lpw.info("Pretrained weights processed.")
        return model
    except Exception as e:
        logger_lpw.error(f"Error loading pretrained {checkpoint_path}: {e}", exc_info=True); return model

def get_parameter_groups(model: nn.Module, base_lr: float, weight_decay_val: float, # Renamed weight_decay
                         llrd_rate_val: Optional[float] = None, # Renamed llrd_rate
                         head_lr_factor_val: float = 1.0, # Renamed head_lr_factor
                         is_frozen_phase_val: bool = False): # Renamed is_frozen_phase
    logger_pg = logging.getLogger(f"{__name__}.param_groups") # Specific logger
    param_groups_list = [] # Renamed param_groups
    no_decay_keywords = ['bias', 'norm.weight', 'norm.bias', 'ln_'] # Renamed no_decay_list

    if is_frozen_phase_val:
        logger_pg.info(f"Frozen phase: Head LR factor {head_lr_factor_val}.")
        for name, param in model.named_parameters():
            if param.requires_grad:
                group_lr = base_lr * head_lr_factor_val
                group_wd = weight_decay_val if not any(nd in name for nd in no_decay_keywords) else 0.0
                param_groups_list.append({'params': [param], 'lr': group_lr, 'weight_decay': group_wd, 'name': f"frozen_head_{name}"})
        return param_groups_list

    if llrd_rate_val is not None and llrd_rate_val < 1.0:
        logger_pg.info(f"Applying LLRD with decay rate: {llrd_rate_val}")
        num_layers = sum(len(stage) for stage in model.rgb_stages) if hasattr(model, 'rgb_stages') else 12 # Approx depth
        assigned_params_set = set() # Renamed assigned_params

        # Head (highest LR, or factor * base_lr)
        head_lr = base_lr * head_lr_factor_val # head_lr_factor applies even with LLRD for emphasis
        for name_prefix in ['head.', 'head_norm.']:
            for name, param in model.named_parameters():
                if name.startswith(name_prefix) and param.requires_grad:
                    pg_wd = weight_decay_val if not any(nd in name for nd in no_decay_keywords) else 0.0
                    param_groups_list.append({'params': [param], 'lr': head_lr, 'weight_decay': pg_wd, 'name': name})
                    assigned_params_set.add(name)
        
        # Embeddings (lowest LR)
        embed_lr = base_lr * (llrd_rate_val ** num_layers)
        for name_prefix in ['rgb_patch_embed', 'spectral_patch_embed', 'rgb_pos_embed', 'spectral_pos_embed']:
            for name, param in model.named_parameters():
                if name.startswith(name_prefix) and name not in assigned_params_set and param.requires_grad:
                    pg_wd = weight_decay_val if not any(nd in name for nd in no_decay_keywords) else 0.0
                    param_groups_list.append({'params': [param], 'lr': embed_lr, 'weight_decay': pg_wd, 'name': name})
                    assigned_params_set.add(name)

        # Transformer Stages (intermediate LRs)
        # Iterate stages from earliest to latest; deeper means less LR decay from llrd_rate^0
        # Depth 0 (earliest blocks) get lr * llrd_rate^(num_layers-1)
        # Depth num_layers-1 (latest blocks) get lr * llrd_rate^0 = lr
        # This requires careful calculation of absolute depth of each block
        block_base_idx = 0
        if hasattr(model, 'rgb_stages') and hasattr(model, 'spectral_stages'):
            for i_stage in range(model.num_stages):
                for i_block in range(len(model.rgb_stages[i_stage])):
                    abs_depth = block_base_idx + i_block
                    lr_scale = llrd_rate_val ** (num_layers - 1 - abs_depth)
                    block_lr = base_lr * lr_scale
                    # RGB block
                    for name, param in model.rgb_stages[i_stage][i_block].named_parameters():
                        full_name = f"rgb_stages.{i_stage}.{i_block}.{name}"
                        if full_name not in assigned_params_set and param.requires_grad:
                            pg_wd = weight_decay_val if not any(nd in name for nd in no_decay_keywords) else 0.0
                            param_groups_list.append({'params': [param], 'lr': block_lr, 'weight_decay': pg_wd, 'name': full_name})
                            assigned_params_set.add(full_name)
                    # Spectral block
                    if i_block < len(model.spectral_stages[i_stage]): # Ensure spectral stage/block exists
                        for name, param in model.spectral_stages[i_stage][i_block].named_parameters():
                            full_name = f"spectral_stages.{i_stage}.{i_block}.{name}"
                            if full_name not in assigned_params_set and param.requires_grad:
                                pg_wd = weight_decay_val if not any(nd in name for nd in no_decay_keywords) else 0.0
                                param_groups_list.append({'params': [param], 'lr': block_lr, 'weight_decay': pg_wd, 'name': full_name})
                                assigned_params_set.add(full_name)
                block_base_idx += len(model.rgb_stages[i_stage]) # Increment base index

                # DFCA modules (LR based on their position, after the stage)
                if hasattr(model, 'dfca_modules') and i_stage < len(model.dfca_modules):
                    dfca_depth = block_base_idx # Effective depth after current stage blocks
                    lr_scale = llrd_rate_val ** (num_layers - 1 - min(dfca_depth, num_layers-1)) # Cap depth for scale
                    dfca_lr = base_lr * lr_scale
                    for name, param in model.dfca_modules[i_stage].named_parameters():
                        full_name = f"dfca_modules.{i_stage}.{name}"
                        if full_name not in assigned_params_set and param.requires_grad:
                            pg_wd = weight_decay_val if not any(nd in name for nd in no_decay_keywords) else 0.0
                            param_groups_list.append({'params': [param], 'lr': dfca_lr, 'weight_decay': pg_wd, 'name': full_name})
                            assigned_params_set.add(full_name)
        
        # Remaining parameters (final norms etc.) get base LR
        for name, param in model.named_parameters():
            if name not in assigned_params_set and param.requires_grad:
                logger_pg.info(f"LLRD: Assigning base LR ({base_lr:.2e}) to unassigned param: {name}")
                pg_wd = weight_decay_val if not any(nd in name for nd in no_decay_keywords) else 0.0
                param_groups_list.append({'params': [param], 'lr': base_lr, 'weight_decay': pg_wd, 'name': f"remaining_{name}"})
    else: # No LLRD
        logger_pg.info("Using standard parameter groups (decay / no_decay).")
        decay_params_list, no_decay_params_list = [], [] # Renamed
        for name, param in model.named_parameters():
            if param.requires_grad:
                target_list = no_decay_params_list if any(nd in name for nd in no_decay_keywords) else decay_params_list
                target_list.append(param)
        
        head_specific_lr = base_lr * head_lr_factor_val if head_lr_factor_val != 1.0 else base_lr
        
        # Separate head params if head_lr_factor is different and not LLRD
        if head_lr_factor_val != 1.0:
            head_decay, head_no_decay = [], []
            body_decay, body_no_decay = [], []
            for name, param in model.named_parameters():
                if not param.requires_grad: continue
                is_head = name.startswith(('head.', 'head_norm.'))
                is_no_decay = any(nd in name for nd in no_decay_keywords)
                if is_head: (head_no_decay if is_no_decay else head_decay).append(param)
                else: (body_no_decay if is_no_decay else body_decay).append(param)
            if head_decay: param_groups_list.append({'params': head_decay, 'lr': head_specific_lr, 'weight_decay': weight_decay_val, 'name': 'head_decay'})
            if head_no_decay: param_groups_list.append({'params': head_no_decay, 'lr': head_specific_lr, 'weight_decay': 0.0, 'name': 'head_no_decay'})
            if body_decay: param_groups_list.append({'params': body_decay, 'lr': base_lr, 'weight_decay': weight_decay_val, 'name': 'body_decay'})
            if body_no_decay: param_groups_list.append({'params': body_no_decay, 'lr': base_lr, 'weight_decay': 0.0, 'name': 'body_no_decay'})
        else: # Standard grouping
            if decay_params_list: param_groups_list.append({'params': decay_params_list, 'lr': base_lr, 'weight_decay': weight_decay_val, 'name': 'decay_group'})
            if no_decay_params_list: param_groups_list.append({'params': no_decay_params_list, 'lr': base_lr, 'weight_decay': 0.0, 'name': 'no_decay_group'})

    # Log created groups
    # for i, group_item in enumerate(param_groups_list): logger_pg.debug(f"PG {i}: name={group_item.get('name','N/A')}, lr={group_item['lr']:.2e}, wd={group_item['weight_decay']}, #params={len(group_item['params'])}")
    return param_groups_list


# --- Main Training Function ---
def run_training_session(config_run: dict) -> float: # Renamed config to config_run
    logger_run = logging.getLogger(f"{__name__}.run_session") # Specific logger
    set_seed(config_run["seed"])
    device = torch.device(config_run["device"])

    img_h, img_w = config_run["img_size"] if isinstance(config_run["img_size"], (list, tuple)) else (config_run["img_size"], config_run["img_size"])
    img_size_tuple_run = (img_h, img_w) # Renamed img_size_tuple

    logger_run.info("Setting up datasets...")
    dataset_kwargs = {
        "root_dir": config_run["data_root"], "img_size": img_size_tuple_run,
        "normalize_for_model": config_run["normalize_data"], "use_spectral": config_run["hvt_spectral_channels"] > 0,
        "spectral_channels": config_run["hvt_spectral_channels"], "random_seed": config_run["seed"]
    }
    train_dataset = SARCLD2024Dataset(**dataset_kwargs, split="train", train_split_ratio=config_run["train_split_ratio"])
    val_dataset = SARCLD2024Dataset(**dataset_kwargs, split="val", train_split_ratio=config_run["train_split_ratio"])
    
    computed_class_weights_run, sampler_run = None, None # Renamed variables
    if config_run.get("use_weighted_sampler", False):
        computed_class_weights_run = train_dataset.get_class_weights()
        if computed_class_weights_run is not None:
            logger_run.info("Attempting WeightedRandomSampler.")
            train_labels_tensor = train_dataset.get_current_split_labels_as_tensor() # Renamed train_labels
            if train_labels_tensor is not None:
                sample_weights_tensor = torch.zeros(len(train_labels_tensor), dtype=torch.double) # Renamed sample_weights
                for i, label_val in enumerate(train_labels_tensor): sample_weights_tensor[i] = computed_class_weights_run[label_val.item()]
                sampler_run = WeightedRandomSampler(sample_weights_tensor, len(sample_weights_tensor), replacement=True)
                logger_run.info("WeightedRandomSampler configured."); config_run["use_weighted_loss"] = False
            else: logger_run.warning("Could not get train labels for WeightedRandomSampler. Disabled.")
        else: logger_run.warning("Could not compute class weights for WeightedRandomSampler. Disabled.")
    if config_run.get("use_weighted_loss", True) and not sampler_run:
        if computed_class_weights_run is None: computed_class_weights_run = train_dataset.get_class_weights()
        if computed_class_weights_run is not None: logger_run.info("Class weights will be used in loss.")
        else: logger_run.warning("Could not compute class weights for loss. Unweighted loss."); config_run["use_weighted_loss"] = False

    train_loader = DataLoader(train_dataset, batch_size=config_run["batch_size"], sampler=sampler_run, shuffle=(sampler_run is None), num_workers=config_run["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config_run["batch_size"], shuffle=False, num_workers=config_run["num_workers"], pin_memory=True, drop_last=False)
    class_names_run = train_dataset.get_class_names() # Renamed class_names
    logger_run.info(f"Loaders: Train {len(train_loader)} batches, Val {len(val_loader)} batches. Workers: {config_run['num_workers']}")

    logger_run.info(f"Initializing model: {config_run['model_name']}")
    model_params_run = { # Renamed model_params
        "img_size": img_size_tuple_run[0], "patch_size": config_run['hvt_patch_size'], "in_chans": 3, 
        "spectral_chans": config_run['hvt_spectral_channels'], "num_classes": config_run['num_classes'], 
        "embed_dim_rgb": config_run['hvt_embed_dim_rgb'], "embed_dim_spectral": config_run['hvt_embed_dim_spectral'],
        "depths": config_run['hvt_depths'], "num_heads": config_run['hvt_num_heads'], "mlp_ratio": config_run['hvt_mlp_ratio'],
        "qkv_bias": config_run['hvt_qkv_bias'], "drop_rate": config_run['hvt_model_drop_rate'], 
        "attn_drop_rate": config_run['hvt_attn_drop_rate'], "drop_path_rate": config_run['hvt_drop_path_rate'],
        "use_dfca": config_run['hvt_use_dfca'], "dfca_heads": config_run['hvt_dfca_heads'],
        "use_gradient_checkpointing": config_run.get("use_gradient_checkpointing", False)
    }
    model_run = DiseaseAwareHVT_XL(**model_params_run) # Renamed model
    if config_run.get("use_gradient_checkpointing"): logger_run.info("Gradient checkpointing enabled for model.")
    if config_run["load_pretrained"]: model_run = load_pretrained_weights(model_run, config_run["pretrained_checkpoint_path"], config_run)
    model_run = model_run.to(device)

    freeze_epochs_run = config_run.get("freeze_backbone_epochs", 0) # Renamed freeze_epochs
    if freeze_epochs_run > 0:
        logger_run.info(f"Freezing backbone for initial {freeze_epochs_run} epochs.")
        for name, param in model_run.named_parameters(): param.requires_grad = not not name.startswith(('head.', 'head_norm.'))
    else:
        logger_run.info("Full model training from start."); [param.requires_grad_(True) for param in model_run.parameters()]

    loss_config_run = { # Renamed loss_config
        "type": config_run["loss_type"], "label_smoothing": config_run["label_smoothing"],
        "focal_gamma": config_run.get("focal_gamma", 2.0), "focal_alpha": config_run.get("focal_alpha", None),
        "class_weights": computed_class_weights_run if config_run.get("use_weighted_loss", False) else None
    }

    current_base_lr_run = config_run["learning_rate"] # Renamed current_base_lr
    optimizer_grouped_params_run = get_parameter_groups( # Renamed optimizer_grouped_parameters
        model_run, base_lr=current_base_lr_run, weight_decay_val=config_run["weight_decay"],
        llrd_rate_val=config_run["llrd_rate"] if config_run.get("use_llrd") and freeze_epochs_run == 0 else None,
        head_lr_factor_val=config_run.get("head_lr_factor", 1.0), is_frozen_phase_val=(freeze_epochs_run > 0)
    )
    if not optimizer_grouped_params_run: logger_run.error("No trainable parameters found!"); return -1.0
    
    opt_name_run = config_run["optimizer"].lower() # Renamed opt_name
    optimizer_run = None # Renamed optimizer
    common_opt_kwargs = {'lr': current_base_lr_run, 'weight_decay': config_run["weight_decay"], 'betas': config_run.get("optimizer_params", {}).get("betas", (0.9, 0.999))}
    if opt_name_run == "lamb": optimizer_run = Lamb(optimizer_grouped_params_run, **common_opt_kwargs); logger_run.info("Using LAMB optimizer.")
    else: optimizer_run = torch.optim.AdamW(optimizer_grouped_params_run, **common_opt_kwargs); logger_run.info("Using AdamW optimizer (default or specified).")

    schedulers_list_run, main_scheduler_run, final_scheduler_run = [], None, None # Renamed variables
    if config_run["warmup_epochs"] > 0:
        schedulers_list_run.append(LinearLR(optimizer_run, start_factor=config_run["warmup_lr_init_factor"], total_iters=config_run["warmup_epochs"]))
        logger_run.info(f"Warmup: LinearLR for {config_run['warmup_epochs']} epochs, factor {config_run['warmup_lr_init_factor']}.")
    
    epochs_after_warmup_run = config_run["epochs"] - config_run["warmup_epochs"] # Renamed epochs_after_warmup
    sched_name_run = config_run["scheduler"].lower() # Renamed scheduler_name
    if sched_name_run == "cosineannealinglr": main_scheduler_run = CosineAnnealingLR(optimizer_run, T_max=max(1, epochs_after_warmup_run), eta_min=config_run["min_lr"]); logger_run.info(f"Main Sched: CosineAnnealingLR.")
    elif sched_name_run == "cosineannealingwarmrestarts": main_scheduler_run = CosineAnnealingWarmRestarts(optimizer_run, T_0=config_run["cosine_t_0"], T_mult=config_run["cosine_t_mult"], eta_min=config_run["min_lr"]); logger_run.info(f"Main Sched: CosineAnnealingWarmRestarts.")
    elif sched_name_run == "reducelronplateau": main_scheduler_run = ReduceLROnPlateau(optimizer_run, mode='max', factor=config_run["reducelr_factor"], patience=config_run["reducelr_patience"], min_lr=config_run["min_lr"], verbose=True); logger_run.info(f"Main Sched: ReduceLROnPlateau.")
    else: logger_run.warning(f"Scheduler '{config_run['scheduler']}' not recognized. No main sched beyond warmup.")

    if main_scheduler_run: final_scheduler_run = SequentialLR(optimizer_run, schedulers=schedulers_list_run + [main_scheduler_run], milestones=[config_run["warmup_epochs"]]) if schedulers_list_run else main_scheduler_run
    elif schedulers_list_run: final_scheduler_run = schedulers_list_run[0]
    if final_scheduler_run and schedulers_list_run : logger_run.info("Using SequentialLR (Warmup + Main).")
    
    scaler_run = GradScaler(enabled=config_run["amp_enabled"]) # Renamed scaler

    trainer_run = Finetuner( # Renamed trainer
        model=model_run, optimizer=optimizer_run, criterion_config=loss_config_run, device=device, scaler=scaler_run,
        scheduler=final_scheduler_run if not isinstance(main_scheduler_run, ReduceLROnPlateau) else None,
        model_config_params=model_params_run, # Pass all model params for EMA
        # Other Finetuner params will default to global_train_config if not specified here
        img_size=img_size_tuple_run, # Explicitly pass current img_size
        accumulation_steps=config_run["accumulation_steps"],
        clip_grad_norm=config_run["clip_grad_norm"],
        num_classes=config_run["num_classes"]
        # multi_scale related params are read by Finetuner from global_train_config
    )
    
    start_epoch_run, best_val_metric_run = 1, -1.0 # Renamed variables
    resume_path_run = config_run.get("resume_checkpoint_path", None) or (config_run["best_model_path"] if config_run.get("resume_from_best") and os.path.exists(config_run["best_model_path"]) else None)
    if resume_path_run: logger_run.info(f"Attempting to resume from: {resume_path_run}"); start_epoch_run, best_val_metric_run = trainer_run.load_model_checkpoint(resume_path_run)

    metric_to_monitor_run, patience_counter_run, early_stop_patience_run = config_run.get("metric_to_monitor",'accuracy'), 0, config_run.get("early_stopping_patience", float('inf')) # Renamed variables
    logger_run.info(f"Starting training from E{start_epoch_run} for {config_run['epochs']} total epochs... Monitor: {metric_to_monitor_run}")
    os.makedirs(os.path.dirname(config_run["best_model_path"]), exist_ok=True)

    for epoch_idx in range(start_epoch_run, config_run["epochs"] + 1): # Renamed epoch
        if freeze_epochs_run > 0 and epoch_idx == freeze_epochs_run + 1:
            logger_run.info(f"Epoch {epoch_idx}: *** Unfreezing backbone ***"); [param.requires_grad_(True) for param in model_run.parameters()]
            unfreeze_lr_run = config_run["learning_rate"] # Renamed unfreeze_base_lr
            opt_params_unfreeze = get_parameter_groups(model_run, base_lr=unfreeze_lr_run, weight_decay_val=config_run["weight_decay"], llrd_rate_val=config_run["llrd_rate"] if config_run.get("use_llrd") else None, head_lr_factor_val=1.0, is_frozen_phase_val=False) # Renamed optimizer_grouped_parameters
            opt_name_unfreeze_run = config_run["optimizer"].lower() # Renamed opt_name_unfreeze
            common_opt_kwargs_unfreeze = {'lr': unfreeze_lr_run, 'weight_decay': config_run["weight_decay"], 'betas': config_run.get("optimizer_params", {}).get("betas", (0.9, 0.999))}
            if opt_name_unfreeze_run == "lamb": trainer_run.optimizer = Lamb(opt_params_unfreeze, **common_opt_kwargs_unfreeze)
            else: trainer_run.optimizer = torch.optim.AdamW(opt_params_unfreeze, **common_opt_kwargs_unfreeze)
            logger_run.info(f"Optimizer re-init for full model with base LR: {unfreeze_lr_run}.")
            
            epochs_left_run = config_run["epochs"] - epoch_idx + 1 # Renamed epochs_left_for_main_sched
            main_sched_unfreeze_run = None # Renamed main_scheduler_unfreeze
            if sched_name_run == "cosineannealinglr": main_sched_unfreeze_run = CosineAnnealingLR(trainer_run.optimizer, T_max=max(1, epochs_left_run), eta_min=config_run["min_lr"])
            elif sched_name_run == "cosineannealingwarmrestarts": main_sched_unfreeze_run = CosineAnnealingWarmRestarts(trainer_run.optimizer, T_0=max(1, epochs_left_run // 2 if epochs_left_run >1 else 1), T_mult=config_run["cosine_t_mult"], eta_min=config_run["min_lr"])
            elif isinstance(main_scheduler_run, ReduceLROnPlateau): main_sched_unfreeze_run = ReduceLROnPlateau(trainer_run.optimizer, mode='max', factor=config_run["reducelr_factor"], patience=config_run["reducelr_patience"], min_lr=config_run["min_lr"], verbose=True)
            trainer_run.scheduler = main_sched_unfreeze_run
            if trainer_run.scheduler: logger_run.info(f"Scheduler re-init for unfreeze: {trainer_run.scheduler.__class__.__name__}")
        
        ep_start_time = time.time(); avg_train_loss = trainer_run.train_one_epoch(train_loader, epoch_idx, config_run["epochs"]); ep_train_time = time.time() - ep_start_time # Renamed variables
        ep_val_start_time = time.time(); avg_val_loss, val_metrics = trainer_run.validate_one_epoch(val_loader, class_names=class_names_run); ep_val_time = time.time() - ep_val_start_time # Renamed variables
        current_val_metric_val = val_metrics.get(metric_to_monitor_run, 0.0) # Renamed current_val_metric_value

        active_rlp = trainer_run.scheduler if isinstance(trainer_run.scheduler, ReduceLROnPlateau) else (main_scheduler_run if isinstance(main_scheduler_run, ReduceLROnPlateau) else None)
        if active_rlp: active_rlp.step(current_val_metric_val); logger_run.info(f"RLP step with {metric_to_monitor_run}: {current_val_metric_val:.4f}")
        elif trainer_run.scheduler: trainer_run.scheduler.step()

        logger_run.info(f"E{epoch_idx}/{config_run['epochs']} Sum | TrL:{avg_train_loss:.4f} | VlL:{avg_val_loss:.4f} | VAcc:{val_metrics.get('accuracy',0.0):.4f} | VF1w:{val_metrics.get('f1_weighted',0.0):.4f} | Mon({metric_to_monitor_run}):{current_val_metric_val:.4f} | LR:{trainer_run.optimizer.param_groups[0]['lr']:.2e} | TrT:{ep_train_time:.1f}s | VlT:{ep_val_time:.1f}s")
        if current_val_metric_val > best_val_metric_run:
            best_val_metric_run = current_val_metric_val
            if config_run["best_model_path"]: trainer_run.save_model_checkpoint(config_run["best_model_path"], epoch=epoch_idx, best_metric=best_val_metric_run)
            logger_run.info(f"E{epoch_idx}: New best val {metric_to_monitor_run}: {best_val_metric_run:.4f}. Ckpt saved.")
            patience_counter_run = 0
        else:
            patience_counter_run += 1
            logger_run.info(f"E{epoch_idx}: Val {metric_to_monitor_run} ({current_val_metric_val:.4f}) !improve from best ({best_val_metric_run:.4f}). Patience: {patience_counter_run}/{early_stop_patience_run}")
        if patience_counter_run >= early_stop_patience_run: logger_run.info(f"Early stopping triggered."); break
        sys.stdout.flush(); [h.flush() for h in logger_run.handlers] # Renamed variables

    logger_run.info(f"Training finished. Best val {metric_to_monitor_run}: {best_val_metric_run:.4f}")
    if config_run["final_model_path"]: trainer_run.save_model_checkpoint(config_run["final_model_path"], epoch=config_run["epochs"], best_metric=best_val_metric_run); logger_run.info(f"Final model saved: {config_run['final_model_path']}")
    return best_val_metric_run


# --- Main Execution ---
if __name__ == "__main__":
    # If OOM issues persist, try setting this ENV var before running:
    # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    cli_args_main = parse_args() # Renamed cli_args
    config_main = default_config.copy() # Renamed config
    if cli_args_main.config: config_main = load_config_yaml(cli_args_main.config, base_config_dict=config_main)
    
    # Override with CLI arguments
    img_h_cli, img_w_cli = cli_args_main.img_size_h, cli_args_main.img_size_w
    if img_h_cli is not None and img_w_cli is not None:
        config_main["img_size"] = (img_h_cli, img_w_cli)
        print(f"Overriding img_size with CLI: ({img_h_cli}, {img_w_cli})")
    elif img_h_cli is not None or img_w_cli is not None:
        print("Warning: Both --img_size_h and --img_size_w must be provided to override img_size.")

    for arg_name, arg_val in vars(cli_args_main).items(): # Renamed arg_value
        if arg_val is not None and arg_name not in ["config", "img_size_h", "img_size_w"]:
            if arg_name in config_main: config_main[arg_name] = arg_val; print(f"Override: {arg_name} = {arg_val}")
            else: print(f"Warn: CLI arg '{arg_name}' not in config.")
            
    log_dir_abs_main = os.path.join(current_dir, config_main["log_dir"]); os.makedirs(log_dir_abs_main, exist_ok=True) # Renamed log_dir_abs
    setup_logging(log_file_name=config_main["log_file_finetune"], log_dir=log_dir_abs_main, log_level=logging.INFO, logger_name=None)
    
    main_script_logger = logging.getLogger(__name__) # Renamed main_logger
    main_script_logger.info("--------- Starting New Training Session ---------")
    main_script_logger.info(f"Using device: {config_main['device']}"); main_script_logger.info(f"Random seed: {config_main['seed']}")
    main_script_logger.info("Effective Configuration:"); [main_script_logger.info(f"  {k_item}: {v_item}") for k_item, v_item in config_main.items()] # Renamed key, value

    if config_main.get("hpo_enabled", False): main_script_logger.warning("HPO enabled. Run HPO script for optimization.")
    else:
        try: run_training_session(config_main)
        except Exception as e_main: main_script_logger.exception(f"Critical error during training: {e_main}"); sys.exit(1) # Renamed e
    main_script_logger.info("--------- Training Session Ended ---------")