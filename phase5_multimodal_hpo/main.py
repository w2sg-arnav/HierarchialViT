# phase4_finetuning/main.py - Enhanced for 90%+ accuracy with advanced techniques

print("EXECUTING THE VERY LATEST MAIN.PY - VERSION CHECK XYZ123") # Add a unique string
import sys
sys.exit("Exiting early for version check") # Temporarily exit

from collections import OrderedDict, defaultdict
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import argparse
import yaml
import torch.nn.functional as F
import math # Keep for potential future use
import sys
from typing import Tuple, Optional, Dict, Any, List
import traceback
from datetime import datetime
import time
import random
from torch.optim import AdamW, SGD
import warnings
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms.v2 as T_v2 # For eval transform

logger_main: Optional[logging.Logger] = None # Global logger for this module

try:
    # Corrected import for config to match the one used later in __main__
    from .config import config as main_config_from_phase4
    from .dataset import SARCLD2024Dataset
    # utils.augmentations (augmentations (10).py) exports CottonLeafDiseaseAugmentation and create_cotton_leaf_augmentation
    from .utils.augmentations import create_cotton_leaf_augmentation
    from .utils.logging_setup import setup_logging
    from .finetune.trainer import EnhancedFinetuner
    from phase2_model.models.hvt import create_disease_aware_hvt
    from .utils.metrics import compute_metrics
except ImportError as e_imp:
    print(f"CRITICAL IMPORT ERROR in main.py: {e_imp}. Check paths and ensure all modules are accessible.", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

class EMA:
    """Exponential Moving Average with warmup and decay scheduling"""
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 0, use_num_updates: bool = True):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.use_num_updates = use_num_updates
        self.num_updates = 0
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        self.num_updates += 1
        current_decay = self.decay
        if self.use_num_updates: # Adaptive decay during early stages
            current_decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        if self.warmup_steps > 0 and self.num_updates <= self.warmup_steps:
            warmup_factor = self.num_updates / self.warmup_steps
            effective_decay = current_decay * warmup_factor
        else:
            effective_decay = current_decay

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param_data_type = param.data.dtype
                # Ensure shadow param is same dtype and device for the operation
                shadow_param = self.shadow[name].to(device=param.data.device, dtype=param_data_type)
                new_average = (1.0 - effective_decay) * param.data + effective_decay * shadow_param
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].to(param.data.dtype) # Ensure correct dtype

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss_val = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss_val.mean()
        elif self.reduction == 'sum':
            return focal_loss_val.sum()
        else:
            return focal_loss_val

class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1, focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 ce_weight: float = 0.5, focal_weight: float = 0.5, class_weights_tensor: Optional[torch.Tensor] = None):
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=smoothing, weight=class_weights_tensor)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        global logger_main
        if logger_main and abs(ce_weight + focal_weight - 1.0) > 1e-6 and (ce_weight > 0 and focal_weight > 0):
            logger_main.warning(f"Sum of ce_weight ({ce_weight}) and focal_weight ({focal_weight}) is not 1.0. Losses will be scaled accordingly.")

    def forward(self, inputs, targets):
        loss = torch.tensor(0.0, device=inputs.device)
        if self.ce_weight > 0:
            loss += self.ce_weight * self.ce_loss(inputs, targets)
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal_loss(inputs, targets)
        return loss

def create_weighted_sampler(dataset: SARCLD2024Dataset, cfg: Dict[str,Any]) -> Optional[WeightedRandomSampler]:
    global logger_main
    mode = cfg.get('weighted_sampler_mode', 'inv_count')
    beta = 0.9999 # Standard for effective number mode

    if not hasattr(dataset, 'get_targets'):
        if logger_main: logger_main.warning("Dataset does not have 'get_targets' method. Cannot create weighted sampler.")
        return None

    targets = np.array(dataset.get_targets())
    if len(targets) == 0:
        if logger_main: logger_main.warning("No targets found for weighted sampler.")
        return None

    num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else (np.max(targets) + 1)
    class_counts = np.bincount(targets, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1e-9) # Avoid division by zero

    weights_per_class: np.ndarray
    if mode == 'inv_count':
        weights_per_class = 1.0 / class_counts
    elif mode == 'effective_number':
        effective_num = 1.0 - np.power(beta, class_counts)
        weights_per_class = (1.0 - beta) / effective_num
    else: # uniform
        if logger_main: logger_main.warning(f"Unknown sampler mode: {mode}. Using uniform sampling weights.")
        weights_per_class = np.ones_like(class_counts, dtype=np.float32)

    weights_per_class = weights_per_class / np.sum(weights_per_class) # Normalize
    sample_weights = weights_per_class[targets]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )
    if logger_main: logger_main.info(f"WeightedRandomSampler created with mode: {mode}.")
    return sampler

def get_class_weights_for_loss(dataset: SARCLD2024Dataset, device: torch.device, cfg: Dict[str,Any]) -> Optional[torch.Tensor]:
    global logger_main
    if not cfg.get('use_weighted_loss', False):
        return None
    if not hasattr(dataset, 'get_targets'):
        if logger_main: logger_main.warning("Dataset does not have 'get_targets' method for class_weights.")
        return None
    targets = np.array(dataset.get_targets())
    if len(targets) == 0:
        if logger_main: logger_main.warning("No targets found for class_weights.")
        return None
    num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else (np.max(targets) + 1)
    class_counts = np.bincount(targets, minlength=num_classes)
    weights = 1.0 / (class_counts + 1e-6) # Add epsilon for stability
    weights = weights / np.sum(weights) * num_classes # Normalize to make sum = num_classes
    if logger_main: logger_main.info(f"Calculated class weights for loss: {weights}")
    return torch.tensor(weights, dtype=torch.float32).to(device)

def get_data_loaders(cfg: Dict[str, Any], current_img_size: Tuple[int, int], class_names: Optional[List[str]] = None, dataset_base_path: Optional[str]=None) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    global logger_main
    if logger_main: logger_main.info(f"Creating DataLoaders with image size: {current_img_size}")

    # Use the factory function from utils.augmentations
    train_aug_obj = create_cotton_leaf_augmentation(
        strategy=cfg.get('augmentation_strategy', 'cotton_disease'), # From config (24).py
        img_size=current_img_size,
        severity=cfg.get('augmentation_severity', 'moderate'),
        # Mixup/Cutmix are handled by EnhancedFinetuner based on config flags like 'mixup_alpha'.
        # CottonLeafDiseaseAugmentation's internal mixup/cutmix should be False here.
        use_mixup=False,
        use_cutmix=False
    )

    eval_transform = T_v2.Compose([
        T_v2.Resize(current_img_size, interpolation=T_v2.InterpolationMode.BILINEAR, antialias=True),
        T_v2.ToDtype(torch.float32, scale=True), # Assumes input is uint8 [0,255]
        T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    _DATA_ROOT = dataset_base_path if dataset_base_path else cfg['data_root']
    actual_dataset_path = os.path.join(_DATA_ROOT, cfg.get('original_dataset_name', "Original Dataset"))
    if not os.path.isdir(actual_dataset_path):
        if logger_main: logger_main.error(f"Dataset path not found: {actual_dataset_path}")
        raise FileNotFoundError(f"Dataset path not found: {actual_dataset_path}")

    try:
        # SARCLD2024Dataset needs to handle these splits and parameters
        train_dataset = SARCLD2024Dataset(root_dir=actual_dataset_path, split='train', transform=train_aug_obj, img_size=current_img_size, num_classes=cfg['num_classes'], class_names=class_names, train_split_ratio=cfg.get('train_split_ratio', 0.8))
        val_dataset = SARCLD2024Dataset(root_dir=actual_dataset_path, split='val', transform=eval_transform, img_size=current_img_size, num_classes=cfg['num_classes'], class_names=class_names, train_split_ratio=cfg.get('train_split_ratio', 0.8))
        test_dataset = None
        try:
            test_dataset = SARCLD2024Dataset(root_dir=actual_dataset_path, split='test', transform=eval_transform, img_size=current_img_size, num_classes=cfg['num_classes'], class_names=class_names)
        except Exception as e_test_ds:
            if logger_main: logger_main.warning(f"Could not create test dataset (may not exist or error): {e_test_ds}")
    except Exception as e:
        if logger_main: logger_main.error(f"Error creating datasets: {e}. Check SARCLD2024Dataset and paths.")
        raise

    train_sampler = None
    if cfg.get('use_weighted_sampler', False) and train_dataset and len(train_dataset) > 0:
        train_sampler = create_weighted_sampler(train_dataset, cfg)

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=(train_sampler is None), num_workers=cfg['num_workers'], pin_memory=(cfg['device']=='cuda'), sampler=train_sampler, prefetch_factor=cfg.get('prefetch_factor') if cfg.get('num_workers',0) > 0 else None, drop_last=True) if train_dataset and len(train_dataset) > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'] * 2, shuffle=False, num_workers=cfg['num_workers'], pin_memory=(cfg['device']=='cuda'), prefetch_factor=cfg.get('prefetch_factor') if cfg.get('num_workers',0) > 0 else None) if val_dataset and len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'] * 2, shuffle=False, num_workers=cfg['num_workers'], pin_memory=(cfg['device']=='cuda'), prefetch_factor=cfg.get('prefetch_factor') if cfg.get('num_workers',0) > 0 else None) if test_dataset and len(test_dataset) > 0 else None

    return train_loader, val_loader, test_loader

def get_optimizer_param_groups(model: nn.Module, cfg: Dict[str, Any]):
    global logger_main
    weight_decay = cfg.get('weight_decay', 0.01)
    default_max_lr_for_onecycle = cfg.get('onecycle_max_lr', 1e-3)

    param_groups = []
    if cfg.get('use_layer_wise_lr', False) and 'layer_wise_lr_multipliers' in cfg:
        if logger_main: logger_main.info("Using layer-wise LR multipliers.")
        multipliers = cfg['layer_wise_lr_multipliers']

        hvt_depths_config = cfg.get('hvt_params_for_model_init', {}).get('depths', [])
        num_total_blocks_from_depths = sum(hvt_depths_config) if isinstance(hvt_depths_config, list) else 12

        param_name_to_group_key = { 'patch_embed': 'layer_1' }
        if num_total_blocks_from_depths > 0:
            q1 = num_total_blocks_from_depths // 4
            q2 = num_total_blocks_from_depths // 2
            q3 = 3 * num_total_blocks_from_depths // 4
            param_name_to_group_key.update(**{f'blocks.{i}.': 'layer_1' for i in range(0, q1)})
            param_name_to_group_key.update(**{f'blocks.{i}.': 'layer_2' for i in range(q1, q2)})
            param_name_to_group_key.update(**{f'blocks.{i}.': 'layer_3' for i in range(q2, q3)})
            param_name_to_group_key.update(**{f'blocks.{i}.': 'layer_4' for i in range(q3, num_total_blocks_from_depths)})
        else:
            if logger_main: logger_main.warning("HVT block depths not found or zero for layer-wise LR mapping. Block LRs might not be specific.")

        param_name_to_group_key.update({
            'norm.': 'layer_4',
            'head.': 'head',
            'classifier.': 'head'
        })

        grouped_params_dict = defaultdict(lambda: {'params': [], 'lr_multiplier_key': None, 'has_wd': True, 'names': []})
        param_names_no_decay = ['bias', '.norm.weight', 'LayerNorm.weight']

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            is_no_decay = any(nd_pattern in name for nd_pattern in param_names_no_decay)
            assigned_group_key_for_mult = None
            current_param_lr_multiplier = 1.0
            best_match_len = 0

            for pattern, mult_key in param_name_to_group_key.items():
                if name.startswith(pattern):
                    if len(pattern) > best_match_len:
                        best_match_len = len(pattern)
                        assigned_group_key_for_mult = mult_key
                        current_param_lr_multiplier = multipliers.get(mult_key, 1.0)

            if assigned_group_key_for_mult is None:
                assigned_group_key_for_mult = 'default_backbone'
                current_param_lr_multiplier = multipliers.get(assigned_group_key_for_mult, 0.5)
                if logger_main: logger_main.debug(f"Param {name} assigned to default LR multiplier group '{assigned_group_key_for_mult}' with mult {current_param_lr_multiplier}.")

            group_dict_key = (assigned_group_key_for_mult, not is_no_decay)
            grouped_params_dict[group_dict_key]['params'].append(param)
            grouped_params_dict[group_dict_key]['lr_multiplier_key'] = assigned_group_key_for_mult
            grouped_params_dict[group_dict_key]['has_wd'] = not is_no_decay

        for (mult_key, has_wd_flag), data in grouped_params_dict.items():
            lr_mult = multipliers.get(data['lr_multiplier_key'], 1.0)
            group_lr = default_max_lr_for_onecycle * lr_mult
            param_groups.append({
                'params': data['params'],
                'lr': group_lr,
                'weight_decay': weight_decay if has_wd_flag else 0.0,
                'name': f"{mult_key}_{'wd' if has_wd_flag else 'no_wd'}"
            })
            if logger_main: logger_main.debug(f"Created param group: {param_groups[-1]['name']} with initial (max) LR: {group_lr:.2e}, WD: {param_groups[-1]['weight_decay']}" )
        if logger_main: logger_main.info(f"Created {len(param_groups)} parameter groups for layer-wise LR.")
    else:
        lr_head = cfg.get('lr_head_unfrozen_phase', default_max_lr_for_onecycle)
        lr_backbone = cfg.get('lr_backbone_unfrozen_phase', default_max_lr_for_onecycle * 0.1)

        head_params, backbone_params, no_decay_params_list = [], [], []
        param_names_no_decay = ['bias', '.norm.weight', 'LayerNorm.weight']
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            is_no_decay = any(nd in name for nd in param_names_no_decay)
            is_head_param = 'head' in name or 'classifier' in name

            if is_no_decay: no_decay_params_list.append(param)
            elif is_head_param: head_params.append(param)
            else: backbone_params.append(param)

        if head_params: param_groups.append({'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay, 'name': 'head'})
        if backbone_params: param_groups.append({'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay, 'name': 'backbone'})
        if no_decay_params_list: param_groups.append({'params': no_decay_params_list, 'lr': lr_head, 'weight_decay': 0.0, 'name': 'no_decay'})
        if logger_main: logger_main.info("Using standard head/backbone parameter groups for LR.")

    if not param_groups:
        if logger_main: logger_main.warning("No specific parameter groups created, all params in one group.")
        return [{'params': model.parameters(), 'lr': default_max_lr_for_onecycle, 'weight_decay': weight_decay}]
    return param_groups

def setup_model_optimizer_scheduler(cfg: Dict[str, Any], device: torch.device, num_classes: int,
                                    steps_per_epoch: Optional[int] = None, model_to_load_state: Optional[dict]=None):
    global logger_main
    hvt_params = cfg.get('hvt_params_for_model_init', {}).copy()
    hvt_params.update({'img_size': cfg['img_size'], 'num_classes': num_classes})
    if 'spectral_channels' not in hvt_params: hvt_params['spectral_channels'] = 0

    model = create_disease_aware_hvt(**hvt_params)
    if model_to_load_state:
        current_model_state = model.state_dict()
        filtered_load_state = {}
        for k, v in model_to_load_state.items():
            if k in current_model_state and current_model_state[k].shape == v.shape:
                filtered_load_state[k] = v
            else:
                if logger_main: logger_main.warning(f"Skipping key {k} from pretrained weights due to shape mismatch or absence in current model.")

        missing_keys, unexpected_keys = model.load_state_dict(filtered_load_state, strict=False)
        if logger_main:
            if missing_keys: logger_main.warning(f"Loading model state: Missing keys: {missing_keys}")
            if unexpected_keys: logger_main.warning(f"Loading model state: Unexpected keys from filtered state (should be rare): {unexpected_keys}")
            logger_main.info(f"Model state loaded from filtered pretrained weights (strict=False). {len(filtered_load_state)} keys loaded.")
    model = model.to(device)

    if cfg.get('channels_last', False):
        try:
            model = model.to(memory_format=torch.channels_last)
            if logger_main: logger_main.info("Converted model to channels_last memory format.")
        except Exception as e_cl:
            if logger_main: logger_main.warning(f"Failed to convert model to channels_last: {e_cl}")

    if cfg.get('enable_torch_compile', True) and hasattr(torch, 'compile'):
        compile_mode = cfg.get('torch_compile_mode', 'default')
        try:
            if logger_main: logger_main.info(f"Compiling model with torch.compile(mode='{compile_mode}')...")
            if 'matmul_precision' in cfg and hasattr(torch, 'set_float32_matmul_precision'):
                if logger_main: logger_main.info(f"Setting matmul_precision: '{cfg['matmul_precision']}'")
                torch.set_float32_matmul_precision(cfg['matmul_precision'])
            model = torch.compile(model, mode=compile_mode)
        except Exception as e:
            if logger_main: logger_main.warning(f"Model compilation failed: {e}. No compilation.")

    optimizer_param_config = get_optimizer_param_groups(model, cfg)
    opt_cfg_params = cfg.get('optimizer_params', {})
    optimizer_name = cfg.get('optimizer', "AdamW").lower()

    if optimizer_name == "adamw":
        optimizer = AdamW(optimizer_param_config,
                          betas=opt_cfg_params.get('betas', (0.9, 0.999)),
                          eps=opt_cfg_params.get('eps', 1e-8),
                          amsgrad=opt_cfg_params.get('amsgrad', False))
    elif optimizer_name == "sgd":
        optimizer = SGD(optimizer_param_config,
                        momentum=opt_cfg_params.get('momentum', 0.9),
                        nesterov=opt_cfg_params.get('nesterov', True))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    if logger_main: logger_main.info(f"Optimizer: {optimizer_name} configured with {len(optimizer.param_groups)} param groups.")

    scheduler_name = cfg.get('scheduler', 'OneCycleLR').lower()
    scheduler = None
    if scheduler_name == 'onecyclelr':
        if steps_per_epoch is None: raise ValueError("OneCycleLR needs steps_per_epoch.")
        max_lrs_for_groups = [pg['lr'] for pg in optimizer.param_groups]
        scheduler = OneCycleLR(optimizer, max_lr=max_lrs_for_groups,
                               epochs=cfg['epochs'], steps_per_epoch=steps_per_epoch,
                               pct_start=cfg.get('onecycle_pct_start', 0.3),
                               div_factor=cfg.get('onecycle_div_factor', 25),
                               final_div_factor=cfg.get('onecycle_final_div_factor', 1e4),
                               three_phase=cfg.get('onecycle_three_phase', False),
                               anneal_strategy=cfg.get('onecycle_anneal_strategy', 'cos'))
        if logger_main: logger_main.info(f"Using OneCycleLR scheduler with max_lrs per group: {max_lrs_for_groups}")
    elif scheduler_name == 'reduceonplateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                      factor=cfg.get('plateau_factor', 0.2),
                                      patience=cfg.get('plateau_patience', 10),
                                      verbose=True, min_lr=cfg.get('eta_min_lr',1e-7))
        if logger_main: logger_main.info("Using ReduceLROnPlateau scheduler.")

    return model, optimizer, scheduler

def progressive_unfreezing_hvt(model: nn.Module, epoch: int, cfg: Dict[str, Any]):
    global logger_main
    if not cfg.get('progressive_unfreezing', False):
        if epoch == 0 and logger_main: logger_main.info("Progressive unfreezing disabled.")
        return

    initial_freeze_epochs = cfg.get('freeze_backbone_epochs', 0)
    unfreeze_schedule_epochs = cfg.get('unfreezing_schedule', [])

    target_model_for_params = model.module if hasattr(model, 'module') else model

    for param in target_model_for_params.parameters():
        param.requires_grad = False

    head_trained_this_pass = False
    head_name_patterns = ['head', 'classifier', 'fc']

    if hasattr(target_model_for_params, 'head') and target_model_for_params.head is not None:
        for param in target_model_for_params.head.parameters():
            param.requires_grad = True
        head_trained_this_pass = True
    else:
        for name, param in target_model_for_params.named_parameters():
            if any(hn_pattern in name.lower() for hn_pattern in head_name_patterns):
                param.requires_grad = True
                head_trained_this_pass = True
    if not head_trained_this_pass and logger_main and epoch==0:
        logger_main.warning("Could not identify model head parameters by attribute or common name patterns for unfreezing.")

    if epoch >= initial_freeze_epochs:
        num_transformer_blocks = 0
        if hasattr(target_model_for_params, 'blocks') and isinstance(target_model_for_params.blocks, nn.ModuleList):
            num_transformer_blocks = len(target_model_for_params.blocks)

        if hasattr(target_model_for_params, 'patch_embed'):
            for param in target_model_for_params.patch_embed.parameters(): param.requires_grad = True
        if hasattr(target_model_for_params, 'norm') and target_model_for_params.norm is not None and not isinstance(target_model_for_params.norm, nn.Identity):
             for param in target_model_for_params.norm.parameters(): param.requires_grad = True

        if num_transformer_blocks > 0:
            blocks_to_unfreeze_count = 0
            if not unfreeze_schedule_epochs:
                blocks_to_unfreeze_count = num_transformer_blocks
            else:
                stages_passed = sum(1 for scheduled_epoch in unfreeze_schedule_epochs if epoch >= scheduled_epoch)
                if stages_passed > 0 and len(unfreeze_schedule_epochs) > 0: # Avoid division by zero
                    blocks_to_unfreeze_count = math.ceil(num_transformer_blocks * (stages_passed / len(unfreeze_schedule_epochs)))
                elif stages_passed == 0 and len(unfreeze_schedule_epochs) > 0 : # Before first schedule point but after initial freeze
                     blocks_to_unfreeze_count = 0 # Or some small fraction like 1 block if desired
                else: # No schedule or no stages passed.
                    blocks_to_unfreeze_count = num_transformer_blocks # Default to unfreezing all after initial

            start_block_idx_to_unfreeze = num_transformer_blocks - blocks_to_unfreeze_count
            for i in range(max(0, start_block_idx_to_unfreeze), num_transformer_blocks):
                if i < len(target_model_for_params.blocks):
                    for param in target_model_for_params.blocks[i].parameters():
                        param.requires_grad = True

            if logger_main and (epoch == initial_freeze_epochs or epoch in unfreeze_schedule_epochs):
                logger_main.info(f"Epoch {epoch}: Unfreezing. Total blocks: {num_transformer_blocks}. Unfrozen from top: {blocks_to_unfreeze_count}.")
    elif logger_main and epoch == 0 :
         logger_main.info(f"Epoch {epoch}: Initial freeze phase. Only head trainable for {initial_freeze_epochs} epochs.")

    if epoch == 0 or epoch == initial_freeze_epochs or epoch in unfreeze_schedule_epochs:
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if logger_main: logger_main.info(f"Epoch {epoch}: Number of trainable parameters: {num_trainable:,}")

def plot_training_curves(history: Dict[str, List[float]], output_dir: str, cfg_metric_name: str):
    global logger_main
    epochs_ran = range(1, len(history['train_loss']) + 1)
    if not epochs_ran:
        if logger_main: logger_main.warning("No history to plot for training curves.")
        return

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1); plt.plot(epochs_ran, history['train_loss'], 'bo-', label='Training Loss'); plt.plot(epochs_ran, history['val_loss'], 'ro-', label='Validation Loss'); plt.title('Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

    base_metric_name = cfg_metric_name.replace('val_', '')
    train_metric_key = f"train_{base_metric_name}"
    val_metric_key = f"val_{base_metric_name}"
    metric_label = base_metric_name.replace("_", " ").title()

    # Ensure keys exist and have data before plotting
    if train_metric_key in history and val_metric_key in history and \
       history[train_metric_key] and history[val_metric_key] and \
       len(history[train_metric_key]) == len(epochs_ran) and len(history[val_metric_key]) == len(epochs_ran):
        plt.subplot(1, 3, 2); plt.plot(epochs_ran, history[train_metric_key], 'bo-', label=f'Training {metric_label}'); plt.plot(epochs_ran, history[val_metric_key], 'ro-', label=f'Validation {metric_label}'); plt.title(f'{metric_label}'); plt.xlabel('Epochs'); plt.ylabel(metric_label.split(' ')[-1]); plt.legend(); plt.grid(True)
    elif 'train_accuracy' in history and 'val_accuracy' in history and \
         history['train_accuracy'] and history['val_accuracy'] and \
         len(history['train_accuracy']) == len(epochs_ran) and len(history['val_accuracy']) == len(epochs_ran):
        if logger_main: logger_main.debug(f"Plotting fallback accuracy as specific metric '{base_metric_name}' not fully in history or mismatched length.")
        plt.subplot(1, 3, 2); plt.plot(epochs_ran, history['train_accuracy'], 'bo-', label='Training Accuracy'); plt.plot(epochs_ran, history['val_accuracy'], 'ro-', label='Validation Accuracy'); plt.title('Accuracy'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    else:
        if logger_main: logger_main.warning(f"Could not find sufficient or consistently sized data for train/val metric '{base_metric_name}' or 'accuracy' in history for plotting.")

    if 'lr' in history and history['lr'] and len(history['lr']) == len(epochs_ran):
        plt.subplot(1, 3, 3); plt.plot(epochs_ran, history['lr'], 'go-', label='Learning Rate'); plt.title('Learning Rate'); plt.xlabel('Epochs'); plt.ylabel('LR'); plt.legend(); plt.grid(True);
        if any(lr_val > 1e-9 for lr_val in history['lr']):
            plt.gca().set_yscale('log')
    else:
         if logger_main: logger_main.warning("No 'lr' data or mismatched length in history for plotting.")

    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "training_curves.png")); plt.close()
    if logger_main: logger_main.info(f"Training curves plotted to {os.path.join(output_dir, 'training_curves.png')}")

def plot_confusion_matrix_main(cm: np.ndarray, class_names: List[str], output_path: str, filename="confusion_matrix.png"):
    global logger_main
    plt.figure(figsize=(max(8, len(class_names)*0.9), max(6, len(class_names)*0.7)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.xlabel('Predicted Labels'); plt.ylabel('True Labels'); plt.title('Confusion Matrix'); plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename)); plt.close()
    if logger_main: logger_main.info(f"Confusion matrix saved to {os.path.join(output_path, filename)}")

def main(cfg: Dict[str, Any]):
    global logger_main

    output_dir = os.path.join(cfg['PACKAGE_ROOT_PATH'], cfg['log_dir'])
    checkpoints_save_dir = os.path.join(output_dir, cfg['checkpoint_save_dir_name'])
    os.makedirs(output_dir, exist_ok=True); os.makedirs(checkpoints_save_dir, exist_ok=True)

    log_file_path = os.path.join(output_dir, cfg['log_file_finetune'])
    try:
        logger_main = setup_logging(log_file_path) # CORRECTED: Removed 'level' argument
    except NameError: # setup_logging not defined/imported
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)])
        logger_main = logging.getLogger(__name__)
        logger_main.warning("setup_logging function not found. Using basicConfig.")
    except TypeError as e_setup_logging: # Catch specific error if definition is wrong
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)])
        logger_main = logging.getLogger(__name__)
        logger_main.error(f"TypeError in setup_logging: {e_setup_logging}. Fallback to basicConfig. Ensure setup_logging function definition is correct (e.g., should not take a 'level' argument if this error persists).")

    logger_main.info("Starting finetuning process (High Performance V2 Strategy):")
    logged_cfg_keys = ['seed', 'device', 'log_dir', 'resume_from_checkpoint', 'data_root', 'img_size', 'num_classes',
                       'epochs', 'batch_size', 'accumulation_steps', 'optimizer', 'scheduler', 'loss_function',
                       'metric_to_monitor_early_stopping', 'early_stopping_patience',
                       'progressive_unfreezing', 'freeze_backbone_epochs', 'use_ema', 'use_swa', 'amp_enabled']
    for key in logged_cfg_keys:
        if key in cfg: logger_main.info(f"  {key}: {cfg[key]}")
    if 'hvt_params_for_model_init' in cfg:
        logger_main.info(f"  HVT Model Key Params: patch_size={cfg['hvt_params_for_model_init'].get('patch_size')}, depths={cfg['hvt_params_for_model_init'].get('depths')}, embed_dim_rgb={cfg['hvt_params_for_model_init'].get('embed_dim_rgb')}")

    seed = cfg.get('seed', 42); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = cfg.get('cudnn_benchmark', True)
    logger_main.info(f"Device: {device}. Seed: {seed}. CudNN Benchmark: {torch.backends.cudnn.benchmark if device.type == 'cuda' else 'N/A'}.")

    try:
        _dummy_ds_path = os.path.join(cfg['data_root'], cfg.get('original_dataset_name', "Original Dataset"))
        _dummy_dataset = SARCLD2024Dataset(root_dir=_dummy_ds_path, split='train', transform=None, img_size=cfg['img_size'], num_classes=cfg['num_classes'], class_names=None, train_split_ratio=cfg.get('train_split_ratio',0.8))
        class_names = _dummy_dataset.class_names if hasattr(_dummy_dataset, 'class_names') and _dummy_dataset.class_names else [f"C{i}" for i in range(cfg['num_classes'])]
        del _dummy_dataset
    except Exception as e_dummy:
        logger_main.warning(f"Could not infer class_names from dummy dataset ({e_dummy}). Using generic names: C0, C1, ...")
        class_names = [f"C{i}" for i in range(cfg['num_classes'])]
    logger_main.info(f"Class names ({len(class_names)} total): {class_names}")

    current_img_size = tuple(cfg['img_size'])
    if cfg.get('progressive_resizing_main_loop_enabled'): # This flag would control complex main loop resizing
        logger_main.warning("Main loop progressive resizing is configured but NOT YET FULLY IMPLEMENTED in this script. Current image size will be fixed as per cfg['img_size'] unless trainer internally resizes.")

    train_loader, val_loader, test_loader = get_data_loaders(cfg, current_img_size, class_names, dataset_base_path=cfg['data_root'])
    if not train_loader or not val_loader: logger_main.error("Train/Val DataLoaders failed. Exiting."); return
    steps_per_epoch = len(train_loader)

    model_initial_state_dict = None
    resume_finetune_path = cfg.get('resume_from_checkpoint')
    ssl_path = cfg.get('pretrained_checkpoint_path')

    if not (resume_finetune_path and os.path.exists(resume_finetune_path)):
        if cfg.get('load_pretrained_backbone', True) and ssl_path and os.path.exists(ssl_path):
            logger_main.info(f"No finetune checkpoint to resume, or path invalid. Will attempt to load SSL backbone: {ssl_path}")
            ssl_checkpoint = torch.load(ssl_path, map_location='cpu')
            model_initial_state_dict = ssl_checkpoint.get('model_state', ssl_checkpoint.get('state_dict', ssl_checkpoint.get('model', ssl_checkpoint)))
            if model_initial_state_dict is None:
                 logger_main.warning(f"Could not find model state in SSL checkpoint {ssl_path} under common keys.")
        elif cfg.get('load_pretrained_backbone', True): # load_pretrained_backbone is True but path is invalid/missing
            logger_main.warning(f"SSL backbone loading configured, but path '{ssl_path}' not found or not specified. Training from scratch.")
        else: # Not resuming and not loading SSL backbone
            logger_main.info("Training from scratch (no resume, no SSL backbone specified or enabled).")

    model, optimizer, scheduler = setup_model_optimizer_scheduler(
        cfg, device, cfg['num_classes'], steps_per_epoch,
        model_initial_state_dict
    )

    class_weights_for_loss = get_class_weights_for_loss(train_loader.dataset, device, cfg) if train_loader else None
    criterion = CombinedLoss(num_classes=cfg['num_classes'],
                             smoothing=cfg.get('loss_label_smoothing', 0.1),
                             focal_alpha=cfg.get('focal_loss_alpha', 0.25),
                             focal_gamma=cfg.get('focal_loss_gamma', 2.0),
                             ce_weight=cfg.get('loss_weights',{}).get('ce_weight',0.5),
                             focal_weight=cfg.get('loss_weights',{}).get('focal_weight',0.5),
                             class_weights_tensor=class_weights_for_loss).to(device)
    logger_main.info(f"Criterion: CombinedLoss (CE_w={criterion.ce_weight:.2f}, Focal_w={criterion.focal_weight:.2f}, LS={cfg.get('loss_label_smoothing',0.1):.2f}), ClassWeights: {'Enabled' if class_weights_for_loss is not None else 'Disabled'}")

    scaler = GradScaler(enabled=cfg.get('amp_enabled', True))
    ema_model = EMA(model, decay=cfg.get('ema_decay',0.9999), warmup_steps=cfg.get('ema_warmup_steps', 0)) if cfg.get('use_ema') else None
    if ema_model: logger_main.info(f"EMA enabled (decay={cfg.get('ema_decay',0.9999)}, warmup_steps={cfg.get('ema_warmup_steps',0)}).")

    trainer_input_size = current_img_size[0]
    trainer_max_input_size = current_img_size[0]
    # This logic is for trainer's internal progressive resizing, if enabled via trainer_use_progressive_resize
    if cfg.get('trainer_use_progressive_resize', False) and 'progressive_resize_schedule' in cfg:
        all_prog_sizes = [s[0] for s in cfg.get('progressive_resize_schedule', {0:current_img_size}).values()]
        if all_prog_sizes: trainer_max_input_size = max(all_prog_sizes)

    finetuner = EnhancedFinetuner(
        model=model, optimizer=optimizer, criterion=criterion, device=device, scaler=scaler, scheduler=scheduler,
        lr_scheduler_on_batch=(cfg.get('scheduler','').lower() == 'onecyclelr'),
        accumulation_steps=cfg.get('accumulation_steps',1),
        clip_grad_norm=cfg.get('clip_grad_norm'),
        num_classes=cfg['num_classes'],
        tta_enabled_val=cfg.get('tta_enabled_val',False),
        use_enhanced_tta=cfg.get('use_enhanced_tta_trainer', True), # Configurable TTA in trainer
        debug_nan_detection=cfg.get('debug_nan_detection',False),
        stop_on_nan_threshold=cfg.get('stop_on_nan_threshold',5),
        monitor_gradients=cfg.get('monitor_gradients',False),
        gradient_log_interval=cfg.get('gradient_log_interval',50),
        mixup_alpha=cfg.get('mixup_alpha',0.0),
        cutmix_alpha=cfg.get('cutmix_alpha',0.0),
        mixup_prob=cfg.get('mixup_prob',0.5) if cfg.get('mixup_alpha',0.0)>0 else 0.0,
        cutmix_prob=cfg.get('cutmix_prob',0.5) if cfg.get('cutmix_alpha',0.0)>0 else 0.0,
        label_smoothing=cfg.get('loss_label_smoothing',0.0),
        confidence_penalty=cfg.get('trainer_confidence_penalty', 0.0),
        use_progressive_resize=cfg.get('trainer_use_progressive_resize', False),
        input_size=trainer_input_size,
        max_input_size=trainer_max_input_size,
        swa_enabled=cfg.get('use_swa',False),
        swa_start_epoch=cfg.get('swa_start_epoch', cfg['epochs'] - max(1, int(cfg['epochs']*0.2)) ),
        multiscale_training=cfg.get('trainer_multiscale_training_enabled', False),
        scale_range=tuple(cfg.get('trainer_multiscale_scale_range', [int(trainer_input_size*0.75), int(trainer_input_size*1.25)]))
    )
    finetuner.augmentations = None # Dataset's transform handles augmentations

    if 'dropout_schedule' in cfg and logger_main:
        logger_main.warning("Config contains 'dropout_schedule', but dynamic application is not implemented in this script.")

    start_epoch = 0
    metric_to_monitor = cfg.get('metric_to_monitor_early_stopping', 'f1_macro')
    effective_metric_key_for_val_dict = metric_to_monitor.replace('val_', '')
    best_val_metric = -float('inf') if "loss" not in metric_to_monitor.lower() else float('inf')
    epochs_no_improve = 0
    history = defaultdict(list)

    if resume_finetune_path and os.path.exists(resume_finetune_path):
        logger_main.info(f"Attempting to load full finetuner state from: {resume_finetune_path}")
        start_epoch, best_val_metric = finetuner.load_checkpoint(resume_finetune_path, effective_metric_key_for_val_dict)
        try:
            chkpt_content_ext = torch.load(resume_finetune_path, map_location='cpu')
            epochs_no_improve = chkpt_content_ext.get('epochs_no_improve', 0)
            if ema_model and 'ema_state_dict' in chkpt_content_ext and chkpt_content_ext['ema_state_dict']:
                try:
                    ema_model.shadow = chkpt_content_ext['ema_state_dict']['shadow']
                    ema_model.num_updates = chkpt_content_ext['ema_state_dict']['num_updates']
                    for name_sh in ema_model.shadow:
                        if name_sh in dict(model.named_parameters()):
                            ref_param = dict(model.named_parameters())[name_sh]
                            ema_model.shadow[name_sh] = ema_model.shadow[name_sh].to(device=ref_param.device, dtype=ref_param.dtype)
                    logger_main.info("Resumed EMA state from extended checkpoint data.")
                except Exception as e_ema_load: logger_main.warning(f"Could not load EMA state from extended data: {e_ema_load}")
            resumed_hist = chkpt_content_ext.get('history')
            if resumed_hist: history = defaultdict(list, resumed_hist); logger_main.info("Resumed training history from extended checkpoint data.")
            del chkpt_content_ext
        except Exception as e_load_ext:
            logger_main.warning(f"Could not load extended checkpoint data (history, EMA) from {resume_finetune_path}: {e_load_ext}")
        logger_main.info(f"Resumed. Start epoch: {start_epoch}, Best '{effective_metric_key_for_val_dict}': {best_val_metric:.4f}, Epochs no improve: {epochs_no_improve}")
    else:
        logger_main.info(f"No valid resume_from_checkpoint path ('{resume_finetune_path}') found or specified. Starting fresh or from SSL backbone.")

    for epoch in range(start_epoch, cfg['epochs']):
        logger_main.info(f"--- Epoch {epoch}/{cfg['epochs']-1} ---")
        progressive_unfreezing_hvt(model, epoch, cfg)

        train_loss, nan_train_flag = finetuner.train_one_epoch(train_loader, epoch, cfg['epochs'])
        if nan_train_flag:
            logger_main.error(f"NaNs encountered during training in epoch {epoch}. Training stopped."); break
        history['train_loss'].append(train_loss)
        # Train metrics are not computed per epoch by default in this setup to save time.
        # You can add a validation-like pass on train data if needed.
        history['train_accuracy'].append(0.0) # Placeholder
        history[f"train_{effective_metric_key_for_val_dict}"].append(0.0) # Placeholder

        use_swa_for_current_val = cfg.get('use_swa',False) and epoch >= cfg.get('swa_start_epoch', cfg['epochs'])
        
        if ema_model and not use_swa_for_current_val: # Apply EMA to base model if not using SWA for validation
            ema_model.apply_shadow()

        val_loss, val_metrics_dict = finetuner.validate_one_epoch(val_loader, class_names, use_swa=use_swa_for_current_val)
        
        if ema_model and not use_swa_for_current_val: # Restore base model if EMA was applied
            ema_model.restore()
        
        history['val_loss'].append(val_loss)
        current_val_metric_value = val_metrics_dict.get(effective_metric_key_for_val_dict, -float('inf'))
        for m_key, m_val in val_metrics_dict.items(): history[f"val_{m_key}"].append(m_val)

        if scheduler and not finetuner.lr_scheduler_on_batch: # Epoch-wise schedulers
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(current_val_metric_value)
            else:
                scheduler.step()
        current_lr_log = optimizer.param_groups[0]['lr']; history['lr'].append(current_lr_log)
        logger_main.info(f"Epoch {epoch} Val: Loss={val_loss:.4f}, Monitored '{effective_metric_key_for_val_dict}'={current_val_metric_value:.4f}. LR: {current_lr_log:.2e}")

        if ema_model: ema_model.update()

        min_improvement_delta = cfg.get('min_delta_early_stopping', 1e-5)
        is_better_metric = False
        if "loss" in metric_to_monitor.lower():
            is_better_metric = current_val_metric_value < best_val_metric - min_improvement_delta
        else:
            is_better_metric = current_val_metric_value > best_val_metric + min_improvement_delta

        if is_better_metric:
            best_val_metric = current_val_metric_value; epochs_no_improve = 0
            best_model_save_path = os.path.join(checkpoints_save_dir, cfg['best_model_filename'])
            finetuner.save_checkpoint(best_model_save_path, epoch, best_val_metric, metric_to_monitor)
            checkpoint_data_to_add = {
                'ema_state_dict': {'shadow': ema_model.shadow, 'num_updates': ema_model.num_updates} if ema_model else None,
                'history': dict(history), 'config_runtime': cfg, 'epochs_no_improve': epochs_no_improve,
                'class_names': class_names, 'val_metrics_at_best': val_metrics_dict
            }
            try:
                current_checkpoint_content = torch.load(best_model_save_path); current_checkpoint_content.update(checkpoint_data_to_add); torch.save(current_checkpoint_content, best_model_save_path)
                logger_main.info(f"Best model extended and saved to {best_model_save_path} (Epoch {epoch}, Best {metric_to_monitor}: {best_val_metric:.4f})")
            except Exception as e_save_best: logger_main.error(f"Error extending best checkpoint: {e_save_best}")
        else:
            epochs_no_improve += 1

        save_periodic_freq = cfg.get('save_checkpoint_every_n_epochs',0)
        if save_periodic_freq > 0 and (epoch + 1) % save_periodic_freq == 0:
            periodic_ckpt_path = os.path.join(checkpoints_save_dir, f"checkpoint_epoch_{epoch}.pth")
            finetuner.save_checkpoint(periodic_ckpt_path, epoch, current_val_metric_value, metric_to_monitor)
            periodic_data_to_add = {
                'ema_state_dict': {'shadow': ema_model.shadow, 'num_updates': ema_model.num_updates} if ema_model else None,
                'history': dict(history), 'config_runtime': cfg, 'epochs_no_improve': epochs_no_improve,
                'overall_best_val_metric': best_val_metric, 'class_names': class_names,
                'val_metrics_current_epoch': val_metrics_dict
            }
            try:
                periodic_ckpt_content = torch.load(periodic_ckpt_path); periodic_ckpt_content.update(periodic_data_to_add); torch.save(periodic_ckpt_content, periodic_ckpt_path)
                logger_main.info(f"Periodic checkpoint extended and saved to {periodic_ckpt_path}")
            except Exception as e_save_periodic: logger_main.error(f"Error extending periodic checkpoint: {e_save_periodic}")

        if (epoch + 1) % cfg.get('plot_every_n_epochs', 5) == 0 or (epoch + 1) == cfg['epochs']:
             plot_training_curves(history, output_dir, metric_to_monitor)

        early_stopping_patience = cfg.get('early_stopping_patience', 0)
        if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
            logger_main.info(f"Early stopping triggered at epoch {epoch} after {epochs_no_improve} epochs with no improvement on '{metric_to_monitor}'. Best val metric: {best_val_metric:.4f}."); break

    logger_main.info("Training finished.")
    if cfg.get('use_swa',False) and val_loader:
        finetuner.finalize_swa(val_loader); logger_main.info("SWA model BN statistics finalized.")
    plot_training_curves(history, output_dir, metric_to_monitor)

    if test_loader:
        logger_main.info("--- Running Final Test ---")
        test_model_to_use = None
        test_model_source_info = "Unknown (Error in logic)"

        if cfg.get('use_swa', False) and finetuner.swa_enabled:
            logger_main.info("Using SWA model for final test.")
            test_model_to_use = finetuner.get_model_for_inference(use_swa=True).to(device)
            test_model_source_info = "SWA Model"
        else:
            best_model_final_path = os.path.join(checkpoints_save_dir, cfg['best_model_filename'])
            if os.path.exists(best_model_final_path):
                logger_main.info(f"Loading best model from {best_model_final_path} for testing.")
                chkpt_test = torch.load(best_model_final_path, map_location=device)
                
                test_img_size_cfg = chkpt_test.get('config_runtime', {}).get('img_size', cfg['img_size'])
                test_img_size = tuple(test_img_size_cfg) if isinstance(test_img_size_cfg, list) else test_img_size_cfg

                hvt_params_test = cfg.get('hvt_params_for_model_init',{});
                hvt_params_test.update({'img_size':test_img_size, 'num_classes':cfg['num_classes']})
                if 'spectral_channels' not in hvt_params_test: hvt_params_test['spectral_channels'] = 0

                _temp_test_model = create_disease_aware_hvt(**hvt_params_test).to(device)
                _temp_test_model.load_state_dict(chkpt_test['model_state_dict'])
                test_model_to_use = _temp_test_model
                test_model_source_info = "Best Checkpoint (Base Model)"

                if cfg.get('use_ema',False) and chkpt_test.get('ema_state_dict') and chkpt_test['ema_state_dict']['shadow']:
                    logger_main.info("Applying EMA weights from best checkpoint for testing.")
                    ema_for_test = EMA(test_model_to_use, decay=cfg.get('ema_decay',0.9999))
                    ema_for_test.shadow = chkpt_test['ema_state_dict']['shadow']
                    for name_sh_test in ema_for_test.shadow:
                        if name_sh_test in dict(test_model_to_use.named_parameters()):
                            ref_param_test = dict(test_model_to_use.named_parameters())[name_sh_test]
                            ema_for_test.shadow[name_sh_test] = ema_for_test.shadow[name_sh_test].to(device=ref_param_test.device, dtype=ref_param_test.dtype)
                    ema_for_test.apply_shadow()
                    test_model_source_info = "Best Checkpoint (EMA Applied)"
            else:
                logger_main.warning("No best model checkpoint found. Using last state of trainer's base model (if available).")
                test_model_to_use = finetuner.get_model_for_inference(use_swa=False).to(device)
                test_model_source_info = "Trainer's Last Base Model State"
                if cfg.get('use_ema',False) and ema_model:
                    ema_model.apply_shadow()
                    test_model_source_info += " (Current EMA Applied)"

        if test_model_to_use:
            logger_main.info(f"Testing with model source: {test_model_source_info}")
            original_trainer_model_ref_for_test = finetuner.model
            finetuner.model = test_model_to_use
            finetuner.model.eval()

            _, test_metrics_results = finetuner.validate_one_epoch(test_loader, class_names, use_swa=False)
            
            finetuner.model = original_trainer_model_ref_for_test
            if cfg.get('use_ema',False) and ema_model and "Current EMA Applied" in test_model_source_info:
                ema_model.restore()

            logger_main.info(f"Final Test Results: Accuracy={test_metrics_results.get('accuracy',0):.4f}, F1_Macro={test_metrics_results.get('f1_macro',0):.4f}, F1_Weighted={test_metrics_results.get('f1_weighted',0):.4f}")
            if 'confusion_matrix' in test_metrics_results:
                plot_confusion_matrix_main(test_metrics_results['confusion_matrix'], class_names, output_dir, "test_confusion_matrix.png")
            
            with open(os.path.join(output_dir, "test_metrics_summary.txt"), 'w') as f_report_final:
                f_report_final.write(f"Tested Model Source: {test_model_source_info}\n")
                for k_met_final, v_met_final in test_metrics_results.items():
                    if k_met_final != 'confusion_matrix': f_report_final.write(f"{k_met_final}: {v_met_final}\n")
            logger_main.info(f"Test metrics summary saved to {os.path.join(output_dir, 'test_metrics_summary.txt')}")
        else:
            logger_main.error("No model could be prepared for final testing.")
    else:
        logger_main.info("Test loader not available. Skipping final test.")

    logger_main.info(f"Process completed. All results, logs, and checkpoints are in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune HVT for Cotton Leaf Diseases - High Performance V2")
    parser.add_argument('--user_config_yaml', type=str, help="Path to user's YAML (overrides default config from module)")
    parser.add_argument('--data_root_override', type=str, help="Override data_root path")
    parser.add_argument('--output_dir_base_override', type=str, help="Override base for output_dir (PACKAGE_ROOT_PATH in config)")
    parser.add_argument('--resume_checkpoint_override', type=str, nargs='?', const=None, help="Override resume_from_checkpoint path. No arg or empty string disables resume.")
    parser.add_argument('--epochs_override', type=int, help="Override number of epochs")
    parser.add_argument('--batch_size_override', type=int, help="Override batch size")
    parser.add_argument('--lr_override', type=float, help="Override onecycle_max_lr (if using OneCycleLR)")

    args = parser.parse_args()

    cfg = main_config_from_phase4.copy() # Start with default config from the imported .config module
    
    # Apply user YAML if provided
    if args.user_config_yaml:
        try:
            with open(args.user_config_yaml, 'r') as f: user_yaml_cfg = yaml.safe_load(f)
            cfg.update(user_yaml_cfg) # User YAML overrides defaults from module config
            print(f"Applied user config from YAML: {args.user_config_yaml}")
        except FileNotFoundError:
            print(f"Warning: User YAML config file '{args.user_config_yaml}' not found. Using module defaults or other CLI overrides.")
        except Exception as e_yaml:
            print(f"Error loading user YAML config '{args.user_config_yaml}': {e_yaml}. Using module defaults or other CLI overrides.")

    # Apply CLI overrides (these take highest precedence)
    if args.data_root_override: cfg['data_root'] = args.data_root_override
    if args.output_dir_base_override: cfg['PACKAGE_ROOT_PATH'] = args.output_dir_base_override
    if args.resume_checkpoint_override is not None: # Check if arg was present at all
        cfg['resume_from_checkpoint'] = args.resume_checkpoint_override # Value can be path or None
    if args.epochs_override is not None: cfg['epochs'] = args.epochs_override
    if args.batch_size_override is not None: cfg['batch_size'] = args.batch_size_override
    if args.lr_override is not None: cfg['onecycle_max_lr'] = args.lr_override # Specific to OneCycleLR

    # Sanity checks for common issues
    if cfg['device'] == 'cpu' and cfg.get('num_workers',0) > 0:
        if cfg.get('num_workers',0) > 1 :
             print(f"Warning: num_workers ({cfg.get('num_workers',0)}) > 1 for CPU training. This can sometimes lead to slowdowns or errors depending on the system. Consider setting to 0 or 1 for CPU.")
    if cfg.get('num_workers',0) == 0 and cfg.get('prefetch_factor'):
        cfg['prefetch_factor'] = None

    main(cfg) # Pass the fully processed config dictionary