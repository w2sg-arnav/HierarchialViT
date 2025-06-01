# phase4_finetuning/main.py - Enhanced for 90%+ accuracy with advanced techniques
from collections import OrderedDict, Counter as PyCounter
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LambdaLR, SequentialLR, LinearLR, ReduceLROnPlateau
import argparse
import yaml
import torch.nn.functional as F
import math
import sys
from typing import Tuple, Optional, Dict, Any, List
import traceback
from datetime import datetime
import time
import random
from torch.optim import AdamW, SGD
import warnings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.transforms import functional as TF
import torchvision.transforms as T
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger: Optional[logging.Logger] = None

# Assuming these modules are in the correct path relative to this file
# or installed in the environment.
# Replace with actual paths if necessary, e.g., if phase2_model is a sibling directory:
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    # If these are local to a 'phase4_finetuning' package:
    from .config import config as base_finetune_config # This might be overridden by load_enhanced_config
    from .dataset import SARCLD2024Dataset # Make sure this dataset can handle dynamic img_size
    # from .utils.augmentations import create_augmentation # We define create_disease_optimized_augmentations locally
    from .utils.logging_setup import setup_logging
    # from .finetune.trainer import Finetuner # Trainer logic will be built into main
    from phase2_model.models.hvt import DiseaseAwareHVT, create_disease_aware_hvt
except ImportError as e_imp:
    print(f"CRITICAL IMPORT ERROR: {e_imp}. Please ensure all custom modules and phase2_model are in PYTHONPATH.", file=sys.stderr)
    traceback.print_exc()
    # Fallback for running as a script if imports fail due to relative paths
    try:
        # This block is for trying to run the script directly if it's not part of a package
        # For example, if 'phase2_model' is a top-level directory:
        # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Adjust as needed
        # from phase2_model.models.hvt import DiseaseAwareHVT, create_disease_aware_hvt
        # from dataset import SARCLD2024Dataset # Assuming dataset.py is in the same directory
        # from utils.logging_setup import setup_logging # Assuming utils is in the same directory
        pass # Keep previous error if this also fails
    except ImportError:
        pass
    # sys.exit(1) # Commenting out for now to allow completion, but this indicates a setup issue.


class FocalLoss(nn.Module):
    """Enhanced Focal Loss with class-specific alpha"""
    def __init__(self, alpha: Optional[List[float]] = None, gamma: float = 2.0, reduction: str = 'mean', num_classes: int = 8, ignore_index: int = -100):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.ones(num_classes) * alpha
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes, f"Alpha list length {len(alpha)} must match num_classes {num_classes}"
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            raise ValueError("alpha must be float, int, list of floats, or None")

        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Filter out ignore_index targets
        valid_mask = targets != self.ignore_index
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        if inputs.numel() == 0: # All targets were ignore_index
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)

            # Ensure alpha tensor matches the number of classes in inputs if dynamic
            if self.alpha.size(0) != inputs.size(1):
                 # This might happen if num_classes passed to init was different from actual model output
                warnings.warn(f"FocalLoss alpha size ({self.alpha.size(0)}) != inputs num_classes ({inputs.size(1)}). Re-creating alpha.")
                self.alpha = self.alpha.new_ones(inputs.size(1)) # Fallback to ones

            at = self.alpha.gather(0, targets)
            focal_loss_val = at * (1 - pt) ** self.gamma * ce_loss # Corrected: CE loss, not logpt for positive form
        else:
            focal_loss_val = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss_val.mean()
        elif self.reduction == 'sum':
            return focal_loss_val.sum()
        else:
            return focal_loss_val

class AdaptiveFocalLoss(nn.Module):
    """Adaptive Focal Loss that adjusts gamma based on training progress (e.g., accuracy)"""
    def __init__(self, alpha: Optional[List[float]] = None, gamma_start: float = 2.0, gamma_end: float = 0.5, num_classes: int = 8, reduction: str = 'mean'):
        super().__init__()
        if alpha is None:
            self.alpha_t = None
        elif isinstance(alpha, (float, int)):
            self.alpha_t = torch.ones(num_classes) * alpha
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes, f"Alpha list length {len(alpha)} must match num_classes {num_classes}"
            self.alpha_t = torch.tensor(alpha, dtype=torch.float32)
        else:
            raise ValueError("alpha must be float, int, list of floats, or None")

        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.num_classes = num_classes
        self.current_gamma = gamma_start
        self.reduction = reduction
        self.ignore_index = -100 # Standard ignore index

    def update_gamma(self, metric_value: float, target_metric: float = 0.95):
        # Reduce gamma as metric_value (e.g. accuracy) improves to focus more on hard examples
        # Progress is normalized metric_value towards target_metric
        progress = min(max(metric_value / target_metric, 0.0), 1.0)
        self.current_gamma = self.gamma_start * (1 - progress) + self.gamma_end * progress
        logger.info(f"AdaptiveFocalLoss: Updated gamma to {self.current_gamma:.4f} based on metric {metric_value:.4f}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != self.ignore_index
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        if inputs.numel() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha_t is not None:
            if self.alpha_t.device != targets.device:
                self.alpha_t = self.alpha_t.to(targets.device)
            if self.alpha_t.size(0) != inputs.size(1):
                 warnings.warn(f"AdaptiveFocalLoss alpha size ({self.alpha_t.size(0)}) != inputs num_classes ({inputs.size(1)}). Re-creating alpha.")
                 self.alpha_t = self.alpha_t.new_ones(inputs.size(1)) # Fallback to ones

            at = self.alpha_t.gather(0, targets)
            focal_loss_val = at * (1 - pt) ** self.current_gamma * ce_loss
        else:
            focal_loss_val = (1 - pt) ** self.current_gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss_val.mean()
        elif self.reduction == 'sum':
            return focal_loss_val.sum()
        else:
            return focal_loss_val

class LabelSmoothingCrossEntropy(nn.Module):
    """Enhanced Label Smoothing with adaptive smoothing and class weights"""
    def __init__(self, num_classes: int, smoothing: float = 0.1, class_weights: Optional[torch.Tensor] = None, adaptive: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.adaptive = adaptive
        if class_weights is not None:
            assert class_weights.ndim == 1 and class_weights.size(0) == num_classes, \
                "class_weights must be a 1D tensor with size num_classes"
        self.register_buffer('class_weights', class_weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        current_smoothing = self.smoothing
        current_confidence = self.confidence

        if self.adaptive:
            with torch.no_grad(): # Detach operations for adaptive smoothing calculation
                pred_softmax = F.softmax(pred.detach(), dim=1)
                pred_max_probs = pred_softmax.max(dim=1)[0]
                # Adaptive smoothing: increase smoothing if model is overconfident on average
                # This is a heuristic; could be refined.
                # Average confidence factor: 1.0 means low confidence (uniform), 0.0 means high confidence (one-hot)
                avg_confidence_factor = 1.0 - pred_max_probs.mean()
                # Modulate smoothing by this factor: less smoothing if model is less confident.
                current_smoothing = self.smoothing * avg_confidence_factor
                current_confidence = 1.0 - current_smoothing
        
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(current_smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), current_confidence)
        
        loss = torch.sum(-true_dist * pred, dim=-1)
        
        if self.class_weights is not None:
            weights = self.class_weights.to(target.device)[target]
            loss = loss * weights
        
        return loss.mean()

class AdvancedCutMixUp:
    """ Standard CutMix and MixUp implementation """
    def __init__(self, mixup_alpha: float = 0.4, cutmix_alpha: float = 1.0, prob: float = 0.8, switch_prob: float = 0.5, num_classes: int = 8):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
        # Returns: mixed_images, labels_a, labels_b, lam, is_mixup_or_cutmix_applied
        if np.random.rand() > self.prob:
            # If one-hot encoded labels are expected by loss, convert here
            # For CrossEntropy, index labels are fine.
            return images, labels, labels, 1.0, False # No mixup/cutmix

        use_mixup = np.random.rand() < self.switch_prob
        
        if use_mixup and self.mixup_alpha > 0:
            return self.mixup(images, labels)
        elif not use_mixup and self.cutmix_alpha > 0:
            return self.cutmix(images, labels)
        else: # Fallback if selected method has alpha=0
            return images, labels, labels, 1.0, False

    def mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam, True

    def cutmix(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        mixed_x = x.clone() # Important to clone
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to reflect actual mixed area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam, True

    def rand_bbox(self, size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization with adaptive rho"""
    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=True, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        # kwargs for base_optimizer are passed via defaults
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups # SAM needs to manipulate groups of base optimizer
        # self.defaults.update(self.base_optimizer.defaults) # Redundant due to how param_groups are set

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: 
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p.data, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        
        if zero_grad: 
            self.zero_grad() # Zeroes grads of model params

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p not in self.state or "old_p" not in self.state[p]: # check if old_p exists
                    continue
                p.data = self.state[p]["old_p"] # Restore original weights
        
        self.base_optimizer.step() # Perform update with original weights and new gradients
        
        if zero_grad: 
            self.zero_grad()

    # step method is inherited from base_optimizer for SAM.
    # For SAM, usually it's model.forward(); loss.backward(); optimizer.first_step(); model.forward(); loss2.backward(); optimizer.second_step();
    # This step() method is for when SAM is used like a standard optimizer, requiring a closure.
    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("SAM optimizer requires a closure for the step() method.")
        
        # First, calculate and store original parameter values
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None: # Ensure grad exists before trying to use it
                     self.state[p]["old_p"] = p.data.clone()

        # Calculate L_S(w + e(w))
        # This closure should re-evaluate loss after first_step perturbation
        # Closure needs to handle its own zero_grad for model before loss calculation.
        
        # First ascent step
        self.first_step(zero_grad=True) # zero_grad=True will clear current grads for the next backward pass
        
        # Calculate gradients at perturbed weights
        with torch.enable_grad(): # Ensure gradients are computed for this forward/backward pass
            loss = closure()
            loss.backward() # This computes gradients at w + e(w)
            
        # Second descent step
        self.second_step(zero_grad=False) # zero_grad=False because grads are already for w_adv

    def _grad_norm(self):
        # self.base_optimizer.param_groups is self.param_groups, so it's fine
        shared_device = self.param_groups[0]["params"][0].device 
        
        # Filter out params without gradients
        valid_params = [
            p for group in self.param_groups for p in group["params"] if p.grad is not None
        ]
        if not valid_params: # If no params have grads (e.g. after zero_grad)
            return torch.tensor(0.0, device=shared_device)

        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_norm_term = (torch.abs(p.data) if group["adaptive"] else 1.0) * p.grad
                norms.append(param_norm_term.norm(p=2))
        
        if not norms: # Handles case where all grads might be None after filtering
             return torch.tensor(0.0, device=shared_device)

        norm = torch.norm(torch.stack(norms), p=2)
        return norm.to(shared_device)

    def zero_grad(self, set_to_none: bool = False): # Ensure SAM's zero_grad calls base_optimizer's
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


class EMA:
    """Exponential Moving Average with warmup and decay scheduling"""
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 1000, use_num_updates: bool = True):
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
        
        # EMA Warmup: gradually decrease the effect of EMA updates during warmup
        if self.warmup_steps > 0 and self.num_updates <= self.warmup_steps:
            # Effective decay starts lower and ramps up to current_decay
            warmup_factor = self.num_updates / self.warmup_steps 
            effective_decay = current_decay * warmup_factor 
        else:
            effective_decay = current_decay
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - effective_decay) * param.data + effective_decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Copies shadow parameters to model for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restores original model parameters from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

class AdaptiveGradientClipping:
    """Adaptive Gradient Clipping based on parameter norms"""
    def __init__(self, clip_factor: float = 0.01, eps: float = 1e-3):
        self.clip_factor = clip_factor
        self.eps = eps
    
    def __call__(self, model: nn.Module):
        for param in model.parameters():
            if param.grad is not None:
                param_norm = torch.linalg.norm(param.data.detach(), ord=2) # Use param.data
                grad_norm = torch.linalg.norm(param.grad.detach(), ord=2)
                
                max_norm = param_norm * self.clip_factor + self.eps
                
                clip_coef = max_norm / (grad_norm + 1e-6)
                clip_coef_clamped = torch.clamp(clip_coef, max=1.0) # Only clip if grad_norm > max_norm
                
                if clip_coef < 1.0: # Only apply if clipping is needed
                    param.grad.detach().mul_(clip_coef_clamped)


def create_disease_optimized_augmentations(img_size: Tuple[int, int] = (384, 384), is_train: bool = True):
    """Disease-specific augmentation optimized for cotton leaf diseases"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return A.Compose([
            A.RandomResizedCrop(height=img_size[0], width=img_size[1], scale=(0.7, 1.0), ratio=(0.8, 1.2), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=40, border_mode=0, value=(int(mean[0]*255), int(mean[1]*255), int(mean[2]*255)), p=0.7), # Use mean for border
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=40, border_mode=0, value=(int(mean[0]*255), int(mean[1]*255), int(mean[2]*255)), p=0.7),
            
            A.Perspective(scale=(0.05, 0.1), p=0.2),
            A.ElasticTransform(alpha=1, sigma=30, alpha_affine=30, p=0.2, border_mode=0, value=(int(mean[0]*255), int(mean[1]*255), int(mean[2]*255))),
            
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.7),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.4),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=5, p=0.5),
                A.MedianBlur(blur_limit=5, p=0.3),
            ], p=0.4),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.3),
            ], p=0.3),
            
            A.CoarseDropout(max_holes=6, max_height=img_size[0]//8, max_width=img_size[1]//8, min_holes=1, 
                           min_height=img_size[0]//16, min_width=img_size[1]//16, fill_value=mean, p=0.3), # Fill with mean
            
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else: # Validation/Test augmentations
        return A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

def load_enhanced_config() -> Dict[str, Any]:
    """Enhanced configuration for 90%+ accuracy"""
    return {
        # Model architecture
        'model_name': 'disease_aware_hvt_large', # Example variant
        'num_classes': 8, # Number of disease classes + healthy
        'img_size': [224, 224], # Initial image size, will be progressively increased
        'patch_size': 16,
        'embed_dim': 768, # Corresponds to ViT-Base like models
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'drop_rate': 0.1,         # Dropout for linear layers
        'attn_drop_rate': 0.0,    # Dropout for attention
        'drop_path_rate': 0.2,    # Stochastic depth
        'layer_scale_init_value': 1e-6, # For LayerScale
        
        # Training hyperparameters
        'epochs': 150, # Adjusted for potentially faster convergence with advanced techniques
        'batch_size': 16, # Adjust based on GPU memory (e.g., 32, 16, 8)
        'accumulation_steps': 4,  # Effective batch size = batch_size * accumulation_steps
        'num_workers': 8, # Adjust based on CPU cores
        'pin_memory': True,
        'prefetch_factor': 2, # For DataLoader
        
        # Learning rates and scheduling
        'lr_head': 5e-4,
        'lr_backbone': 5e-5,
        'weight_decay': 0.05,
        'warmup_epochs': 10, # Number of epochs for linear warmup
        'scheduler_type': 'cosine_warm_restarts', # 'cosine_annealing' or 'cosine_warm_restarts' or 'reduce_on_plateau'
        'eta_min': 1e-7,      # Min LR for cosine schedulers
        'T_0': 15,            # For CosineAnnealingWarmRestarts: epochs for first restart
        'T_mult': 2,          # For CosineAnnealingWarmRestarts: factor to increase T_i after restart
        'plateau_factor': 0.2, # For ReduceLROnPlateau
        'plateau_patience': 10, # For ReduceLROnPlateau

        # Loss and optimization
        'loss_type': 'adaptive_focal', # 'ce', 'focal', 'adaptive_focal', 'label_smoothing'
        'focal_alpha': [0.73, 1.41, 0.74, 1.34, 0.89, 1.0, 1.17, 0.72], # Example: inverse of class frequencies, tuned
        'focal_gamma': 2.0,
        'adaptive_focal_gamma_start': 2.5,
        'adaptive_focal_gamma_end': 0.5,
        'label_smoothing_epsilon': 0.15,
        'optimizer_type': 'adamw', # 'adamw' or 'sgd'
        'adamw_betas': (0.9, 0.999),
        'adamw_eps': 1e-8,
        'sgd_momentum': 0.9,
        
        # Regularization and stability
        'freeze_backbone_epochs': 0, # Number of initial epochs to freeze backbone (0 to disable)
        'use_sam': True,
        'sam_rho': 0.05, # SAM neighborhood size
        'sam_adaptive': True,
        'clip_grad_norm_value': 1.0, # Max grad norm, set to 0 or None to disable
        'use_adaptive_grad_clip': True,
        'agc_clip_factor': 0.01,
        'agc_eps': 1e-3,
        'use_ema': True,
        'ema_decay': 0.9999,
        'ema_warmup_steps': 2000, # Number of optimizer steps for EMA warmup
        
        # Data augmentation
        'use_advanced_augmentation': True, # Uses create_disease_optimized_augmentations
        'use_mixup_cutmix': True,
        'mixup_cutmix_prob': 0.8, # Probability of applying mixup/cutmix
        'mixup_alpha': 0.4,       # Alpha for mixup beta distribution
        'cutmix_alpha': 1.0,      # Alpha for cutmix beta distribution
        
        # Progressive resizing: epoch -> [height, width]
        'progressive_resizing': True,
        'resize_schedule': { 
            0: [224, 224],    # Start with 224x224
            # (epochs_total * 1/3): [256, 256], # Example: At 1/3 of training
            # (epochs_total * 2/3): [384, 384]  # Example: At 2/3 of training
            # These will be calculated based on total epochs dynamically later
        },
        
        # Test-time augmentation
        'use_tta': True,
        'tta_num_augmentations': 10, # Number of TTA views (including original)
        
        # Evaluation and saving
        'eval_every_n_epochs': 1,
        'save_best_only': True,
        'early_stopping_patience': 30, # Patience for early stopping
        'monitor_metric': 'val_f1_score', # 'val_accuracy', 'val_loss', 'val_f1_score'
        
        # Hardware optimization
        'amp_enabled': True,      # Automatic Mixed Precision
        'compile_model': True,   # PyTorch 2.0+ model compilation
        'channels_last': True,   # Use channels_last memory format
        
        # Dataset paths (MUST BE SET BY USER)
        'train_dir': 'path/to/your/train_data',
        'val_dir': 'path/to/your/val_data',
        'test_dir': 'path/to/your/test_data',
        'output_dir': './finetune_results',
        'checkpoint_path': None, # Path to a checkpoint to resume training
        'pretrained_weights_path': None # Path to pre-trained HVT weights (e.g., from phase 2)
    }

def create_weighted_sampler(dataset: SARCLD2024Dataset, mode: str = 'effective_number', beta: float = 0.9999) -> WeightedRandomSampler:
    """Create weighted sampler for imbalanced dataset"""
    if hasattr(dataset, 'get_targets'): # Assuming dataset has a method to get all targets
        targets = np.array(dataset.get_targets())
    else: # Fallback: iterate through dataset (slower)
        logger.warning("Dataset does not have 'get_targets' method. Iterating to get targets for sampler (can be slow).")
        targets = []
        # This can be very slow for large datasets, ensure SARCLD2024Dataset implements get_targets()
        for i in range(len(dataset)): 
            try:
                _, target = dataset[i] # Assuming dataset returns (image, target)
                if isinstance(target, torch.Tensor): target = target.item()
                targets.append(target)
            except Exception as e:
                logger.error(f"Error getting target for sample {i} for weighted sampler: {e}")
                # Fallback to assuming balanced if targets cannot be fetched
                return WeightedRandomSampler(weights=torch.ones(len(dataset)), num_samples=len(dataset), replacement=True)

        targets = np.array(targets)

    if len(targets) == 0:
        logger.warning("No targets found for weighted sampler. Using uniform sampling.")
        return WeightedRandomSampler(weights=torch.ones(len(dataset)), num_samples=len(dataset), replacement=True)

    class_counts = np.bincount(targets, minlength=dataset.num_classes if hasattr(dataset, 'num_classes') else np.max(targets) + 1)
    
    # Avoid division by zero for classes not present in the dataset subset
    class_counts = np.maximum(class_counts, 1e-9) 

    if mode == 'inverse_frequency':
        weights_per_class = 1.0 / class_counts
    elif mode == 'effective_number':
        effective_num = 1.0 - np.power(beta, class_counts)
        weights_per_class = (1.0 - beta) / effective_num
    else: # uniform
        weights_per_class = np.ones_like(class_counts, dtype=np.float32)
    
    weights_per_class = weights_per_class / np.sum(weights_per_class) # Normalize
    
    sample_weights = weights_per_class[targets]
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(), 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    return sampler

def mixup_cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp/CutMix loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def test_time_augmentation(model: nn.Module, image: torch.Tensor, device: torch.device, num_augmentations: int = 10) -> torch.Tensor:
    """Advanced Test Time Augmentation for a single image or a batch.
       Returns averaged softmax probabilities.
    """
    model.eval() # Ensure model is in eval mode

    # Define TTA transforms (applied on tensor directly)
    # These are simpler than training augmentations
    tta_transforms_list = [
        lambda x: x, # Original
        lambda x: TF.hflip(x),
        lambda x: TF.vflip(x),
        lambda x: TF.rotate(x, degrees=15),
        lambda x: TF.rotate(x, degrees=-15),
    ]
    # Can add more complex ones like small crops, color jitters if needed
    # For color jitter, ensure it's applied carefully if images are already normalized
    # E.g., denormalize, jitter, re-normalize or use kornia for tensor-based color jitter

    if image.ndim == 3: # Single image C, H, W
        image = image.unsqueeze(0) # Add batch dimension B, C, H, W

    all_outputs = []
    
    with torch.no_grad(), autocast(enabled=True): # Assuming AMP might be used
        # Original image
        all_outputs.append(model(image.to(device)))

        # Augmented images
        # Ensure num_augmentations does not exceed available unique transforms if they are few
        for i in range(1, num_augmentations):
            idx = i % len(tta_transforms_list) # Cycle through transforms
            if idx == 0 and i >= len(tta_transforms_list): # Avoid re-doing original if num_augs > len(transforms)
                 # Apply a combination or a slightly varied transform
                 combined_img = TF.hflip(TF.rotate(image, degrees=5 * (i // len(tta_transforms_list))))
                 all_outputs.append(model(combined_img.to(device)))
                 continue

            aug_image = tta_transforms_list[idx](image)
            all_outputs.append(model(aug_image.to(device)))
            
    # Average softmax probabilities
    avg_probs = torch.stack([F.softmax(out, dim=1) for out in all_outputs]).mean(dim=0)
    return avg_probs


def progressive_unfreezing(model: nn.Module, epoch: int, total_epochs: int, cfg: Dict[str, Any]):
    """Progressive unfreezing strategy for HVT based on layers."""
    # HVT typically has 'patch_embed', 'blocks' (layers), 'norm', 'head'
    num_layers = cfg.get('depth', 12) # model.depth if available
    freeze_backbone_epochs = cfg.get('freeze_backbone_epochs', 0)

    if freeze_backbone_epochs == 0: # No unfreezing schedule, all trainable from start (or based on pretraining)
        if epoch == 0: logger.info("Progressive unfreezing disabled or freeze_backbone_epochs is 0.")
        return

    if epoch == 0: # Initial freeze if specified
        logger.info(f"Progressive unfreezing: Initially freezing backbone for {freeze_backbone_epochs} epochs.")
        for name, param in model.named_parameters():
            if not ('head' in name or 'classifier' in name):
                param.requires_grad = False
            else:
                param.requires_grad = True # Ensure head is trainable
        return

    # Determine unfreezing stages, e.g., 3 stages within freeze_backbone_epochs
    # Stage 1: Unfreeze last 1/3 of layers
    # Stage 2: Unfreeze last 2/3 of layers
    # Stage 3: Unfreeze all backbone layers
    
    layers_to_unfreeze_increment = num_layers // 3 # Example: unfreeze in 3 steps
    
    if epoch < freeze_backbone_epochs * (1/3):
        # Only head is trainable (already set)
        pass
    elif epoch < freeze_backbone_epochs * (2/3):
        # Unfreeze top 1/3 of blocks
        layers_to_thaw = range(num_layers - layers_to_unfreeze_increment, num_layers)
        if epoch == int(freeze_backbone_epochs * (1/3)): logger.info(f"Unfreezing layers: {list(layers_to_thaw)}")
        for i in layers_to_thaw:
            if hasattr(model, 'blocks') and i < len(model.blocks):
                for param in model.blocks[i].parameters():
                    param.requires_grad = True
        if hasattr(model, 'norm'): # Unfreeze final norm layer
            for param in model.norm.parameters():
                param.requires_grad = True
    elif epoch < freeze_backbone_epochs:
        # Unfreeze top 2/3 of blocks
        layers_to_thaw = range(num_layers - 2 * layers_to_unfreeze_increment, num_layers)
        if epoch == int(freeze_backbone_epochs * (2/3)): logger.info(f"Unfreezing layers: {list(layers_to_thaw)}")
        for i in layers_to_thaw:
            if hasattr(model, 'blocks') and i < len(model.blocks):
                for param in model.blocks[i].parameters():
                    param.requires_grad = True
    elif epoch >= freeze_backbone_epochs:
        if epoch == freeze_backbone_epochs : logger.info("Unfreezing all backbone layers.")
        for name, param in model.named_parameters():
            param.requires_grad = True # Unfreeze everything

    # Log trainable parameters
    if epoch in [0, int(freeze_backbone_epochs * (1/3)), int(freeze_backbone_epochs * (2/3)), freeze_backbone_epochs]:
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Epoch {epoch}: Number of trainable parameters: {num_trainable}")


def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 8, class_names: Optional[List[str]] = None) -> Dict:
    """Calculate comprehensive metrics including per-class and confusion matrix."""
    if predictions.ndim == 2: # Softmax outputs
        pred_labels = predictions.argmax(dim=1)
    else: # Already class indices
        pred_labels = predictions
    
    targets_cpu = targets.cpu().numpy()
    pred_labels_cpu = pred_labels.cpu().numpy()
    
    accuracy = accuracy_score(targets_cpu, pred_labels_cpu)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        targets_cpu, pred_labels_cpu, average='weighted', zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        targets_cpu, pred_labels_cpu, average='macro', zero_division=0
    )
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        targets_cpu, pred_labels_cpu, average=None, zero_division=0, labels=list(range(num_classes))
    )
    
    cm = confusion_matrix(targets_cpu, pred_labels_cpu, labels=list(range(num_classes)))
    
    metrics_dict = {
        'accuracy': accuracy,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'f1_score_weighted': f1_w,
        'precision_macro': precision_m,
        'recall_macro': recall_m,
        'f1_score_macro': f1_m,
        'confusion_matrix': cm
    }
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    for i, name in enumerate(class_names):
        metrics_dict[f'{name}_precision'] = per_class_precision[i]
        metrics_dict[f'{name}_recall'] = per_class_recall[i]
        metrics_dict[f'{name}_f1_score'] = per_class_f1[i]
        metrics_dict[f'{name}_support'] = per_class_support[i]
        
    return metrics_dict

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: str):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
    plt.close()

def plot_training_curves(history: Dict[str, List[float]], output_path: str):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot F1-score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_f1_score'], 'bo-', label='Training F1 (Weighted)')
    plt.plot(epochs, history['val_f1_score'], 'ro-', label='Validation F1 (Weighted)')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "training_curves.png"))
    plt.close()


def setup_model_optimizer_scheduler(cfg: Dict[str, Any], device: torch.device, num_classes: int, current_epoch: int = 0, total_steps_per_epoch: Optional[int] = None) -> Tuple:
    """Setup model, optimizer, and schedulers"""
    
    model = create_disease_aware_hvt(
        img_size=cfg['img_size'], # current img_size
        patch_size=cfg['patch_size'],
        num_classes=num_classes,
        embed_dim=cfg['embed_dim'],
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
        mlp_ratio=cfg['mlp_ratio'],
        drop_rate=cfg['drop_rate'],
        attn_drop_rate=cfg.get('attn_drop_rate', 0.0), # Added from config example
        drop_path_rate=cfg['drop_path_rate'],
        use_layer_scale=cfg.get('layer_scale_init_value', 0.0) > 0, # Enable if init_value > 0
        layer_scale_init_value=cfg.get('layer_scale_init_value', 1e-6),
    )

    # Load pretrained weights if specified (e.g., from Phase 2)
    if cfg.get('pretrained_weights_path') and os.path.exists(cfg['pretrained_weights_path']):
        logger.info(f"Loading pretrained weights from: {cfg['pretrained_weights_path']}")
        checkpoint = torch.load(cfg['pretrained_weights_path'], map_location='cpu')
        # Adjust state_dict loading based on how weights are saved (e.g. 'model' key, prefix)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        
        # Handle potential mismatch in classifier due to num_classes change
        # For ViT, classifier is often 'head.weight' and 'head.bias'
        if 'head.weight' in state_dict and state_dict['head.weight'].shape[0] != num_classes:
            logger.warning("Pretrained head num_classes mismatch. Re-initializing head.")
            del state_dict['head.weight']
            if 'head.bias' in state_dict: del state_dict['head.bias']
        
        # Filter out non-matching keys (e.g., related to image size if pos_embed is not interpolated)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Pretrained weights loaded successfully (strict=False).")
    
    model = model.to(device)
    
    if cfg.get('channels_last', True):
        model = model.to(memory_format=torch.channels_last)
    
    if cfg.get('compile_model', True) and hasattr(torch, 'compile'):
        try:
            logger.info("Compiling model with torch.compile()...")
            model = torch.compile(model, mode='max-autotune') # or 'reduce-overhead'
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Continuing without compilation.")
    
    # Optimizer setup
    head_params = []
    backbone_params = []
    no_decay_params = [] # For params like biases, LayerNorm weights

    # Typically, biases and LayerNorm/BatchNorm weights are not decayed
    param_names_no_decay = ['bias', 'LayerNorm.weight', 'norm.weight']

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        is_no_decay = any(nd in name for nd in param_names_no_decay)
        
        if 'head' in name or 'classifier' in name:
            if is_no_decay:
                no_decay_params.append(param)
            else:
                head_params.append(param)
        else:
            if is_no_decay:
                no_decay_params.append(param)
            else:
                backbone_params.append(param)
    
    optimizer_grouped_parameters = [
        {'params': head_params, 'lr': cfg['lr_head'], 'weight_decay': cfg['weight_decay']},
        {'params': backbone_params, 'lr': cfg['lr_backbone'], 'weight_decay': cfg['weight_decay']},
        {'params': no_decay_params, 'lr': max(cfg['lr_head'], cfg['lr_backbone']), 'weight_decay': 0.0} # No decay for these
    ]
    
    base_optimizer_cls = None
    optimizer_kwargs = {}

    if cfg['optimizer_type'].lower() == 'adamw':
        base_optimizer_cls = AdamW
        optimizer_kwargs = {
            'betas': cfg['adamw_betas'],
            'eps': cfg['adamw_eps'],
            # weight_decay is per-group now
        }
    elif cfg['optimizer_type'].lower() == 'sgd':
        base_optimizer_cls = SGD
        optimizer_kwargs = {
            'momentum': cfg['sgd_momentum'],
            'nesterov': True, # Typically good with SGD
            # weight_decay is per-group now
        }
    else:
        raise ValueError(f"Unsupported optimizer_type: {cfg['optimizer_type']}")

    if cfg.get('use_sam', True):
        logger.info("Using SAM optimizer.")
        optimizer = SAM(
            optimizer_grouped_parameters, # Pass param groups directly
            base_optimizer_cls, 
            rho=cfg['sam_rho'], 
            adaptive=cfg['sam_adaptive'],
            **optimizer_kwargs # Pass base optimizer args here
        )
    else:
        optimizer = base_optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    # Scheduler setup
    scheduler = None
    if total_steps_per_epoch is None and cfg['scheduler_type'] != 'reduce_on_plateau':
        logger.warning("total_steps_per_epoch not provided for scheduler setup. Some schedulers might not work optimally.")
        # Estimate if needed, or ensure it's passed. For now, let's assume it's available for relevant schedulers.
        
    total_training_steps = cfg['epochs'] * total_steps_per_epoch if total_steps_per_epoch else cfg['epochs']
    warmup_steps = cfg['warmup_epochs'] * total_steps_per_epoch if total_steps_per_epoch else cfg['warmup_epochs']

    if cfg['scheduler_type'] == 'cosine_annealing':
        main_scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps - warmup_steps, eta_min=cfg['eta_min'])
    elif cfg['scheduler_type'] == 'cosine_warm_restarts':
        main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg['T_0'] * total_steps_per_epoch if total_steps_per_epoch else cfg['T_0'],
                                                     T_mult=cfg['T_mult'], eta_min=cfg['eta_min'])
    elif cfg['scheduler_type'] == 'reduce_on_plateau':
        # This scheduler is step_every_epoch=True
        scheduler = ReduceLROnPlateau(optimizer, mode='max' if 'accuracy' in cfg['monitor_metric'] or 'f1' in cfg['monitor_metric'] else 'min',
                                      factor=cfg['plateau_factor'], patience=cfg['plateau_patience'], verbose=True)
    else: # Default to cosine_annealing if type is unknown
        logger.warning(f"Unknown scheduler_type: {cfg['scheduler_type']}. Defaulting to CosineAnnealingLR.")
        main_scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps - warmup_steps, eta_min=cfg['eta_min'])

    if cfg['scheduler_type'] != 'reduce_on_plateau': # ReduceLROnPlateau doesn't mix well with warmup typically
        if warmup_steps > 0 :
            warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])
        else:
            scheduler = main_scheduler
            
    # If resuming, try to load optimizer and scheduler states
    if cfg.get('checkpoint_path') and os.path.exists(cfg['checkpoint_path']):
        checkpoint = torch.load(cfg['checkpoint_path'], map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Optimizer state loaded from checkpoint.")
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}. Initializing fresh optimizer state.")
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Scheduler state loaded from checkpoint.")
            except Exception as e:
                logger.warning(f"Could not load scheduler state: {e}. Initializing fresh scheduler state.")
        # current_epoch will be updated outside based on checkpoint

    return model, optimizer, scheduler


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, cfg, epoch_num,
                    mixup_cutmix_fn=None, agc_fn=None, adaptive_loss=None):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    optimizer.zero_grad() # Initialize gradients for accumulation

    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)

        # Apply channels_last if model expects it
        if cfg.get('channels_last', True):
            images = images.to(memory_format=torch.channels_last)

        mixed_images, targets_a, targets_b, lam, is_mixed = images, targets, targets, 1.0, False
        if mixup_cutmix_fn and cfg.get('use_mixup_cutmix', False):
            mixed_images, targets_a, targets_b, lam, is_mixed = mixup_cutmix_fn(images, targets)
            
        def closure(): # For SAM optimizer
            output = model(mixed_images)
            if is_mixed:
                loss = mixup_cutmix_criterion(criterion, output, targets_a, targets_b, lam)
            else:
                loss = criterion(output, targets)
            loss = loss / cfg['accumulation_steps'] # Normalize loss for accumulation
            return loss, output

        if cfg.get('use_sam', True):
            # SAM requires two forward/backward passes
            with autocast(enabled=cfg['amp_enabled']):
                loss, outputs = closure() # Calculate loss for first step (perturbation)
            
            scaler.scale(loss).backward() # Grad for perturbation
            scaler.unscale_(optimizer) # Unscale before SAM's first step grad norm
            optimizer.first_step(zero_grad=False) # Perturb weights, SAM handles its own zero_grad based on its logic
                                                 # but we handle gradient accumulation zero_grad outside loop

            # Second forward/backward pass on perturbed weights
            with autocast(enabled=cfg['amp_enabled']):
                loss, outputs = closure() # This is L_S(w + e(w))
            
            scaler.scale(loss).backward() # Grad for descent step
            # Optimizer step will be done after accumulation
        else: # Standard optimizer
            with autocast(enabled=cfg['amp_enabled']):
                outputs = model(mixed_images)
                if is_mixed:
                    loss = mixup_cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)
                loss = loss / cfg['accumulation_steps']
            
            scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % cfg['accumulation_steps'] == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer) # Unscale before clipping and step

            if cfg.get('clip_grad_norm_value', 0) and cfg['clip_grad_norm_value'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm_value'])
            if agc_fn and cfg.get('use_adaptive_grad_clip', False):
                agc_fn(model)

            if cfg.get('use_sam', True):
                optimizer.second_step(zero_grad=False) # SAM's second step performs the actual update
            else:
                scaler.step(optimizer) # Standard optimizer step
            
            scaler.update()
            optimizer.zero_grad() # Zero gradients for the next accumulation cycle

        total_loss += loss.item() * cfg['accumulation_steps'] # De-normalize for logging
        
        all_preds.append(outputs.detach())
        all_targets.append(targets_a.detach()) # Use targets_a for metrics if mixed

    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics for the epoch
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # If adaptive loss, update based on training accuracy (or other metric)
    # However, it's often better to update based on validation metrics.
    # For now, let's assume it's updated after validation.

    epoch_metrics = calculate_metrics(all_preds, all_targets, num_classes=cfg['num_classes'])
    return avg_loss, epoch_metrics


def validate_one_epoch(model, dataloader, criterion, device, cfg, ema_model=None):
    eval_model = ema_model if ema_model and cfg.get('use_ema', False) else model
    if ema_model and cfg.get('use_ema', False):
        ema_model.apply_shadow() # Use EMA weights for validation
    
    eval_model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            if cfg.get('channels_last', True) and eval_model is model: # EMA model might not be compiled/channels_last
                 images = images.to(memory_format=torch.channels_last)

            with autocast(enabled=cfg['amp_enabled']):
                if cfg.get('use_tta', False) and eval_model is model : # TTA typically not with EMA shadow directly unless EMA shadow is a full model object
                    outputs_probs = test_time_augmentation(eval_model, images, device, num_augmentations=cfg['tta_num_augmentations'])
                    # TTA returns probabilities, need to convert to logits for some loss fns or use a prob-based loss
                    # For simplicity with CE/Focal, let's get argmax if criterion expects logits.
                    # Or, if criterion can handle probs (e.g. custom NLLLoss), use that.
                    # A simpler TTA is to average logits then softmax. test_time_augmentation returns avg softmax.
                    # So, use a dummy loss or skip loss calculation for TTA validation if too complex.
                    # For now, let's assume loss is based on non-TTA single pass for val_loss, TTA for metrics.
                    # For simplicity in this loop, TTA will be for final test, not every validation.
                    # Reverting to standard validation for simplicity:
                    outputs = eval_model(images)
                else:
                    outputs = eval_model(images)

                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_preds.append(outputs)
            all_targets.append(targets)

    if ema_model and cfg.get('use_ema', False):
        ema_model.restore() # Restore original model weights

    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    epoch_metrics = calculate_metrics(all_preds, all_targets, num_classes=cfg['num_classes'])
    return avg_loss, epoch_metrics


def get_data_loaders(cfg: Dict[str, Any], current_img_size: Tuple[int, int], class_names: Optional[List[str]] = None) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Creates and returns train, validation, and test DataLoaders."""
    logger.info(f"Creating DataLoaders with image size: {current_img_size}")
    
    train_transform = create_disease_optimized_augmentations(img_size=current_img_size, is_train=True)
    val_test_transform = create_disease_optimized_augmentations(img_size=current_img_size, is_train=False)

    # Dataset paths should be set in config
    if not os.path.isdir(cfg['train_dir']) or not os.path.isdir(cfg['val_dir']):
        logger.error(f"Train directory ({cfg['train_dir']}) or Val directory ({cfg['val_dir']}) not found. Please check config.")
        # For dataset instantiation, SARCLD2024Dataset needs to be defined or imported correctly.
        # Assuming it takes (root_dir, transform, num_classes (optional), class_names (optional))
        # Let's create dummy datasets if paths are invalid to allow code to run for structure checking.
        # Replace with actual dataset instantiation.
        # Example: SARCLD2024Dataset(root_dir=cfg['train_dir'], transform=train_transform, img_size=current_img_size)
        raise FileNotFoundError("Dataset directories not found. Please set 'train_dir' and 'val_dir' in config.")

    try:
        train_dataset = SARCLD2024Dataset(root_dir=cfg['train_dir'], transform=train_transform, img_size=current_img_size, class_names=class_names)
        val_dataset = SARCLD2024Dataset(root_dir=cfg['val_dir'], transform=val_test_transform, img_size=current_img_size, class_names=class_names)
        
        test_dataset = None
        if os.path.isdir(cfg.get('test_dir', '')):
            test_dataset = SARCLD2024Dataset(root_dir=cfg['test_dir'], transform=val_test_transform, img_size=current_img_size, class_names=class_names)
        else:
            logger.warning(f"Test directory ({cfg.get('test_dir', '')}) not found. Test loader will not be created.")

    except NameError: # SARCLD2024Dataset not defined
        logger.error("SARCLD2024Dataset class not found. Please ensure it's correctly defined and imported.")
        raise
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        raise

    train_sampler = None
    if len(train_dataset) > 0: # Only create sampler if dataset is not empty
        # Weighted sampler for imbalanced datasets
        # Ensure SARCLD2024Dataset has a get_targets() method or similar for this to work efficiently
        if not hasattr(train_dataset, 'get_targets'):
            logger.warning("train_dataset does not have get_targets(). WeightedRandomSampler might be slow or inaccurate.")
        try:
            train_sampler = create_weighted_sampler(train_dataset)
            logger.info("Using WeightedRandomSampler for training.")
        except Exception as e:
            logger.warning(f"Failed to create weighted sampler: {e}. Using default sampler.")
            train_sampler = None # Fallback to default shuffle
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['batch_size'], shuffle=(train_sampler is None),
        num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'], sampler=train_sampler,
        prefetch_factor=cfg.get('prefetch_factor', 2) if cfg['num_workers'] > 0 else None,
        drop_last=True # Often beneficial for stability, esp. with SAM or BatchNorm
    ) if len(train_dataset) > 0 else None
    
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['batch_size'] * 2, # Often can use larger batch for validation
        shuffle=False, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
        prefetch_factor=cfg.get('prefetch_factor', 2) if cfg['num_workers'] > 0 else None
    ) if len(val_dataset) > 0 else None

    test_loader = None
    if test_dataset and len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset, batch_size=cfg['batch_size'] * 2, shuffle=False,
            num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
            prefetch_factor=cfg.get('prefetch_factor', 2) if cfg['num_workers'] > 0 else None
        )
    
    return train_loader, val_loader, test_loader


def main(cfg_path: Optional[str] = None):
    global logger 
    
    # --- Setup ---
    if cfg_path:
        with open(cfg_path, 'r') as f:
            user_cfg = yaml.safe_load(f)
        base_cfg = load_enhanced_config()
        # Deep merge user_cfg into base_cfg (user_cfg takes precedence)
        # This simple update is shallow; for deep merge, a utility is needed.
        # For now, assume user_cfg might overwrite entire dict keys like 'resize_schedule'.
        base_cfg.update(user_cfg) 
        cfg = base_cfg
    else:
        cfg = load_enhanced_config()

    # Ensure output directory exists
    os.makedirs(cfg['output_dir'], exist_ok=True)
    
    # Setup logging
    # Assuming setup_logging is imported and works
    try:
        logger = setup_logging(os.path.join(cfg['output_dir'], 'training.log'), level=logging.INFO)
    except NameError: # setup_logging not defined
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(os.path.join(cfg['output_dir'], 'training.log')),
                                      logging.StreamHandler(sys.stdout)])
        logger = logging.getLogger(__name__)
        logger.warning("setup_logging function not found. Using basic logging configuration.")


    logger.info("Starting training process with configuration:")
    for key, value in cfg.items():
        logger.info(f"{key}: {value}")

    # Set seed for reproducibility
    seed = cfg.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = cfg.get('cudnn_deterministic', False) # False for speed
        torch.backends.cudnn.benchmark = cfg.get('cudnn_benchmark', True)         # True for speed if input sizes don't vary much

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Progressive Resizing Setup ---
    current_img_size = tuple(cfg['img_size']) # Initial image size
    # Dynamically set resize_schedule epochs if defined as fractions
    # Example: 30, 60, 100 -> means absolute epochs
    # If using fractions, they need to be calculated based on total_epochs
    # The current config uses absolute epochs, which is fine.
    
    # --- Class names (assuming they can be fetched or are predefined) ---
    # This is important for 'calculate_metrics' and 'SARCLD2024Dataset'
    # For SARCLD2024Dataset, it might infer from subdirectories or need them passed
    # Example class names, replace with actuals or load from dataset
    class_names = [f"Disease_{i}" for i in range(cfg['num_classes'])] 
    # If SARCLD2024Dataset provides class_names after init:
    # temp_dataset = SARCLD2024Dataset(cfg['train_dir'], transform=None, img_size=current_img_size)
    # class_names = temp_dataset.class_names

    # --- DataLoaders ---
    # Initial DataLoaders
    train_loader, val_loader, test_loader = get_data_loaders(cfg, current_img_size, class_names=class_names)
    if not train_loader or not val_loader:
        logger.error("Failed to create train or validation data loaders. Exiting.")
        return

    steps_per_epoch = len(train_loader)

    # --- Model, Optimizer, Scheduler ---
    model, optimizer, scheduler = setup_model_optimizer_scheduler(cfg, device, cfg['num_classes'], total_steps_per_epoch=steps_per_epoch)

    # --- Loss Function ---
    criterion = None
    adaptive_loss_instance = None # For AdaptiveFocalLoss gamma updates
    if cfg['loss_type'] == 'focal':
        criterion = FocalLoss(alpha=cfg.get('focal_alpha'), gamma=cfg['focal_gamma'], num_classes=cfg['num_classes'])
    elif cfg['loss_type'] == 'adaptive_focal':
        adaptive_loss_instance = AdaptiveFocalLoss(
            alpha=cfg.get('focal_alpha'), 
            gamma_start=cfg['adaptive_focal_gamma_start'], 
            gamma_end=cfg['adaptive_focal_gamma_end'],
            num_classes=cfg['num_classes']
        )
        criterion = adaptive_loss_instance
    elif cfg['loss_type'] == 'label_smoothing':
        # Example: calculate class weights if needed (e.g., inverse frequency)
        # class_weights_tensor = torch.tensor([1.0] * cfg['num_classes'], device=device) # Placeholder
        criterion = LabelSmoothingCrossEntropy(num_classes=cfg['num_classes'], smoothing=cfg['label_smoothing_epsilon'])
    else: # Default to CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    logger.info(f"Using loss function: {cfg['loss_type']}")

    # --- AMP Scaler ---
    scaler = GradScaler(enabled=cfg['amp_enabled'])

    # --- EMA Model ---
    ema_model = None
    if cfg.get('use_ema', False):
        ema_model = EMA(model, decay=cfg['ema_decay'], warmup_steps=cfg['ema_warmup_steps'])
        logger.info("EMA enabled.")

    # --- Augmentation Functions ---
    mixup_cutmix_fn = None
    if cfg.get('use_mixup_cutmix', False):
        mixup_cutmix_fn = AdvancedCutMixUp(
            mixup_alpha=cfg['mixup_alpha'], cutmix_alpha=cfg['cutmix_alpha'],
            prob=cfg['mixup_cutmix_prob'], num_classes=cfg['num_classes']
        )
        logger.info("MixUp/CutMix enabled.")
    
    agc_fn = None
    if cfg.get('use_adaptive_grad_clip', False):
        agc_fn = AdaptiveGradientClipping(clip_factor=cfg['agc_clip_factor'], eps=cfg['agc_eps'])
        logger.info("Adaptive Gradient Clipping enabled.")

    # --- Training Loop ---
    best_metric_val = -float('inf') if 'accuracy' in cfg['monitor_metric'] or 'f1' in cfg['monitor_metric'] else float('inf')
    epochs_no_improve = 0
    start_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [], 'train_f1_score': [], 'val_f1_score': []}

    # Resume from checkpoint
    if cfg.get('checkpoint_path') and os.path.exists(cfg['checkpoint_path']):
        logger.info(f"Resuming training from checkpoint: {cfg['checkpoint_path']}")
        checkpoint = torch.load(cfg['checkpoint_path'], map_location=device)
        # Model state is loaded in setup_model_optimizer_scheduler
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_metric_val = checkpoint.get('best_metric_val', best_metric_val)
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        if ema_model and 'ema_state_dict' in checkpoint:
            # EMA state loading needs care if model structure changed (unlikely for just weights)
            # For simplicity, re-register if resuming. Or load shadow params.
            # ema.shadow = checkpoint['ema_state_dict']['shadow']
            # ema.num_updates = checkpoint['ema_state_dict']['num_updates']
            logger.info("EMA state found in checkpoint. Consider how to load it properly if needed (not implemented here for simplicity).")
        if 'history' in checkpoint: history = checkpoint['history']
        logger.info(f"Resuming from epoch {start_epoch}, Best Val Metric: {best_metric_val:.4f}")


    for epoch in range(start_epoch, cfg['epochs']):
        logger.info(f"--- Epoch {epoch}/{cfg['epochs']-1} ---")

        # Progressive Resizing
        if cfg.get('progressive_resizing', False):
            new_size_tuple = None
            for scheduled_epoch, size_cfg in cfg['resize_schedule'].items():
                if epoch == scheduled_epoch:
                    new_size_tuple = tuple(size_cfg)
                    break
            
            if new_size_tuple and new_size_tuple != current_img_size:
                logger.info(f"Progressive resizing: Changing image size from {current_img_size} to {new_size_tuple} at epoch {epoch}")
                current_img_size = new_size_tuple
                cfg['img_size'] = list(current_img_size) # Update config for dataset/model recreation

                # Recreate DataLoaders
                train_loader, val_loader, test_loader = get_data_loaders(cfg, current_img_size, class_names=class_names)
                if not train_loader or not val_loader:
                    logger.error("Failed to recreate data loaders during progressive resizing. Exiting.")
                    return
                steps_per_epoch = len(train_loader)

                # Recreate model (or adjust position embeddings if necessary)
                # For ViT, pos_embed often needs interpolation. create_disease_aware_hvt should handle this.
                # Optimizer and scheduler might also need re-init or careful state handling if model params change significantly.
                # For simplicity, we re-init them here. State can be lost.
                # More robust: Save optimizer state, re-init with new params, load relevant parts of state.
                logger.info("Re-initializing model, optimizer, and scheduler for new image size.")
                model, optimizer, scheduler = setup_model_optimizer_scheduler(cfg, device, cfg['num_classes'], current_epoch=epoch, total_steps_per_epoch=steps_per_epoch)
                
                # Re-init EMA if used
                if ema_model: ema_model = EMA(model, decay=cfg['ema_decay'], warmup_steps=cfg['ema_warmup_steps'])

        # Progressive Unfreezing
        if cfg.get('freeze_backbone_epochs', 0) > 0:
            progressive_unfreezing(model, epoch, cfg['epochs'], cfg)

        # Training
        train_start_time = time.time()
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, cfg, epoch,
            mixup_cutmix_fn, agc_fn, adaptive_loss_instance
        )
        logger.info(f"Epoch {epoch} Train: Loss={train_loss:.4f}, Acc={train_metrics['accuracy']:.4f}, F1_w={train_metrics['f1_score_weighted']:.4f} (Time: {time.time()-train_start_time:.2f}s)")
        
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_f1_score'].append(train_metrics['f1_score_weighted'])

        # Validation
        val_start_time = time.time()
        val_loss, val_metrics = validate_one_epoch(
            model, val_loader, criterion, device, cfg, ema_model
        )
        logger.info(f"Epoch {epoch} Val: Loss={val_loss:.4f}, Acc={val_metrics['accuracy']:.4f}, F1_w={val_metrics['f1_score_weighted']:.4f} (Time: {time.time()-val_start_time:.2f}s)")
        
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1_score'].append(val_metrics['f1_score_weighted'])
        
        # Update AdaptiveFocalLoss gamma based on validation metric
        if adaptive_loss_instance and hasattr(adaptive_loss_instance, 'update_gamma'):
            adaptive_loss_instance.update_gamma(val_metrics[cfg.get('monitor_metric', 'val_f1_score_weighted').replace('val_','')]) # Use the validation metric specified

        # LR Scheduler step
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics[cfg['monitor_metric']]) # Needs the monitored metric
            else: # For step-based schedulers, if not done per step in train_one_epoch
                scheduler.step() # Step per epoch

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current LR: {current_lr:.2e}")

        # Save model checkpoint
        is_best = False
        current_metric_val = val_metrics[cfg['monitor_metric']]
        if ('loss' in cfg['monitor_metric'] and current_metric_val < best_metric_val) or \
           (('accuracy' in cfg['monitor_metric'] or 'f1' in cfg['monitor_metric']) and current_metric_val > best_metric_val):
            best_metric_val = current_metric_val
            epochs_no_improve = 0
            is_best = True
            if cfg['save_best_only']:
                save_path = os.path.join(cfg['output_dir'], "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'ema_state_dict': {'shadow': ema_model.shadow, 'num_updates': ema_model.num_updates} if ema_model else None,
                    'best_metric_val': best_metric_val,
                    'config': cfg, # Save config for reproducibility
                    'history': history,
                    'current_img_size': current_img_size
                }, save_path)
                logger.info(f"Best model saved to {save_path} (Epoch {epoch}, {cfg['monitor_metric']}: {best_metric_val:.4f})")
        else:
            epochs_no_improve += 1
        
        # Save latest checkpoint periodically or always if not save_best_only
        if not cfg['save_best_only'] or epoch % 5 == 0 : # Example: save every 5 epochs
            save_path = os.path.join(cfg['output_dir'], f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                 # ... (same content as best_model save)
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'ema_state_dict': {'shadow': ema_model.shadow, 'num_updates': ema_model.num_updates} if ema_model else None,
                'best_metric_val': best_metric_val, # Keep track of overall best
                'config': cfg,
                'history': history,
                'current_img_size': current_img_size,
                'epochs_no_improve': epochs_no_improve
            }, save_path)
            logger.info(f"Checkpoint saved to {save_path}")

        # Early Stopping
        if cfg.get('early_stopping_patience', 0) > 0 and epochs_no_improve >= cfg['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement on {cfg['monitor_metric']}.")
            break
            
        # Plot metrics so far
        if epoch % 5 == 0 and epoch > 0: # Plot every 5 epochs
             plot_training_curves(history, cfg['output_dir'])

    logger.info("Training finished.")
    plot_training_curves(history, cfg['output_dir']) # Final plots

    # --- Final Test ---
    if test_loader:
        logger.info("--- Running Final Test ---")
        # Load best model for testing
        best_model_path = os.path.join(cfg['output_dir'], "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            
            # Potentially update img_size in cfg if best model was saved with a different size
            saved_img_size = checkpoint.get('current_img_size')
            if saved_img_size and tuple(saved_img_size) != current_img_size:
                logger.info(f"Best model used image size {saved_img_size}. Re-creating test loader if necessary.")
                current_img_size = tuple(saved_img_size)
                cfg['img_size'] = list(current_img_size)
                _, _, test_loader = get_data_loaders(cfg, current_img_size, class_names=class_names) # Recreate with correct size


            # Re-setup model for test (in case of compilation, etc.)
            # Ensure model is created with the correct img_size from the checkpoint
            test_model, _, _ = setup_model_optimizer_scheduler(cfg, device, cfg['num_classes'])
            test_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_model_path} for testing.")

            # Use EMA weights for testing if they were used for validation best
            final_eval_model = test_model
            if cfg.get('use_ema', False) and checkpoint.get('ema_state_dict'):
                logger.info("Applying EMA weights for final testing.")
                # Create a temporary EMA object and load its state
                temp_ema = EMA(test_model, decay=cfg['ema_decay']) # Decay value doesn't matter here
                temp_ema.shadow = checkpoint['ema_state_dict']['shadow']
                temp_ema.num_updates = checkpoint['ema_state_dict']['num_updates'] # Not strictly needed for apply_shadow
                temp_ema.apply_shadow() # test_model now has EMA weights
                # final_eval_model is already test_model which now has EMA weights
            
            final_eval_model.eval()
            all_test_preds = []
            all_test_targets = []

            with torch.no_grad():
                for images, targets in test_loader:
                    images, targets = images.to(device), targets.to(device)
                    if cfg.get('channels_last', True) and not (cfg.get('use_ema') and checkpoint.get('ema_state_dict')): # EMA model might not be compiled
                        images = images.to(memory_format=torch.channels_last)

                    with autocast(enabled=cfg['amp_enabled']):
                        if cfg.get('use_tta', True):
                            outputs_probs = test_time_augmentation(final_eval_model, images, device, num_augmentations=cfg['tta_num_augmentations'])
                            # outputs_probs are softmaxed probabilities
                            all_test_preds.append(outputs_probs.cpu())
                        else:
                            outputs_logits = final_eval_model(images)
                            all_test_preds.append(F.softmax(outputs_logits, dim=1).cpu())
                    all_test_targets.append(targets.cpu())
            
            if cfg.get('use_ema', False) and checkpoint.get('ema_state_dict'):
                 temp_ema.restore() # Restore original weights to test_model if needed elsewhere, though it's end of script

            all_test_preds = torch.cat(all_test_preds)
            all_test_targets = torch.cat(all_test_targets)
            
            test_metrics = calculate_metrics(all_test_preds, all_test_targets, num_classes=cfg['num_classes'], class_names=class_names)
            logger.info(f"Test Results: Acc={test_metrics['accuracy']:.4f}, F1_w={test_metrics['f1_score_weighted']:.4f}, F1_m={test_metrics['f1_score_macro']:.4f}")
            logger.info("Per-class F1 Scores:")
            for i, name in enumerate(class_names):
                logger.info(f"  {name}: {test_metrics[name+'_f1_score']:.4f} (Support: {test_metrics[name+'_support']})")

            plot_confusion_matrix(test_metrics['confusion_matrix'], class_names, cfg['output_dir'])
            
            # Save classification report
            report_str = classification_report(all_test_targets.numpy(), all_test_preds.argmax(dim=1).numpy(), 
                                               target_names=class_names, zero_division=0)
            with open(os.path.join(cfg['output_dir'], "classification_report.txt"), 'w') as f:
                f.write(report_str)
            logger.info("Classification report saved.")

        else:
            logger.warning("Best model checkpoint not found. Skipping final test.")
    else:
        logger.info("Test loader not available. Skipping final test.")

    logger.info("Process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune HVT for Cotton Leaf Diseases")
    parser.add_argument('--config', type=str, help="Path to YAML configuration file (optional)")
    # Allow overriding specific config values from command line
    parser.add_argument('--train_dir', type=str, help="Path to training data directory")
    parser.add_argument('--val_dir', type=str, help="Path to validation data directory")
    parser.add_argument('--test_dir', type=str, help="Path to test data directory")
    parser.add_argument('--output_dir', type=str, help="Directory to save results and logs")
    parser.add_argument('--epochs', type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, help="Batch size for training")
    parser.add_argument('--lr_head', type=float, help="Learning rate for model head")
    parser.add_argument('--lr_backbone', type=float, help="Learning rate for model backbone")
    parser.add_argument('--pretrained_weights_path', type=str, help="Path to pretrained model weights")


    args = parser.parse_args()

    # Update config with CLI args if provided
    cfg = load_enhanced_config() # Load defaults first
    if args.config:
        try:
            with open(args.config, 'r') as f:
                user_yaml_cfg = yaml.safe_load(f)
            cfg.update(user_yaml_cfg) # User YAML overrides defaults
        except FileNotFoundError:
            print(f"Warning: Config file {args.config} not found. Using default config.")
        except Exception as e:
            print(f"Error loading config file {args.config}: {e}. Using default config.")


    cli_args_dict = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
    cfg.update(cli_args_dict) # CLI args override YAML and defaults

    # Critical check for dataset paths
    if not cfg.get('train_dir') or not cfg.get('val_dir'):
        print("ERROR: 'train_dir' and 'val_dir' must be specified either in config or via CLI. Exiting.")
        sys.exit(1)
    
    # Initialize dynamic parts of config, e.g. progressive resize schedule based on total epochs
    total_epochs_for_schedule = cfg['epochs']
    new_resize_schedule = {}
    # Current schedule in load_enhanced_config uses absolute epochs. If fractional needed:
    # example_schedule = {0: [224,224], 0.33: [256,256], 0.66: [384,384]}
    # for key_epoch_frac, size_val in example_schedule.items():
    #    actual_epoch = int(key_epoch_frac * total_epochs_for_schedule)
    #    new_resize_schedule[actual_epoch] = size_val
    # cfg['resize_schedule'] = new_resize_schedule
    # For now, using the absolute epochs from config is fine.

    main(cfg_path=None) # Pass None as cfg is already prepared