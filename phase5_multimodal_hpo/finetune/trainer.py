import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import logging
import sys
import random
from typing import Optional, Dict, Tuple, Any, List

import torchvision.transforms.v2 as T_v2

from ..utils.metrics import compute_metrics
from ..config import config as global_train_config # Used for defaults

logger = logging.getLogger(__name__)

# --- EMA Implementation (UNCHANGED from previous version) ---
class ModelEma:
    def __init__(self, model, decay=0.9999, device=None):
        self.module = model 
        self.decay = decay
        self.device = device
        if self.device is None:
            self.device = next(model.parameters()).device
        self.ema_params = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.ema_params[name] = param.data.clone().detach()
    def update(self, model_to_update_from):
        with torch.no_grad():
            for name, param in model_to_update_from.named_parameters():
                if param.requires_grad and name in self.ema_params:
                    self.ema_params[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    @torch.no_grad()
    def get_model_for_eval(self):
        eval_model = type(self.module)(**self.module.config_params).to(self.device)
        temp_state_dict = self.module.state_dict()
        for name, ema_param_val in self.ema_params.items():
            if name in temp_state_dict:
                 temp_state_dict[name] = ema_param_val
        for name, buffer_val in self.module.named_buffers():
            if name in temp_state_dict:
                temp_state_dict[name] = buffer_val.data.clone().detach()
        eval_model.load_state_dict(temp_state_dict)
        eval_model.eval()
        return eval_model

# --- Focal Loss Implementation (UNCHANGED from previous version) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[Any] = None, gamma: float = 2.0, 
                 reduction: str = 'mean', label_smoothing: float = 0.0,
                 num_classes: Optional[int] = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        if isinstance(alpha, (float, int)): self.alpha = torch.tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.tensor(alpha)
        if self.label_smoothing > 0.0 and self.num_classes is None:
            raise ValueError("num_classes must be provided for label smoothing with Focal Loss.")
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing if not targets.ndim == 2 else 0.0)
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        if self.alpha is not None:
            alpha_t = torch.zeros_like(targets, dtype=torch.float).to(inputs.device)
            if targets.ndim == 1:
                alpha_t = self.alpha.to(inputs.device)[targets.data.view(-1)]
            elif targets.ndim == 2:
                true_classes = targets.argmax(dim=1)
                alpha_t = self.alpha.to(inputs.device)[true_classes]
            else: raise ValueError("targets ndim must be 1 or 2")
            F_loss = alpha_t * F_loss
        if self.reduction == 'mean': return F_loss.mean()
        elif self.reduction == 'sum': return F_loss.sum()
        else: return F_loss


class Finetuner:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion_config: dict,
                 device: str,
                 scaler: GradScaler,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 lr_scheduler_on_batch: bool = False,
                 accumulation_steps: int = 1,
                 clip_grad_norm: Optional[float] = None,
                 num_classes: int = global_train_config['num_classes'],
                 img_size: Tuple[int, int] = tuple(global_train_config['img_size']), # Ensure tuple
                 augmentations_enabled: bool = global_train_config['augmentations_enabled'],
                 rand_augment_num_ops: int = global_train_config['rand_augment_num_ops'],
                 rand_augment_magnitude: int = global_train_config['rand_augment_magnitude'],
                 color_jitter_params: dict = global_train_config['color_jitter_params'],
                 random_erase_prob: float = global_train_config['random_erase_prob'],
                 mixup_alpha: float = global_train_config['mixup_alpha'],
                 cutmix_alpha: float = global_train_config['cutmix_alpha'],
                 mixup_cutmix_prob: float = global_train_config['mixup_cutmix_prob'],
                 auto_augment_policy: Optional[str] = global_train_config['auto_augment_policy'],
                 ema_decay: Optional[float] = global_train_config['ema_decay'],
                 tta_enabled: bool = global_train_config['tta_enabled'],
                 tta_augmentations: list = global_train_config['tta_augmentations'],
                 multi_scale_training_epoch_interval: int = global_train_config['multi_scale_training_epoch_interval'],
                 multi_scale_training_factors: List[float] = global_train_config['multi_scale_training_factors'],
                 multi_scale_max_factor_after_unfreeze: float = global_train_config['multi_scale_max_factor_after_unfreeze'], # NEW
                 model_config_params: Optional[dict] = None
                 ):
        self.model = model.to(device)
        self.model.config_params = model_config_params if model_config_params else self._get_model_config_params(global_train_config)

        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.scheduler = scheduler
        self.lr_scheduler_on_batch = lr_scheduler_on_batch
        self.accumulation_steps = max(1, accumulation_steps)
        self.clip_grad_norm = clip_grad_norm
        self.num_classes = num_classes
        self.img_size = img_size # Original/base image size

        self.ema_model = None
        if ema_decay is not None and ema_decay > 0:
            self.ema_model = ModelEma(self.model, decay=ema_decay, device=self.device)
            logger.info(f"EMA enabled with decay: {ema_decay}")

        self.loss_type = criterion_config.get("type", "cross_entropy")
        # ... (Loss setup remains the same)
        if self.loss_type == "focal":
            self.criterion = FocalLoss(
                alpha=criterion_config.get('focal_alpha', None),
                gamma=criterion_config.get('focal_gamma', 2.0),
                label_smoothing=criterion_config.get('label_smoothing', 0.0),
                num_classes=self.num_classes
            ).to(device)
            logger.info(f"Using Focal Loss with gamma={criterion_config.get('focal_gamma', 2.0)}, label_smoothing={criterion_config.get('label_smoothing', 0.0)}")
        else:
            class_weights = criterion_config.get("class_weights", None)
            if class_weights is not None: class_weights = class_weights.to(device)
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=criterion_config.get('label_smoothing', 0.0),
                weight=class_weights
            ).to(device)
            logger.info(f"Using CrossEntropyLoss with label_smoothing={criterion_config.get('label_smoothing', 0.0)}")


        self.augmentations_enabled = augmentations_enabled
        # ... (Augmentation setup remains the same, uses passed-in or default global_train_config values)
        if self.augmentations_enabled:
            logger.info("Augmentations enabled for training.")
            self.geometric_augmentations = T_v2.Compose([
                T_v2.RandomHorizontalFlip(p=0.5), T_v2.RandomVerticalFlip(p=0.5),
                T_v2.RandomRotation(degrees=30), 
                T_v2.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
            ])
            self.color_augmentations = T_v2.Compose([T_v2.ColorJitter(**color_jitter_params)])
            if auto_augment_policy: self.advanced_augment = T_v2.AutoAugment(policy=T_v2.AutoAugmentPolicy(auto_augment_policy)); logger.info(f"Using AutoAugment: {auto_augment_policy}")
            else: self.advanced_augment = T_v2.RandAugment(num_ops=rand_augment_num_ops, magnitude=rand_augment_magnitude); logger.info(f"Using RandAugment: ops={rand_augment_num_ops}, mag={rand_augment_magnitude}")
            self.random_erase = T_v2.RandomErasing(p=random_erase_prob, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0); logger.info(f"Random Erasing: p={random_erase_prob}")
            self.use_mixup_cutmix = (mixup_alpha > 0.0 or cutmix_alpha > 0.0) and mixup_cutmix_prob > 0.0
            if self.use_mixup_cutmix:
                mixers = []
                if mixup_alpha > 0: mixers.append(T_v2.MixUp(alpha=mixup_alpha, num_classes=self.num_classes))
                if cutmix_alpha > 0: mixers.append(T_v2.CutMix(alpha=cutmix_alpha, num_classes=self.num_classes))
                if mixers: self.mixup_cutmix_transform = T_v2.RandomChoice(mixers); self.mixup_cutmix_prob = mixup_cutmix_prob; logger.info(f"Mixup/Cutmix: p={self.mixup_cutmix_prob}, mix_a={mixup_alpha}, cut_a={cutmix_alpha}")
                else: self.use_mixup_cutmix = False
            else: self.mixup_cutmix_transform = None
        else: # Augs disabled
            logger.info("Augmentations disabled.")
            self.geometric_augmentations = nn.Identity(); self.color_augmentations = nn.Identity(); self.advanced_augment = nn.Identity(); self.random_erase = nn.Identity()
            self.use_mixup_cutmix = False; self.mixup_cutmix_transform = None


        self.tta_enabled = tta_enabled
        # ... (TTA setup remains the same)
        self.tta_transforms = []
        if self.tta_enabled:
            self.tta_transforms.append(nn.Identity())
            if 'hflip' in tta_augmentations: self.tta_transforms.append(T_v2.RandomHorizontalFlip(p=1.0))
            if 'vflip' in tta_augmentations: self.tta_transforms.append(T_v2.RandomVerticalFlip(p=1.0))
            if 'rotate' in tta_augmentations: self.tta_transforms.append(T_v2.RandomRotation(degrees=90, expand=False))
            logger.info(f"TTA enabled with {len(self.tta_transforms)} views: {tta_augmentations}")


        self.multi_scale_training_epoch_interval = multi_scale_training_epoch_interval
        self.multi_scale_training_factors = multi_scale_training_factors
        self.multi_scale_max_factor_after_unfreeze = multi_scale_max_factor_after_unfreeze # NEW
        self.current_img_size = self.img_size # Initialize current_img_size

        logger.info(f"Finetuner initialized. Device={device}, AccumSteps={self.accumulation_steps}, BaseImgSize={self.img_size}")

    def _get_model_config_params(self, cfg):
        # Helper to extract model parameters from the global config for EMA reconstruction
        return {
            "img_size": cfg['img_size'][0] if isinstance(cfg['img_size'], (list,tuple)) else cfg['img_size'],
            "patch_size": cfg['hvt_patch_size'],
            "in_chans": 3,
            "spectral_chans": cfg['hvt_spectral_channels'],
            "num_classes": cfg['num_classes'],
            "embed_dim_rgb": cfg['hvt_embed_dim_rgb'],
            "embed_dim_spectral": cfg['hvt_embed_dim_spectral'],
            "depths": cfg['hvt_depths'],
            "num_heads": cfg['hvt_num_heads'],
            "mlp_ratio": cfg['hvt_mlp_ratio'],
            "qkv_bias": cfg['hvt_qkv_bias'],
            "drop_rate": 0.0, # For EMA model, usually use eval mode drop rates
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.0,
            "use_dfca": cfg['hvt_use_dfca'],
            "dfca_heads": cfg['hvt_dfca_heads'],
            "use_gradient_checkpointing": False # EMA model is for eval, no checkpointing needed
        }

    def _apply_augmentations(self, rgb_batch, spectral_batch, labels_batch):
        # This method uses the already corrected logic for MixUp/CutMix from the previous step
        # It correctly passes (B,) labels to self.mixup_cutmix_transform
        # ... (UNCHANGED from the previous correct version)
        if spectral_batch is not None:
            combined_input = torch.cat((rgb_batch, spectral_batch), dim=1)
            combined_input_aug = self.geometric_augmentations(combined_input)
            rgb_batch_aug, spectral_batch_aug = torch.split(combined_input_aug, [rgb_batch.shape[1], spectral_batch.shape[1]], dim=1)
        else:
            rgb_batch_aug = self.geometric_augmentations(rgb_batch)
            spectral_batch_aug = None
        rgb_batch_aug = self.color_augmentations(rgb_batch_aug)
        rgb_batch_aug = self.advanced_augment(rgb_batch_aug)
        rgb_batch_aug = self.random_erase(rgb_batch_aug)
        labels_final = labels_batch
        if self.use_mixup_cutmix and torch.rand(1).item() < self.mixup_cutmix_prob:
            if labels_batch.ndim == 2:
                # This should ideally not happen if dataset returns (B,) labels.
                # logger.warning("Mixup/Cutmix received labels_batch with ndim=2. Using argmax.")
                labels_for_mixup = labels_batch.argmax(dim=1)
            else:
                labels_for_mixup = labels_batch # Should be (B,)
            rgb_batch_mixed, labels_mixed_one_hot = self.mixup_cutmix_transform(rgb_batch_aug, labels_for_mixup)
            return rgb_batch_mixed, spectral_batch_aug, labels_mixed_one_hot
        return rgb_batch_aug, spectral_batch_aug, labels_final


    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int, total_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0; processed_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        # --- MODIFIED Multi-scale training logic ---
        self.current_img_size = self.img_size # Default to base size
        if self.multi_scale_training_epoch_interval > 0 and \
           epoch % self.multi_scale_training_epoch_interval == 0 and \
           len(self.multi_scale_training_factors) > 0:
            
            current_epoch_factors = list(self.multi_scale_training_factors) # Make a mutable copy
            # After unfreezing, cap the max scale factor
            if epoch > global_train_config["freeze_backbone_epochs"]:
                 current_epoch_factors = [f for f in current_epoch_factors if f <= self.multi_scale_max_factor_after_unfreeze]
                 if not current_epoch_factors: # Ensure there's at least one factor (e.g., 1.0)
                     current_epoch_factors = [min(1.0, self.multi_scale_max_factor_after_unfreeze)]


            scale_factor = random.choice(current_epoch_factors)
            new_h = int(self.img_size[0] * scale_factor)
            new_w = int(self.img_size[1] * scale_factor)
            
            patch_s = self.model.rgb_patch_embed.kernel_size[0] # Get patch size from model
            # Ensure new size is divisible by patch size
            self.current_img_size = (new_h // patch_s * patch_s, new_w // patch_s * patch_s)
            if self.current_img_size[0] == 0 or self.current_img_size[1] == 0: # Safety for too small scales
                logger.warning(f"Multi-scale resulted in zero dim for size {self.current_img_size}, reverting to base {self.img_size}")
                self.current_img_size = self.img_size
            else:
                logger.info(f"Epoch {epoch}: Multi-scale training. Resizing to {self.current_img_size} (factor {scale_factor:.2f})")
        # --- END MODIFIED Multi-scale ---

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{total_epochs} [Training]", file=sys.stdout, dynamic_ncols=True)
        for batch_idx, batch_data in pbar:
            if len(batch_data) == 3: rgb_images, spectral_images, labels = batch_data
            elif len(batch_data) == 2: rgb_images, labels = batch_data; spectral_images = None
            else: logger.error(f"Unexpected batch_data len: {len(batch_data)}"); continue

            rgb_images = rgb_images.to(self.device, non_blocking=True)
            if spectral_images is not None: spectral_images = spectral_images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.current_img_size != self.img_size: # Apply if multi-scale changed size
                rgb_images = F.interpolate(rgb_images, size=self.current_img_size, mode='bilinear', align_corners=False)
                if spectral_images is not None:
                    spectral_images = F.interpolate(spectral_images, size=self.current_img_size, mode='bilinear', align_corners=False)

            if self.augmentations_enabled:
                rgb_images, spectral_images, labels = self._apply_augmentations(rgb_images, spectral_images, labels)

            with autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(rgb_images, spectral_images)
                loss = self.criterion(outputs, labels)
                if self.accumulation_steps > 1: loss = loss / self.accumulation_steps
            if not torch.isfinite(loss): logger.error(f"E{epoch} B{batch_idx}: Non-finite loss ({loss.item()}). Skip."); self.optimizer.zero_grad(set_to_none=True); continue
            
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.clip_grad_norm is not None: self.scaler.unscale_(self.optimizer); torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler and self.lr_scheduler_on_batch: self.scheduler.step()
            
            if self.ema_model is not None: self.ema_model.update(self.model)

            current_lr = self.optimizer.param_groups[0]['lr']; batch_loss = loss.item() * self.accumulation_steps
            total_loss += batch_loss; processed_batches += 1
            pbar.set_postfix({"Loss": f"{batch_loss:.4f}", "LR": f"{current_lr:.2e}", "Size": f"{self.current_img_size[0]}x{self.current_img_size[1]}"})
            
        avg_epoch_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
        if self.scheduler and not self.lr_scheduler_on_batch:
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): self.scheduler.step()
        logger.info(f"Epoch {epoch} training finished. Avg Loss: {avg_epoch_loss:.4f}, Current LR: {current_lr:.2e}")
        return avg_epoch_loss

    def validate_one_epoch(self, val_loader: torch.utils.data.DataLoader, class_names: Optional[list] = None) -> Tuple[float, Dict]:
        # ... (Validation logic remains largely UNCHANGED from previous correct version) ...
        val_model = self.ema_model.get_model_for_eval() if self.ema_model is not None else self.model
        val_model.eval()
        total_val_loss = 0.0; all_batch_outputs = []; all_labels = []; num_valid_batches = 0
        pbar = tqdm(val_loader, desc="Validation", file=sys.stdout, leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for rgb_images, spectral_images_maybe, labels in pbar:
                if spectral_images_maybe is not None and spectral_images_maybe.nelement() > 0 : spectral_images = spectral_images_maybe.to(self.device, non_blocking=True)
                else: spectral_images = None
                rgb_images = rgb_images.to(self.device, non_blocking=True); labels = labels.to(self.device, non_blocking=True)
                batch_tta_outputs_logits = []
                num_tta_views = len(self.tta_transforms) if self.tta_enabled else 1
                for tta_idx in range(num_tta_views):
                    aug_rgb = rgb_images; aug_spec = spectral_images
                    if self.tta_enabled:
                        transform = self.tta_transforms[tta_idx].to(self.device)
                        if not isinstance(transform, nn.Identity):
                            aug_rgb = transform(rgb_images)
                            if spectral_images is not None and isinstance(transform, (T_v2.RandomHorizontalFlip, T_v2.RandomVerticalFlip, T_v2.RandomRotation)):
                                 aug_spec = transform(spectral_images)
                    with autocast(enabled=self.scaler.is_enabled()): outputs_logits = val_model(aug_rgb, aug_spec)
                    if not torch.isfinite(outputs_logits).all(): logger.warning(f"Val Batch: Non-finite TTA view {tta_idx}. Skip."); continue
                    batch_tta_outputs_logits.append(outputs_logits)
                if not batch_tta_outputs_logits: logger.warning("Val Batch: All TTA views failed. Skip."); continue
                avg_outputs_logits = torch.stack(batch_tta_outputs_logits, dim=0).mean(dim=0)
                with autocast(enabled=self.scaler.is_enabled()): loss = self.criterion(avg_outputs_logits, labels)
                if torch.isfinite(loss):
                    total_val_loss += loss.item(); all_batch_outputs.append(avg_outputs_logits.cpu()); all_labels.append(labels.cpu()); num_valid_batches += 1
                else: logger.warning(f"Val Batch: Non-finite loss post TTA. Skip.")
        avg_val_loss = total_val_loss / num_valid_batches if num_valid_batches > 0 else 0.0
        metrics = {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, "precision_macro":0.0, "recall_macro":0.0}
        if not all_batch_outputs or not all_labels: logger.warning("Validation: No valid results.")
        else:
            all_preds_logits = torch.cat(all_batch_outputs); all_true_labels_np = torch.cat(all_labels).numpy()
            all_preds_np = torch.argmax(all_preds_logits, dim=1).numpy()
            metrics = compute_metrics(all_preds_np, all_true_labels_np, self.num_classes, class_names)
        log_str = f"Validation Summary -- Avg Loss: {avg_val_loss:.4f}"; 
        for k, v_met in metrics.items(): log_str += f", {k}: {v_met:.4f}" # Renamed v to v_met
        if self.tta_enabled: log_str += f" (TTA: {len(self.tta_transforms)} views)"
        logger.info(log_str)
        return avg_val_loss, metrics

    def save_model_checkpoint(self, path: str, epoch: int, best_metric: float):
        # ... (Save checkpoint logic remains UNCHANGED from previous version) ...
        try:
            state = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scaler_state_dict': self.scaler.state_dict(), 'best_metric': best_metric}
            if self.scheduler is not None: state['scheduler_state_dict'] = self.scheduler.state_dict()
            if self.ema_model is not None: state['ema_params'] = self.ema_model.ema_params
            torch.save(state, path)
            logger.info(f"Checkpoint saved: {path} (Epoch {epoch}, Best {best_metric:.4f})")
        except Exception as e: logger.error(f"Error saving ckpt {path}: {e}", exc_info=True)

    def load_model_checkpoint(self, path: str):
        # ... (Load checkpoint logic remains UNCHANGED from previous version) ...
        if not os.path.exists(path): logger.warning(f"Ckpt not found: {path}. Start fresh."); return 0, -1.0
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1; best_metric = checkpoint.get('best_metric', -1.0)
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.ema_model is not None and 'ema_params' in checkpoint:
                loaded_ema_params = checkpoint['ema_params']
                for name, param_val in loaded_ema_params.items():
                    if name in self.ema_model.ema_params: self.ema_model.ema_params[name] = param_val.to(self.device)
                logger.info("EMA params loaded from ckpt.")
            logger.info(f"Ckpt loaded: {path}. Resume E{start_epoch}, BestMetric: {best_metric:.4f}")
            return start_epoch, best_metric
        except Exception as e: logger.error(f"Error loading ckpt {path}: {e}. Start fresh.", exc_info=True); return 0, -1.0