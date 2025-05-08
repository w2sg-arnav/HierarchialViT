# phase5_multimodal_hpo/finetune/trainer.py
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import logging
import sys
from typing import Optional, Dict, Tuple 

import torchvision.transforms.v2 as T_v2 

# Use relative import for metrics within the same package
from ..utils.metrics import compute_metrics 
# Import config dict for defaults if needed (e.g. num_classes)
from ..config import config as default_config 

logger = logging.getLogger(__name__)

class Finetuner:
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module, 
                 device: str, 
                 scaler: GradScaler, 
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
                 lr_scheduler_on_batch: bool = False, 
                 accumulation_steps: int = 1,
                 clip_grad_norm: Optional[float] = None,
                 num_classes: int = default_config['num_classes'],
                 mixup_alpha: float = 0.0, 
                 cutmix_alpha: float = 0.0, 
                 mixup_cutmix_prob: float = 0.0, 
                 img_size: Tuple[int, int] = default_config['img_size'],
                 tta_enabled: bool = False,
                 tta_augmentations: list = ['hflip'],
                 augmentations_enabled: bool = True 
                 ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device) 
        self.device = device
        self.scaler = scaler
        self.scheduler = scheduler
        self.lr_scheduler_on_batch = lr_scheduler_on_batch
        self.accumulation_steps = max(1, accumulation_steps) 
        self.clip_grad_norm = clip_grad_norm
        self.num_classes = num_classes
        self.augmentations_enabled = augmentations_enabled

        # --- Setup Internal Augmentations ---
        # Use config values passed during initialization
        self.geometric_augmentations = T_v2.Compose([
            T_v2.RandomHorizontalFlip(p=0.5), T_v2.RandomVerticalFlip(p=0.5),
            T_v2.RandomRotation(degrees=20), 
            T_v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        ]) if self.augmentations_enabled else nn.Identity()

        self.color_augmentations = T_v2.Compose([
             T_v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), 
             T_v2.RandomApply([T_v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
        ]) if self.augmentations_enabled else nn.Identity()
        
        # Mixup/Cutmix Setup using passed parameters
        self.use_mixup_cutmix = mixup_alpha > 0.0 or cutmix_alpha > 0.0
        if self.use_mixup_cutmix and self.augmentations_enabled:
            self.mixup_cutmix = T_v2.RandomChoice([
                T_v2.MixUp(alpha=mixup_alpha, num_classes=self.num_classes),
                T_v2.CutMix(alpha=cutmix_alpha, num_classes=self.num_classes)])
            self.mixup_cutmix_prob = mixup_cutmix_prob
            logger.info(f"Mixup/Cutmix enabled: p={self.mixup_cutmix_prob}, mixup_alpha={mixup_alpha}, cutmix_alpha={cutmix_alpha}")
        else: 
            self.mixup_cutmix = None
            self.mixup_cutmix_prob = 0.0 # Ensure prob is 0 if disabled

        # TTA Setup using passed parameters
        self.tta_enabled = tta_enabled
        self.tta_transforms = []
        if self.tta_enabled:
             self.tta_transforms.append(nn.Identity()) # Original
             if 'hflip' in tta_augmentations: self.tta_transforms.append(T_v2.RandomHorizontalFlip(p=1.0))
             if 'vflip' in tta_augmentations: self.tta_transforms.append(T_v2.RandomVerticalFlip(p=1.0))
             logger.info(f"TTA enabled with {len(self.tta_transforms)} views: {tta_augmentations}")
        
        logger.info(f"Finetuner initialized. Device={device}, AccumSteps={self.accumulation_steps}, Augs={self.augmentations_enabled}, Mixup/Cutmix={self.use_mixup_cutmix}, TTA={self.tta_enabled}")

    def _apply_consistent_geometric_aug(self, rgb_batch, spectral_batch):
        """ Applies the same geometric augmentation to both modalities """
        # (Keep unchanged)
        if not isinstance(self.geometric_augmentations, nn.Identity): # Only apply if augs enabled
            if spectral_batch is not None:
                if rgb_batch.shape[0] == spectral_batch.shape[0] and rgb_batch.shape[2:] == spectral_batch.shape[2:]:
                    combined = torch.cat((rgb_batch, spectral_batch), dim=1)
                    augmented_combined = self.geometric_augmentations(combined)
                    rgb_batch, spectral_batch = torch.split(augmented_combined, [rgb_batch.shape[1], spectral_batch.shape[1]], dim=1)
                else: logger.warning("RGB/Spectral shape mismatch for consistent geometric aug. Skipping spectral.")
            else: rgb_batch = self.geometric_augmentations(rgb_batch)
        return rgb_batch, spectral_batch


    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int, total_epochs: int) -> float:
        """ Trains the model for one epoch. Returns average training loss. """
        # (Keep training loop logic mostly unchanged, Mixup/Cutmix application verified)
        self.model.train(); total_loss = 0.0; processed_batches = 0
        self.optimizer.zero_grad(set_to_none=True) 
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{total_epochs} [Fine-tuning]", file=sys.stdout)

        for batch_idx, batch_data in pbar: 
            if len(batch_data) == 3: rgb_images, spectral_images, labels = batch_data
            else: rgb_images, labels = batch_data; spectral_images = None
            rgb_images = rgb_images.to(self.device, non_blocking=True)
            spectral_images = spectral_images.to(self.device, non_blocking=True) if spectral_images is not None else None
            labels = labels.to(self.device, non_blocking=True)

            # --- Apply Augmentations ---
            if self.augmentations_enabled:
                 rgb_images, spectral_images = self._apply_consistent_geometric_aug(rgb_images, spectral_images)
                 rgb_images = self.color_augmentations(rgb_images)
                 if self.mixup_cutmix is not None and torch.rand(1).item() < self.mixup_cutmix_prob:
                      # Apply only to RGB and labels (assumes mixup/cutmix not needed/compatible with spectral)
                      rgb_images, labels = self.mixup_cutmix(rgb_images, labels)
            
            # --- Forward and Backward ---
            with autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(rgb_images, spectral_images) 
                loss = self.criterion(outputs, labels) # CrossEntropy handles mixup labels
                if self.accumulation_steps > 1: loss = loss / self.accumulation_steps
            if not torch.isfinite(loss): logger.error(f"E{epoch} B{batch_idx}: Non-finite loss ({loss.item()}). Skip."); continue 
            self.scaler.scale(loss).backward()

            # --- Optimizer Step ---
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.clip_grad_norm is not None: self.scaler.unscale_(self.optimizer); torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad(set_to_none=True) 
                if self.scheduler and self.lr_scheduler_on_batch: self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']; batch_loss = loss.item() * self.accumulation_steps 
            total_loss += batch_loss; processed_batches += 1 
            pbar.set_postfix({"Loss": f"{batch_loss:.4f}", "LR": f"{current_lr:.1e}"})
            
        avg_epoch_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
        if self.scheduler and not self.lr_scheduler_on_batch:
             if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): self.scheduler.step()
        logger.info(f"Epoch {epoch} training finished. Average Loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

    def validate_one_epoch(self, val_loader: torch.utils.data.DataLoader, class_names: Optional[list] = None) -> Tuple[float, Dict]:
        """ Validates the model, optionally using TTA. """
        # (Keep validation logic with TTA unchanged)
        self.model.eval(); total_val_loss = 0.0; all_batch_outputs = []; all_labels = []
        num_batches = len(val_loader)
        pbar = tqdm(val_loader, desc="Validation", file=sys.stdout, leave=False)
        with torch.no_grad():
            for rgb_images, spectral_images, labels in pbar:
                rgb_images = rgb_images.to(self.device, non_blocking=True); spectral_images = spectral_images.to(self.device, non_blocking=True) if spectral_images is not None else None; labels = labels.to(self.device, non_blocking=True)
                batch_tta_outputs = []; num_tta_views = len(self.tta_transforms) if self.tta_enabled else 1
                for tta_idx in range(num_tta_views):
                    aug_rgb = rgb_images; aug_spec = spectral_images
                    if self.tta_enabled:
                        transform = self.tta_transforms[tta_idx]
                        aug_rgb = transform(rgb_images) # Apply TTA only to RGB for now
                    with autocast(enabled=self.scaler.is_enabled()): outputs = self.model(aug_rgb, aug_spec) 
                    if not torch.isfinite(outputs).all(): logger.warning(f"Non-finite val output TTA view {tta_idx}. Skip."); continue 
                    batch_tta_outputs.append(outputs) 
                if not batch_tta_outputs: logger.warning("Skip val batch: all TTA views failed."); num_batches = max(1, num_batches - 1); continue
                avg_outputs = torch.stack(batch_tta_outputs, dim=0).mean(dim=0)
                with autocast(enabled=self.scaler.is_enabled()): loss = self.criterion(avg_outputs, labels)
                if torch.isfinite(loss): total_val_loss += loss.item()
                all_batch_outputs.append(avg_outputs.cpu()); all_labels.append(labels.cpu())
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
        metrics = {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0} 
        if not all_batch_outputs or not all_labels: logger.warning("Validation yielded no results.")
        else: all_preds_logits = torch.cat(all_batch_outputs); all_labels_np = torch.cat(all_labels).numpy(); all_preds_np = torch.argmax(all_preds_logits, dim=1).numpy(); metrics = compute_metrics(all_preds_np, all_labels_np, self.num_classes, class_names)
        log_str = f"Validation finished. Avg Loss: {avg_val_loss:.4f}"; 
        for k, v in metrics.items(): log_str += f", {k}: {v:.4f}"
        if self.tta_enabled: log_str += " (TTA Enabled)"
        logger.info(log_str)       
        return avg_val_loss, metrics

    def save_model_checkpoint(self, path: str):
        # (Keep unchanged)
        try: torch.save(self.model.state_dict(), path); logger.info(f"Model checkpoint saved to {path}")
        except Exception as e: logger.error(f"Error saving checkpoint to {path}: {e}")