# phase4_finetuning/finetune/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T_v2
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import logging
import sys
import os
import random
import math
from typing import Optional, Dict, Tuple, List, Any
from collections import defaultdict

from ..utils.metrics import compute_metrics

logger = logging.getLogger(__name__)

class EnhancedFinetuner:
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
                 augmentations = None,
                 num_classes: int = 7,
                 tta_enabled_val: bool = False,
                 debug_nan_detection: bool = False,
                 stop_on_nan_threshold: int = 5,
                 monitor_gradients: bool = False,
                 gradient_log_interval: int = 50,
                 # Enhanced features
                 mixup_alpha: float = 0.2,
                 cutmix_alpha: float = 1.0,
                 mixup_prob: float = 0.5,
                 cutmix_prob: float = 0.5,
                 label_smoothing: float = 0.1,
                 use_enhanced_tta: bool = True,
                 focal_loss_alpha: float = 1.0,
                 focal_loss_gamma: float = 2.0,
                 use_focal_loss: bool = False,
                 confidence_penalty: float = 0.1,
                 use_progressive_resize: bool = False,
                 input_size: int = 224,
                 max_input_size: int = 384,
                 swa_enabled: bool = False,
                 swa_start_epoch: int = 15,
                 multiscale_training: bool = True,
                 scale_range: Tuple[int, int] = (192, 320)
                 ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.scheduler = scheduler
        self.lr_scheduler_on_batch = lr_scheduler_on_batch
        self.accumulation_steps = max(1, accumulation_steps)
        self.clip_grad_norm = clip_grad_norm
        self.num_classes = num_classes
        self.tta_enabled_val = tta_enabled_val

        # Enhanced features
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.use_enhanced_tta = use_enhanced_tta
        self.confidence_penalty = confidence_penalty
        self.use_progressive_resize = use_progressive_resize
        self.input_size = input_size
        self.max_input_size = max_input_size
        self.swa_enabled = swa_enabled
        self.swa_start_epoch = swa_start_epoch
        self.multiscale_training = multiscale_training
        self.scale_range = scale_range
        
        # Initialize SWA model if enabled
        if self.swa_enabled:
            self.swa_model = torch.optim.swa_utils.AveragedModel(model)
            self.swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.05)
        
        # Enhanced criterion with multiple loss components
        if use_focal_loss:
            self.criterion = self._create_focal_loss(focal_loss_alpha, focal_loss_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
        
        # Enhanced augmentations for cotton leaf diseases
        self.augmentations = self._create_enhanced_augmentations() if augmentations is None else augmentations
        
        # Debugging
        self.debug_nan_detection = debug_nan_detection
        self.stop_on_nan_threshold = stop_on_nan_threshold
        self.nan_loss_counter_epoch = 0
        self.monitor_gradients = monitor_gradients
        self.gradient_log_interval = gradient_log_interval
        
        # Store model for saving
        self._model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        
        # Class frequency weights for balanced training
        self.class_weights = None
        
        logger.info(f"Enhanced Finetuner initialized with advanced features:")
        logger.info(f"  Device: {device}, Accumulation: {self.accumulation_steps}, AMP: {self.scaler.is_enabled()}")
        logger.info(f"  Mixup α: {mixup_alpha}, CutMix α: {cutmix_alpha}")
        logger.info(f"  Label smoothing: {label_smoothing}, Focal loss: {use_focal_loss}")
        logger.info(f"  Enhanced TTA: {use_enhanced_tta}, SWA: {swa_enabled}")
        logger.info(f"  Multiscale training: {multiscale_training}, Progressive resize: {use_progressive_resize}")

    def _create_enhanced_augmentations(self):
        """Create comprehensive augmentation pipeline for cotton leaf diseases"""
        return T_v2.Compose([
            T_v2.RandomResizedCrop(size=(self.input_size, self.input_size), 
                                 scale=(0.75, 1.0), ratio=(0.8, 1.2)),
            T_v2.RandomHorizontalFlip(p=0.5),
            T_v2.RandomVerticalFlip(p=0.3),
            T_v2.RandomRotation(degrees=45, interpolation=T_v2.InterpolationMode.BILINEAR),
            T_v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            
            # Color augmentations crucial for disease detection
            T_v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            T_v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            T_v2.RandomAutocontrast(p=0.3),
            T_v2.RandomEqualize(p=0.2),
            
            # Noise and blur for robustness
            T_v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5), p=0.2),
            
            # Occlusion simulation
            T_v2.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            
            # Normalize (you may need to adjust these values for your dataset)
            T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _create_focal_loss(self, alpha, gamma):
        """Create focal loss for handling class imbalance"""
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction
                
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                
                if self.reduction == 'mean':
                    return focal_loss.mean()
                elif self.reduction == 'sum':
                    return focal_loss.sum()
                else:
                    return focal_loss
                    
        return FocalLoss(alpha, gamma).to(self.device)

    def _mixup_data(self, x, y, alpha=1.0):
        """Mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _cutmix_data(self, x, y, alpha=1.0):
        """CutMix augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam

    def _rand_bbox(self, size, lam):
        """Generate random bounding box for CutMix"""
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

    def _mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Mixed loss for mixup/cutmix"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def _confidence_penalty_loss(self, outputs):
        """Confidence penalty to prevent overconfidence"""
        probs = F.softmax(outputs, dim=1)
        log_probs = F.log_softmax(outputs, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        return -self.confidence_penalty * entropy

    def _get_current_input_size(self, epoch, total_epochs):
        """Progressive resizing: start small, end large"""
        if not self.use_progressive_resize:
            return self.input_size
            
        progress = epoch / total_epochs
        size = int(self.input_size + (self.max_input_size - self.input_size) * progress)
        return min(size, self.max_input_size)

    def _get_multiscale_size(self):
        """Random scale for multiscale training"""
        if not self.multiscale_training:
            return self.input_size
        return random.randint(self.scale_range[0], self.scale_range[1])

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader, 
                       current_epoch: int, total_epochs: int) -> Tuple[float, bool]:
        self.model.train()
        total_loss = 0.0
        actual_optimizer_steps = 0
        processed_batches_for_avg_loss = 0
        self.optimizer.zero_grad(set_to_none=True)
        self.nan_loss_counter_epoch = 0
        nan_threshold_exceeded_this_epoch = False
        
        # Progressive input size
        current_input_size = self._get_current_input_size(current_epoch, total_epochs)
        if current_input_size != self.input_size:
            logger.info(f"Epoch {current_epoch}: Using progressive input size {current_input_size}")

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {current_epoch}/{total_epochs} [Enhanced Fine-tuning]", 
                    file=sys.stdout, dynamic_ncols=True)

        for batch_idx, (rgb_images, labels) in pbar:
            rgb_images = rgb_images.to(self.device, non_blocking=(self.device == 'cuda'))
            labels = labels.to(self.device, non_blocking=(self.device == 'cuda'))

            # Multiscale training
            if self.multiscale_training and random.random() < 0.3:
                scale_size = self._get_multiscale_size()
                rgb_images = F.interpolate(rgb_images, size=(scale_size, scale_size), 
                                         mode='bilinear', align_corners=False)

            # Apply augmentations
            if self.augmentations:
                rgb_images = self.augmentations(rgb_images)
                if not torch.isfinite(rgb_images).all():
                    logger.error(f"E{current_epoch} B{batch_idx}: Non-finite values in augmented images. Skipping batch.")
                    self.nan_loss_counter_epoch += 1
                    if self.debug_nan_detection and self.nan_loss_counter_epoch >= self.stop_on_nan_threshold:
                        nan_threshold_exceeded_this_epoch = True
                        break
                    continue

            # Apply Mixup or CutMix
            use_mixup = random.random() < self.mixup_prob
            use_cutmix = random.random() < self.cutmix_prob and not use_mixup
            
            if use_mixup:
                rgb_images, labels_a, labels_b, lam = self._mixup_data(rgb_images, labels, self.mixup_alpha)
                mixed_labels = True
            elif use_cutmix:
                rgb_images, labels_a, labels_b, lam = self._cutmix_data(rgb_images, labels, self.cutmix_alpha)
                mixed_labels = True
            else:
                mixed_labels = False

            loss_value_this_iteration = 0.0
            try:
                with autocast(enabled=self.scaler.is_enabled()):
                    outputs = self.model(rgb_img=rgb_images, spectral_img=None, mode='classify')
                    if isinstance(outputs, tuple) and len(outputs) > 0:
                        outputs = outputs[0]
                    
                    # Calculate loss
                    if mixed_labels:
                        loss = self._mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)
                    
                    # Add confidence penalty
                    if self.confidence_penalty > 0:
                        loss += self._confidence_penalty_loss(outputs)
                    
                    loss_value_this_iteration = loss.item()
                    if self.accumulation_steps > 1:
                        loss = loss / self.accumulation_steps
                        
            except Exception as e_fwd:
                logger.error(f"E{current_epoch} B{batch_idx}: Forward/loss error: {e_fwd}", exc_info=True)
                if self.accumulation_steps > 1 and (batch_idx + 1) % self.accumulation_steps != 0:
                    self.optimizer.zero_grad(set_to_none=True)
                self.nan_loss_counter_epoch += 1
                if self.debug_nan_detection and self.nan_loss_counter_epoch >= self.stop_on_nan_threshold:
                    nan_threshold_exceeded_this_epoch = True
                    break
                continue

            if not torch.isfinite(loss):
                logger.error(f"E{current_epoch} B{batch_idx}: Non-finite loss ({loss.item()}). Skipping grad.")
                if self.accumulation_steps > 1 and (batch_idx + 1) % self.accumulation_steps != 0:
                    self.optimizer.zero_grad(set_to_none=True)
                self.nan_loss_counter_epoch += 1
                if self.debug_nan_detection and self.nan_loss_counter_epoch >= self.stop_on_nan_threshold:
                    nan_threshold_exceeded_this_epoch = True
                    break
                continue

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                actual_optimizer_steps += 1
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)

                if self.monitor_gradients and actual_optimizer_steps % self.gradient_log_interval == 0:
                    total_norm = 0.0
                    params_to_check = self._model_to_save.parameters() if hasattr(self._model_to_save, 'parameters') else self.model.parameters()
                    for p in params_to_check:
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    logger.debug(f"E{current_epoch} OptStep {actual_optimizer_steps}: Grad Norm: {total_norm:.4f}")

                if self.clip_grad_norm is not None:
                    params_to_clip = self._model_to_save.parameters() if hasattr(self._model_to_save, 'parameters') else self.model.parameters()
                    torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=self.clip_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # SWA update
                if self.swa_enabled and current_epoch >= self.swa_start_epoch:
                    self.swa_model.update_parameters(self.model)

                if self.scheduler and self.lr_scheduler_on_batch:
                    if self.swa_enabled and current_epoch >= self.swa_start_epoch:
                        self.swa_scheduler.step()
                    else:
                        self.scheduler.step()

            total_loss += loss_value_this_iteration
            processed_batches_for_avg_loss += 1
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                "Loss": f"{loss_value_this_iteration:.4f}", 
                "LR": f"{current_lr:.2e}", 
                "NaNs": self.nan_loss_counter_epoch,
                "Size": f"{rgb_images.shape[-1]}px" if self.multiscale_training else ""
            })

            if nan_threshold_exceeded_this_epoch:
                break

        pbar.close()
        
        if nan_threshold_exceeded_this_epoch:
            logger.error(f"Epoch {current_epoch} terminated early due to exceeding NaN loss threshold ({self.stop_on_nan_threshold}).")
            return float('nan'), True

        avg_epoch_loss = total_loss / processed_batches_for_avg_loss if processed_batches_for_avg_loss > 0 else float('nan')
        final_lr_epoch = self.optimizer.param_groups[0]['lr']
        
        # Update SWA scheduler if enabled
        if self.swa_enabled and current_epoch >= self.swa_start_epoch and not self.lr_scheduler_on_batch:
            self.swa_scheduler.step()
        
        logger.info(f"Epoch {current_epoch} training finished. Avg Loss: {avg_epoch_loss:.4f}, "
                   f"Final LR: {final_lr_epoch:.2e}, Opt Steps: {actual_optimizer_steps}, "
                   f"NaN count: {self.nan_loss_counter_epoch}")
        
        if self.swa_enabled and current_epoch >= self.swa_start_epoch:
            logger.info(f"SWA active since epoch {self.swa_start_epoch}")
            
        return avg_epoch_loss, False

    def _enhanced_tta_inference(self, rgb_images):
        """Enhanced Test-Time Augmentation with multiple transformations"""
        outputs_list = []
        
        # Original image
        with autocast(enabled=self.scaler.is_enabled()):
            outputs = self.model(rgb_img=rgb_images, spectral_img=None, mode='classify')
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs_list.append(outputs)
        
        # Horizontal and vertical flips
        for flip_fn in [T_v2.functional.horizontal_flip, T_v2.functional.vertical_flip]:
            aug_images = flip_fn(rgb_images)
            with autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(rgb_img=aug_images, spectral_img=None, mode='classify')
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                outputs_list.append(outputs)
        
        # Rotations (90, 180, 270 degrees)
        for angle in [90, 180, 270]:
            aug_images = T_v2.functional.rotate(rgb_images, angle)
            with autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(rgb_img=aug_images, spectral_img=None, mode='classify')
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                outputs_list.append(outputs)
        
        # Multi-scale inference (if image is large enough)
        if rgb_images.shape[-1] >= 288:
            for scale in [0.8, 1.2]:
                size = int(rgb_images.shape[-1] * scale)
                if size >= 224:  # Minimum size
                    aug_images = F.interpolate(rgb_images, size=(size, size), 
                                             mode='bilinear', align_corners=False)
                    # Center crop back to original size for consistency
                    if size > rgb_images.shape[-1]:
                        crop_start = (size - rgb_images.shape[-1]) // 2
                        aug_images = aug_images[:, :, crop_start:crop_start+rgb_images.shape[-1], 
                                               crop_start:crop_start+rgb_images.shape[-1]]
                    
                    with autocast(enabled=self.scaler.is_enabled()):
                        outputs = self.model(rgb_img=aug_images, spectral_img=None, mode='classify')
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        outputs_list.append(outputs)
        
        # Average all predictions
        final_outputs = torch.stack(outputs_list).mean(dim=0)
        return final_outputs

    def validate_one_epoch(self, val_loader: torch.utils.data.DataLoader, 
                          class_names: Optional[List[str]] = None, 
                          use_swa: bool = False) -> Tuple[float, Dict[str, float]]:
        
        # Choose model for validation
        model_to_eval = self.swa_model if (use_swa and self.swa_enabled) else self.model
        model_to_eval.eval()
        
        total_val_loss = 0.0
        all_final_outputs_list: List[torch.Tensor] = []
        all_labels_list: List[torch.Tensor] = []
        confidence_scores = []
        
        num_val_batches_initial = len(val_loader)
        num_valid_batches_for_loss_avg = num_val_batches_initial

        if num_val_batches_initial == 0:
            logger.warning("Validation loader is empty. Returning zero metrics.")
            metrics = {"val_loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0,
                      "precision_macro": 0.0, "precision_weighted": 0.0, 
                      "recall_macro": 0.0, "recall_weighted": 0.0}
            if class_names:
                for cn_idx, cn in enumerate(class_names):
                    metrics[f"f1_{cn.replace(' ', '_')}"] = 0.0
            return 0.0, metrics

        desc = "Validation (SWA)" if use_swa else "Validation"
        pbar_val = tqdm(val_loader, desc=desc, file=sys.stdout, leave=False, dynamic_ncols=True)
        
        with torch.no_grad():
            for rgb_images, labels in pbar_val:
                rgb_images = rgb_images.to(self.device, non_blocking=(self.device == 'cuda'))
                labels = labels.to(self.device, non_blocking=(self.device == 'cuda'))

                # Apply TTA if enabled
                if self.tta_enabled_val and self.use_enhanced_tta:
                    final_outputs = self._enhanced_tta_inference(rgb_images)
                elif self.tta_enabled_val:
                    # Original TTA (backward compatibility)
                    with autocast(enabled=self.scaler.is_enabled()):
                        outputs_original = model_to_eval(rgb_img=rgb_images, spectral_img=None, mode='classify')
                        if isinstance(outputs_original, tuple):
                            outputs_original = outputs_original[0]
                        
                        rgb_images_hflip = T_v2.functional.horizontal_flip(rgb_images)
                        outputs_hflip = model_to_eval(rgb_img=rgb_images_hflip, spectral_img=None, mode='classify')
                        if isinstance(outputs_hflip, tuple):
                            outputs_hflip = outputs_hflip[0]
                        
                        rgb_images_vflip = T_v2.functional.vertical_flip(rgb_images)
                        outputs_vflip = model_to_eval(rgb_img=rgb_images_vflip, spectral_img=None, mode='classify')
                        if isinstance(outputs_vflip, tuple):
                            outputs_vflip = outputs_vflip[0]
                        
                        final_outputs = (outputs_original + outputs_hflip + outputs_vflip) / 3.0
                else:
                    # No TTA
                    with autocast(enabled=self.scaler.is_enabled()):
                        final_outputs = model_to_eval(rgb_img=rgb_images, spectral_img=None, mode='classify')
                        if isinstance(final_outputs, tuple):
                            final_outputs = final_outputs[0]

                # Calculate validation loss
                with autocast(enabled=self.scaler.is_enabled()):
                    loss = self.criterion(final_outputs, labels)

                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite validation loss detected ({loss.item()}). Skipping this batch for loss avg.")
                    num_valid_batches_for_loss_avg = max(1, num_valid_batches_for_loss_avg - 1)
                    continue

                total_val_loss += loss.item()
                all_final_outputs_list.append(final_outputs.cpu())
                all_labels_list.append(labels.cpu())
                
                # Track confidence scores
                probs = F.softmax(final_outputs, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                confidence_scores.extend(max_probs.cpu().numpy())

        pbar_val.close()

        avg_val_loss = total_val_loss / num_valid_batches_for_loss_avg if num_valid_batches_for_loss_avg > 0 else float('nan')
        metrics: Dict[str, float] = {"val_loss": avg_val_loss}

        if not all_final_outputs_list:
            logger.warning("Validation yielded no valid outputs. Returning zero metrics beyond val_loss.")
            metrics.update({"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, 
                           "precision_macro": 0.0, "recall_macro": 0.0})
            if class_names:
                for cn_idx, cn in enumerate(class_names):
                    metrics[f"f1_{cn.replace(' ', '_')}"] = 0.0
        else:
            all_logits_stacked = torch.cat(all_final_outputs_list)
            all_preds_np = torch.argmax(all_logits_stacked, dim=1).numpy()
            all_labels_np = torch.cat(all_labels_list).numpy()
            
            computed_metrics = compute_metrics(all_preds_np, all_labels_np, 
                                             num_classes=self.num_classes, class_names=class_names)
            metrics.update(computed_metrics)
            
            # Add confidence statistics
            metrics["avg_confidence"] = np.mean(confidence_scores)
            metrics["min_confidence"] = np.min(confidence_scores)

        log_parts = [f"Validation finished. Avg Loss: {avg_val_loss:.4f}"]
        for k, v_metric in metrics.items():
            if k != "val_loss":
                log_parts.append(f"{k}: {v_metric:.4f}")
        logger.info(", ".join(log_parts))

        return avg_val_loss, metrics

    def save_checkpoint(self, path: str, epoch: int, best_val_metric: float, metric_name: str):
        """Enhanced checkpoint saving with SWA support"""
        self._model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self._model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': best_val_metric,
            'metric_name_val': metric_name,
        }
        
        # Save SWA model if enabled
        if self.swa_enabled:
            checkpoint_data['swa_model_state_dict'] = self.swa_model.state_dict()
            checkpoint_data['swa_scheduler_state_dict'] = self.swa_scheduler.state_dict()
        
        if self.scheduler:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler.is_enabled():
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
        
        try:
            torch.save(checkpoint_data, path)
            logger.info(f"Enhanced checkpoint saved to {path} (Epoch {epoch}, Best {metric_name}: {best_val_metric:.4f})")
        except Exception as e:
            logger.error(f"Error saving checkpoint to {path}: {e}", exc_info=True)

    def load_checkpoint(self, path: str, cfg_metric_to_monitor: str) -> Tuple[int, float]:
        """Enhanced checkpoint loading with SWA support"""
        start_epoch = 1
        best_val_metric_resumed = 0.0 if cfg_metric_to_monitor != "val_loss" else float('inf')

        if not os.path.exists(path):
            logger.warning(f"Checkpoint path {path} does not exist. Cannot load.")
            return start_epoch, best_val_metric_resumed

        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            model_to_load_into = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
            
            if 'model_state_dict' in checkpoint:
                missing_keys, unexpected_keys = model_to_load_into.load_state_dict(
                    checkpoint['model_state_dict'], strict=False)
                if missing_keys:
                    logger.warning(f"Resuming: Missing keys in model_state_dict: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Resuming: Unexpected keys in model_state_dict: {unexpected_keys}")
                logger.info("Resumed model state from checkpoint.")
            else:
                logger.warning(f"Checkpoint {path} does not contain 'model_state_dict'. Model weights not resumed.")

            # Load SWA model if available
            if self.swa_enabled and 'swa_model_state_dict' in checkpoint:
                self.swa_model.load_state_dict(checkpoint['swa_model_state_dict'])
                logger.info("Resumed SWA model state from checkpoint.")
                
            if self.swa_enabled and 'swa_scheduler_state_dict' in checkpoint:
                self.swa_scheduler.load_state_dict(checkpoint['swa_scheduler_state_dict'])
                logger.info("Resumed SWA scheduler state from checkpoint.")

            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Resumed optimizer state from checkpoint.")
            else:
                logger.warning("Optimizer state not found in checkpoint or optimizer not initialized. Optimizer not resumed.")

            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Resumed scheduler state from checkpoint.")
            else:
                logger.warning("Scheduler state not found in checkpoint or scheduler not initialized. Scheduler not resumed.")
            
            if 'scaler_state_dict' in checkpoint and self.scaler.is_enabled():
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logger.info("Resumed GradScaler state from checkpoint.")
            elif 'scaler_state_dict' in checkpoint and not self.scaler.is_enabled():
                logger.info("GradScaler state found in checkpoint, but AMP is currently disabled. Scaler state not loaded.")

            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                logger.info(f"Resuming from epoch: {start_epoch - 1}. Training will start at epoch {start_epoch}.")
            else:
                logger.warning("Epoch number not found in checkpoint. Will start from epoch 1 (or as configured).")
            
            if 'best_val_metric' in checkpoint and 'metric_name_val' in checkpoint:
                if checkpoint['metric_name_val'] == cfg_metric_to_monitor:
                    best_val_metric_resumed = checkpoint['best_val_metric']
                    logger.info(f"Resumed best_val_metric ({cfg_metric_to_monitor}): {best_val_metric_resumed:.4f}")
                else:
                    logger.warning(f"Checkpoint's best metric ({checkpoint['metric_name_val']}) doesn't match config's metric_to_monitor ({cfg_metric_to_monitor}). Resetting best_val_metric.")
            else:
                logger.warning("Best validation metric not found in checkpoint. Resetting.")

            logger.info("LRs in optimizer after loading checkpoint states:")
            for i_pg, optim_pg in enumerate(self.optimizer.param_groups):
                sched_base_lr_info = ""
                if self.scheduler and hasattr(self.scheduler, 'base_lrs') and i_pg < len(self.scheduler.base_lrs):
                    sched_base_lr_info = f", Scheduler Base LR {self.scheduler.base_lrs[i_pg]:.2e}"
                logger.info(f"  Opt Group {i_pg} ('{optim_pg.get('name')}'): Optimizer LR {optim_pg['lr']:.2e}{sched_base_lr_info}")

            return start_epoch, best_val_metric_resumed

        except Exception as e:
            logger.error(f"Error loading checkpoint from {path}: {e}", exc_info=True)
            return 1, (0.0 if cfg_metric_to_monitor != "val_loss" else float('inf'))

    def finalize_swa(self, val_loader: torch.utils.data.DataLoader):
        """Finalize SWA training by updating batch norm statistics"""
        if not self.swa_enabled:
            logger.warning("SWA not enabled. Cannot finalize.")
            return
        
        logger.info("Finalizing SWA: Updating BatchNorm statistics...")
        torch.optim.swa_utils.update_bn(val_loader, self.swa_model, device=self.device)
        logger.info("SWA finalization complete.")

    def get_model_for_inference(self, use_swa: bool = False):
        """Get the appropriate model for inference"""
        if use_swa and self.swa_enabled:
            return self.swa_model
        return self.model

    def set_class_weights(self, class_counts: List[int]):
        """Set class weights for handling imbalanced datasets"""
        total_samples = sum(class_counts)
        weights = [total_samples / (len(class_counts) * count) for count in class_counts]
        self.class_weights = torch.FloatTensor(weights).to(self.device)
        
        # Update criterion with class weights
        if hasattr(self.criterion, 'weight'):
            self.criterion.weight = self.class_weights
        else:
            # Recreate criterion with weights
            if isinstance(self.criterion, nn.CrossEntropyLoss):
                self.criterion = nn.CrossEntropyLoss(
                    weight=self.class_weights, 
                    label_smoothing=getattr(self.criterion, 'label_smoothing', 0.0)
                ).to(self.device)
        
        logger.info(f"Set class weights: {weights}")

    def enable_advanced_scheduler(self, patience: int = 5, factor: float = 0.5, 
                                min_lr: float = 1e-7, warmup_epochs: int = 5):
        """Enable advanced learning rate scheduling with warmup and reduce on plateau"""
        
        class WarmupReduceLROnPlateau:
            def __init__(self, optimizer, warmup_epochs, main_scheduler):
                self.optimizer = optimizer
                self.warmup_epochs = warmup_epochs
                self.main_scheduler = main_scheduler
                self.epoch = 0
                self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
                
            def step(self, metric=None):
                self.epoch += 1
                if self.epoch <= self.warmup_epochs:
                    # Warmup phase
                    warmup_factor = self.epoch / self.warmup_epochs
                    for i, group in enumerate(self.optimizer.param_groups):
                        group['lr'] = self.initial_lrs[i] * warmup_factor
                else:
                    # Main scheduler phase
                    if metric is not None:
                        self.main_scheduler.step(metric)
                    else:
                        self.main_scheduler.step()
        
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=patience, factor=factor, 
            min_lr=min_lr, verbose=True
        )
        
        self.advanced_scheduler = WarmupReduceLROnPlateau(
            self.optimizer, warmup_epochs, plateau_scheduler
        )
        
        logger.info(f"Advanced scheduler enabled: Warmup={warmup_epochs} epochs, "
                   f"Plateau patience={patience}, factor={factor}, min_lr={min_lr}")

    def step_advanced_scheduler(self, val_loss: float):
        """Step the advanced scheduler with validation loss"""
        if hasattr(self, 'advanced_scheduler'):
            self.advanced_scheduler.step(val_loss)

# Backward compatibility alias
Finetuner = EnhancedFinetuner