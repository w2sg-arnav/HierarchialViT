# phase4_finetuning/finetune/trainer.py
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T_v2 # For TTA
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import logging
import sys
import os
from typing import Optional, Dict, Tuple, List, Any

from ..utils.metrics import compute_metrics

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
                 augmentations = None,
                 num_classes: int = 7,
                 tta_enabled_val: bool = False,
                 debug_nan_detection: bool = False,
                 stop_on_nan_threshold: int = 5,
                 monitor_gradients: bool = False,
                 gradient_log_interval: int = 50
                 ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.scaler = scaler
        self.scheduler = scheduler
        self.lr_scheduler_on_batch = lr_scheduler_on_batch
        self.augmentations = augmentations
        self.accumulation_steps = max(1, accumulation_steps)
        self.clip_grad_norm = clip_grad_norm
        self.num_classes = num_classes
        self.tta_enabled_val = tta_enabled_val

        self.debug_nan_detection = debug_nan_detection
        self.stop_on_nan_threshold = stop_on_nan_threshold
        self.nan_loss_counter_epoch = 0
        self.monitor_gradients = monitor_gradients
        self.gradient_log_interval = gradient_log_interval
        
        # Store model for saving (in case of torch.compile wrapping)
        self._model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model


        logger.info(f"Finetuner initialized: device={device}, accum_steps={self.accumulation_steps}, AMP={self.scaler.is_enabled()}")
        if self.augmentations: logger.info(f"Using training augmentations: {self.augmentations.__class__.__name__}")
        if self.tta_enabled_val: logger.info("Test-Time Augmentation (TTA) enabled for validation.")
        if self.debug_nan_detection: logger.info(f"NaN detection enabled. Stop threshold: {self.stop_on_nan_threshold} NaNs/epoch.")
        if self.monitor_gradients: logger.info(f"Gradient monitoring enabled. Log interval: {self.gradient_log_interval} optimizer steps.")


    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader, current_epoch: int, total_epochs: int) -> Tuple[float, bool]:
        self.model.train()
        total_loss = 0.0
        actual_optimizer_steps = 0
        processed_batches_for_avg_loss = 0
        self.optimizer.zero_grad(set_to_none=True)
        self.nan_loss_counter_epoch = 0 # Reset for the new epoch
        nan_threshold_exceeded_this_epoch = False

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {current_epoch}/{total_epochs} [Fine-tuning]", file=sys.stdout, dynamic_ncols=True)

        for batch_idx, (rgb_images, labels) in pbar:
            rgb_images = rgb_images.to(self.device, non_blocking=(self.device == 'cuda'))
            labels = labels.to(self.device, non_blocking=(self.device == 'cuda'))

            if self.augmentations:
                rgb_images = self.augmentations(rgb_images)
                if not torch.isfinite(rgb_images).all(): # Check after augmentation
                    logger.error(f"E{current_epoch} B{batch_idx}: Non-finite values in augmented images. Skipping batch.")
                    self.nan_loss_counter_epoch +=1 # Count this as a NaN-related issue
                    if self.debug_nan_detection and self.nan_loss_counter_epoch >= self.stop_on_nan_threshold:
                        nan_threshold_exceeded_this_epoch = True; break
                    continue


            loss_value_this_iteration = 0.0
            try:
                with autocast(enabled=self.scaler.is_enabled()):
                    outputs = self.model(rgb_img=rgb_images, spectral_img=None, mode='classify')
                    if isinstance(outputs, tuple) and len(outputs) > 0: 
                        outputs = outputs[0]
                    loss = self.criterion(outputs, labels)
                    loss_value_this_iteration = loss.item() # Store pre-accumulation loss
                    if self.accumulation_steps > 1:
                        loss = loss / self.accumulation_steps
            except Exception as e_fwd:
                logger.error(f"E{current_epoch} B{batch_idx}: Forward/loss error: {e_fwd}", exc_info=True)
                if self.accumulation_steps > 1 and (batch_idx +1) % self.accumulation_steps != 0 :
                    self.optimizer.zero_grad(set_to_none=True) # Clear any partial grads
                self.nan_loss_counter_epoch +=1
                if self.debug_nan_detection and self.nan_loss_counter_epoch >= self.stop_on_nan_threshold:
                    nan_threshold_exceeded_this_epoch = True; break
                continue

            if not torch.isfinite(loss):
                logger.error(f"E{current_epoch} B{batch_idx}: Non-finite loss ({loss.item()}). Skipping grad.")
                if self.accumulation_steps > 1 and (batch_idx +1) % self.accumulation_steps != 0 :
                    self.optimizer.zero_grad(set_to_none=True)
                self.nan_loss_counter_epoch +=1
                if self.debug_nan_detection and self.nan_loss_counter_epoch >= self.stop_on_nan_threshold:
                    nan_threshold_exceeded_this_epoch = True; break
                continue

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                actual_optimizer_steps +=1
                if self.scaler.is_enabled(): self.scaler.unscale_(self.optimizer)

                if self.monitor_gradients and actual_optimizer_steps % self.gradient_log_interval == 0:
                    total_norm = 0.0
                    # Use self._model_to_save for parameters if model is compiled
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

                if self.scheduler and self.lr_scheduler_on_batch:
                    self.scheduler.step()

            total_loss += loss_value_this_iteration 
            processed_batches_for_avg_loss += 1
            current_lr = self.optimizer.param_groups[0]['lr'] 
            pbar.set_postfix({"Loss": f"{loss_value_this_iteration:.4f}", "LR": f"{current_lr:.2e}", "NaNs": self.nan_loss_counter_epoch})
            
            if nan_threshold_exceeded_this_epoch: break 

        pbar.close()
        if nan_threshold_exceeded_this_epoch:
            logger.error(f"Epoch {current_epoch} terminated early due to exceeding NaN loss threshold ({self.stop_on_nan_threshold}).")
            return float('nan'), True 

        avg_epoch_loss = total_loss / processed_batches_for_avg_loss if processed_batches_for_avg_loss > 0 else float('nan')
        final_lr_epoch = self.optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {current_epoch} training finished. Avg Loss: {avg_epoch_loss:.4f}, Final LR: {final_lr_epoch:.2e}, Opt Steps: {actual_optimizer_steps}, NaN count: {self.nan_loss_counter_epoch}")
        return avg_epoch_loss, False

    def validate_one_epoch(self, val_loader: torch.utils.data.DataLoader, class_names: Optional[List[str]] = None) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_val_loss = 0.0
        all_final_outputs_list: List[torch.Tensor] = [] 
        all_labels_list: List[torch.Tensor] = []
        
        num_val_batches_initial = len(val_loader)
        num_valid_batches_for_loss_avg = num_val_batches_initial

        if num_val_batches_initial == 0:
            logger.warning("Validation loader is empty. Returning zero metrics.")
            metrics = {"val_loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, 
                       "precision_macro":0.0, "precision_weighted": 0.0, "recall_macro":0.0, "recall_weighted":0.0}
            if class_names:
                for cn_idx, cn in enumerate(class_names): metrics[f"f1_{cn.replace(' ', '_')}"] = 0.0
            return 0.0, metrics

        pbar_val = tqdm(val_loader, desc="Validation", file=sys.stdout, leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for rgb_images, labels in pbar_val:
                rgb_images = rgb_images.to(self.device, non_blocking=(self.device == 'cuda'))
                labels = labels.to(self.device, non_blocking=(self.device == 'cuda'))

                with autocast(enabled=self.scaler.is_enabled()): 
                    if self.tta_enabled_val:
                        outputs_original = self.model(rgb_img=rgb_images, spectral_img=None, mode='classify')
                        if isinstance(outputs_original, tuple): outputs_original = outputs_original[0]
                        
                        rgb_images_hflip = T_v2.functional.horizontal_flip(rgb_images)
                        outputs_hflip = self.model(rgb_img=rgb_images_hflip, spectral_img=None, mode='classify')
                        if isinstance(outputs_hflip, tuple): outputs_hflip = outputs_hflip[0]
                        
                        rgb_images_vflip = T_v2.functional.vertical_flip(rgb_images)
                        outputs_vflip = self.model(rgb_img=rgb_images_vflip, spectral_img=None, mode='classify')
                        if isinstance(outputs_vflip, tuple): outputs_vflip = outputs_vflip[0]
                        
                        final_outputs = (outputs_original + outputs_hflip + outputs_vflip) / 3.0
                    else:
                        final_outputs = self.model(rgb_img=rgb_images, spectral_img=None, mode='classify')
                        if isinstance(final_outputs, tuple): final_outputs = final_outputs[0]
                    
                    loss = self.criterion(final_outputs, labels)

                if not torch.isfinite(loss):
                     logger.warning(f"Non-finite validation loss detected ({loss.item()}). Skipping this batch for loss avg.")
                     num_valid_batches_for_loss_avg = max(1, num_valid_batches_for_loss_avg - 1) 
                     continue 

                total_val_loss += loss.item()
                all_final_outputs_list.append(final_outputs.cpu()) 
                all_labels_list.append(labels.cpu())
        pbar_val.close()

        avg_val_loss = total_val_loss / num_valid_batches_for_loss_avg if num_valid_batches_for_loss_avg > 0 else float('nan')
        metrics: Dict[str, float] = {"val_loss": avg_val_loss}

        if not all_final_outputs_list:
             logger.warning("Validation yielded no valid outputs. Returning zero metrics beyond val_loss.")
             metrics.update({"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, "precision_macro":0.0, "recall_macro":0.0})
             if class_names:
                for cn_idx, cn in enumerate(class_names): metrics[f"f1_{cn.replace(' ', '_')}"] = 0.0
        else:
            all_logits_stacked = torch.cat(all_final_outputs_list)
            all_preds_np = torch.argmax(all_logits_stacked, dim=1).numpy()
            all_labels_np = torch.cat(all_labels_list).numpy()
            
            computed_metrics = compute_metrics(all_preds_np, all_labels_np, num_classes=self.num_classes, class_names=class_names)
            metrics.update(computed_metrics)

        log_parts = [f"Validation finished. Avg Loss: {avg_val_loss:.4f}"]
        for k, v_metric in metrics.items():
            if k != "val_loss": log_parts.append(f"{k}: {v_metric:.4f}")
        logger.info(", ".join(log_parts))

        return avg_val_loss, metrics

    def save_checkpoint(self, path: str, epoch: int, best_val_metric: float, metric_name: str):
        """Saves model, optimizer, scheduler, epoch, and best metric."""
        # Ensure self._model_to_save is up-to-date if model was re-assigned (e.g. torch.compile)
        self._model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self._model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': best_val_metric,
            'metric_name_val': metric_name, # Store which metric this "best_val_metric" refers to
        }
        if self.scheduler:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler.is_enabled(): # Save scaler state only if AMP was active
             checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
        
        try:
            torch.save(checkpoint_data, path)
            logger.info(f"Checkpoint saved to {path} (Epoch {epoch}, Best {metric_name}: {best_val_metric:.4f})")
        except Exception as e:
            logger.error(f"Error saving checkpoint to {path}: {e}", exc_info=True)

    # load_checkpoint might be better placed in main.py or as a static method
    # if it needs to interact with config before Finetuner is fully initialized.
    # For now, keeping it here implies Finetuner object exists.
    def load_checkpoint(self, path: str, cfg_metric_to_monitor: str) -> Tuple[int, float]:
        """Loads checkpoint data. Returns start_epoch and best_val_metric."""
        start_epoch = 1
        best_val_metric_resumed = 0.0 if cfg_metric_to_monitor != "val_loss" else float('inf')

        if not os.path.exists(path):
            logger.warning(f"Checkpoint path {path} does not exist. Cannot load.")
            return start_epoch, best_val_metric_resumed

        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            model_to_load_into = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
            
            if 'model_state_dict' in checkpoint:
                # Ensure the model structure (esp. classifier head for num_classes) is correct before loading.
                # This should be handled by how `model` is created in `main.py` before Finetuner init.
                missing_keys, unexpected_keys = model_to_load_into.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if missing_keys: logger.warning(f"Resuming: Missing keys in model_state_dict: {missing_keys}")
                if unexpected_keys: logger.warning(f"Resuming: Unexpected keys in model_state_dict: {unexpected_keys}")
                logger.info("Resumed model state from checkpoint.")
            else:
                logger.warning(f"Checkpoint {path} does not contain 'model_state_dict'. Model weights not resumed.")

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
                logger.info(f"Resuming from epoch: {start_epoch -1}. Training will start at epoch {start_epoch}.")
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
            return 1, (0.0 if cfg_metric_to_monitor != "val_loss" else float('inf')) # Fallback