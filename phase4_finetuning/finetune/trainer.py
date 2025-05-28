# phase4_finetuning/finetune/trainer.py
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T_v2 # For TTA
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import logging
import sys
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
                 tta_enabled_val: bool = False # New flag for TTA
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

        logger.info(f"Finetuner initialized: device={device}, accum_steps={self.accumulation_steps}, lr_sched_on_batch={self.lr_scheduler_on_batch}, AMP={self.scaler.is_enabled()}")
        if self.augmentations:
            logger.info(f"Using training augmentations: {self.augmentations.__class__.__name__}")
        if self.tta_enabled_val:
            logger.info("Test-Time Augmentation (TTA) enabled for validation.")


    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader, current_epoch: int, total_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        actual_optimizer_steps = 0
        processed_batches_for_avg_loss = 0
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {current_epoch}/{total_epochs} [Fine-tuning]", file=sys.stdout, dynamic_ncols=True)

        for batch_idx, (rgb_images, labels) in pbar:
            rgb_images = rgb_images.to(self.device, non_blocking=(self.device == 'cuda'))
            labels = labels.to(self.device, non_blocking=(self.device == 'cuda'))

            if self.augmentations:
                rgb_images = self.augmentations(rgb_images)

            loss_value_this_iteration = 0.0
            try:
                with autocast(enabled=self.scaler.is_enabled()):
                    outputs = self.model(rgb_img=rgb_images, spectral_img=None, mode='classify')
                    if isinstance(outputs, tuple) and len(outputs) > 0: 
                        outputs = outputs[0]

                    loss = self.criterion(outputs, labels)
                    loss_value_this_iteration = loss.item()
                    if self.accumulation_steps > 1:
                        loss = loss / self.accumulation_steps
            except Exception as e_fwd:
                logger.error(f"E{current_epoch} B{batch_idx}: Forward/loss error: {e_fwd}", exc_info=True)
                if self.accumulation_steps > 1 and (batch_idx +1) % self.accumulation_steps != 0 :
                    self.optimizer.zero_grad(set_to_none=True)
                continue 

            if not torch.isfinite(loss):
                logger.error(f"E{current_epoch} B{batch_idx}: Non-finite loss ({loss.item()}). Skipping grad.")
                if self.accumulation_steps > 1 and (batch_idx +1) % self.accumulation_steps != 0 :
                    self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                actual_optimizer_steps +=1
                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer) 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True) 

                if self.scheduler and self.lr_scheduler_on_batch:
                    self.scheduler.step()

            total_loss += loss_value_this_iteration
            processed_batches_for_avg_loss += 1
            current_lr = self.optimizer.param_groups[0]['lr'] 
            pbar.set_postfix({"Loss": f"{loss_value_this_iteration:.4f}", "LR": f"{current_lr:.2e}"})

        pbar.close()
        avg_epoch_loss = total_loss / processed_batches_for_avg_loss if processed_batches_for_avg_loss > 0 else 0.0
        final_lr_epoch = self.optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {current_epoch} training finished. Avg Loss: {avg_epoch_loss:.4f}, Final LR for epoch: {final_lr_epoch:.2e}, Opt Steps: {actual_optimizer_steps}")
        return avg_epoch_loss

    def validate_one_epoch(self, val_loader: torch.utils.data.DataLoader, class_names: Optional[List[str]] = None) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_val_loss = 0.0
        all_final_outputs_list: List[torch.Tensor] = [] # Store final (potentially TTA-averaged) logits
        all_labels_list: List[torch.Tensor] = []
        
        num_val_batches = len(val_loader)
        if num_val_batches == 0:
            logger.warning("Validation loader is empty. Returning zero metrics.")
            # Ensure all expected metric keys are returned for consistency
            metrics = {"val_loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, 
                       "precision_macro":0.0, "precision_weighted": 0.0, "recall_macro":0.0, "recall_weighted":0.0}
            # Add per-class F1s if class_names are provided
            if class_names:
                for cn in class_names: metrics[f"f1_{cn.replace(' ', '_')}"] = 0.0
            return 0.0, metrics


        pbar_val = tqdm(val_loader, desc="Validation", file=sys.stdout, leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for rgb_images, labels in pbar_val:
                rgb_images = rgb_images.to(self.device, non_blocking=(self.device == 'cuda'))
                labels = labels.to(self.device, non_blocking=(self.device == 'cuda'))

                with autocast(enabled=self.scaler.is_enabled()):
                    if self.tta_enabled_val:
                        # TTA: hflip, vflip (simple example, can be extended)
                        outputs_original = self.model(rgb_img=rgb_images, spectral_img=None, mode='classify')
                        if isinstance(outputs_original, tuple): outputs_original = outputs_original[0]

                        outputs_hflip = self.model(rgb_img=T_v2.functional.horizontal_flip(rgb_images), spectral_img=None, mode='classify')
                        if isinstance(outputs_hflip, tuple): outputs_hflip = outputs_hflip[0]
                        
                        # For vflip, ensure model is robust or use it cautiously
                        outputs_vflip = self.model(rgb_img=T_v2.functional.vertical_flip(rgb_images), spectral_img=None, mode='classify')
                        if isinstance(outputs_vflip, tuple): outputs_vflip = outputs_vflip[0]

                        # Average logits (or probabilities after softmax, logits usually preferred)
                        final_outputs = (outputs_original + outputs_hflip + outputs_vflip) / 3.0
                    else:
                        final_outputs = self.model(rgb_img=rgb_images, spectral_img=None, mode='classify')
                        if isinstance(final_outputs, tuple): final_outputs = final_outputs[0]
                    
                    loss = self.criterion(final_outputs, labels)

                if not torch.isfinite(loss):
                     logger.warning(f"Non-finite validation loss detected ({loss.item()}). Skipping this batch for loss avg.")
                     num_val_batches = max(1, num_val_batches - 1)
                     continue

                total_val_loss += loss.item()
                all_final_outputs_list.append(final_outputs.cpu()) # Store logits
                all_labels_list.append(labels.cpu())

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
        metrics: Dict[str, float] = {"val_loss": avg_val_loss}

        if not all_final_outputs_list:
             logger.warning("Validation yielded no outputs (all batches may have had issues). Returning zero metrics beyond val_loss.")
             metrics.update({"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, "precision_macro":0.0, "recall_macro":0.0})
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

    def save_model_checkpoint(self, path: str):
        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model checkpoint saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model checkpoint to {path}: {e}", exc_info=True)