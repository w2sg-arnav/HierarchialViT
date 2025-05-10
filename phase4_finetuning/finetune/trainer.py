# phase4_finetuning/finetune/trainer.py
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import logging
import sys
from typing import Optional, Dict, Tuple, List, Any # Added Any, List

# Use relative import for metrics within the same package
from ..utils.metrics import compute_metrics
# Config import for defaults like num_classes if not passed explicitly
# from ..config import config as default_trainer_config # Not strictly needed if all passed

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
                 augmentations = None, # Augmentation transform to apply per batch
                 num_classes: int = 7 # Fallback default
                 ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.scaler = scaler
        self.scheduler = scheduler
        self.lr_scheduler_on_batch = lr_scheduler_on_batch
        self.augmentations = augmentations # This is an augmentation object/callable
        self.accumulation_steps = max(1, accumulation_steps)
        self.clip_grad_norm = clip_grad_norm
        self.num_classes = num_classes

        logger.info(f"Finetuner initialized: device={device}, accum_steps={self.accumulation_steps}, lr_sched_on_batch={self.lr_scheduler_on_batch}, AMP={self.scaler.is_enabled()}")
        if self.augmentations:
            logger.info(f"Using augmentations: {self.augmentations.__class__.__name__}")

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
                # Augmentations are applied on the GPU if images are already there, or CPU then moved.
                # Ensure augmentations handle batched tensors correctly.
                rgb_images = self.augmentations(rgb_images)

            loss_value_this_iteration = 0.0 # For logging actual loss per iteration
            try:
                with autocast(enabled=self.scaler.is_enabled()):
                    # Assuming HVT model takes rgb_img, spectral_img, mode
                    # For fine-tuning, usually spectral_img=None if pre-trained on RGB only
                    outputs = self.model(rgb_img=rgb_images, spectral_img=None, mode='classify')
                    if isinstance(outputs, tuple) and len(outputs) > 0: # Handle if model returns (main_logits, aux_logits)
                        outputs = outputs[0]

                    loss = self.criterion(outputs, labels)
                    loss_value_this_iteration = loss.item()
                    if self.accumulation_steps > 1:
                        loss = loss / self.accumulation_steps
            except Exception as e_fwd:
                logger.error(f"E{current_epoch} B{batch_idx}: Forward/loss error: {e_fwd}", exc_info=True)
                # Reset accumulation if an error occurs mid-cycle
                if self.accumulation_steps > 1 and (batch_idx +1) % self.accumulation_steps != 0 :
                    self.optimizer.zero_grad(set_to_none=True) # zero previously accumulated grads
                continue # Skip this batch

            if not torch.isfinite(loss):
                logger.error(f"E{current_epoch} B{batch_idx}: Non-finite loss ({loss.item()}). Skipping grad.")
                if self.accumulation_steps > 1 and (batch_idx +1) % self.accumulation_steps != 0 :
                    self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                actual_optimizer_steps +=1
                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer) # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True) # Zero grads after step

                if self.scheduler and self.lr_scheduler_on_batch: # e.g. WarmupCosine
                    self.scheduler.step()

            total_loss += loss_value_this_iteration
            processed_batches_for_avg_loss += 1
            current_lr = self.optimizer.param_groups[0]['lr'] # Get LR from first param group
            pbar.set_postfix({"Loss": f"{loss_value_this_iteration:.4f}", "LR": f"{current_lr:.2e}"})

        pbar.close()
        avg_epoch_loss = total_loss / processed_batches_for_avg_loss if processed_batches_for_avg_loss > 0 else 0.0
        final_lr_epoch = self.optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {current_epoch} training finished. Avg Loss: {avg_epoch_loss:.4f}, Final LR for epoch: {final_lr_epoch:.2e}, Opt Steps: {actual_optimizer_steps}")
        return avg_epoch_loss

    def validate_one_epoch(self, val_loader: torch.utils.data.DataLoader, class_names: Optional[List[str]] = None) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_val_loss = 0.0
        all_preds_list: List[torch.Tensor] = []
        all_labels_list: List[torch.Tensor] = []
        num_val_batches = len(val_loader)
        if num_val_batches == 0:
            logger.warning("Validation loader is empty. Returning zero metrics.")
            return 0.0, {"val_loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, "precision_macro":0.0, "recall_macro":0.0}


        pbar_val = tqdm(val_loader, desc="Validation", file=sys.stdout, leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for rgb_images, labels in pbar_val:
                rgb_images = rgb_images.to(self.device, non_blocking=(self.device == 'cuda'))
                labels = labels.to(self.device, non_blocking=(self.device == 'cuda'))

                with autocast(enabled=self.scaler.is_enabled()):
                    outputs = self.model(rgb_img=rgb_images, spectral_img=None, mode='classify')
                    if isinstance(outputs, tuple): outputs = outputs[0]
                    loss = self.criterion(outputs, labels)

                if not torch.isfinite(loss):
                     logger.warning(f"Non-finite validation loss detected ({loss.item()}). Skipping this batch for loss avg.")
                     num_val_batches = max(1, num_val_batches - 1) # Avoid division by zero
                     continue

                total_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds_list.append(preds.cpu())
                all_labels_list.append(labels.cpu())

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
        metrics: Dict[str, float] = {"val_loss": avg_val_loss} # Initialize with val_loss

        if not all_preds_list:
             logger.warning("Validation yielded no predictions (all batches may have had issues). Returning zero metrics beyond val_loss.")
             metrics.update({"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, "precision_macro":0.0, "recall_macro":0.0})
        else:
            all_preds_np = torch.cat(all_preds_list).numpy()
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
            # For fine-tuning, saving the whole model (including the new head) is common
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model checkpoint saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model checkpoint to {path}: {e}", exc_info=True)