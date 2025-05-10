# phase4_finetuning/finetune/trainer.py
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import logging
import sys
from typing import Optional, Dict, Tuple, List # Added List

# Use relative import for metrics within the same package
from ..utils.metrics import compute_metrics
# Import config dict for defaults if needed (e.g. num_classes)
# from ..config import config as default_config # Not strictly needed if params passed

logger = logging.getLogger(__name__)

class Finetuner:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module, # Loss function
                 device: str,
                 scaler: GradScaler,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 lr_scheduler_on_batch: bool = False, # True if scheduler steps per batch (e.g. WarmupCosine)
                 accumulation_steps: int = 1,
                 clip_grad_norm: Optional[float] = None,
                 augmentations = None, # Augmentation transform
                 num_classes: int = 7 # Fallback if not in config from main
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

        logger.info(f"Finetuner initialized: device={device}, accum_steps={self.accumulation_steps}, lr_sched_on_batch={self.lr_scheduler_on_batch}")

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int, total_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        processed_batches_for_avg_loss = 0 # Use this for averaging loss
        self.optimizer.zero_grad(set_to_none=True) # Zero at start of epoch / first accum cycle

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch}/{total_epochs} [Fine-tuning]", file=sys.stdout, dynamic_ncols=True)

        for batch_idx, (rgb_images, labels) in pbar:
            rgb_images = rgb_images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.augmentations:
                rgb_images = self.augmentations(rgb_images)

            loss_this_iter = 0.0 # For logging the loss of this specific iteration
            try:
                with autocast(enabled=self.scaler.is_enabled()):
                    # Assuming your HVT model's forward for classification takes only rgb_img
                    # or handles spectral_img=None correctly if it's still an argument.
                    # For fine-tuning, it's common to adapt the pre-trained model
                    # to only take RGB if SimCLR was RGB-only.
                    # The HVT model from phase2 uses mode='classify' and can take spectral=None.
                    outputs = self.model(rgb_img=rgb_images, spectral_img=None, mode='classify')
                    # If the model's classify mode returns (main_logits, aux_logits), take main_logits
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    loss = self.criterion(outputs, labels)
                    loss_this_iter = loss.item() # Store raw loss for this iteration
                    if self.accumulation_steps > 1:
                        loss = loss / self.accumulation_steps
            except Exception as e_fwd:
                logger.error(f"E{epoch} B{batch_idx}: Forward/loss error: {e_fwd}", exc_info=True); continue


            if not torch.isfinite(loss):
                logger.error(f"E{epoch} B{batch_idx}: Non-finite loss ({loss.item()}). Skipping backward/step.")
                # If grads were accumulated, they should be zeroed before next official step
                if (batch_idx + 1) % self.accumulation_steps == 0 : self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler and self.lr_scheduler_on_batch:
                    self.scheduler.step() # Step per-batch schedulers here

            current_lr = self.optimizer.param_groups[0]['lr']
            total_loss += loss_this_iter # Accumulate the raw loss for averaging
            processed_batches_for_avg_loss += 1
            pbar.set_postfix({"Loss": f"{loss_this_iter:.4f}", "LR": f"{current_lr:.2e}"})
        
        pbar.close()
        avg_epoch_loss = total_loss / processed_batches_for_avg_loss if processed_batches_for_avg_loss > 0 else 0.0
        
        # Epoch-based schedulers are stepped in main.py after validation
        # (Except ReduceLROnPlateau which needs metric)

        logger.info(f"Epoch {epoch} training finished. Average Loss: {avg_epoch_loss:.4f}, Current LR: {current_lr:.2e}")
        return avg_epoch_loss

    def validate_one_epoch(self, val_loader: torch.utils.data.DataLoader, class_names: Optional[List[str]] = None) -> Tuple[float, Dict]:
        self.model.eval()
        total_val_loss = 0.0
        all_preds_list: List[torch.Tensor] = [] # Use list of tensors
        all_labels_list: List[torch.Tensor] = []
        num_val_batches = len(val_loader)

        pbar_val = tqdm(val_loader, desc="Validation", file=sys.stdout, leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for rgb_images, labels in pbar_val:
                rgb_images = rgb_images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(enabled=self.scaler.is_enabled()): # autocast can be used for inference too
                    outputs = self.model(rgb_img=rgb_images, spectral_img=None, mode='classify')
                    if isinstance(outputs, tuple): outputs = outputs[0] # Take main logits
                    loss = self.criterion(outputs, labels)

                if not torch.isfinite(loss):
                     logger.warning(f"Non-finite validation loss detected ({loss.item()}). Skipping this batch for loss avg.")
                     num_val_batches = max(1, num_val_batches -1) # Adjust divisor
                     continue

                total_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds_list.append(preds.cpu())
                all_labels_list.append(labels.cpu())

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
        metrics = {"val_loss": avg_val_loss, "accuracy": 0.0, "f1_macro": 0.0} # Init with val_loss

        if not all_preds_list: # Check if list is empty
             logger.warning("Validation yielded no predictions. Returning zero metrics.")
        else:
            all_preds_np = torch.cat(all_preds_list).numpy()
            all_labels_np = torch.cat(all_labels_list).numpy()
            # Ensure num_classes passed to compute_metrics is correct
            computed_metrics = compute_metrics(all_preds_np, all_labels_np, num_classes=self.num_classes, class_names=class_names)
            metrics.update(computed_metrics) # Add computed metrics

        log_str = f"Validation finished. Avg Loss: {avg_val_loss:.4f}"
        for k, v_metric in metrics.items(): # Iterate through updated metrics dict
            if k != "val_loss": log_str += f", {k}: {v_metric:.4f}" # Avoid double printing val_loss
        logger.info(log_str)

        return avg_val_loss, metrics # Return the dict which includes val_loss

    def save_model_checkpoint(self, path: str):
        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model checkpoint saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model checkpoint to {path}: {e}", exc_info=True)