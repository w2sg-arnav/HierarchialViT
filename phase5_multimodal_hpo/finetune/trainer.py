# phase5_multimodal_hpo/finetune/trainer.py
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import logging
import sys
from typing import Optional, Dict, Tuple 

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
                 augmentations = None, 
                 num_classes: int = default_config['num_classes'] 
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
        
        logger.info(f"Finetuner initialized with device={device}, accumulation_steps={self.accumulation_steps}.")

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int, total_epochs: int) -> float:
        """ Trains the model for one epoch. Returns average training loss. """
        self.model.train()
        total_loss = 0.0
        processed_batches = 0
        self.optimizer.zero_grad(set_to_none=True) # More efficient zeroing at start

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                    desc=f"Epoch {epoch}/{total_epochs} [Fine-tuning]", file=sys.stdout)

        # MODIFIED: Unpack spectral data
        for batch_idx, (rgb_images, spectral_images, labels) in pbar: 
            rgb_images = rgb_images.to(self.device, non_blocking=True)
            # Move spectral to device only if it's not None
            spectral_images = spectral_images.to(self.device, non_blocking=True) if spectral_images is not None else None
            labels = labels.to(self.device, non_blocking=True)

            # Apply augmentations if they exist (apply consistently to both modalities if needed?)
            # Current FinetuneAugmentation likely only handles RGB. Spectral might need separate/no aug.
            if self.augmentations:
                rgb_images = self.augmentations(rgb_images) 
                # TODO: Consider if spectral needs augmentation too

            # Mixed precision forward pass
            with autocast(enabled=self.scaler.is_enabled()):
                # MODIFIED: Pass both RGB and Spectral to model
                outputs = self.model(rgb_images, spectral_images) 
                
                loss = self.criterion(outputs, labels)
                if self.accumulation_steps > 1:
                    loss = loss / self.accumulation_steps

            if not torch.isfinite(loss):
                logger.error(f"Epoch {epoch}, Batch {batch_idx}: Non-finite loss ({loss.item()}). Skipping.")
                # self.optimizer.zero_grad(set_to_none=True) # Already zeroed before step
                continue 

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer) 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True) 

                if self.scheduler and self.lr_scheduler_on_batch: self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            batch_loss = loss.item() * self.accumulation_steps 
            total_loss += batch_loss
            processed_batches += 1 
            pbar.set_postfix({"Loss": f"{batch_loss:.4f}", "LR": f"{current_lr:.1e}"})
            
        avg_epoch_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
        
        if self.scheduler and not self.lr_scheduler_on_batch:
             if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                 self.scheduler.step()
        
        logger.info(f"Epoch {epoch} training finished. Average Loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

    def validate_one_epoch(self, val_loader: torch.utils.data.DataLoader, class_names: Optional[list] = None) -> Tuple[float, Dict]:
        """ Validates the model for one epoch. Returns avg loss and metrics dict. """
        self.model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_labels = []
        num_batches = len(val_loader)

        pbar = tqdm(val_loader, desc="Validation", file=sys.stdout, leave=False)
        with torch.no_grad():
            # MODIFIED: Unpack spectral data
            for rgb_images, spectral_images, labels in pbar:
                rgb_images = rgb_images.to(self.device, non_blocking=True)
                spectral_images = spectral_images.to(self.device, non_blocking=True) if spectral_images is not None else None
                labels = labels.to(self.device, non_blocking=True)

                with autocast(enabled=self.scaler.is_enabled()):
                    # MODIFIED: Pass both RGB and Spectral to model for validation
                    outputs = self.model(rgb_images, spectral_images) 
                    loss = self.criterion(outputs, labels)
                
                if not torch.isfinite(loss):
                     logger.warning(f"Non-finite validation loss detected ({loss.item()}). Skipping.")
                     num_batches = max(1, num_batches - 1) 
                     continue

                total_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
        
        metrics = {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0} 
        if not all_preds or not all_labels:
             logger.warning("Validation yielded no results. Returning zero metrics.")
        else:
            all_preds_np = torch.cat(all_preds).numpy()
            all_labels_np = torch.cat(all_labels).numpy()
            metrics = compute_metrics(all_preds_np, all_labels_np, self.num_classes, class_names)

        log_str = f"Validation finished. Avg Loss: {avg_val_loss:.4f}"
        for k, v in metrics.items(): log_str += f", {k}: {v:.4f}"
        logger.info(log_str)
                    
        return avg_val_loss, metrics

    def save_model_checkpoint(self, path: str):
        """ Saves the model's state dictionary. """
        # (Keep as is)
        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model checkpoint saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model checkpoint to {path}: {e}")