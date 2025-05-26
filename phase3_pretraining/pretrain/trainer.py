# phase3_pretraining/pretrain/trainer.py
from typing import Optional, Union, Tuple, Dict, Any
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import logging
import sys
from tqdm import tqdm
import os
import math
import time

from ..models.hvt_wrapper import HVTForPretraining
from ..config import config as runner_config

logger = logging.getLogger(__name__)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Creates a learning rate scheduler with a linear warmup phase followed by a cosine decay.
    Args:
        last_epoch (int): The index of the last epoch. Used for resuming.
                          Should be -1 for a new scheduler, or (start_epoch - 1 - 1) if start_epoch is 1-based.
                          More accurately, if resuming at epoch `e` (1-based), `last_epoch` should be `(e-1)*batches_per_epoch -1`.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

class Pretrainer:
    def __init__(self, model: HVTForPretraining,
                 augmentations, loss_fn, device: str,
                 train_loader_for_probe: Optional[DataLoader] = None,
                 val_loader_for_probe: Optional[DataLoader] = None):
        logger.info("Initializing Pretrainer...")
        self.model = model
        self.augmentations = augmentations
        self.loss_fn = loss_fn
        self.device = device
        self.train_loader_for_probe = train_loader_for_probe
        self.val_loader_for_probe = val_loader_for_probe
        self.run_config = runner_config

        optimizer_name = self.run_config.get("pretrain_optimizer", "AdamW").lower()
        lr = self.run_config.get('pretrain_lr', 5e-4)
        weight_decay = self.run_config.get('pretrain_weight_decay', 0.05)
        parameters_to_optimize = filter(lambda p: p.requires_grad, self.model.parameters())

        if optimizer_name == "adamw": self.optimizer = AdamW(parameters_to_optimize, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd": self.optimizer = SGD(parameters_to_optimize, lr=lr, momentum=self.run_config.get("pretrain_momentum",0.9), weight_decay=weight_decay)
        else: logger.warning(f"Unsupported optimizer: {optimizer_name}. Default AdamW."); self.optimizer = AdamW(parameters_to_optimize, lr=lr, weight_decay=weight_decay)
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}, BaseLR: {lr}, WD: {weight_decay}")

        self.scheduler = None # Will be initialized in _initialize_scheduler_if_needed
        self.scheduler_name = self.run_config.get("pretrain_scheduler", "None").lower()
        self.pretrain_epochs_total = self.run_config.get("pretrain_epochs", 100)
        self.warmup_epochs_val = self.run_config.get("warmup_epochs", 0)

        self.scaler = GradScaler(enabled=(self.device == 'cuda' and torch.cuda.is_available()))
        self.accum_steps = self.run_config.get('accumulation_steps', 1)
        self.current_step_in_accumulation = 0
        self.clip_grad_norm_val = self.run_config.get('clip_grad_norm', None)
        logger.info(f"Pretrainer init complete. AMP: {self.scaler.is_enabled()}, Accum: {self.accum_steps}, ClipGrad: {self.clip_grad_norm_val}")

    def _initialize_scheduler_if_needed(self, batches_per_epoch: int, last_completed_epoch_for_resume: int = 0):
        # last_completed_epoch_for_resume is 0-based (0 means starting from scratch or epoch 1)
        if self.scheduler is not None: return

        # Calculate last_step for LambdaLR. If resuming at start_epoch (1-based),
        # last_epoch for LambdaLR should be start_epoch - 1 (0-based).
        # If scheduler is per iteration, last_step = (start_epoch_0_based * batches_per_epoch) -1
        # If scheduler is per epoch, last_epoch_for_scheduler = start_epoch_0_based -1
        
        current_scheduler_step = -1 # Default for new scheduler
        if last_completed_epoch_for_resume > 0 : # Resuming
            if self.scheduler_name == "warmupcosine": # Per-iteration scheduler
                current_scheduler_step = last_completed_epoch_for_resume * batches_per_epoch -1
            elif self.scheduler_name == "cosineannealinglr": # Per-epoch scheduler
                current_scheduler_step = last_completed_epoch_for_resume -1
            logger.info(f"Resuming: Scheduler will be initialized with last_step/epoch around: {current_scheduler_step}")


        if self.scheduler_name == "cosineannealinglr":
            # T_max is total epochs *after* warmup
            t_max_epochs = self.pretrain_epochs_total - self.warmup_epochs_val
            if t_max_epochs <= 0: t_max_epochs = self.pretrain_epochs_total
            self.scheduler = CosineAnnealingLR(self.optimizer,
                                               T_max=t_max_epochs,
                                               eta_min=self.run_config.get("eta_min_lr", 1e-6),
                                               last_epoch=current_scheduler_step if current_scheduler_step >=0 and self.warmup_epochs_val == 0 else -1) # only if no warmup
            logger.info(f"Scheduler: CosineAnnealingLR (T_max={t_max_epochs} effective epochs). Last epoch for init: {self.scheduler.last_epoch}")
        elif self.scheduler_name == "warmupcosine":
            num_training_steps_total = self.pretrain_epochs_total * batches_per_epoch
            num_warmup_steps_total = self.warmup_epochs_val * batches_per_epoch
            if num_training_steps_total == 0: logger.error("Total training steps 0. Cannot init WarmupCosine."); return
            if num_warmup_steps_total >= num_training_steps_total:
                logger.warning(f"Warmup steps ({num_warmup_steps_total}) >= Total steps ({num_training_steps_total}). Adjusting.")
                num_warmup_steps_total = max(1, int(0.1 * num_training_steps_total))
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps_total, num_training_steps_total, last_epoch=current_scheduler_step)
            logger.info(f"Scheduler: WarmupCosine (WarmupSteps={num_warmup_steps_total}, TotalTrainSteps={num_training_steps_total}). Last epoch for init: {self.scheduler.last_epoch}")
        elif self.scheduler_name != "none":
            logger.warning(f"Unknown scheduler: {self.scheduler_name}. No scheduler used.")

    def train_one_epoch(self, train_loader: DataLoader, current_epoch: int, total_epochs: int, batches_per_epoch: int, last_completed_epoch_for_resume: int):
        self.model.train()
        # Pass last_completed_epoch_for_resume for correct scheduler initialization if resuming
        self._initialize_scheduler_if_needed(batches_per_epoch, last_completed_epoch_for_resume)

        # ... (rest of train_one_epoch remains largely the same) ...
        # Key change: Manual warmup for CosineAnnealingLR needs to consider current_epoch correctly
        total_loss = 0.0; actual_optimizer_steps = 0; num_finite_loss_batches = 0
        pbar = tqdm(enumerate(train_loader), total=batches_per_epoch, desc=f"Epoch {current_epoch}/{total_epochs} [SSL Pre-train]", dynamic_ncols=True, mininterval=1.0)
        
        # Zero grad once at the beginning of the epoch if not accumulating from previous leftovers (should be handled by end of epoch)
        if self.current_step_in_accumulation == 0:
            self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch_data in pbar:
            if not batch_data: logger.warning(f"E{current_epoch} B{batch_idx+1}: Empty batch. Skipping."); continue
            rgb_images, _ = batch_data # Labels not used for SimCLR
            if rgb_images is None or rgb_images.numel() == 0: logger.warning(f"E{current_epoch} B{batch_idx+1}: Empty images tensor. Skipping."); continue
            
            rgb_images = rgb_images.to(self.device, non_blocking=True)
            view1, view2 = self.augmentations(rgb_images)

            loss_value_iter = 0.0
            with autocast(enabled=self.scaler.is_enabled()):
                try:
                    projected1 = self.model(rgb_img=view1, mode='pretrain')
                    projected2 = self.model(rgb_img=view2, mode='pretrain')
                    loss = self.loss_fn(projected1, projected2)
                    loss_value_iter = loss.item()
                except Exception as e_fwd:
                    logger.error(f"E{current_epoch} B{batch_idx+1}: Forward/Loss error: {e_fwd}", exc_info=True)
                    continue 

            if not torch.isfinite(loss):
                logger.error(f"E{current_epoch} B{batch_idx+1}: Non-finite loss ({loss.item()}). Skipping update.")
                if self.current_step_in_accumulation == 0 : self.optimizer.zero_grad(set_to_none=True)
                continue

            loss_to_backward = loss / self.accum_steps if self.accum_steps > 1 else loss
            self.scaler.scale(loss_to_backward).backward()
            
            self.current_step_in_accumulation += 1
            if self.current_step_in_accumulation >= self.accum_steps:
                if self.clip_grad_norm_val:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                actual_optimizer_steps += 1
                self.current_step_in_accumulation = 0
                if self.scheduler and self.scheduler_name == "warmupcosine": self.scheduler.step()

            total_loss += loss_value_iter; num_finite_loss_batches +=1
            pbar.set_postfix({"Loss": f"{loss_value_iter:.4f}", "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}"})

        pbar.close()
        if self.current_step_in_accumulation > 0: # Should ideally not happen if drop_last=True and batches_per_epoch is accurate
            logger.warning(f"E{current_epoch}: Applying leftover gradients from accumulation ({self.current_step_in_accumulation} steps).")
            if self.clip_grad_norm_val: self.scaler.unscale_(self.optimizer); torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_val)
            self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler and self.scheduler_name == "warmupcosine": self.scheduler.step()
            self.current_step_in_accumulation = 0; actual_optimizer_steps +=1
        
        # Step epoch-based schedulers (like CosineAnnealingLR) OR handle manual warmup for it
        if self.scheduler and self.scheduler_name == "cosineannealinglr":
            if self.warmup_epochs_val > 0 and current_epoch <= self.warmup_epochs_val: # Manual warmup phase
                base_lr = self.run_config.get('pretrain_lr', 5e-4)
                # Ensure current_epoch is 1-based for this calculation
                warmup_factor = float(current_epoch) / float(self.warmup_epochs_val) if self.warmup_epochs_val > 0 else 1.0
                for group in self.optimizer.param_groups: group['lr'] = base_lr * warmup_factor
                logger.info(f"E{current_epoch} Manual CosineAnnealingLR warmup. LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            elif current_epoch > self.warmup_epochs_val: # Step scheduler after warmup
                self.scheduler.step()
                logger.info(f"E{current_epoch} CosineAnnealingLR stepped. LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            # If current_epoch == warmup_epochs_val, LR is set by warmup, next scheduler step will use this.

        avg_epoch_loss = total_loss / num_finite_loss_batches if num_finite_loss_batches > 0 else float('inf')
        logger.info(f"Epoch {current_epoch} Training Summary: AvgLoss={avg_epoch_loss:.4f}, OptSteps={actual_optimizer_steps}, FinalLR={self.optimizer.param_groups[0]['lr']:.2e}")
        return avg_epoch_loss

    # ... (evaluate_linear_probe and save_checkpoint remain mostly the same, but ensure save_checkpoint saves the correct epoch)
    def save_checkpoint(self, current_epoch_completed: Union[int, str], best_metric: float = -1.0, file_name_override: Optional[str] = None):
        # ... (Path construction as before) ...
        package_root = self.run_config.get("PACKAGE_ROOT_PATH", ".")
        checkpoint_dir_name = self.run_config.get("checkpoint_dir_name", "pretrain_checkpoints")
        abs_checkpoint_dir = os.path.join(package_root, checkpoint_dir_name)
        os.makedirs(abs_checkpoint_dir, exist_ok=True)

        file_name = file_name_override if file_name_override else \
                    f"{self.run_config.get('model_arch_name_for_ckpt','hvt_simclr')}_epoch_{current_epoch_completed}.pth"
        path = os.path.join(abs_checkpoint_dir, file_name)
        logger.info(f"Saving checkpoint to {path} (reflecting completion of epoch {current_epoch_completed})...")
        try:
            save_content = {
                'epoch': current_epoch_completed, # Save the epoch that was just COMPLETED
                'best_probe_metric': best_metric,
                'model_backbone_state_dict': self.model.backbone.state_dict(),
                'model_backbone_init_config': self.model.backbone_init_config,
                'projection_head_state_dict': self.model.projection_head.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'scaler_state_dict': self.scaler.state_dict(),
                'run_config_snapshot': self.run_config # Save the config dict used for this run
            }
            torch.save(save_content, path)
            logger.info(f"Checkpoint for epoch {current_epoch_completed} saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint for epoch {current_epoch_completed} to {path}: {e}", exc_info=True)


    def evaluate_linear_probe(self, current_ssl_epoch: int):
        # ... (this function can remain largely the same) ...
        # Ensure it correctly uses self.run_config for probe parameters
        if not (self.train_loader_for_probe and self.val_loader_for_probe and \
                len(self.train_loader_for_probe.dataset) > 0 and len(self.val_loader_for_probe.dataset) > 0):
            logger.warning(f"Probe E{current_ssl_epoch}: Loaders/datasets insufficient. Skipping probe."); return -1.0

        logger.info(f"--- Starting Linear Probe (after SSL Epoch {current_ssl_epoch}) ---")
        self.model.eval()
        
        # Determine feature_dim from the projection head's input, which is backbone output
        # Assumes projection_head.head[0] is the first nn.Linear layer
        try:
            feature_dim = self.model.projection_head.head[0].in_features
        except (AttributeError, IndexError) as e:
            logger.error(f"Could not determine feature_dim from projection_head for probe: {e}. Attempting fallback from backbone.")
            if hasattr(self.model.backbone, 'final_encoded_dim_rgb') and self.model.backbone.final_encoded_dim_rgb > 0 :
                 feature_dim = self.model.backbone.final_encoded_dim_rgb
            else:
                 logger.error("Fallback for feature_dim also failed. Cannot run probe."); self.model.train(); return -1.0

        if feature_dim <= 0: logger.error(f"Probe E{current_ssl_epoch}: Invalid feature_dim: {feature_dim}."); self.model.train(); return -1.0
        logger.info(f"Probe E{current_ssl_epoch}: Feature dim for linear classifier: {feature_dim}")

        num_classes = self.run_config.get('num_classes', 7)
        classifier = nn.Linear(feature_dim, num_classes).to(self.device)
        
        probe_optim_name = self.run_config.get("probe_optimizer", "SGD").lower()
        probe_lr = self.run_config.get('linear_probe_lr', 0.1)
        probe_wd = self.run_config.get('probe_weight_decay',0.0)
        probe_mom = self.run_config.get('probe_momentum',0.9)

        if probe_optim_name == "adamw": optimizer = AdamW(classifier.parameters(), lr=probe_lr, weight_decay=probe_wd)
        else: optimizer = SGD(classifier.parameters(), lr=probe_lr, momentum=probe_mom, weight_decay=probe_wd) # Default SGD
        
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        for epoch in range(1, self.run_config.get('linear_probe_epochs', 10) + 1):
            classifier.train()
            pbar_probe_train = tqdm(self.train_loader_for_probe, desc=f"ProbeTrain EP {epoch} (SSL E{current_ssl_epoch})", leave=False, dynamic_ncols=True)
            for rgb_imgs, labels in pbar_probe_train:
                rgb_imgs, labels = rgb_imgs.to(self.device), labels.to(self.device)
                with torch.no_grad(): features = self.model(rgb_img=rgb_imgs, mode='probe_extract')
                if features.shape[1] != feature_dim: logger.warning(f"Probe train feature dim mismatch. Exp {feature_dim}, Got {features.shape[1]}. Skipping batch."); continue
                logits = classifier(features); loss = criterion(logits, labels)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                pbar_probe_train.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        classifier.eval(); correct = 0; total = 0
        with torch.no_grad():
            pbar_probe_val = tqdm(self.val_loader_for_probe, desc=f"ProbeVal (SSL E{current_ssl_epoch})", leave=False, dynamic_ncols=True)
            for rgb_imgs, labels in pbar_probe_val:
                rgb_imgs, labels = rgb_imgs.to(self.device), labels.to(self.device)
                features = self.model(rgb_img=rgb_imgs, mode='probe_extract')
                if features.shape[1] != feature_dim: logger.warning(f"Probe val feature dim mismatch. Skipping batch."); continue
                logits = classifier(features); _, predicted = torch.max(logits.data, 1)
                total += labels.size(0); correct += (predicted == labels).sum().item()
        
        accuracy = (100 * correct / total) if total > 0 else 0.0
        logger.info(f"Linear Probe (SSL E{current_ssl_epoch}) Validation Accuracy: {accuracy:.2f}% ({correct}/{total})")
        self.model.train(); return accuracy