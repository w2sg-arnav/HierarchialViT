# phase3_pretraining/pretrain/trainer.py
from typing import Optional, Union, Tuple, Dict, Any
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import logging
import sys
from tqdm import tqdm
import os
import math
import time # For timing batch processing

from ..models.hvt_wrapper import HVTForPretraining

logger = logging.getLogger(__name__)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
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
                 val_loader_for_probe: Optional[DataLoader] = None,
                 h100_optim_config: Optional[Dict[str, Any]] = None):
        logger.info("[TRAINER INIT] Initializing Pretrainer...")
        self.model = model
        self.augmentations = augmentations
        self.loss_fn = loss_fn
        self.device = device
        self.train_loader_for_probe = train_loader_for_probe
        self.val_loader_for_probe = val_loader_for_probe
        self.config = h100_optim_config if h100_optim_config is not None else {}
        logger.debug(f"[TRAINER INIT] Config used by trainer: {self.config}")

        optimizer_name = self.config.get("pretrain_optimizer", "AdamW").lower()
        lr = self.config.get('pretrain_lr', 5e-4)
        weight_decay = self.config.get('pretrain_weight_decay', 0.05)
        parameters_to_optimize = filter(lambda p: p.requires_grad, self.model.parameters())

        if optimizer_name == "adamw":
            self.optimizer = AdamW(parameters_to_optimize, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
             momentum = self.config.get("pretrain_momentum", 0.9)
             self.optimizer = SGD(parameters_to_optimize, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            logger.warning(f"[TRAINER INIT] Unsupported pretrain_optimizer: {optimizer_name}. Defaulting to AdamW.")
            self.optimizer = AdamW(parameters_to_optimize, lr=lr, weight_decay=weight_decay)
        logger.info(f"[TRAINER INIT] Optimizer: {self.optimizer.__class__.__name__}, LR: {lr}, Weight Decay: {weight_decay}")

        self.scheduler = None
        self.scheduler_name = self.config.get("pretrain_scheduler", "None").lower()
        self.pretrain_epochs_total = self.config.get("pretrain_epochs", 100)
        self.warmup_epochs_val = self.config.get("warmup_epochs", 0)

        if self.scheduler_name == "cosineannealinglr":
            t_max_epochs = self.pretrain_epochs_total - self.warmup_epochs_val
            if t_max_epochs <= 0: t_max_epochs = 1 # Ensure T_max is at least 1
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max_epochs, eta_min=self.config.get("eta_min_lr", 0))
            logger.info(f"[TRAINER INIT] Scheduler: CosineAnnealingLR, T_max={t_max_epochs} epochs (after warmup if any).")
        elif self.scheduler_name == "warmupcosine":
            logger.info(f"[TRAINER INIT] Scheduler: WarmupCosine selected. Will be initialized at start of first epoch.")
        elif self.scheduler_name != "none":
            logger.warning(f"[TRAINER INIT] Unknown scheduler name: {self.scheduler_name}. No scheduler will be used.")


        self.scaler = GradScaler(enabled=(self.device == 'cuda' and torch.cuda.is_available()))
        self.accum_steps = self.config.get('accumulation_steps', 1)
        self.current_step_in_accumulation = 0
        self.clip_grad_norm_val = self.config.get('clip_grad_norm', None)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"[TRAINER INIT] Pretrainer initialized. Wrapper params: Total={total_params}, Trainable={trainable_params}")
        if self.clip_grad_norm_val: logger.info(f"[TRAINER INIT] Gradient clipping enabled: Max norm {self.clip_grad_norm_val}")
        logger.info(f"[TRAINER INIT] Accumulation steps: {self.accum_steps}")
        logger.info(f"[TRAINER INIT] AMP (scaler) enabled: {self.scaler.is_enabled()}")


    def train_one_epoch(self, train_loader: DataLoader, current_epoch: int, total_epochs: int, batches_per_epoch: int):
        self.model.train()
        logger.info(f"[TRAINER E{current_epoch}/{total_epochs}] train_one_epoch started. Batches: {batches_per_epoch}. Accum: {self.accum_steps}")
        total_loss = 0.0; actual_optimizer_steps = 0; processed_batches_for_avg = 0
        batch_times = []

        # Initialize WarmupCosine scheduler here if it's the first epoch and not already initialized
        if current_epoch == 1 and self.scheduler_name == "warmupcosine" and self.scheduler is None:
            num_training_steps = total_epochs * batches_per_epoch # Total steps for all epochs
            num_warmup_steps = self.warmup_epochs_val * batches_per_epoch
            if num_training_steps == 0: # Should not happen if dataloader is not empty
                 logger.error("Total training steps is 0. Cannot initialize WarmupCosine scheduler. Disabling scheduler.")
                 self.scheduler_name = "none"
            elif num_warmup_steps >= num_training_steps: # Warn if warmup is too long
                logger.warning(f"Warmup steps ({num_warmup_steps}) >= Total steps ({num_training_steps}). Setting warmup to 10% of total or 1 batch if total > 0.")
                num_warmup_steps = max(1, int(0.1 * num_training_steps)) if num_training_steps > 0 else 0
                if num_warmup_steps == 0: # If still 0, disable.
                    logger.error("Effective warmup steps is 0 after adjustment. Disabling WarmupCosine scheduler.")
                    self.scheduler_name = "none"
                else:
                    self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
                    logger.info(f"[TRAINER E{current_epoch}] WarmupCosine scheduler initialized: WarmupSteps={num_warmup_steps}, TotalSteps={num_training_steps}")
            else:
                self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
                logger.info(f"[TRAINER E{current_epoch}] WarmupCosine scheduler initialized: WarmupSteps={num_warmup_steps}, TotalSteps={num_training_steps}")


        pbar = tqdm(enumerate(train_loader), total=batches_per_epoch, desc=f"Epoch {current_epoch}/{total_epochs} [SSL Pre-training]", file=sys.stdout, dynamic_ncols=True, mininterval=1.0)
        self.optimizer.zero_grad(set_to_none=True) # Zero grads once before the loop

        for batch_idx, batch_data in pbar:
            batch_start_time = time.time()
            if not batch_data:
                logger.warning(f"[TRAINER E{current_epoch} B{batch_idx+1}/{batches_per_epoch}] Empty batch. Skipping.")
                continue
            
            rgb_images, _ = batch_data # Assuming labels are not used in pretrain
            if rgb_images is None or rgb_images.numel() == 0:
                logger.warning(f"[TRAINER E{current_epoch} B{batch_idx+1}/{batches_per_epoch}] Empty images tensor in batch. Skipping.")
                continue
            
            data_load_time = time.time() - batch_start_time
            
            # --- Data to device ---
            to_device_start_time = time.time()
            try:
                rgb_images = rgb_images.to(self.device, non_blocking=(self.device == 'cuda'))
            except Exception as e_device:
                logger.error(f"[TRAINER E{current_epoch} B{batch_idx+1}] Error moving images to device: {e_device}. Skipping batch.", exc_info=True)
                continue
            to_device_time = time.time() - to_device_start_time
            # logger.debug(f"[TRAINER E{current_epoch} B{batch_idx+1}] Images to device. Shape: {rgb_images.shape}")

            # --- Augmentations ---
            aug_start_time = time.time()
            try:
                view1, view2 = self.augmentations(rgb_images)
                # logger.debug(f"[TRAINER E{current_epoch} B{batch_idx+1}] Augmentations applied. V1:{view1.shape}, V2:{view2.shape}")
            except Exception as e_aug:
                logger.error(f"[TRAINER E{current_epoch} B{batch_idx+1}] Augmentation error: {e_aug}. Skipping batch.", exc_info=True)
                continue
            aug_time = time.time() - aug_start_time

            loss_value_this_iter = 0.0
            # --- Forward pass, loss, backward pass ---
            fwd_loss_bwd_start_time = time.time()
            try:
                with autocast(enabled=self.scaler.is_enabled()):
                    # logger.debug(f"[TRAINER E{current_epoch} B{batch_idx+1}] Autocast. Model forward view1...")
                    projected_features1 = self.model(rgb_img=view1, mode='pretrain')
                    # logger.debug(f"[TRAINER E{current_epoch} B{batch_idx+1}] Model forward view2...")
                    projected_features2 = self.model(rgb_img=view2, mode='pretrain')
                    loss = self.loss_fn(projected_features1, projected_features2)
                
                loss_value_this_iter = loss.item() # Store unscaled loss for logging
                # logger.debug(f"[TRAINER E{current_epoch} B{batch_idx+1}] Raw loss: {loss_value_this_iter:.4f}")

                if self.accum_steps > 1:
                    loss = loss / self.accum_steps
                    # logger.debug(f"Accum loss: {loss.item():.4f}")
            
            except Exception as e_fwd:
                logger.error(f"[TRAINER E{current_epoch} B{batch_idx+1}] Forward/loss error: {e_fwd}. Skipping batch.", exc_info=True)
                continue

            if not torch.isfinite(loss):
                logger.error(f"[TRAINER E{current_epoch} B{batch_idx+1}] Non-finite loss ({loss.item()}). Skipping gradient update for this iteration.")
                # Optionally, zero_grad here if not accumulating, or handle carefully
                # self.optimizer.zero_grad(set_to_none=True) # Might be too aggressive if part of accumulation
                continue # Skip backward and step
            
            self.scaler.scale(loss).backward()
            fwd_loss_bwd_time = time.time() - fwd_loss_bwd_start_time
            
            self.current_step_in_accumulation += 1

            # --- Optimizer step ---
            opt_step_time = 0.0
            if self.current_step_in_accumulation >= self.accum_steps:
                opt_step_start_time = time.time()
                if self.clip_grad_norm_val is not None:
                    self.scaler.unscale_(self.optimizer) # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm_val)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True) # Zero grads after step
                
                actual_optimizer_steps +=1
                self.current_step_in_accumulation = 0
                
                # Step per-iteration schedulers (like WarmupCosine)
                if self.scheduler and self.scheduler_name == "warmupcosine":
                    self.scheduler.step()
                opt_step_time = time.time() - opt_step_start_time
            
            total_loss += loss_value_this_iter # Accumulate the original (un-scaled by accum_steps) loss for average
            processed_batches_for_avg += 1
            
            batch_total_time = time.time() - batch_start_time
            batch_times.append(batch_total_time)
            
            pbar.set_postfix({
                "Loss": f"{loss_value_this_iter:.4f}", 
                "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                "Time/batch": f"{batch_total_time:.3f}s"
            })
            if batch_idx < 3 or batch_idx % 50 == 0 : # Log details for first few batches and then periodically
                logger.debug(
                    f"[TRAINER E{current_epoch} B{batch_idx+1}/{batches_per_epoch}] "
                    f"Loss: {loss_value_this_iter:.4f} (accum {loss.item():.4f}), "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, "
                    f"Times(s): Tot={batch_total_time:.3f} (Load={data_load_time:.3f}, Device={to_device_time:.3f}, Aug={aug_time:.3f}, Fwd/Bwd={fwd_loss_bwd_time:.3f}, Opt={opt_step_time:.3f})"
                )
        
        pbar.close()
        
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        logger.info(f"[TRAINER E{current_epoch}] Batch loop finished. Actual optimizer steps: {actual_optimizer_steps}. Avg batch time: {avg_batch_time:.3f}s.")

        # Handle any remaining gradients if epoch ends mid-accumulation (should be rare with drop_last=True)
        if self.current_step_in_accumulation > 0:
            logger.warning(f"[TRAINER E{current_epoch}] Epoch ended mid-accumulation ({self.current_step_in_accumulation}/{self.accum_steps} steps). Applying remaining gradients.")
            if self.clip_grad_norm_val is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm_val)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler and self.scheduler_name == "warmupcosine": # Step scheduler if optimizer stepped
                 self.scheduler.step()
            self.current_step_in_accumulation = 0


        # Step epoch-based schedulers (like CosineAnnealingLR)
        if self.scheduler and self.scheduler_name == "cosineannealinglr":
            # Manual linear warmup for CosineAnnealingLR if warmup_epochs is set
            if self.warmup_epochs_val > 0 and current_epoch <= self.warmup_epochs_val:
                 initial_lr = self.config.get('pretrain_lr', 5e-4) # Base LR for warmup
                 # Ensure lr is not negative or zero if current_epoch is 0 or warmup_epochs_val is 0
                 warmup_factor = current_epoch / self.warmup_epochs_val if self.warmup_epochs_val > 0 else 1.0
                 for param_group in self.optimizer.param_groups:
                     param_group['lr'] = initial_lr * warmup_factor
                 logger.info(f"[TRAINER E{current_epoch}] CosineAnnealingLR manual warmup. New LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            elif current_epoch > self.warmup_epochs_val: # Step scheduler after warmup
                self.scheduler.step()
                logger.info(f"[TRAINER E{current_epoch}] CosineAnnealingLR (epoch-based) stepped. New LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            # If current_epoch == self.warmup_epochs_val, LR has been set by warmup, scheduler.step() will use this as starting point.

        avg_epoch_loss = total_loss / processed_batches_for_avg if processed_batches_for_avg > 0 else 0.0
        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"[TRAINER E{current_epoch}] train_one_epoch finished. AvgLoss: {avg_epoch_loss:.4f}, FinalLR: {current_lr:.2e}")
        return avg_epoch_loss

    def evaluate_linear_probe(self, current_epoch: int):
        if self.train_loader_for_probe is None or self.val_loader_for_probe is None:
            logger.warning(f"[PROBE E{current_epoch}] Loaders not provided. Skipping linear probe.")
            return -1.0
        if len(self.train_loader_for_probe.dataset) == 0 or len(self.val_loader_for_probe.dataset) == 0:
            logger.warning(f"[PROBE E{current_epoch}] Train or Val loader for probe is empty. Skipping.")
            return -1.0

        logger.info(f"[PROBE E{current_epoch}] --- Starting Linear Probe ---")
        self.model.eval() # Set backbone to eval mode for feature extraction
        
        feature_dim = -1
        try:
            # Determine feature dimension from a sample batch
            sample_rgb, _ = next(iter(self.val_loader_for_probe))
            sample_rgb = sample_rgb.to(self.device)
            with torch.no_grad():
                # Use a small slice to avoid large memory use if batch is big
                sample_features = self.model(rgb_img=sample_rgb[:2], mode='probe_extract') 
            feature_dim = sample_features.shape[1]
            logger.info(f"[PROBE E{current_epoch}] Feature dim determined as: {feature_dim}")
        except StopIteration:
            logger.error(f"[PROBE E{current_epoch}] Validation loader for probe is empty. Cannot determine feature dim.")
            self.model.train() # Set model back to train mode
            return -1.0
        except Exception as e:
            logger.error(f"[PROBE E{current_epoch}] Error determining feature dim for probe: {e}", exc_info=True)
            self.model.train()
            return -1.0

        if feature_dim <= 0:
            logger.error(f"[PROBE E{current_epoch}] Invalid feature_dim: {feature_dim}. Cannot proceed with probe.")
            self.model.train(); return -1.0

        # Linear classifier
        num_classes = self.config.get('num_classes', 7) # Default to 7 if not in config
        linear_probe_classifier = nn.Linear(feature_dim, num_classes).to(self.device)
        
        # Optimizer for the probe
        probe_optim_name = self.config.get("probe_optimizer", "AdamW").lower()
        probe_lr = self.config.get('linear_probe_lr', 1e-3)
        probe_wd = self.config.get('probe_weight_decay', 0.0)
        probe_momentum = self.config.get('probe_momentum', 0.9)

        if probe_optim_name == "adamw":
            probe_optimizer = AdamW(linear_probe_classifier.parameters(), lr=probe_lr, weight_decay=probe_wd)
        elif probe_optim_name == "sgd":
            probe_optimizer = SGD(linear_probe_classifier.parameters(), lr=probe_lr, momentum=probe_momentum, weight_decay=probe_wd)
        else:
            logger.warning(f"[PROBE E{current_epoch}] Unsupported probe optimizer: {probe_optim_name}. Defaulting to AdamW.")
            probe_optimizer = AdamW(linear_probe_classifier.parameters(), lr=probe_lr, weight_decay=probe_wd)
        logger.info(f"[PROBE E{current_epoch}] Probe Optimizer: {probe_optimizer.__class__.__name__}, LR: {probe_lr}, WD: {probe_wd}")

        probe_criterion = nn.CrossEntropyLoss().to(self.device)
        num_probe_epochs = self.config.get('linear_probe_epochs', 10)
        logger.info(f"[PROBE E{current_epoch}] Training probe for {num_probe_epochs} epochs using {num_classes} classes.")

        for probe_epoch in range(1, num_probe_epochs + 1):
            linear_probe_classifier.train()
            probe_epoch_loss = 0.0
            probe_batches = 0
            probe_pbar_train = tqdm(self.train_loader_for_probe, desc=f"ProbeTrain EP {probe_epoch}/{num_probe_epochs} (SSL E{current_epoch})", file=sys.stdout, leave=False, dynamic_ncols=True, mininterval=1.0)
            for rgb_images, labels in probe_pbar_train:
                rgb_images, labels = rgb_images.to(self.device), labels.to(self.device)
                with torch.no_grad(): # Backbone features are frozen
                    features = self.model(rgb_img=rgb_images, mode='probe_extract')
                
                if features.shape[1] != feature_dim:
                    logger.error(f"[PROBE Train EP{probe_epoch}] Feature dimension mismatch! Expected {feature_dim}, Got {features.shape[1]}. Skipping batch.")
                    continue
                
                logits = linear_probe_classifier(features)
                loss = probe_criterion(logits, labels)
                
                probe_optimizer.zero_grad()
                loss.backward()
                probe_optimizer.step()
                
                probe_epoch_loss += loss.item()
                probe_batches += 1
                probe_pbar_train.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            avg_probe_train_loss = probe_epoch_loss / probe_batches if probe_batches > 0 else 0
            logger.debug(f"[PROBE E{current_epoch} - Train EP {probe_epoch}] Avg Train Loss: {avg_probe_train_loss:.4f}")

        # Validation for the linear probe
        linear_probe_classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            probe_pbar_val = tqdm(self.val_loader_for_probe, desc=f"ProbeVal (SSL E{current_epoch})", file=sys.stdout, leave=False, dynamic_ncols=True, mininterval=1.0)
            for rgb_images, labels in probe_pbar_val:
                rgb_images, labels = rgb_images.to(self.device), labels.to(self.device)
                features = self.model(rgb_img=rgb_images, mode='probe_extract')
                if features.shape[1] != feature_dim:
                    logger.error(f"[PROBE Val] Feature dimension mismatch! Expected {feature_dim}, Got {features.shape[1]}. Skipping batch.")
                    continue
                logits = linear_probe_classifier(features)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = (100 * correct / total) if total > 0 else 0.0
        logger.info(f"[PROBE E{current_epoch}] --- Linear Probe Val Accuracy: {accuracy:.2f}% ({correct}/{total}) ---")
        
        self.model.train() # Set backbone model back to train mode
        return accuracy

    def save_checkpoint(self, epoch: Union[int, str], best_metric: float = -1.0, file_name: Optional[str] = None):
        checkpoint_dir = self.config.get("checkpoint_dir", "pretrain_checkpoints")
        if not os.path.isabs(checkpoint_dir): # If relative, make it relative to project root or package root
            # Assuming project root is parent of 'phase3_pretraining' package
            package_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # .../phase3_pretraining
            project_root_path = os.path.dirname(package_root_path) # .../
            # Prefer checkpoint_dir relative to project_root if possible, or package_root
            # For simplicity, let's assume config "checkpoint_dir" is relative to where script is run or an absolute path.
            # If always relative to project root, it should be constructed there.
            # Here, if relative, it will be relative to current working directory.
            # Let's ensure it's in the project space, same as config.py logic
            if not os.path.isabs(self.config.get("PROJECT_ROOT_PATH", "")): #PROJECT_ROOT_PATH should be absolute
                 pass # Use config's PROJECT_ROOT_PATH as base
            
            # Using the same logic as in config.py for consistency:
            # This path is relative to project root if PROJECT_ROOT_PATH is defined and checkpoint_dir is not absolute
            project_root = self.config.get("PROJECT_ROOT_PATH", None)
            if project_root and not os.path.isabs(checkpoint_dir):
                 checkpoint_dir = os.path.join(project_root, checkpoint_dir)


        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger.info(f"[SAVE CKPT] Created checkpoint directory: {checkpoint_dir}")
        
        base_model_name_from_cfg = self.config.get("model_name", "hvt").replace("DiseaseAware", "").lower()
        if file_name is None:
            file_name = f"{base_model_name_from_cfg}_pretrain_epoch_{epoch}.pth"
        path = os.path.join(checkpoint_dir, file_name)
        
        logger.info(f"[SAVE CKPT] Attempting to save checkpoint to {path}...")
        try:
            # Prefer saving backbone state_dict and its init config
            backbone_state_dict = self.model.backbone.state_dict()
            backbone_init_config = self.model.backbone_init_config if hasattr(self.model, 'backbone_init_config') else {}
            
            optimizer_state_dict = self.optimizer.state_dict()
            scaler_state_dict = self.scaler.state_dict()
            scheduler_state_dict = self.scheduler.state_dict() if self.scheduler else None
            
            # Include the h100_optim_config (which is self.config in trainer) used for this run
            # This is important for reproducibility and loading.
            run_config = self.config

            save_dict = {
                'epoch': epoch,
                'best_probe_metric': best_metric,
                'model_state_dict': backbone_state_dict, # Backbone state dict
                'model_init_config': backbone_init_config, # Config used to init HVTBackbone
                'optimizer_state_dict': optimizer_state_dict,
                'scheduler_state_dict': scheduler_state_dict,
                'scaler_state_dict': scaler_state_dict,
                'run_config': run_config # The full config dict used for this training run
            }
            torch.save(save_dict, path)
            logger.info(f"[SAVE CKPT] Pretrained backbone checkpoint saved to {path} (SSL Epoch {epoch}, BestProbeMetric {best_metric:.4f})")

        except AttributeError as e:
            logger.error(f"[SAVE CKPT] Attribute error during checkpoint saving. Potentially model structure issue or missing 'backbone' attribute on self.model. Error: {e}. Saving full wrapper state dict as fallback.", exc_info=True)
            # Fallback: save the entire HVTForPretraining wrapper state_dict
            torch.save({
                'epoch': epoch,
                'best_probe_metric': best_metric,
                'model_state_dict': self.model.state_dict(), # Full wrapper
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'scaler_state_dict': self.scaler.state_dict(),
                'run_config': self.config,
                'is_wrapper_fallback': True # Flag to indicate this is a fallback save
            }, path.replace(".pth", "_wrapper_fallback.pth"))
            logger.info(f"[SAVE CKPT] Fallback: Full wrapper checkpoint saved to {path.replace('.pth', '_wrapper_fallback.pth')}")
        except Exception as e:
            logger.error(f"[SAVE CKPT] General error saving checkpoint to {path}: {e}", exc_info=True)