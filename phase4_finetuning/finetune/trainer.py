# phase4_finetuning/finetune/trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
import re
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import math
from typing import Dict, Any, List, Tuple

# Use relative imports to access modules within the same package
from ..utils.metrics import compute_metrics
from ..utils.ema import EMA
from ..utils.losses import FocalLoss, CombinedLoss

class EnhancedFinetuner:
    """
    An advanced, configuration-driven trainer for fine-tuning models.
    This class manages the entire training lifecycle, including model setup,
    training loops, validation, advanced techniques (SWA, EMA, Mixup),
    and checkpointing. It is designed to be orchestrated by a main script
    that provides the configuration.
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: Dict[str, Any], output_dir: str):
        self.cfg = config
        self.device = config['device']
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)

        self.start_epoch = 1
        self.best_metric = 0.0 if self.cfg['evaluation']['early_stopping']['metric'] != "val_loss" else float('inf')
        self.history = defaultdict(list)

        # Initialize core components
        self.model = self._setup_model(model)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        self.scaler = GradScaler(enabled=self.cfg['amp_enabled'])

        # Initialize advanced technique handlers
        self.ema = EMA(self.model, decay=self.cfg['evaluation']['ema_decay']) if self.cfg['evaluation']['use_ema'] else None
        self.swa_model = torch.optim.swa_utils.AveragedModel(self.model) if self.cfg['evaluation']['use_swa'] else None
        self.swa_scheduler = torch.optim.swa_utils.SWALR(self.optimizer, swa_lr=self.cfg['evaluation']['swa_lr']) if self.cfg['evaluation']['use_swa'] else None

        # Load states if resuming a fine-tuning run
        if self.cfg['model'].get('resume_finetune_path'):
            self._load_finetune_checkpoint(self.cfg['model']['resume_finetune_path'])
        
        self.logger.info(f"Trainer initialized. Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters.")

    def run(self):
        """ The main entry point to start the training and validation process. """
        self.logger.info(f"Starting fine-tuning run. Monitoring '{self.cfg['evaluation']['early_stopping']['metric']}' for best model.")
        patience_counter = 0

        for epoch in range(self.start_epoch, self.cfg['training']['epochs'] + 1):
            self.logger.info(f"--- Starting Epoch {epoch}/{self.cfg['training']['epochs']} ---")
            
            self._set_parameter_groups_for_epoch(epoch)
            
            train_loss = self._train_one_epoch(epoch)
            
            model_for_eval = self.ema.shadow if self.ema else self.model
            val_loss, metrics = self._validate_one_epoch(epoch, model_for_eval)

            # Update history and scheduler
            self._update_history(train_loss, val_loss, metrics)
            
            # Early stopping and checkpointing logic
            current_metric_val = metrics.get(self.cfg['evaluation']['early_stopping']['metric'])
            if current_metric_val is not None:
                is_better = self._is_metric_better(current_metric_val)
                if is_better:
                    self.logger.info(f"Epoch {epoch}: New best metric! {self.cfg['evaluation']['early_stopping']['metric']} = {current_metric_val:.4f}")
                    self.best_metric = current_metric_val
                    patience_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    patience_counter += 1
                    self.logger.info(f"Epoch {epoch}: No improvement. Patience: {patience_counter}/{self.cfg['evaluation']['early_stopping']['patience']}")

            if patience_counter >= self.cfg['evaluation']['early_stopping']['patience']:
                self.logger.info("Early stopping triggered. Ending training.")
                break
        
        self.logger.info("Training finished. Finalizing...")
        self._finalize_run()

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg['training']['epochs']} [Train]")
        
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            is_mixed = False
            
            # Apply Mixup or CutMix
            if np.random.rand() < self.cfg['augmentations']['mixup_alpha']:
                images, labels_a, labels_b, lam = self._mixup_data(images, labels)
                is_mixed = True
            elif np.random.rand() < self.cfg['augmentations']['cutmix_prob']:
                images, labels_a, labels_b, lam = self._cutmix_data(images, labels)
                is_mixed = True

            with autocast(enabled=self.cfg['amp_enabled']):
                # Assuming model might return a tuple (main_logits, aux_logits)
                outputs = self.model(rgb_img=images, mode='classify')
                main_logits = outputs[0] if isinstance(outputs, tuple) else outputs

                if is_mixed:
                    loss = lam * self.criterion(main_logits, labels_a) + (1 - lam) * self.criterion(main_logits, labels_b)
                else:
                    loss = self.criterion(main_logits, labels)
                
                loss_scaled = loss / self.cfg['training']['accumulation_steps']
            
            self.scaler.scale(loss_scaled).backward()

            if (i + 1) % self.cfg['training']['accumulation_steps'] == 0:
                if self.cfg['training']['clip_grad_norm']:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['training']['clip_grad_norm'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                if self.scheduler: self.scheduler.step()
                if self.ema: self.ema.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
        
        if self.swa_model and epoch >= self.cfg['training']['epochs'] * self.cfg['evaluation']['swa_start_epoch_ratio']:
            self.swa_model.update_parameters(self.model)
            if self.swa_scheduler: self.swa_scheduler.step()
            
        return total_loss / len(self.train_loader)

    def _validate_one_epoch(self, epoch: int, model_to_eval: nn.Module) -> Tuple[float, Dict[str, Any]]:
        model_to_eval.eval()
        if self.ema: self.ema.apply_shadow() # Use EMA weights for validation

        total_loss = 0.0
        all_preds, all_labels = [], []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Validate]")
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                with autocast(enabled=self.cfg['amp_enabled']):
                    if self.cfg['evaluation']['tta_enabled']:
                        outputs = self._tta_inference(model_to_eval, images)
                    else:
                        raw_outputs = model_to_eval(rgb_img=images, mode='classify')
                        outputs = raw_outputs[0] if isinstance(raw_outputs, tuple) else raw_outputs

                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                all_preds.append(torch.argmax(outputs, dim=1).cpu())
                all_labels.append(labels.cpu())
        
        if self.ema: self.ema.restore() # Restore original model weights

        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(
            torch.cat(all_preds).numpy(),
            torch.cat(all_labels).numpy(),
            self.cfg['data']['num_classes'],
            self.train_loader.dataset.get_class_names()
        )
        self.logger.info(f"Validation Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={metrics.get('accuracy',0):.4f}, F1-Macro={metrics.get('f1_macro',0):.4f}")
        return avg_loss, metrics

    # --- Helper Methods ---
    def _setup_model(self, model):
        if not self.cfg['model'].get('resume_finetune_path'):
            self._load_ssl_weights(model)
        
        if self.cfg['torch_compile']['enable']:
            try:
                model = torch.compile(model, mode=self.cfg['torch_compile']['mode'])
                self.logger.info(f"Model compiled with mode '{self.cfg['torch_compile']['mode']}'.")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
        return model.to(self.device)
    
    def _create_optimizer(self):
        param_groups = self._get_param_groups()
        opt_cfg = self.cfg['training']['optimizer']
        return AdamW(param_groups, weight_decay=opt_cfg['weight_decay'])

    def _get_param_groups(self):
        head_name = 'classifier_head'
        head_params = [p for n, p in self.model.named_parameters() if head_name in n and p.requires_grad]
        backbone_params = [p for n, p in self.model.named_parameters() if head_name not in n and p.requires_grad]
        opt_cfg = self.cfg['training']['optimizer']
        return [
            {'params': backbone_params, 'lr': opt_cfg['lr_backbone_unfrozen'], 'name': 'backbone'},
            {'params': head_params, 'lr': opt_cfg['lr_head_unfrozen'], 'name': 'head'}
        ]
        
    def _create_scheduler(self):
        sched_cfg = self.cfg['training']['scheduler']
        if sched_cfg['name'].lower() == 'onecyclelr':
            return OneCycleLR(
                self.optimizer,
                max_lr=[pg['lr'] for pg in self.optimizer.param_groups],
                total_steps=self.cfg['training']['epochs'] * len(self.train_loader) // self.cfg['training']['accumulation_steps'],
                pct_start=float(sched_cfg['pct_start']),
                div_factor=float(sched_cfg['div_factor']),
                final_div_factor=float(sched_cfg['final_div_factor']),
            )
        return None

    def _create_criterion(self):
        loss_cfg = self.cfg['training']['loss']
        class_weights = None
        if loss_cfg['use_class_weights']:
            class_weights = self.train_loader.dataset.get_class_weights().to(self.device)
            
        if loss_cfg['type'].lower() == 'combined':
            return CombinedLoss(
                num_classes=self.cfg['data']['num_classes'],
                smoothing=loss_cfg['label_smoothing'],
                focal_alpha=loss_cfg['focal_alpha'],
                focal_gamma=loss_cfg['focal_gamma'],
                ce_weight=loss_cfg['weights'].get('ce', 0.5),
                focal_weight=loss_cfg['weights'].get('focal', 0.5),
                class_weights_tensor=class_weights
            ).to(self.device)
        return nn.CrossEntropyLoss(label_smoothing=loss_cfg['label_smoothing'], weight=class_weights).to(self.device)

    def _set_parameter_groups_for_epoch(self, epoch):
        is_frozen_phase = epoch <= self.cfg['training']['freeze_backbone_epochs']
        opt_cfg = self.cfg['training']['optimizer']
        
        # This logic is now primarily for requires_grad, as OneCycleLR handles LR interpolation
        for group in self.optimizer.param_groups:
            if group['name'] == 'backbone':
                for param in group['params']:
                    param.requires_grad = not is_frozen_phase
        
        if is_frozen_phase and epoch == 1:
            self.logger.info("Epoch 1: Backbone is FROZEN.")
            # Set backbone LR to 0 for initial frozen epochs
            self.optimizer.param_groups[0]['lr'] = 0.0
        elif epoch == self.cfg['training']['freeze_backbone_epochs'] + 1:
            self.logger.info(f"Epoch {epoch}: Backbone UNFROZEN. Restoring LR.")
            self.optimizer.param_groups[0]['lr'] = opt_cfg['lr_backbone_unfrozen']
            # Recreate scheduler to use the new learning rates
            self.scheduler = self._create_scheduler()
            self.logger.info(f"Trainable params after unfreeze: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def _update_history(self, train_loss, val_loss, metrics):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        for k, v in metrics.items():
            self.history[f"val_{k}"].append(v)
        if self.scheduler:
            # Log LR for each parameter group
            for i, pg in enumerate(self.optimizer.param_groups):
                self.history[f"lr_group_{pg.get('name', i)}"].append(pg['lr'])

    def _is_metric_better(self, new_metric):
        metric_name = self.cfg['evaluation']['early_stopping']['metric']
        delta = self.cfg['evaluation']['early_stopping']['min_delta']
        if metric_name == "val_loss":
            return new_metric < self.best_metric - delta
        else:
            return new_metric > self.best_metric + delta

    def _load_ssl_weights(self, model):
        path = self.cfg['model']['ssl_pretrained_path']
        self.logger.info(f"Loading SSL backbone weights from: {path}")
        try:
            ckpt = torch.load(path, map_location=self.device)
            ssl_sd = ckpt.get('model_backbone_state_dict', ckpt.get('model_state_dict'))
            
            # This is a robust way to load, ignoring the classification head
            msg = model.load_state_dict(ssl_sd, strict=False)
            self.logger.info(f"SSL weights loaded. Mismatched/missing keys: {msg}")
        except Exception as e:
            self.logger.error(f"Failed to load SSL weights from {path}: {e}", exc_info=True)

    def _load_finetune_checkpoint(self, path):
        self.logger.info(f"Resuming fine-tune state from: {path}")
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if self.ema and 'ema_state_dict' in ckpt:
                self.ema.load_state_dict(ckpt['ema_state_dict'])
            if self.swa_model and 'swa_model_state_dict' in ckpt:
                self.swa_model.load_state_dict(ckpt['swa_model_state_dict'])
            
            self.start_epoch = ckpt.get('epoch', 1) + 1
            self.best_metric = ckpt.get('best_val_metric', self.best_metric)
            self.logger.info(f"Resumed from epoch {self.start_epoch - 1}. Best metric so far: {self.best_metric:.4f}")
        except Exception as e:
            self.logger.error(f"Failed to load fine-tune checkpoint from {path}: {e}", exc_info=True)

    def _save_checkpoint(self, epoch, is_best=False):
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        data = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_metric,
            'config': self.cfg
        }
        if self.scheduler: data['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.ema: data['ema_state_dict'] = self.ema.state_dict()
        if self.swa_model: data['swa_model_state_dict'] = self.swa_model.state_dict()

        path = os.path.join(self.checkpoints_dir, 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth')
        torch.save(data, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def _finalize_run(self):
        if self.swa_model:
            self.logger.info("Finalizing SWA: updating batch norm stats...")
            torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device=self.device)
            # Final validation on SWA model
            self.logger.info("Performing final validation on SWA model...")
            self._validate_one_epoch(self.cfg['training']['epochs'], self.swa_model)
        
        self.logger.info("Plotting training curves...")
        # Add plotting logic here if needed

    def _mixup_data(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply Mixup augmentation to the batch."""
        batch_size = images.size(0)
        lam = np.random.beta(self.cfg['augmentations']['mixup_alpha'], self.cfg['augmentations']['mixup_alpha'])
        index = torch.randperm(batch_size).to(self.device)
        mixed_images = lam * images + (1 - lam) * images[index, :]
        return mixed_images, labels, labels[index], lam

    def _cutmix_data(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation to the batch."""
        batch_size = images.size(0)
        lam = np.random.beta(1.0, 1.0)  # Using a beta distribution for CutMix
        index = torch.randperm(batch_size).to(self.device)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        images_cloned = images.clone()
        images_cloned[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        return images_cloned, labels, labels[index], lam

    def _rand_bbox(self, size: Tuple[int, ...], lam: float) -> Tuple[int, int, int, int]:
        """Generate random bounding box for CutMix."""
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)
        bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
        bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2