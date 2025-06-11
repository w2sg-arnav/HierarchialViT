# phase4_finetuning/finetune/trainer.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Any, Tuple
import torchvision.transforms.v2 as T_v2

# Use relative imports
from ..utils.metrics import compute_metrics
from ..utils.ema import EMA
from ..utils.losses import CombinedLoss

class EnhancedFinetuner:
    """ An advanced, configuration-driven trainer for fine-tuning models. """
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
        self.best_metric = -1.0 if self.cfg['evaluation']['early_stopping']['metric'] != 'val_loss' else float('inf')
        self.history = defaultdict(list)

        self.model = self._setup_model(model)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        self.scaler = GradScaler(enabled=self.cfg['amp_enabled'])
        self.ema = EMA(self.model, decay=self.cfg['evaluation']['ema_decay']) if self.cfg['evaluation']['use_ema'] else None

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
            val_loss, metrics = self._validate_one_epoch(epoch)
            self._update_history(train_loss, val_loss, metrics)
            
            # Note: OneCycleLR is a step-based scheduler, handled in the training loop.
            # Other schedulers would be stepped here.
            # if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            #     self.scheduler.step()

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
            
            if patience_counter >= self.cfg['evaluation']['early_stopping']['patience']:
                self.logger.info(f"Early stopping triggered at epoch {epoch}. Ending training.")
                break
        
        self.logger.info("Training finished.")

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg['training']['epochs']} [Train]")
        
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            is_mixed = False
            
            # FIXED: Apply Mixup/Cutmix here
            if np.random.rand() < self.cfg['augmentations'].get('mixup_alpha', 0.0):
                images, labels_a, labels_b, lam = self._mixup_data(images, labels, alpha=self.cfg['augmentations']['mixup_alpha'])
                is_mixed = True
            elif np.random.rand() < self.cfg['augmentations'].get('cutmix_prob', 0.0):
                images, labels_a, labels_b, lam = self._cutmix_data(images, labels)
                is_mixed = True

            with autocast(enabled=self.cfg['amp_enabled']):
                outputs = self.model(rgb_img=images, mode='classify')
                main_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # FIXED: Correct loss calculation for mixed batches
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
                self.optimizer.zero_grad(set_to_none=True)
                
                if self.scheduler: self.scheduler.step()
                if self.ema: self.ema.update()

            total_loss += loss.item()
            lr_val = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}", lr=f"{lr_val:.2e}")
            
        return total_loss / len(self.train_loader)

    def _validate_one_epoch(self, epoch: int) -> Tuple[float, Dict[str, Any]]:
        model_to_eval = self.ema.shadow_model if self.ema else self.model
        model_to_eval.eval()
        
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
                # If TTA is used, outputs are probabilities, so need argmax
                preds = torch.argmax(outputs, dim=1) if outputs.ndim == 2 else outputs
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(
            torch.cat(all_preds), torch.cat(all_labels),
            self.cfg['data']['num_classes'], self.train_loader.dataset.get_class_names()
        )
        metrics['val_loss'] = avg_loss
        self.logger.info(f"Validation Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={metrics.get('accuracy',0):.4f}, F1-Macro={metrics.get('f1_macro',0):.4f}")
        return avg_loss, metrics

    # --- IMPLEMENTED HELPER METHODS ---
    def _tta_inference(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        original_logits = model(rgb_img=images, mode='classify')
        original_logits = original_logits[0] if isinstance(original_logits, tuple) else original_logits
        
        flipped_images = T_v2.functional.hflip(images)
        flipped_logits = model(rgb_img=flipped_images, mode='classify')
        flipped_logits = flipped_logits[0] if isinstance(flipped_logits, tuple) else flipped_logits
        
        # Average the softmax probabilities
        return (torch.softmax(original_logits, dim=1) + torch.softmax(flipped_logits, dim=1)) / 2.0

    def _mixup_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, y, y[index], lam

    def _cutmix_data(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = np.random.beta(1.0, 1.0)
        rand_index = torch.randperm(x.size(0)).to(self.device)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        x_clone = x.clone()
        x_clone[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
        lam_adjusted = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x_clone, y, y[rand_index], lam_adjusted

    def _rand_bbox(self, size: Tuple[int, ...], lam: float) -> Tuple[int, int, int, int]:
        W, H = size[3], size[2] # Corrected order for H, W
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
        
    # --- REFACTORED SETUP METHODS ---
    def _setup_model(self, model):
        if self.cfg['model'].get('ssl_pretrained_path'):
            self._load_ssl_weights(model)
        if self.cfg['torch_compile']['enable'] and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode=self.cfg['torch_compile']['mode'])
                self.logger.info(f"Model compiled with mode '{self.cfg['torch_compile']['mode']}'.")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
        return model.to(self.device)

    def _create_optimizer(self) -> AdamW:
        param_groups = self._get_param_groups()
        # LR will be managed by the scheduler, so initial value here is less critical for OneCycle
        return AdamW(param_groups, lr=1e-3)

    def _get_param_groups(self) -> list:
        # Differential learning rates for backbone and head
        head_name = 'classifier_head'
        head_params = [p for n, p in self.model.named_parameters() if head_name in n and p.requires_grad]
        backbone_params = [p for n, p in self.model.named_parameters() if head_name not in n and p.requires_grad]
        opt_cfg = self.cfg['training']['optimizer']
        return [
            {'params': backbone_params, 'lr': opt_cfg['lr_backbone_unfrozen'], 'name': 'backbone'},
            {'params': head_params, 'lr': opt_cfg['lr_head_unfrozen'], 'name': 'head'}
        ]

    def _create_scheduler(self):
        cfg = self.cfg['training']['scheduler']
        if cfg['name'].lower() == 'onecyclelr':
            max_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
            total_steps = (self.cfg['training']['epochs'] * len(self.train_loader)) // self.cfg['training']['accumulation_steps']
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=max_lrs, total_steps=total_steps,
                pct_start=cfg['pct_start'], div_factor=cfg['div_factor'], final_div_factor=cfg['final_div_factor']
            )
        return None

    def _set_parameter_groups_for_epoch(self, epoch: int):
        # FIXED: Simplified and corrected layer freezing logic
        is_frozen_phase = epoch <= self.cfg['training']['freeze_backbone_epochs']
        backbone_group = next((g for g in self.optimizer.param_groups if g['name'] == 'backbone'), None)
        
        if backbone_group is None: return

        if is_frozen_phase and epoch == 1:
            self.logger.info("Epoch 1: Backbone is FROZEN.")
            for param in backbone_group['params']:
                param.requires_grad = False
        elif epoch == self.cfg['training']['freeze_backbone_epochs'] + 1:
            self.logger.info(f"Epoch {epoch}: Backbone UNFROZEN.")
            for param in backbone_group['params']:
                param.requires_grad = True
            # The optimizer will now pick up these gradients automatically.
            self.logger.info(f"Trainable params after unfreeze: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def _create_criterion(self):
        loss_cfg = self.cfg['training']['loss']
        class_weights = self.train_loader.dataset.get_class_weights().to(self.device) if loss_cfg['use_class_weights'] else None
        
        if loss_cfg['type'].lower() == 'combined':
            return CombinedLoss(
                num_classes=self.cfg['data']['num_classes'],
                smoothing=loss_cfg['label_smoothing'],
                focal_alpha=loss_cfg['focal_alpha'],
                focal_gamma=loss_cfg['focal_gamma'],
                ce_weight=loss_cfg['weights']['ce'],
                focal_weight=loss_cfg['weights']['focal'],
                class_weights_tensor=class_weights
            ).to(self.device)
        return nn.CrossEntropyLoss(label_smoothing=loss_cfg['label_smoothing'], weight=class_weights).to(self.device)

    def _update_history(self, train_loss, val_loss, metrics):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        for k, v in metrics.items(): self.history[f"val_{k}"].append(v)
        if self.scheduler:
            for i, pg in enumerate(self.optimizer.param_groups): self.history[f"lr_group_{pg.get('name', i)}"].append(pg['lr'])

    def _is_metric_better(self, new_metric):
        metric_name = self.cfg['evaluation']['early_stopping']['metric']
        delta = self.cfg['evaluation']['early_stopping']['min_delta']
        return new_metric < self.best_metric - delta if 'loss' in metric_name else new_metric > self.best_metric + delta

    def _load_ssl_weights(self, model):
        path = self.cfg['model']['ssl_pretrained_path']
        self.logger.info(f"Attempting to load SSL backbone weights from: {path}")
        if not path or not os.path.exists(path): self.logger.error(f"SSL checkpoint not found at {path}"); return
        try:
            ckpt = torch.load(path, map_location=self.device)
            ssl_sd = ckpt.get('model_backbone_state_dict')
            if ssl_sd:
                msg = model.load_state_dict(ssl_sd, strict=False)
                self.logger.info(f"SSL weights loaded. Missing keys in model: {msg.missing_keys}. Unexpected keys in checkpoint: {msg.unexpected_keys}")
            else: self.logger.error("Could not find 'model_backbone_state_dict' in the SSL checkpoint.")
        except Exception as e: self.logger.error(f"Failed to load SSL weights from {path}: {e}", exc_info=True)

    def _load_finetune_checkpoint(self, path):
        # ... (Your original fine-tune checkpoint loading logic is good) ...
        pass

    def _save_checkpoint(self, epoch, is_best=False):
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        data = { 'epoch': epoch, 'model_state_dict': model_to_save.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'best_val_metric': self.best_metric, 'config': self.cfg }
        if self.scheduler: data['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.ema: data['ema_state_dict'] = self.ema.state_dict()
        path = os.path.join(self.checkpoints_dir, 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth')
        torch.save(data, path)
        self.logger.info(f"Saved checkpoint to {path}")