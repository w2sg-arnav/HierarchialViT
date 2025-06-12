# phase4_finetuning/finetune/trainer.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ConstantLR
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
from ..utils.losses import CombinedLoss, FocalLoss

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

        # --- RESTRUCTURED INITIALIZATION ---
        # 1. Load weights into the original model BEFORE compiling.
        uncompiled_model = self._load_initial_weights(model).to(self.device)

        # 2. Initialize EMA with the ORIGINAL, UN-COMPILED model.
        self.ema = EMA(uncompiled_model, decay=self.cfg['evaluation']['ema_decay']) if self.cfg['evaluation']['use_ema'] else None
        
        # 3. NOW, compile the model for the trainer to use.
        self.model = self._compile_model_if_enabled(uncompiled_model)
        # --- END OF RESTRUCTURED INITIALIZATION ---

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        self.scaler = GradScaler(enabled=self.cfg['training']['amp_enabled'])

        # Resuming a fine-tune run should load into the uncompiled model's state
        if self.cfg['model'].get('resume_finetune_path'):
            self._load_finetune_checkpoint(uncompiled_model, self.cfg['model']['resume_finetune_path'])
        
        self.logger.info(f"Trainer initialized. Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters.")

    def run(self):
        """ The main entry point to start the training and validation process. """
        self.logger.info(f"Starting fine-tuning run. Monitoring '{self.cfg['evaluation']['early_stopping']['metric']}' for best model.")
        patience_counter = 0
        total_epochs = self.cfg['training']['epochs']
        save_every = self.cfg['training'].get('save_every_n_epochs', 10)

        last_epoch_ran = self.start_epoch - 1
        try:
            for epoch in range(self.start_epoch, total_epochs + 1):
                last_epoch_ran = epoch
                self.logger.info(f"--- Starting Epoch {epoch}/{total_epochs} ---")
                self._set_parameter_groups_for_epoch(epoch)
                train_loss = self._train_one_epoch(epoch)
                val_loss, metrics = self._validate_one_epoch(epoch)
                self._update_history(train_loss, val_loss, metrics)

                metric_to_monitor = self.cfg['evaluation']['early_stopping']['metric']
                current_metric_val = metrics.get(metric_to_monitor)

                if current_metric_val is not None:
                    if self._is_metric_better(current_metric_val):
                        self.logger.info(f"Epoch {epoch}: New best metric! {metric_to_monitor} = {current_metric_val:.4f} (previously {self.best_metric:.4f})")
                        self.best_metric = current_metric_val
                        patience_counter = 0
                        self._save_checkpoint(epoch, is_best=True)
                    else:
                        patience_counter += 1
                        self.logger.info(f"No improvement in {metric_to_monitor}. Best: {self.best_metric:.4f}. Patience: {patience_counter}/{self.cfg['evaluation']['early_stopping']['patience']}")

                if epoch > 0 and epoch % save_every == 0:
                    self._save_checkpoint(epoch, is_best=False)

                if patience_counter >= self.cfg['evaluation']['early_stopping']['patience']:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}. No improvement for {patience_counter} epochs.")
                    break
        
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user.")
        
        except Exception as e:
            self.logger.critical(f"A fatal error occurred during training at epoch {last_epoch_ran}: {e}", exc_info=True)
        
        finally:
            self.logger.info("Training finished. Saving final model state.")
            self._save_checkpoint(last_epoch_ran, is_best=False, final_save=True)

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg['training']['epochs']} [Train]")
        
        mixup_alpha = self.cfg['augmentations'].get('mixup_alpha', 0.0)
        cutmix_prob = self.cfg['augmentations'].get('cutmix_prob', 0.0)
        accumulation_steps = self.cfg['training']['accumulation_steps']
        clip_grad_val = self.cfg['training'].get('clip_grad_norm')
        amp_enabled = self.cfg['training']['amp_enabled']

        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            is_mixed = False
            
            if np.random.rand() < mixup_alpha:
                images, labels_a, labels_b, lam = self._mixup_data(images, labels, alpha=mixup_alpha)
                is_mixed = True
            elif np.random.rand() < cutmix_prob:
                images, labels_a, labels_b, lam = self._cutmix_data(images, labels)
                is_mixed = True

            # Use a generic forward call, as all models (custom or timm) will return logits
            with autocast(enabled=amp_enabled):
                outputs = self.model(images)
                
                if is_mixed:
                    loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    loss = self.criterion(outputs, labels)
            
            loss_scaled = loss / accumulation_steps
            self.scaler.scale(loss_scaled).backward()

            if (i + 1) % accumulation_steps == 0:
                if clip_grad_val:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_val)
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
                with autocast(enabled=self.cfg['training']['amp_enabled']):
                    if self.cfg['evaluation']['tta_enabled']:
                        outputs = self._tta_inference(model_to_eval, images)
                    else:
                        outputs = model_to_eval(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
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

    def _tta_inference(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        original_logits = model(images)
        flipped_images = T_v2.functional.hflip(images)
        flipped_logits = model(flipped_images)
        # Average softmax probabilities
        return (torch.softmax(original_logits, dim=1) + torch.softmax(flipped_logits, dim=1)) / 2.0

    def _mixup_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
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
        W, H = size[3], size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W); bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W); bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
        
    def _load_initial_weights(self, model: nn.Module) -> nn.Module:
        """
        Helper to load SSL pre-trained weights into the model before any other setup.
        This version correctly handles loading backbone weights into the model's backbone.
        """
        if self.cfg['model'].get('ssl_pretrained_path'):
            path = self.cfg['model']['ssl_pretrained_path']
            self.logger.info(f"Attempting to load SSL backbone weights from: {path}")
            if not path or not os.path.exists(path):
                self.logger.error(f"SSL checkpoint not found at {path}. Starting with model's initial weights.")
                return model
            
            try:
                ckpt = torch.load(path, map_location='cpu')
                
                # --- CORRECTED LOGIC ---
                # Check for the key used by our SSL saver
                if 'model_backbone_state_dict' in ckpt:
                    backbone_weights = ckpt['model_backbone_state_dict']
                    
                    self.logger.info("Found 'model_backbone_state_dict'. Loading into the main model with strict=False.")
                    # Load directly into the model. `strict=False` is crucial because the fine-tuning
                    # model has a classification head that the SSL backbone does not.
                    msg = model.load_state_dict(backbone_weights, strict=False)
                    
                    self.logger.info("SSL weights loaded. Load message summary:")
                    if msg.missing_keys:
                        # We EXPECT missing keys (the final classification head), which is a good sign.
                        self.logger.info(f"  > Missing keys (as expected for fine-tuning): {msg.missing_keys}")
                    if msg.unexpected_keys:
                        # We don't expect unexpected keys. This would be a warning.
                        self.logger.warning(f"  > Unexpected keys in model state dict: {msg.unexpected_keys}")
                
                # Fallback for checkpoints that might just have a single model state dict
                elif 'model_state_dict' in ckpt:
                    self.logger.warning("Found 'model_state_dict' instead of 'model_backbone_state_dict'. Attempting to load with strict=False.")
                    msg = model.load_state_dict(ckpt['model_state_dict'], strict=False)
                    self.logger.info(f"Load message: {msg}")
                
                else:
                    self.logger.error("Could not find a valid state_dict ('model_backbone_state_dict' or 'model_state_dict') in the SSL checkpoint.")
                    
            except Exception as e:
                self.logger.error(f"Failed to load SSL weights from {path}: {e}", exc_info=True)
                
        return model

    def _compile_model_if_enabled(self, model: nn.Module) -> nn.Module:
        """
        Applies torch.compile to the model if enabled in the config.
        Returns the compiled model (or the original if compile is disabled/fails).
        """
        if self.cfg['torch_compile']['enable'] and hasattr(torch, 'compile'):
            try:
                self.logger.info(f"Compiling model with mode '{self.cfg['torch_compile']['mode']}'...")
                compiled_model = torch.compile(model, mode=self.cfg['torch_compile']['mode'])
                self.logger.info("Model compiled successfully.")
                return compiled_model
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
        return model

    def _create_optimizer(self) -> AdamW:
        return AdamW(self._get_param_groups())

    def _get_param_groups(self) -> list:
        # List of common names for classifier heads in various model architectures
        head_names = ['head', 'fc', 'classifier']
        
        head_params = [p for n, p in self.model.named_parameters() if any(hn in n.lower() for hn in head_names) and p.requires_grad]
        backbone_params = [p for n, p in self.model.named_parameters() if not any(hn in n.lower() for hn in head_names) and p.requires_grad]

        # A fallback check: if we found no head params, something is wrong or it's a model with an unusual head name.
        if not head_params:
            self.logger.warning(f"Could not find a specific classifier head using names: {head_names}. "
                                "Treating all parameters as 'backbone' for learning rate purposes.")
            backbone_params = [p for p in self.model.parameters() if p.requires_grad]
        
        opt_cfg = self.cfg['training']['optimizer']
        num_head_params = sum(p.numel() for p in head_params)
        num_backbone_params = sum(p.numel() for p in backbone_params)

        self.logger.info(f"Optimizer parameter groups: "
                         f"Head ({num_head_params:,} params) with LR {opt_cfg['lr_head_unfrozen']:.2e}, "
                         f"Backbone ({num_backbone_params:,} params) with LR {opt_cfg['lr_backbone_unfrozen']:.2e}")

        return [
            {'params': backbone_params, 'lr': opt_cfg['lr_backbone_unfrozen'], 'name': 'backbone'},
            {'params': head_params, 'lr': opt_cfg['lr_head_unfrozen'], 'name': 'head'}
        ]

    def _create_scheduler(self):
        cfg = self.cfg['training']['scheduler']
        scheduler_name = cfg.get('name', 'none').lower()
        if scheduler_name == 'onecyclelr':
            max_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
            total_steps = (self.cfg['training']['epochs'] * len(self.train_loader)) // self.cfg['training']['accumulation_steps']
            return OneCycleLR(self.optimizer, max_lr=max_lrs, total_steps=total_steps, pct_start=cfg['pct_start'], div_factor=cfg['div_factor'], final_div_factor=cfg['final_div_factor'])
        elif scheduler_name == 'constantlr':
            self.logger.info("Using ConstantLR scheduler.")
            return ConstantLR(self.optimizer, factor=1.0)
        self.logger.warning(f"Scheduler '{scheduler_name}' not recognized. No scheduler will be used.")
        return None

    def _set_parameter_groups_for_epoch(self, epoch: int):
        is_frozen_phase = epoch <= self.cfg['training']['freeze_backbone_epochs']
        backbone_group = next((g for g in self.optimizer.param_groups if g['name'] == 'backbone'), None)
        if backbone_group is None or not backbone_group['params']: return # Nothing to freeze
        
        current_requires_grad = backbone_group['params'][0].requires_grad
        new_requires_grad = not is_frozen_phase

        if current_requires_grad != new_requires_grad:
            self.logger.info(f"Epoch {epoch}: Setting backbone requires_grad to {new_requires_grad}")
            for param in backbone_group['params']:
                param.requires_grad = new_requires_grad
            self.logger.info(f"Trainable params updated: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def _create_criterion(self):
        loss_cfg = self.cfg['training']['loss']
        class_weights = self.train_loader.dataset.get_class_weights().to(self.device) if loss_cfg.get('use_class_weights') else None
        if class_weights is not None:
            self.logger.info("Using class weights in loss function.")

        loss_type = loss_cfg.get('type', 'CrossEntropyLoss').lower()
        if loss_type == 'combined':
            return CombinedLoss(num_classes=self.cfg['data']['num_classes'], smoothing=loss_cfg.get('label_smoothing', 0.0), focal_alpha=loss_cfg.get('focal_alpha', 0.25), focal_gamma=loss_cfg.get('focal_gamma', 2.0), ce_weight=loss_cfg.get('weights', {}).get('ce', 0.5), focal_weight=loss_cfg.get('weights', {}).get('focal', 0.5), class_weights_tensor=class_weights).to(self.device)
        return nn.CrossEntropyLoss(label_smoothing=loss_cfg.get('label_smoothing', 0.0), weight=class_weights).to(self.device)

    def _update_history(self, train_loss, val_loss, metrics):
        self.history['train_loss'].append(train_loss); self.history['val_loss'].append(val_loss)
        for k, v in metrics.items(): self.history[f"val_{k}"].append(v)
        if self.scheduler:
            for i, pg in enumerate(self.optimizer.param_groups): self.history[f"lr_group_{pg.get('name', i)}"].append(pg['lr'])

    def _is_metric_better(self, new_metric):
        metric_name = self.cfg['evaluation']['early_stopping']['metric']; delta = self.cfg['evaluation']['early_stopping']['min_delta']
        return new_metric < self.best_metric - delta if 'loss' in metric_name else new_metric > self.best_metric + delta

    def _load_finetune_checkpoint(self, model_to_load_into: nn.Module, path: str):
        if not path or not os.path.exists(path):
            self.logger.warning(f"Fine-tune checkpoint path provided but not found: {path}")
            return
        self.logger.info(f"Resuming fine-tune state from: {path}")
        try:
            ckpt = torch.load(path, map_location=self.device)
            model_to_load_into.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict'] is not None:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if self.ema and 'ema_state_dict' in ckpt and ckpt['ema_state_dict'] is not None:
                self.ema.shadow_model.load_state_dict(ckpt['ema_state_dict'])
            self.start_epoch = ckpt.get('epoch', 0) + 1
            self.best_metric = ckpt.get('best_val_metric', self.best_metric)
            self.logger.info(f"Resumed from epoch {self.start_epoch - 1}. Best metric so far: {self.best_metric:.4f}")
        except Exception as e:
            self.logger.error(f"Failed to load fine-tune checkpoint from {path}: {e}", exc_info=True)

    def _save_checkpoint(self, epoch, is_best=False, final_save=False):
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        data = {'epoch': epoch, 'model_state_dict': model_to_save.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'best_val_metric': self.best_metric, 'config': self.cfg}
        if self.scheduler: data['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.ema: data['ema_state_dict'] = self.ema.state_dict()

        if is_best:
            path = os.path.join(self.checkpoints_dir, 'best_model.pth')
            self.logger.info(f"Saving best model checkpoint to {path}")
        elif final_save:
            path = os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{epoch}_final.pth')
            self.logger.info(f"Saving final model checkpoint to {path}")
        else:
            path = os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{epoch}.pth')
            self.logger.info(f"Saving periodic checkpoint to {path}")

        torch.save(data, path)