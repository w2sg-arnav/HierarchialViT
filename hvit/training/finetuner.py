# hvit/training/finetuner.py
"""Fine-tuning trainer for supervised classification.

This module provides the EnhancedFinetuner class for fine-tuning
pre-trained HViT models on classification tasks.
"""

from typing import Dict, Any, Tuple, Optional
from collections import defaultdict
import os
import logging

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ConstantLR
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T_v2
from tqdm import tqdm

from hvit.utils.ema import EMA
from hvit.utils.metrics import compute_metrics
from hvit.training.losses import CombinedLoss, FocalLoss

logger = logging.getLogger(__name__)


class EnhancedFinetuner:
    """Advanced fine-tuner for classification tasks.
    
    Features:
    - EMA model averaging
    - MixUp and CutMix augmentation
    - Test-time augmentation (TTA)
    - Gradient accumulation
    - Backbone freezing with warmup
    - Early stopping
    
    Args:
        model: The model to fine-tune.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration dictionary.
        output_dir: Directory for saving checkpoints and logs.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        output_dir: str
    ) -> None:
        self.cfg = config
        self.device = config['device']
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.start_epoch = 1
        self.best_metric = -1.0 if 'loss' not in config['evaluation']['early_stopping']['metric'] else float('inf')
        self.history = defaultdict(list)

        # Load weights before compiling
        uncompiled_model = self._load_initial_weights(model).to(self.device)

        # Initialize EMA with uncompiled model
        use_ema = config['evaluation'].get('use_ema', True)
        ema_decay = config['evaluation'].get('ema_decay', 0.999)
        self.ema = EMA(uncompiled_model, decay=ema_decay) if use_ema else None

        # Compile model if enabled
        self.model = self._compile_model_if_enabled(uncompiled_model)

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        self.scaler = GradScaler(enabled=config['training']['amp_enabled'])

        # Load fine-tune checkpoint if resuming
        resume_path = config['model'].get('resume_finetune_path')
        if resume_path:
            self._load_finetune_checkpoint(uncompiled_model, resume_path)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainer initialized. {trainable_params:,} trainable parameters.")

    def run(self) -> None:
        """Execute the full training loop."""
        logger.info(
            f"Starting fine-tuning. Monitoring '{self.cfg['evaluation']['early_stopping']['metric']}'"
        )
        patience_counter = 0
        total_epochs = self.cfg['training']['epochs']
        save_every = self.cfg['training'].get('save_every_n_epochs', 10)
        patience = self.cfg['evaluation']['early_stopping']['patience']
        metric_name = self.cfg['evaluation']['early_stopping']['metric']

        last_epoch = self.start_epoch - 1
        
        try:
            for epoch in range(self.start_epoch, total_epochs + 1):
                last_epoch = epoch
                logger.info(f"--- Epoch {epoch}/{total_epochs} ---")
                
                self._set_parameter_groups_for_epoch(epoch)
                train_loss = self._train_one_epoch(epoch)
                val_loss, metrics = self._validate_one_epoch(epoch)
                self._update_history(train_loss, val_loss, metrics)

                current_metric = metrics.get(metric_name)
                if current_metric is not None:
                    if self._is_metric_better(current_metric):
                        logger.info(
                            f"New best! {metric_name}={current_metric:.4f} "
                            f"(prev={self.best_metric:.4f})"
                        )
                        self.best_metric = current_metric
                        patience_counter = 0
                        self._save_checkpoint(epoch, is_best=True)
                    else:
                        patience_counter += 1
                        logger.info(
                            f"No improvement. Best={self.best_metric:.4f}, "
                            f"Patience={patience_counter}/{patience}"
                        )

                if epoch % save_every == 0:
                    self._save_checkpoint(epoch, is_best=False)

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
        except Exception as e:
            logger.critical(f"Fatal error at epoch {last_epoch}: {e}", exc_info=True)
        finally:
            logger.info("Training finished. Saving final checkpoint.")
            self._save_checkpoint(last_epoch, is_best=False, final_save=True)

    def _train_one_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        mixup_alpha = self.cfg['augmentations'].get('mixup_alpha', 0.0)
        cutmix_prob = self.cfg['augmentations'].get('cutmix_prob', 0.0)
        accum_steps = self.cfg['training']['accumulation_steps']
        clip_grad = self.cfg['training'].get('clip_grad_norm')
        amp_enabled = self.cfg['training']['amp_enabled']

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]"
        )

        for i, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            is_mixed = False

            # Apply MixUp or CutMix
            if torch.rand(1).item() < mixup_alpha:
                images, labels_a, labels_b, lam = self._mixup_data(images, labels, mixup_alpha)
                is_mixed = True
            elif torch.rand(1).item() < cutmix_prob:
                images, labels_a, labels_b, lam = self._cutmix_data(images, labels)
                is_mixed = True

            with autocast(device_type='cuda', enabled=amp_enabled):
                outputs = self.model(images)
                if is_mixed:
                    loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    loss = self.criterion(outputs, labels)

            loss_scaled = loss / accum_steps
            self.scaler.scale(loss_scaled).backward()

            if (i + 1) % accum_steps == 0:
                if clip_grad:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                if self.scheduler:
                    self.scheduler.step()
                if self.ema:
                    self.ema.update()

            total_loss += loss.item()
            pbar.set_postfix(
                loss=f"{total_loss / (i + 1):.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}"
            )

        return total_loss / len(self.train_loader)

    def _validate_one_epoch(self, epoch: int) -> Tuple[float, Dict[str, Any]]:
        """Validate for one epoch."""
        model_to_eval = self.ema.shadow_model if self.ema else self.model
        model_to_eval.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        tta_enabled = self.cfg['evaluation'].get('tta_enabled', False)
        amp_enabled = self.cfg['training']['amp_enabled']

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                with autocast(device_type='cuda', enabled=amp_enabled):
                    if tta_enabled:
                        outputs = self._tta_inference(model_to_eval, images)
                    else:
                        outputs = model_to_eval(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(
            torch.cat(all_preds),
            torch.cat(all_labels),
            self.cfg['data']['num_classes'],
            self.train_loader.dataset.get_class_names()
        )
        metrics['val_loss'] = avg_loss

        logger.info(
            f"Val Epoch {epoch}: Loss={avg_loss:.4f}, "
            f"Acc={metrics.get('accuracy', 0):.4f}, "
            f"F1={metrics.get('f1_macro', 0):.4f}"
        )
        return avg_loss, metrics

    def _tta_inference(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """Test-time augmentation inference."""
        original_logits = model(images)
        flipped_logits = model(T_v2.functional.hflip(images))
        return (torch.softmax(original_logits, dim=1) + torch.softmax(flipped_logits, dim=1)) / 2.0

    def _mixup_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply MixUp augmentation."""
        if alpha > 0:
            lam = torch.distributions.Beta(alpha, alpha).sample().item()
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, y, y[index], lam

    def _cutmix_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation."""
        lam = torch.distributions.Beta(1.0, 1.0).sample().item()
        rand_index = torch.randperm(x.size(0), device=self.device)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        x_clone = x.clone()
        x_clone[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
        
        lam_adjusted = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x_clone, y, y[rand_index], lam_adjusted

    def _rand_bbox(
        self,
        size: Tuple[int, ...],
        lam: float
    ) -> Tuple[int, int, int, int]:
        """Generate random bounding box for CutMix."""
        W, H = size[3], size[2]
        cut_rat = (1.0 - lam) ** 0.5
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        
        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()
        
        bbx1 = max(0, min(cx - cut_w // 2, W))
        bby1 = max(0, min(cy - cut_h // 2, H))
        bbx2 = max(0, min(cx + cut_w // 2, W))
        bby2 = max(0, min(cy + cut_h // 2, H))
        
        return bbx1, bby1, bbx2, bby2

    def _load_initial_weights(self, model: nn.Module) -> nn.Module:
        """Load SSL pre-trained weights into the model."""
        ssl_path = self.cfg['model'].get('ssl_pretrained_path')
        if not ssl_path or not os.path.exists(ssl_path):
            if ssl_path:
                logger.error(f"SSL checkpoint not found: {ssl_path}")
            return model

        logger.info(f"Loading SSL weights from: {ssl_path}")
        try:
            ckpt = torch.load(ssl_path, map_location='cpu')
            
            if 'model_backbone_state_dict' in ckpt:
                backbone_weights = ckpt['model_backbone_state_dict']
                msg = model.load_state_dict(backbone_weights, strict=False)
                logger.info(f"SSL weights loaded. Missing keys: {len(msg.missing_keys)}")
            elif 'model_state_dict' in ckpt:
                msg = model.load_state_dict(ckpt['model_state_dict'], strict=False)
                logger.info(f"Model weights loaded. Missing keys: {len(msg.missing_keys)}")
            else:
                logger.error("No valid state_dict found in checkpoint.")
        except Exception as e:
            logger.error(f"Failed to load SSL weights: {e}", exc_info=True)
        
        return model

    def _compile_model_if_enabled(self, model: nn.Module) -> nn.Module:
        """Apply torch.compile if enabled."""
        compile_cfg = self.cfg.get('torch_compile', {})
        if compile_cfg.get('enable', False) and hasattr(torch, 'compile'):
            try:
                mode = compile_cfg.get('mode', 'default')
                logger.info(f"Compiling model with mode '{mode}'")
                return torch.compile(model, mode=mode)
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        return model

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with separate param groups for head and backbone."""
        head_names = ['head', 'fc', 'classifier']
        
        head_params = [
            p for n, p in self.model.named_parameters()
            if any(hn in n.lower() for hn in head_names) and p.requires_grad
        ]
        backbone_params = [
            p for n, p in self.model.named_parameters()
            if not any(hn in n.lower() for hn in head_names) and p.requires_grad
        ]

        if not head_params:
            logger.warning("No head parameters found. Treating all as backbone.")
            backbone_params = [p for p in self.model.parameters() if p.requires_grad]

        opt_cfg = self.cfg['training']['optimizer']
        return AdamW([
            {'params': backbone_params, 'lr': opt_cfg['lr_backbone_unfrozen'], 'name': 'backbone'},
            {'params': head_params, 'lr': opt_cfg['lr_head_unfrozen'], 'name': 'head'}
        ])

    def _create_scheduler(self) -> Optional[OneCycleLR]:
        """Create learning rate scheduler."""
        cfg = self.cfg['training'].get('scheduler', {})
        name = cfg.get('name', 'none').lower()
        
        if name == 'onecyclelr':
            max_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
            total_steps = (
                self.cfg['training']['epochs'] * len(self.train_loader)
            ) // self.cfg['training']['accumulation_steps']
            
            return OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=cfg.get('pct_start', 0.1),
                div_factor=cfg.get('div_factor', 25),
                final_div_factor=cfg.get('final_div_factor', 10000)
            )
        elif name == 'constantlr':
            return ConstantLR(self.optimizer, factor=1.0)
        
        logger.warning(f"Scheduler '{name}' not recognized.")
        return None

    def _create_criterion(self) -> nn.Module:
        """Create loss function."""
        loss_cfg = self.cfg['training']['loss']
        
        use_weights = loss_cfg.get('use_class_weights', False)
        class_weights = None
        if use_weights:
            class_weights = self.train_loader.dataset.get_class_weights()
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
                logger.info("Using class weights in loss.")

        loss_type = loss_cfg.get('type', 'CrossEntropyLoss').lower()
        if loss_type == 'combined':
            return CombinedLoss(
                num_classes=self.cfg['data']['num_classes'],
                smoothing=loss_cfg.get('label_smoothing', 0.0),
                focal_alpha=loss_cfg.get('focal_alpha', 0.25),
                focal_gamma=loss_cfg.get('focal_gamma', 2.0),
                ce_weight=loss_cfg.get('weights', {}).get('ce', 0.5),
                focal_weight=loss_cfg.get('weights', {}).get('focal', 0.5),
                class_weights_tensor=class_weights
            ).to(self.device)
        
        return nn.CrossEntropyLoss(
            label_smoothing=loss_cfg.get('label_smoothing', 0.0),
            weight=class_weights
        ).to(self.device)

    def _set_parameter_groups_for_epoch(self, epoch: int) -> None:
        """Update parameter groups for backbone freezing schedule."""
        freeze_epochs = self.cfg['training'].get('freeze_backbone_epochs', 0)
        is_frozen = epoch <= freeze_epochs
        
        backbone_group = next(
            (g for g in self.optimizer.param_groups if g.get('name') == 'backbone'),
            None
        )
        if backbone_group is None or not backbone_group['params']:
            return

        current_grad = backbone_group['params'][0].requires_grad
        new_grad = not is_frozen

        if current_grad != new_grad:
            logger.info(f"Epoch {epoch}: Setting backbone requires_grad={new_grad}")
            for param in backbone_group['params']:
                param.requires_grad = new_grad
            
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Trainable parameters: {trainable:,}")

    def _update_history(
        self,
        train_loss: float,
        val_loss: float,
        metrics: Dict[str, Any]
    ) -> None:
        """Update training history."""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        for k, v in metrics.items():
            self.history[f"val_{k}"].append(v)

    def _is_metric_better(self, new_metric: float) -> bool:
        """Check if new metric is better than best."""
        metric_name = self.cfg['evaluation']['early_stopping']['metric']
        delta = self.cfg['evaluation']['early_stopping'].get('min_delta', 0.0)
        
        if 'loss' in metric_name:
            return new_metric < self.best_metric - delta
        return new_metric > self.best_metric + delta

    def _load_finetune_checkpoint(self, model: nn.Module, path: str) -> None:
        """Load fine-tuning checkpoint for resuming."""
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return

        logger.info(f"Loading fine-tune checkpoint from: {path}")
        try:
            ckpt = torch.load(path, map_location=self.device)
            model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if self.ema and 'ema_state_dict' in ckpt:
                self.ema.shadow_model.load_state_dict(ckpt['ema_state_dict'])
            
            self.start_epoch = ckpt.get('epoch', 0) + 1
            self.best_metric = ckpt.get('best_val_metric', self.best_metric)
            logger.info(f"Resumed from epoch {self.start_epoch - 1}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)

    def _save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        final_save: bool = False
    ) -> None:
        """Save training checkpoint."""
        # Get uncompiled model state
        model_to_save = getattr(self.model, '_orig_mod', self.model)
        
        data = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_metric,
            'config': self.cfg
        }
        
        if self.scheduler:
            data['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.ema:
            data['ema_state_dict'] = self.ema.state_dict()

        if is_best:
            path = os.path.join(self.checkpoints_dir, 'best_model.pth')
        elif final_save:
            path = os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{epoch}_final.pth')
        else:
            path = os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{epoch}.pth')

        torch.save(data, path)
        logger.info(f"Saved checkpoint: {path}")


__all__ = ["EnhancedFinetuner"]
