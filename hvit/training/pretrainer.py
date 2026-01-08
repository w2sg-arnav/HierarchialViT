# hvit/training/pretrainer.py
"""Self-supervised pre-training trainer using SimCLR.

This module provides the Pretrainer class for SimCLR-based pre-training
of the HViT backbone.
"""

from typing import Optional, Dict, Any
import math
import logging
import os

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> LambdaLR:
    """Create a cosine learning rate scheduler with linear warmup.
    
    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine cycles.
        last_epoch: Last epoch for resuming.
    
    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Pretrainer:
    """SimCLR pre-training trainer for HViT backbone.
    
    Handles the self-supervised pre-training loop with contrastive learning.
    
    Args:
        model: The model to pre-train (HVTForPretraining wrapper).
        augmentations: Augmentation pipeline that generates two views.
        loss_fn: Contrastive loss function (e.g., InfoNCELoss).
        config: Training configuration dictionary.
        device: Device to train on ('cuda' or 'cpu').
        train_loader_for_probe: Optional loader for linear probe evaluation.
        val_loader_for_probe: Optional loader for linear probe validation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        augmentations: Any,
        loss_fn: nn.Module,
        config: Dict[str, Any],
        device: str = 'cuda',
        train_loader_for_probe: Optional[DataLoader] = None,
        val_loader_for_probe: Optional[DataLoader] = None
    ) -> None:
        logger.info("Initializing Pretrainer...")
        self.model = model.to(device)
        self.augmentations = augmentations
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.train_loader_for_probe = train_loader_for_probe
        self.val_loader_for_probe = val_loader_for_probe

        # Setup optimizer
        optimizer_name = config.get("optimizer", "AdamW").lower()
        lr = config.get('learning_rate', 5e-4)
        weight_decay = config.get('weight_decay', 0.05)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        if optimizer_name == "adamw":
            self.optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            momentum = config.get("momentum", 0.9)
            self.optimizer = SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            logger.warning(f"Unsupported optimizer: {optimizer_name}. Using AdamW.")
            self.optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)
        
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}, LR: {lr}, WD: {weight_decay}")

        # Scheduler will be initialized when training starts
        self.scheduler = None
        self.scheduler_name = config.get("scheduler", "warmupcosine").lower()
        self.total_epochs = config.get("epochs", 100)
        self.warmup_epochs = config.get("warmup_epochs", 5)

        # AMP and gradient accumulation
        self.scaler = GradScaler(enabled=(device == 'cuda' and torch.cuda.is_available()))
        self.accum_steps = config.get('accumulation_steps', 1)
        self.clip_grad_norm = config.get('clip_grad_norm', None)
        
        self._current_accum_step = 0
        
        logger.info(
            f"Pretrainer initialized. AMP: {self.scaler.is_enabled()}, "
            f"Accum: {self.accum_steps}, ClipGrad: {self.clip_grad_norm}"
        )

    def _init_scheduler(self, batches_per_epoch: int, start_epoch: int = 0) -> None:
        """Initialize the learning rate scheduler."""
        if self.scheduler is not None:
            return

        if self.scheduler_name == "warmupcosine":
            total_steps = self.total_epochs * batches_per_epoch
            warmup_steps = self.warmup_epochs * batches_per_epoch
            
            last_step = -1
            if start_epoch > 0:
                last_step = start_epoch * batches_per_epoch - 1
            
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                warmup_steps,
                total_steps,
                last_epoch=last_step
            )
            logger.info(
                f"Scheduler: WarmupCosine (warmup_steps={warmup_steps}, "
                f"total_steps={total_steps})"
            )
        elif self.scheduler_name == "cosineannealinglr":
            t_max = self.total_epochs - self.warmup_epochs
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=self.config.get("eta_min_lr", 1e-6)
            )
            logger.info(f"Scheduler: CosineAnnealingLR (T_max={t_max})")

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            epoch: Current epoch number (1-based).
        
        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        batches_per_epoch = len(train_loader)
        self._init_scheduler(batches_per_epoch, epoch - 1)

        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{self.total_epochs} [Pre-train]",
            dynamic_ncols=True
        )

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch_data in enumerate(pbar):
            if not batch_data:
                continue
            
            images, _ = batch_data
            if images is None or images.numel() == 0:
                continue
            
            images = images.to(self.device, non_blocking=True)
            view1, view2 = self.augmentations(images)

            with autocast(device_type='cuda', enabled=self.scaler.is_enabled()):
                proj1 = self.model(rgb_img=view1, mode='pretrain')
                proj2 = self.model(rgb_img=view2, mode='pretrain')
                loss = self.loss_fn(proj1, proj2)

            if not torch.isfinite(loss):
                logger.error(f"Non-finite loss at batch {batch_idx}. Skipping.")
                self.optimizer.zero_grad(set_to_none=True)
                self._current_accum_step = 0
                continue

            loss_scaled = loss / self.accum_steps if self.accum_steps > 1 else loss
            self.scaler.scale(loss_scaled).backward()

            self._current_accum_step += 1
            if self._current_accum_step >= self.accum_steps:
                if self.clip_grad_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self._current_accum_step = 0
                
                if self.scheduler and self.scheduler_name == "warmupcosine":
                    self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

        # Handle leftover gradients
        if self._current_accum_step > 0:
            if self.clip_grad_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self._current_accum_step = 0

        # Step epoch-based scheduler
        if self.scheduler and self.scheduler_name == "cosineannealinglr":
            if epoch > self.warmup_epochs:
                self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(
            f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}, "
            f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
        )
        return avg_loss

    def save_checkpoint(
        self,
        epoch: int,
        checkpoint_dir: str,
        best_probe_accuracy: float = -1.0,
        is_best: bool = False
    ) -> None:
        """Save a training checkpoint.
        
        Args:
            epoch: Current epoch number.
            checkpoint_dir: Directory to save checkpoints.
            best_probe_accuracy: Best linear probe accuracy so far.
            is_best: Whether this is the best checkpoint.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if is_best:
            filename = "best_probe.pth"
        else:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        path = os.path.join(checkpoint_dir, filename)
        
        # Get backbone state dict
        backbone_state = self.model.backbone.state_dict()
        backbone_config = getattr(self.model, 'backbone_init_config', {})
        
        save_content = {
            'epoch': epoch,
            'best_probe_metric': best_probe_accuracy,
            'model_backbone_state_dict': backbone_state,
            'model_backbone_init_config': backbone_config,
            'projection_head_state_dict': self.model.projection_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config
        }
        
        torch.save(save_content, path)
        logger.info(f"Checkpoint saved to {path}")

    def evaluate_linear_probe(self, epoch: int) -> float:
        """Evaluate using a linear probe.
        
        Trains a simple linear classifier on frozen backbone features
        to evaluate representation quality.
        
        Args:
            epoch: Current SSL epoch.
        
        Returns:
            Linear probe validation accuracy (0-100).
        """
        if not self.train_loader_for_probe or not self.val_loader_for_probe:
            logger.warning("Probe loaders not provided. Skipping probe evaluation.")
            return -1.0

        logger.info(f"Starting linear probe evaluation (SSL epoch {epoch})")
        self.model.eval()

        # Get feature dimension
        feature_dim = getattr(self.model.projection_head.head[0], 'in_features', 768)
        num_classes = self.config.get('num_classes', 7)
        
        classifier = nn.Linear(feature_dim, num_classes).to(self.device)
        
        probe_lr = self.config.get('probe_lr', 0.1)
        probe_epochs = self.config.get('probe_epochs', 10)
        
        optimizer = SGD(classifier.parameters(), lr=probe_lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # Train probe
        for probe_epoch in range(1, probe_epochs + 1):
            classifier.train()
            for images, labels in self.train_loader_for_probe:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                with torch.no_grad():
                    features = self.model(rgb_img=images, mode='probe_extract')
                
                logits = classifier(features)
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate probe
        classifier.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader_for_probe:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                features = self.model(rgb_img=images, mode='probe_extract')
                logits = classifier(features)
                _, predicted = torch.max(logits, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0.0
        logger.info(f"Linear probe accuracy (SSL epoch {epoch}): {accuracy:.2f}%")
        
        self.model.train()
        return accuracy


__all__ = [
    "Pretrainer",
    "get_cosine_schedule_with_warmup",
]
