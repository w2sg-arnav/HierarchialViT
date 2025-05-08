# phase3_pretraining/pretrain/trainer.py
from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW # Using AdamW is common for transformers
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
# from torch.utils.checkpoint import checkpoint # Checkpointing is inside the HVT model now

import logging
import sys
from tqdm import tqdm
import time
import os

# Import from project structure
from ..config import (
    PRETRAIN_LR, ACCUMULATION_STEPS, PRETRAIN_WEIGHT_DECAY,
    LINEAR_PROBE_EPOCHS, LINEAR_PROBE_LR, NUM_CLASSES,
    PRETRAIN_CHECKPOINT_DIR
)
# from ..utils.logging_setup import setup_logging # Logging setup done in main script

logger = logging.getLogger(__name__) # Get logger for this module

class Pretrainer:
    def __init__(self, model: nn.Module, 
                 augmentations, # Instance of SimCLRAugmentation
                 loss_fn,       # Instance of InfoNCELoss
                 device: str,
                 train_loader_for_probe: Optional[DataLoader] = None, # For linear probing
                 val_loader_for_probe: Optional[DataLoader] = None    # For linear probing
                 ):
        self.model = model.to(device)
        self.augmentations = augmentations
        self.loss_fn = loss_fn.to(device) # Ensure loss function is on device
        self.device = device
        self.train_loader_for_probe = train_loader_for_probe
        self.val_loader_for_probe = val_loader_for_probe
        
        # Optimizer for the pre-training model (backbone + projection head)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), # Only optimize trainable params
            lr=PRETRAIN_LR,
            weight_decay=PRETRAIN_WEIGHT_DECAY
        )
        self.scaler = GradScaler(enabled=(self.device == 'cuda')) # Enable only for CUDA
        self.accum_steps = ACCUMULATION_STEPS
        self.current_step_in_accumulation = 0
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Pretrainer initialized. Total model params: {total_params}, Trainable: {trainable_params}")
        # for name, param in self.model.named_parameters():
        #     logger.debug(f"  Param: {name}, Requires Grad: {param.requires_grad}, Device: {param.device}")
    
    def train_one_epoch(self, train_loader: DataLoader, current_epoch: int, total_epochs: int):
        self.model.train() # Set model to training mode
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(enumerate(train_loader), total=num_batches, 
                    desc=f"Epoch {current_epoch}/{total_epochs} [Pre-training]", file=sys.stdout)
        
        for batch_idx, (rgb_images, _) in pbar: # Labels are ignored for SimCLR
            rgb_images = rgb_images.to(self.device, non_blocking=True)
            
            # Create two augmented views
            # SimCLRAugmentation now returns a tuple of (view1, view2)
            view1, view2 = self.augmentations(rgb_images) 

            with autocast(enabled=(self.device == 'cuda')):
                # Forward pass through model (which includes projection head)
                # The HVTForPretraining forward method with pretrain_mode=True handles projection.
                projected_features1 = self.model(rgb_img=view1, pretrain_mode=True)
                projected_features2 = self.model(rgb_img=view2, pretrain_mode=True)
                
                loss = self.loss_fn(projected_features1, projected_features2)
                if self.accum_steps > 1:
                    loss = loss / self.accum_steps
            
            if not torch.isfinite(loss):
                logger.error(f"Epoch {current_epoch}, Batch {batch_idx}: Non-finite loss detected ({loss.item()}). Skipping batch.")
                self.optimizer.zero_grad() # Still zero grads for this step
                continue

            self.scaler.scale(loss).backward()
            
            self.current_step_in_accumulation += 1
            if self.current_step_in_accumulation % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer) # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Optional grad clipping
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.current_step_in_accumulation = 0
            
            total_loss += loss.item() * (self.accum_steps if self.accum_steps > 1 else 1)
            pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
            
            # Memory cleanup (optional, can slow down if too frequent)
            # if batch_idx % 50 == 0:
            #    torch.cuda.empty_cache()

        avg_epoch_loss = total_loss / num_batches
        logger.info(f"Epoch {current_epoch}/{total_epochs} completed. Average Pre-training Loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

    def evaluate_linear_probe(self, current_epoch: int):
        if self.train_loader_for_probe is None or self.val_loader_for_probe is None:
            logger.warning("Linear probe loaders not provided. Skipping evaluation.")
            return -1.0

        logger.info(f"--- Starting Linear Probe Evaluation after Epoch {current_epoch} ---")
        self.model.eval() # Set main model to eval to extract features

        # The features for linear probe are from self.model(..., extract_features_for_probe=True)
        # which should be the output of backbone.head_norm
        # The HVTForPretraining model's __init__ infers this dimension.
        feature_dim = self.model.projection_head.head[0].in_features # Get in_dim of the projection head
        
        linear_probe_classifier = nn.Linear(feature_dim, NUM_CLASSES).to(self.device)
        probe_optimizer = AdamW(linear_probe_classifier.parameters(), lr=LINEAR_PROBE_LR)
        probe_criterion = nn.CrossEntropyLoss().to(self.device)
        
        # Train the linear probe
        for probe_epoch in range(1, LINEAR_PROBE_EPOCHS + 1):
            linear_probe_classifier.train()
            probe_epoch_loss = 0.0
            probe_pbar = tqdm(self.train_loader_for_probe, 
                              desc=f"Probe Epoch {probe_epoch}/{LINEAR_PROBE_EPOCHS}", file=sys.stdout, leave=False)
            for rgb_images, labels in probe_pbar:
                rgb_images, labels = rgb_images.to(self.device), labels.to(self.device)
                
                with torch.no_grad(): # Extract features from the frozen backbone
                    # Pass pretrain_mode=True, extract_features_for_probe=True to HVTForPretraining
                    features = self.model(rgb_img=rgb_images, pretrain_mode=True, extract_features_for_probe=True)
                
                logits = linear_probe_classifier(features)
                loss = probe_criterion(logits, labels)
                
                probe_optimizer.zero_grad()
                loss.backward()
                probe_optimizer.step()
                probe_epoch_loss += loss.item()
            avg_probe_train_loss = probe_epoch_loss / len(self.train_loader_for_probe)
            logger.info(f"Linear Probe Train Epoch {probe_epoch}, Avg Loss: {avg_probe_train_loss:.4f}")

        # Evaluate the linear probe on validation set
        linear_probe_classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for rgb_images, labels in tqdm(self.val_loader_for_probe, desc="Probe Validation", file=sys.stdout, leave=False):
                rgb_images, labels = rgb_images.to(self.device), labels.to(self.device)
                features = self.model(rgb_img=rgb_images, pretrain_mode=True, extract_features_for_probe=True)
                logits = linear_probe_classifier(features)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        logger.info(f"Linear Probe Validation Accuracy after Epoch {current_epoch}: {accuracy:.2f}%")
        return accuracy
    
    def save_checkpoint(self, epoch: int, file_name: Optional[str] = None):
        if not os.path.exists(PRETRAIN_CHECKPOINT_DIR):
            os.makedirs(PRETRAIN_CHECKPOINT_DIR)
        
        if file_name is None:
            file_name = f"hvt_pretrain_epoch_{epoch}.pth"
        path = os.path.join(PRETRAIN_CHECKPOINT_DIR, file_name)
        
        # Save only the backbone's state_dict for easier fine-tuning later
        # Assuming self.model is HVTForPretraining, self.model.backbone is the HVTBackbone
        torch.save(self.model.backbone.state_dict(), path)
        logger.info(f"Pretrained backbone checkpoint saved to {path}")

from typing import Optional # Add for Optional type hint