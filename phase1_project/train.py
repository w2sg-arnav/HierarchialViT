# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
from tqdm import tqdm
from typing import Tuple, Optional, Dict, List

import torchmetrics # Ensure torchmetrics is installed: pip install torchmetrics

from config import (
    DEVICE, MODEL_NAME, PRETRAINED, ORIGINAL_DATASET_ROOT, NUM_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, MODEL_SAVE_PATH, METRICS_AVERAGE,
    APPLY_PROGRESSION_TRAIN, USE_SPECTRAL_TRAIN
)
from data_utils import get_dataloaders
from models import get_baseline_model

logger = logging.getLogger(__name__)

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    num_epochs: int,
    metrics_collection: torchmetrics.MetricCollection
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    
    # Reset metrics for the epoch
    metrics_collection.reset()
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    
    for batch_idx, (rgb_images, spectral_images, labels) in enumerate(progress_bar):
        if rgb_images.numel() == 0: # Skip if batch is empty (e.g. from collate_fn error handling)
            logger.warning(f"Skipping empty batch {batch_idx} in training.")
            continue

        rgb_images, labels = rgb_images.to(device), labels.to(device)
        # spectral_images would also be moved to device if used by the model

        optimizer.zero_grad()
        
        # For baselines, we primarily use RGB. If spectral is used, model needs to handle it.
        outputs = model(rgb_images) # Baselines expect RGB
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update metrics
        preds = torch.argmax(outputs, dim=1)
        metrics_collection.update(preds, labels)
        
        if batch_idx % (len(dataloader) // 5) == 0 and batch_idx > 0 : # Log ~5 times per epoch
            current_loss = total_loss / (batch_idx + 1)
            metrics_results = metrics_collection.compute()
            log_str = f"Batch {batch_idx+1}/{len(dataloader)} | Loss: {current_loss:.4f}"
            for name, value in metrics_results.items():
                log_str += f" | {name}: {value.item():.4f}" if isinstance(value, torch.Tensor) else f" | {name}: {value:.4f}"
            progress_bar.set_postfix_str(log_str)

    avg_loss = total_loss / len(dataloader)
    epoch_metrics = metrics_collection.compute()
    
    metrics_dict = {name: value.item() for name, value in epoch_metrics.items()}
    return avg_loss, metrics_dict


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    metrics_collection: torchmetrics.MetricCollection,
    epoch: int = -1, # -1 for final eval, or epoch number
    num_epochs: int = -1
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    
    metrics_collection.reset()
    desc_str = "Validation"
    if epoch != -1 and num_epochs != -1: # During training
        desc_str = f"Epoch {epoch+1}/{num_epochs} [Val]"

    progress_bar = tqdm(dataloader, desc=desc_str, leave=False)
    
    with torch.no_grad():
        for batch_idx, (rgb_images, spectral_images, labels) in enumerate(progress_bar):
            if rgb_images.numel() == 0: # Skip if batch is empty
                logger.warning(f"Skipping empty batch {batch_idx} in evaluation.")
                continue
            
            rgb_images, labels = rgb_images.to(device), labels.to(device)
            outputs = model(rgb_images) # Baselines expect RGB
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            metrics_collection.update(preds, labels)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    epoch_metrics = metrics_collection.compute()

    metrics_dict = {name: value.item() for name, value in epoch_metrics.items()}
    return avg_loss, metrics_dict


def main():
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Selected model: {MODEL_NAME}")

    # --- Data Loading ---
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
        dataset_root=ORIGINAL_DATASET_ROOT, # Or AUGMENTED_DATASET_ROOT
        apply_progression=APPLY_PROGRESSION_TRAIN,
        use_spectral=USE_SPECTRAL_TRAIN # For baseline models, this is typically False
    )
    logger.info(f"Number of classes: {num_classes}")
    if num_classes == 0:
        logger.error("No classes found. Exiting.")
        return
    if train_loader is None or len(train_loader.dataset) == 0:
        logger.error("Training loader is empty. Exiting.")
        return

    # --- Model Initialization ---
    model = get_baseline_model(MODEL_NAME, num_classes=num_classes, pretrained=PRETRAINED)
    model.to(DEVICE)

    # --- Optimizer and Loss Function ---
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # --- Metrics ---
    # Ensure task is "multiclass"
    common_metric_args = {"task": "multiclass", "num_classes": num_classes, "average": METRICS_AVERAGE}
    train_metrics = torchmetrics.MetricCollection({
        'Accuracy': torchmetrics.Accuracy(**common_metric_args),
        'F1Score': torchmetrics.F1Score(**common_metric_args)
    }).to(DEVICE)
    
    val_metrics = torchmetrics.MetricCollection({
        'Accuracy': torchmetrics.Accuracy(**common_metric_args),
        'Precision': torchmetrics.Precision(**common_metric_args),
        'Recall': torchmetrics.Recall(**common_metric_args),
        'F1Score': torchmetrics.F1Score(**common_metric_args),
        # 'ConfusionMatrix': torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes) # Can be large to log every epoch
    }).to(DEVICE)

    # --- Training Loop ---
    best_val_metric = 0.0 # Using F1-score or Accuracy
    best_epoch = -1

    for epoch in range(NUM_EPOCHS):
        train_loss, train_metrics_results = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, epoch, NUM_EPOCHS, train_metrics
        )
        log_msg = f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f}"
        for name, value in train_metrics_results.items(): log_msg += f" | Train {name}: {value:.4f}"
        logger.info(log_msg)

        if val_loader and len(val_loader.dataset)>0:
            val_loss, val_metrics_results = evaluate(
                model, val_loader, criterion, DEVICE, val_metrics, epoch, NUM_EPOCHS
            )
            log_msg = f"Epoch {epoch+1}/{NUM_EPOCHS} | Val Loss: {val_loss:.4f}"
            for name, value in val_metrics_results.items(): log_msg += f" | Val {name}: {value:.4f}"
            logger.info(log_msg)

            # Save best model (e.g., based on F1Score, can be Accuracy too)
            current_val_metric = val_metrics_results.get('F1Score', val_metrics_results.get('Accuracy', 0.0))
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                best_epoch = epoch
                save_path = os.path.join(MODEL_SAVE_PATH, f"{MODEL_NAME}_best_epoch{epoch+1}_val_f1_{best_val_metric:.4f}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'metrics': val_metrics_results,
                    'num_classes': num_classes,
                    'class_names': class_names
                }, save_path)
                logger.info(f"Best model saved to {save_path} (Val F1/Acc: {best_val_metric:.4f})")
        else:
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} | No validation performed.")
            # Save model at the end of training if no validation
            if epoch == NUM_EPOCHS - 1:
                 save_path = os.path.join(MODEL_SAVE_PATH, f"{MODEL_NAME}_epoch{epoch+1}_final.pth")
                 torch.save(model.state_dict(), save_path)
                 logger.info(f"Final model saved to {save_path}")


    logger.info(f"Training complete. Best validation F1/Accuracy: {best_val_metric:.4f} at epoch {best_epoch+1}")

    # --- Optional: Final Evaluation on Test Set ---
    if test_loader and len(test_loader.dataset) > 0:
        logger.info("Evaluating on Test Set...")
        # Load best model for test evaluation
        best_model_path = None
        # Find the best saved model path (simple search, can be made more robust)
        saved_model_files = [f for f in os.listdir(MODEL_SAVE_PATH) if f.startswith(MODEL_NAME) and "_best_" in f and f.endswith(".pth")]
        if saved_model_files:
            # A simple way to get the one with highest metric if it's in filename, or just the latest one.
            # This part might need refinement depending on naming convention and how "best" is determined.
            # For now, let's assume the loop saved the truly best one.
            # If a specific path was stored:
            # best_model_path = specific_path_to_best_model
            # Otherwise, try to load the one saved in the loop if naming implies it's the best one from THIS run
            if best_epoch != -1: # if a best model was found and saved during training
                best_model_path = os.path.join(MODEL_SAVE_PATH, f"{MODEL_NAME}_best_epoch{best_epoch+1}_val_f1_{best_val_metric:.4f}.pth")

        if best_model_path and os.path.exists(best_model_path):
            logger.info(f"Loading best model from: {best_model_path} for test evaluation.")
            checkpoint = torch.load(best_model_path, map_location=DEVICE)
            # Re-initialize model architecture before loading state_dict
            test_model = get_baseline_model(MODEL_NAME, num_classes=checkpoint.get('num_classes', num_classes), pretrained=False) # Do not load pretrained weights again
            test_model.load_state_dict(checkpoint['model_state_dict'])
            test_model.to(DEVICE)

            test_loss, test_metrics_results = evaluate(
                test_model, test_loader, criterion, DEVICE, val_metrics # Can use val_metrics collection
            )
            log_msg = f"Final Test Set Evaluation | Test Loss: {test_loss:.4f}"
            for name, value in test_metrics_results.items(): log_msg += f" | Test {name}: {value:.4f}"
            logger.info(log_msg)
        else:
            logger.warning(f"Best model path not found or specified ({best_model_path}), or no best model was saved. Skipping test set evaluation with best model.")
            logger.info("Evaluating test set with the model from the last epoch instead (if available).")
            test_loss, test_metrics_results = evaluate(
                model, test_loader, criterion, DEVICE, val_metrics
            )
            log_msg = f"Test Set Evaluation (Last Epoch Model) | Test Loss: {test_loss:.4f}"
            for name, value in test_metrics_results.items(): log_msg += f" | Test {name}: {value:.4f}"
            logger.info(log_msg)


if __name__ == "__main__":
    main()