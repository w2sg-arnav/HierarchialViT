# phase4_finetuning/main.py
from collections import OrderedDict, defaultdict
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, LinearLR, SequentialLR, ReduceLROnPlateau, OneCycleLR
import argparse
import yaml
import torch.nn.functional as F
import math
import sys
from typing import Tuple, Optional, Dict, Any, List
import traceback
from datetime import datetime
import re
import time
import random
import torchvision.transforms.v2 as T_v2

logger_main_global: Optional[logging.Logger] = None

try:
    from .config import config as main_config_from_phase4
    from .dataset import SARCLD2024Dataset
    from .utils.augmentations import create_cotton_leaf_augmentation
    from .utils.logging_setup import setup_logging
    from .finetune.trainer import EnhancedFinetuner
    from .utils.metrics import compute_metrics
    from phase2_model.models.hvt import DiseaseAwareHVT, create_disease_aware_hvt
except ImportError as e_imp:
    print(f"CRITICAL IMPORT ERROR in main.py: {e_imp}. Check paths.", file=sys.stderr)
    traceback.print_exc(); sys.exit(1)

# --- EMA and Loss Classes ---
# ... (Keep EMA, FocalLoss, CombinedLoss class definitions exactly as in your last working version) ...
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 0, use_num_updates: bool = True):
        self.model = model; self.decay = decay; self.warmup_steps = warmup_steps
        self.use_num_updates = use_num_updates; self.num_updates = 0
        self.shadow = {}; self.backup = {}; self.register()
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad: self.shadow[name] = param.data.clone()
    def update(self):
        self.num_updates += 1; current_decay = self.decay
        if self.use_num_updates: current_decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        effective_decay = current_decay * (self.num_updates / self.warmup_steps) if self.warmup_steps > 0 and self.num_updates <= self.warmup_steps else current_decay
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param_data_type = param.data.dtype
                shadow_param = self.shadow[name].to(device=param.data.device, dtype=param_data_type)
                new_average = (1.0 - effective_decay) * param.data + effective_decay * shadow_param
                self.shadow[name] = new_average.clone()
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow: self.backup[name] = param.data.clone(); param.data = self.shadow[name].to(param.data.dtype)
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup: param.data = self.backup[name]
        self.backup = {}

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__(); self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none'); pt = torch.exp(-ce_loss)
        focal_loss_val = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss_val.mean()
        elif self.reduction == 'sum': return focal_loss_val.sum()
        return focal_loss_val

class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1, focal_alpha: float = 0.25, focal_gamma: float = 2.0, ce_weight: float = 0.5, focal_weight: float = 0.5, class_weights_tensor: Optional[torch.Tensor] = None):
        super().__init__(); self.ce_weight = ce_weight; self.focal_weight = focal_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=smoothing, weight=class_weights_tensor)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        current_logger = _get_logger();
        if abs(ce_weight + focal_weight - 1.0) > 1e-6 and (ce_weight > 0 and focal_weight > 0) and (ce_weight + focal_weight > 0): current_logger.warning(f"Loss weights sum to {ce_weight + focal_weight:.2f} != 1.0")
    def forward(self, inputs, targets):
        loss = torch.tensor(0.0, device=inputs.device)
        if self.ce_weight > 0: loss += self.ce_weight * self.ce_loss(inputs, targets)
        if self.focal_weight > 0: loss += self.focal_weight * self.focal_loss(inputs, targets)
        return loss


# --- Helper Functions ---
def _get_logger(): return logger_main_global if logger_main_global else logging.getLogger(__name__)

def get_cosine_schedule_with_warmup_step(optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
    _local_logger_sched = _get_logger()
    def lr_lambda(current_step: int):
        if num_warmup_steps > 0 and current_step < num_warmup_steps: return float(current_step) / float(max(1, num_warmup_steps))
        eff_cs = current_step - num_warmup_steps; eff_ts = num_training_steps - num_warmup_steps
        if eff_ts <= 0: return 0.0
        prog = float(eff_cs) / float(max(1, eff_ts)); return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * prog)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def load_config_from_yaml_or_default(yaml_config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = main_config_from_phase4.copy()
    if yaml_config_path and os.path.exists(yaml_config_path):
        try:
            with open(yaml_config_path, 'r') as f: yaml_override = yaml.safe_load(f)
            if yaml_override and isinstance(yaml_override, dict):
                for key, value in yaml_override.items():
                    if isinstance(value, dict) and isinstance(cfg.get(key), dict): cfg[key].update(value)
                    else: cfg[key] = value
                _get_logger().info(f"Applied config overrides from YAML: {yaml_config_path}")
        except Exception as e_yaml: _get_logger().warning(f"Could not load/parse YAML {yaml_config_path}: {e_yaml}.")
    elif yaml_config_path: _get_logger().warning(f"Specified YAML config {yaml_config_path} not found.")
    return cfg

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning HVT for Cotton Leaf Disease - High Performance V2")
    parser.add_argument("--config", type=str, help="Path to YAML config file (optional).")
    parser.add_argument("--run_name_suffix", type=str, default="ft_run", help="Suffix for log/checkpoint directory names.")
    return parser.parse_args()

def set_global_seed(seed_value: int):
    torch.manual_seed(seed_value); np.random.seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)
    _get_logger().info(f"Global random seed set to: {seed_value}")

def _interpolate_positional_embedding( # ... (same as before)
    checkpoint_pos_embed: torch.Tensor, model_pos_embed_param: nn.Parameter,
    hvt_patch_size: int, target_img_size_for_finetune: Tuple[int, int],
    ssl_img_size_for_pos_embed: Tuple[int, int]):
    _local_logger_pe = _get_logger()
    if checkpoint_pos_embed.ndim == 2: checkpoint_pos_embed = checkpoint_pos_embed.unsqueeze(0)
    N_ckpt, C_ckpt = checkpoint_pos_embed.shape[1], checkpoint_pos_embed.shape[2]
    N_model, C_model = model_pos_embed_param.shape[1], model_pos_embed_param.shape[2]
    if C_ckpt != C_model: _local_logger_pe.error(f"PE C-dim mismatch. No interp."); return model_pos_embed_param.data
    if N_ckpt == N_model: _local_logger_pe.debug("PE N-dim match. Using ckpt PE."); return checkpoint_pos_embed.view_as(model_pos_embed_param.data)
    H0_patches = ssl_img_size_for_pos_embed[0] // hvt_patch_size
    W0_patches = ssl_img_size_for_pos_embed[1] // hvt_patch_size
    if H0_patches * W0_patches != N_ckpt:
        if math.isqrt(N_ckpt)**2 == N_ckpt: H0_patches = W0_patches = math.isqrt(N_ckpt)
        else: _local_logger_pe.error(f"PE Interp: Cannot infer source grid. No interp."); return model_pos_embed_param.data
    Ht_patches = target_img_size_for_finetune[0] // hvt_patch_size
    Wt_patches = target_img_size_for_finetune[1] // hvt_patch_size
    if Ht_patches * Wt_patches != N_model: _local_logger_pe.error(f"PE Interp: Target grid != N_model. No interp."); return model_pos_embed_param.data
    _local_logger_pe.info(f"PE Interpolating: {N_ckpt}({H0_patches}x{W0_patches}) from SSL {ssl_img_size_for_pos_embed} to {N_model}({Ht_patches}x{Wt_patches}) for FT {target_img_size_for_finetune}.")
    try:
        pe_to_interp = checkpoint_pos_embed.reshape(1, H0_patches, W0_patches, C_ckpt).permute(0, 3, 1, 2)
        pe_interpolated = F.interpolate(pe_to_interp, size=(Ht_patches, Wt_patches), mode='bicubic', align_corners=False)
        pe_interpolated = pe_interpolated.permute(0, 2, 3, 1).flatten(1, 2)
        if pe_interpolated.shape == model_pos_embed_param.shape: return pe_interpolated
        else: _local_logger_pe.error(f"PE Interp: Final shape mismatch."); return model_pos_embed_param.data
    except Exception as e: _local_logger_pe.error(f"PE Interp Error: {e}", exc_info=True); return model_pos_embed_param.data

def load_initial_ssl_weights(hvt_model_instance: DiseaseAwareHVT, cfg: Dict[str, Any]): # For SSL weights
    # ... (same as before)
    _local_logger_ssl = _get_logger()
    if not cfg.get("load_pretrained_backbone_from_ssl", False): _local_logger_ssl.info("Skipping SSL load: 'load_pretrained_backbone_from_ssl' is False."); return
    ssl_ckpt_path = cfg.get("ssl_pretrained_backbone_path")
    if not (ssl_ckpt_path and os.path.exists(ssl_ckpt_path)): _local_logger_ssl.warning(f"SSL ckpt path invalid: '{ssl_ckpt_path}'. Backbone random."); return
    _local_logger_ssl.info(f"Loading initial SSL backbone weights from: {ssl_ckpt_path}")
    checkpoint = torch.load(ssl_ckpt_path, map_location='cpu'); ssl_backbone_sd = checkpoint.get('model_state_dict', checkpoint.get('model_backbone_state_dict'))
    if not ssl_backbone_sd: _local_logger_ssl.error(f"SSL ckpt missing model state."); return
    current_model_sd = hvt_model_instance.state_dict(); new_sd=OrderedDict(); ld_c,pe_c,hd_s,sh_m=0,0,0,0
    ssl_cfg_snap = checkpoint.get('run_config_snapshot',checkpoint.get('config_runtime', {})); ssl_img_val = ssl_cfg_snap.get('pretrain_img_size', cfg.get("ssl_pretrain_img_size_fallback",cfg["img_size"]))
    ssl_img_pe = tuple(ssl_img_val)
    for k, v in ssl_backbone_sd.items():
        if k not in current_model_sd: continue
        if k.startswith("classifier_head.") or k.startswith("head."): hd_s+=1; continue
        is_pe_k = k in ["rgb_pos_embed","spectral_pos_embed"]; tgt_pe=getattr(hvt_model_instance,k,None) if is_pe_k else None
        if is_pe_k and tgt_pe is not None and v.shape!=tgt_pe.shape:
            interp_v=_interpolate_positional_embedding(v,tgt_pe,cfg['hvt_params_for_model_init']['patch_size'],tuple(cfg["img_size"]),ssl_img_pe)
            if interp_v.shape==tgt_pe.shape: new_sd[k]=interp_v; pe_c+=1
        elif v.shape==current_model_sd[k].shape: new_sd[k]=v; ld_c+=1
        else: _local_logger_ssl.warning(f"Shape mismatch {k}. Skip."); sh_m+=1
    msg=hvt_model_instance.load_state_dict(new_sd,strict=False)
    _local_logger_ssl.info(f"SSL Backbone: {ld_c} ld, {pe_c} PE interp, {hd_s} head skip, {sh_m} mismatch.")
    if msg.missing_keys: _local_logger_ssl.warning(f"Missing keys SSL load: {msg.missing_keys}")
    if hasattr(hvt_model_instance,'classifier_head') and isinstance(hvt_model_instance.classifier_head,nn.Linear):
        in_f=hvt_model_instance.classifier_head.in_features; hvt_model_instance.classifier_head=nn.Linear(in_f,cfg["num_classes"])
        _local_logger_ssl.info(f"Re-init HVT classifier_head for {cfg['num_classes']} FT classes (in_features={in_f}).")
    else: _local_logger_ssl.error("Could not re-init classifier_head post-SSL.")

def resume_finetune_from_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], scaler: Optional[GradScaler], checkpoint_path: str, device: str, cfg: Dict[str, Any]) -> Tuple[int, float, int]:
    # ... (same as before)
    _local_logger_resume = _get_logger(); start_ep=1; best_m=0.0 if cfg.get("metric_to_monitor_early_stopping","f1_macro")!="val_loss" else float('inf'); last_ss=-1
    if not (checkpoint_path and os.path.exists(checkpoint_path)): _local_logger_resume.warning(f"FT resume ckpt not found: '{checkpoint_path}'."); return start_ep,best_m,last_ss
    _local_logger_resume.info(f"Resuming FT from ckpt: {checkpoint_path}")
    try:
        ckpt=torch.load(checkpoint_path,map_location=device)
        if 'model_state_dict' in ckpt: model.load_state_dict(ckpt['model_state_dict']); _local_logger_resume.info("Loaded model_state_dict FT.")
        if optimizer and 'optimizer_state_dict' in ckpt and cfg.get("load_optimizer_scheduler_on_resume", True):
            try: optimizer.load_state_dict(ckpt['optimizer_state_dict']); _local_logger_resume.info("Optimizer state loaded.")
            except: _local_logger_resume.warning("Could not load optimizer state. Optimizer may start fresh.")
        if scaler and 'scaler_state_dict' in ckpt and scaler.is_enabled() and cfg.get("load_optimizer_scheduler_on_resume",True):
            try: scaler.load_state_dict(ckpt['scaler_state_dict']); _local_logger_resume.info("GradScaler state loaded.")
            except: _local_logger_resume.warning("Could not load GradScaler state. Scaler starts fresh.")
        raw_ep=ckpt.get('epoch',0); num_ep_comp=0
        if isinstance(raw_ep,str): match=re.search(r'\d+',raw_ep); num_ep_comp=int(match.group(0)) if match else 0
        elif isinstance(raw_ep,int): num_ep_comp=raw_ep
        start_ep=num_ep_comp+1; best_m=ckpt.get('best_val_metric',best_m)
        train_loader_ref=cfg.get('_train_loader_ref_for_resume')
        if train_loader_ref and hasattr(train_loader_ref,'__len__') and len(train_loader_ref)>0 :
            steps_per_ep_calc=len(train_loader_ref)//max(1,cfg.get("accumulation_steps",1))
            if num_ep_comp>0 and steps_per_ep_calc>0: last_ss=num_ep_comp*steps_per_ep_calc-1
        _local_logger_resume.info(f"Resuming FT from epoch {start_ep}. Best: {best_m:.4f}. Last sched step: {last_ss}")
    except Exception as e: _local_logger_resume.error(f"Error loading FT ckpt {checkpoint_path}: {e}",exc_info=True)
    return start_ep,best_m,last_ss

def create_weighted_sampler(dataset: SARCLD2024Dataset, cfg: Dict[str,Any]) -> Optional[WeightedRandomSampler]:
    # ... (same as before)
    logger_sampler = _get_logger(); mode=cfg.get('weighted_sampler_mode','inv_count'); beta=0.9999
    targets = getattr(dataset, 'current_split_labels', None)
    if targets is None or len(targets) == 0: targets = getattr(dataset, 'labels_np', None) # Fallback for get_targets
    if targets is None or len(targets) == 0: logger_sampler.warning("No targets for weighted sampler."); return None
    targets = np.array(targets)
    num_cls_ds=dataset.num_classes if hasattr(dataset,'num_classes') else (np.max(targets)+1)
    cls_counts=np.bincount(targets,minlength=num_cls_ds); cls_counts=np.maximum(cls_counts,1e-9)
    if mode=='inv_count': w_per_cls=1.0/cls_counts
    elif mode=='effective_number': eff_num=1.0-np.power(beta,cls_counts); w_per_cls=(1.0-beta)/eff_num
    elif mode == 'sqrt_inv_count': w_per_cls = 1.0 / np.sqrt(cls_counts)
    else: logger_sampler.warning(f"Unknown sampler mode: {mode}. Using uniform."); w_per_cls=np.ones_like(cls_counts,dtype=np.float32)
    w_per_cls/=np.sum(w_per_cls); sample_w=w_per_cls[targets]
    sampler=WeightedRandomSampler(weights=torch.from_numpy(sample_w).double(),num_samples=len(sample_w),replacement=True)
    logger_sampler.info(f"WeightedRandomSampler created with mode: {mode}.")
    return sampler

def get_class_weights_for_loss(dataset: SARCLD2024Dataset, device: torch.device, cfg: Dict[str,Any]) -> Optional[torch.Tensor]:
    # ... (same as before)
    logger_cw = _get_logger();
    if not cfg.get('use_weighted_loss',False): return None
    if hasattr(dataset,'get_class_weights') and callable(getattr(dataset,'get_class_weights')):
        weights=dataset.get_class_weights() # This method should use current_split_labels
        if weights is not None: logger_cw.info(f"Using class weights for loss from dataset: {weights.numpy()}"); return weights.to(device)
    logger_cw.warning("Dataset has no 'get_class_weights' or it returned None. Calculating manually.");
    targets = getattr(dataset, 'current_split_labels', None)
    if targets is None or len(targets)==0: logger_cw.warning("No targets for class_weights."); return None
    num_cls_ds=dataset.num_classes if hasattr(dataset,'num_classes') else (np.max(targets)+1)
    cls_counts=np.bincount(np.array(targets),minlength=num_cls_ds)
    weights_val=1.0/(cls_counts+1e-6); weights_val=weights_val/np.sum(weights_val)*num_cls_ds
    logger_cw.info(f"Calculated class weights for loss manually: {weights_val}")
    return torch.tensor(weights_val,dtype=torch.float32).to(device)

def plot_training_curves(history: Dict[str, List[float]], output_dir: str, cfg_metric_name: str):
    # ... (same as before)
    logger_plot = _get_logger(); epochs_ran = range(1, len(history.get('train_loss',[])) + 1)
    if not epochs_ran: logger_plot.warning("No history to plot."); return
    try:
        import matplotlib.pyplot as plt; import seaborn as sns
        plt.figure(figsize=(20,6)); plt.subplot(1,3,1); plt.plot(epochs_ran,history['train_loss'],'bo-',label='Train Loss');
        if 'val_loss' in history and len(history['val_loss'])==len(epochs_ran): plt.plot(epochs_ran,history['val_loss'],'ro-',label='Val Loss')
        plt.title('Loss');plt.xlabel('Epochs');plt.ylabel('Loss');plt.legend();plt.grid(True)
        base_m=cfg_metric_name.replace('val_',''); tr_k,v_k=f"train_{base_m}",f"val_{base_m}";lbl=base_m.replace("_"," ").title();
        if not (tr_k in history and v_k in history and len(history[tr_k])==len(epochs_ran) and len(history[v_k])==len(epochs_ran)):
            if 'train_accuracy' in history and 'val_accuracy' in history and len(history['train_accuracy'])==len(epochs_ran) and len(history['val_accuracy'])==len(epochs_ran):
                tr_k,v_k,lbl='train_accuracy','val_accuracy','Accuracy'; logger_plot.debug("Plotting fallback accuracy.")
            else: logger_plot.warning(f"Cannot plot metric '{base_m}'."); return
        plt.subplot(1,3,2); plt.plot(epochs_ran,history[tr_k],'bo-',label=f'Train {lbl}'); plt.plot(epochs_ran,history[v_k],'ro-',label=f'Val {lbl}'); plt.title(lbl);plt.xlabel('Epochs');plt.ylabel(lbl.split(' ')[-1]);plt.legend();plt.grid(True)
        if 'lr' in history and len(history['lr'])==len(epochs_ran):
            plt.subplot(1,3,3); plt.plot(epochs_ran,history['lr'],'go-',label='LR'); plt.title('LR');plt.xlabel('Epochs');plt.ylabel('LR');plt.legend();plt.grid(True);
            if any(lr_val > 1e-9 for lr_val in history['lr']): plt.gca().set_yscale('log')
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, "training_curves.png")); plt.close()
        logger_plot.info(f"Training curves saved to {os.path.join(output_dir, 'training_curves.png')}")
    except ImportError: logger_plot.warning("Matplotlib/Seaborn not installed. Skipping plots.")
    except Exception as e: logger_plot.error(f"Error plotting curves: {e}", exc_info=True)

def plot_confusion_matrix_main(cm_data: np.ndarray, class_names_list: List[str], output_dir_path: str, filename_cm="confusion_matrix.png"):
    # ... (same as before)
    logger_cm = _get_logger();
    try:
        import matplotlib.pyplot as plt; import seaborn as sns
        plt.figure(figsize=(max(8,len(class_names_list)*0.9),max(6,len(class_names_list)*0.7))); sns.heatmap(cm_data,annot=True,fmt='d',cmap='Blues',xticklabels=class_names_list,yticklabels=class_names_list,cbar=True)
        plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix'); plt.tight_layout(); plt.savefig(os.path.join(output_dir_path,filename_cm)); plt.close()
        logger_cm.info(f"Confusion matrix saved to {os.path.join(output_dir_path, filename_cm)}")
    except ImportError: logger_cm.warning("Matplotlib/Seaborn not installed. Skipping CM plot.")
    except Exception as e: logger_cm.error(f"Error plotting CM: {e}", exc_info=True)

# --- Main Execution Logic ---
def main_execution_logic():
    global logger_main_global
    args = parse_arguments()
    cfg = load_config_from_yaml_or_default(args.config)
    if args.run_name_suffix and args.run_name_suffix != "ft_run": cfg['run_name_suffix'] = args.run_name_suffix # Override from CLI if provided and not default

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.get('run_name_suffix', 'ft_run')}_{run_ts}"
    abs_pkg_root = cfg.get("PACKAGE_ROOT_PATH", os.path.dirname(os.path.abspath(__file__)))
    run_specific_log_dir = os.path.join(abs_pkg_root, cfg.get("log_dir", "logs_finetune"), run_name)
    os.makedirs(run_specific_log_dir, exist_ok=True)
    log_file_base = os.path.splitext(cfg.get("log_file_finetune_base", "finetune.log"))[0]
    final_log_filename = f"{log_file_base}.log"
    
    setup_logging(log_file_name=final_log_filename, log_dir=run_specific_log_dir, log_level=logging.DEBUG, run_timestamp=run_ts)
    logger_main_global = logging.getLogger(__name__) # Assign to global
    logger = _get_logger()

    logger.info(f"======== Starting Phase 4 Fine-tuning (Run: {run_name}) ========")
    logger.info(f"Full effective configuration for this run: {cfg}")
    set_global_seed(cfg["seed"]); device = cfg["device"]
    if device == 'cuda' and not torch.cuda.is_available(): device = 'cpu'; cfg['device'] = 'cpu'; logger.warning("CUDA unavailable.")
    logger.info(f"Device: {device}. GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'N/A'}")
    if cfg.get("cudnn_benchmark", True) and device == 'cuda': torch.backends.cudnn.benchmark = True; logger.info("cudnn.benchmark = True")
    if cfg.get("matmul_precision") and hasattr(torch,'set_float32_matmul_precision'):
        try: torch.set_float32_matmul_precision(cfg["matmul_precision"]); logger.info(f"matmul_precision = '{cfg['matmul_precision']}'")
        except Exception as e: logger.warning(f"Failed to set matmul_precision: {e}")

    # --- Dataset and DataLoaders ---
    current_img_size = tuple(cfg["img_size"])
    base_transforms = T_v2.Compose([
        T_v2.ToImage(), T_v2.ToDtype(torch.float32, scale=True),
        T_v2.Resize(current_img_size, interpolation=T_v2.InterpolationMode.BICUBIC, antialias=True),
        T_v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) if cfg.get("normalize_data", True) else nn.Identity()
    ])
    dataset_args = {"root_dir": cfg["data_root"], "img_size": current_img_size, "train_split_ratio": cfg["train_split_ratio"], "original_dataset_name": cfg["original_dataset_name"], "augmented_dataset_name": cfg.get("augmented_dataset_name", None), "random_seed": cfg["seed"], "transform": base_transforms}
    train_dataset = SARCLD2024Dataset(**dataset_args, split="train"); val_dataset = SARCLD2024Dataset(**dataset_args, split="val")
    class_names = train_dataset.get_class_names(); cfg['num_classes'] = len(class_names) # Ensure num_classes is correct
    
    sampler = create_weighted_sampler(train_dataset, cfg)
    loader_args = {"num_workers": cfg.get('num_workers',4), "pin_memory": (device=='cuda'), "persistent_workers": (device=='cuda' and cfg.get('num_workers',4)>0)}
    if cfg.get('num_workers',4) > 0 and cfg.get('prefetch_factor') is not None : loader_args["prefetch_factor"] = cfg['prefetch_factor']
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], sampler=sampler, shuffle=(sampler is None), drop_last=True, **loader_args)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"]*2, shuffle=False, drop_last=False, **loader_args)
    logger.info(f"Dataloaders: Train batches={len(train_loader)}, Val batches={len(val_loader)}")
    cfg['_train_loader_ref_for_resume'] = train_loader # Used by resume_finetune_from_checkpoint for scheduler steps

    # --- Model, Optimizer, Scheduler Setup ---
    model = create_disease_aware_hvt(current_img_size=current_img_size, num_classes=cfg["num_classes"], model_params_dict=cfg['hvt_params_for_model_init'])
    logger.info(f"Base HVT model created (num_classes={cfg['num_classes']}).")

    head_name = "classifier_head"
    pg_head = {'params': [p for n,p in model.named_parameters() if n.startswith(head_name+".")], 'name': 'head'}
    pg_backbone = {'params': [p for n,p in model.named_parameters() if not n.startswith(head_name+".")], 'name': 'backbone'}
    param_groups_for_optimizer = [pg_head, pg_backbone] if pg_head['params'] and pg_backbone['params'] else \
                                 ([pg_head] if pg_head['params'] else [pg_backbone] if pg_backbone['params'] else [])
    if not param_groups_for_optimizer: logger.critical("No params for optimizer!"); sys.exit(1)

    optim_name = cfg.get("optimizer","AdamW").lower(); opt_kwargs_cfg = cfg.get("optimizer_params",{}); wd_cfg = cfg.get("weight_decay",0.05)
    if optim_name == "adamw": optimizer = AdamW(param_groups_for_optimizer, lr=cfg['lr_head_unfrozen_phase'], weight_decay=wd_cfg, **opt_kwargs_cfg) # Default LR, will be set per epoch
    else: optimizer = SGD(param_groups_for_optimizer, lr=cfg['lr_head_unfrozen_phase'], momentum=0.9, weight_decay=wd_cfg, **opt_kwargs_cfg)
    logger.info(f"Optimizer: {optimizer.__class__.__name__} created. LRs will be staged per epoch.")

    scaler = GradScaler(enabled=(cfg.get("amp_enabled", True) and device == 'cuda'))
    start_epoch = 1; best_val_metric = 0.0 if cfg.get("metric_to_monitor_early_stopping","f1_macro")!="val_loss" else float('inf'); last_completed_scheduler_step = -1
    
    ft_resume_path_val = cfg.get("resume_finetune_checkpoint_path")
    if ft_resume_path_val and os.path.exists(ft_resume_path_val):
        start_epoch, best_val_metric, last_completed_scheduler_step = resume_finetune_from_checkpoint(model, optimizer, scaler, ft_resume_path_val, device, cfg)
    elif cfg.get("load_pretrained_backbone_from_ssl", False):
        load_initial_ssl_weights(model, cfg) # Modifies model in-place (loads SSL weights, re-inits head)
    else: logger.info("Starting FT from scratch (no SSL backbone, no FT resume).")
    model.to(device)
    
    total_p, trainable_p = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model on {device}. Total params: {total_p:,}, Initial Trainable (may change with freezing): {trainable_p:,}")

    scheduler = None; lr_scheduler_on_batch = False
    sched_name_cfg_val = cfg.get("scheduler", "WarmupCosine").lower() # Default to WarmupCosine
    if sched_name_cfg_val != "none":
        steps_per_ep_val = len(train_loader) // max(1, cfg.get("accumulation_steps", 1)) if len(train_loader)>0 else 0
        if steps_per_ep_val > 0:
            total_steps_this_run = (cfg["epochs"] - (start_epoch - 1)) * steps_per_ep_val
            warmup_epochs_this_run = max(0, cfg.get("warmup_epochs",0) - (start_epoch - 1))
            warmup_steps_this_run = warmup_epochs_this_run * steps_per_ep_val
            if warmup_steps_this_run >= total_steps_this_run and total_steps_this_run > 0 : warmup_steps_this_run = max(1, int(0.1 * total_steps_this_run))

            if sched_name_cfg_val == "warmupcosine":
                if total_steps_this_run > 0:
                    scheduler = get_cosine_schedule_with_warmup_step(optimizer, warmup_steps_this_run, total_steps_this_run, last_epoch=last_completed_scheduler_step)
                    lr_scheduler_on_batch = True
                if scheduler: logger.info(f"Scheduler: WarmupCosine. WU Steps(run):{warmup_steps_this_run}, Total Steps(run):{total_steps_this_run}, Resumed step:{scheduler.last_epoch}")
            elif sched_name_cfg_val == "onecyclelr":
                max_lrs_for_onecycle = [pg.get('lr') for pg in optimizer.param_groups] # Base LRs are now set from stage
                scheduler = OneCycleLR(optimizer, max_lr=max_lrs_for_onecycle, epochs=cfg['epochs']-(start_epoch-1), steps_per_epoch=steps_per_ep_val, pct_start=cfg.get('onecycle_pct_start',0.25), div_factor=cfg.get('onecycle_div_factor',10), final_div_factor=cfg.get('onecycle_final_div_factor',1e4), last_epoch=last_completed_scheduler_step)
                lr_scheduler_on_batch = True; logger.info(f"Scheduler: OneCycleLR. Max LRs from groups. Total Steps: {(cfg['epochs']-(start_epoch-1))*steps_per_ep_val}. Resumed step: {scheduler.last_epoch}")
        else: logger.warning(f"Scheduler '{sched_name_cfg_val}' not used as steps_per_epoch is 0.")

    if cfg.get("enable_torch_compile", False) and hasattr(torch, 'compile'):
        try: model = torch.compile(model, mode=cfg.get("torch_compile_mode")); logger.info("Model compiled.")
        except Exception as e: logger.warning(f"torch.compile() failed: {e}.", exc_info=True)
    
    aug_strat_val = cfg.get("augmentation_strategy", "standard_finetune"); aug_sev_val = cfg.get("augmentation_severity", "moderate")
    augmentations_for_trainer = create_cotton_leaf_augmentation(strategy=aug_strat_val, img_size=tuple(cfg["img_size"]), severity=aug_sev_val) if cfg.get("augmentations_enabled",True) else None
    if augmentations_for_trainer: logger.info(f"Using {augmentations_for_trainer.__class__.__name__} (strategy: {aug_strat_val}, severity: {aug_sev_val}).")
    
    cls_weights_loss_val = get_class_weights_for_loss(train_dataset, device, cfg)
    loss_fn_name_val = cfg.get("loss_function", "crossentropy").lower()
    if loss_fn_name_val == "combined":
        criterion = CombinedLoss(num_classes=cfg["num_classes"], smoothing=cfg.get("loss_label_smoothing",0.1), focal_alpha=cfg.get("focal_loss_alpha",0.25), focal_gamma=cfg.get("focal_loss_gamma",2.0), ce_weight=cfg.get("loss_weights",{}).get("ce_weight",0.5), focal_weight=cfg.get("loss_weights",{}).get("focal_weight",0.5), class_weights_tensor=cls_weights_loss_val).to(device)
    else: criterion = nn.CrossEntropyLoss(weight=cls_weights_loss_val, label_smoothing=cfg.get("loss_label_smoothing", 0.0)).to(device)
    logger.info(f"Using {loss_fn_name_val.capitalize()}Loss.")
    
    finetuner = EnhancedFinetuner(model=model, optimizer=optimizer, criterion=criterion, device=device, scaler=scaler, scheduler=scheduler, lr_scheduler_on_batch=lr_scheduler_on_batch, accumulation_steps=max(1,cfg.get("accumulation_steps",1)), clip_grad_norm=cfg.get("clip_grad_norm"), augmentations=augmentations_for_trainer, num_classes=cfg["num_classes"], tta_enabled_val=cfg.get('tta_enabled_val',False), use_enhanced_tta=cfg.get('use_enhanced_tta_trainer',True), mixup_alpha=cfg.get('mixup_alpha',0.0), cutmix_alpha=cfg.get('cutmix_alpha',0.0), mixup_prob=cfg.get('mixup_prob',0.0), cutmix_prob=cfg.get('cutmix_prob',0.0), use_progressive_resize=cfg.get('trainer_use_progressive_resize',False), input_size=tuple(cfg['img_size'])[0], max_input_size=max(s[0] for s in cfg.get('progressive_resize_schedule', {0:cfg['img_size']}).values()) if cfg.get('trainer_use_progressive_resize',False) else tuple(cfg['img_size'])[0], swa_enabled=cfg.get('use_swa',False), swa_start_epoch=cfg.get('swa_start_epoch', cfg['epochs'] - max(1,int(cfg['epochs']*0.2))), multiscale_training=cfg.get('trainer_multiscale_training_enabled', False), scale_range=tuple(cfg.get('trainer_multiscale_scale_range', [int(tuple(cfg['img_size'])[0]*0.75), int(tuple(cfg['img_size'])[0]*1.25)])))
    if cfg.get('use_ema'): finetuner.ema_model = EMA(model, decay=cfg.get('ema_decay',0.9999), warmup_steps=cfg.get('ema_warmup_steps',0))

    metric_to_watch = cfg.get("metric_to_monitor_early_stopping", "f1_macro")
    patience = cfg.get("early_stopping_patience", float('inf')); patience_counter = 0
    abs_ckpt_save_dir = os.path.join(run_specific_log_dir, cfg.get("checkpoint_save_dir_name", "checkpoints"))
    os.makedirs(abs_ckpt_save_dir, exist_ok=True); logger.info(f"Checkpoints for run '{run_name}' will be saved in: {abs_ckpt_save_dir}")
    logger.info(f"Starting fine-tuning from epoch {start_epoch} for {cfg['epochs']} total epochs. Monitoring '{metric_to_watch}'.")
    
    epochs_this_run = 0; history = defaultdict(list)
    try:
        for epoch_1_based in range(start_epoch, cfg["epochs"] + 1):
            epoch_time_start = time.time()
            is_frozen_phase_for_epoch = epoch_1_based <= cfg["freeze_backbone_epochs"]
            model_for_grads = model._orig_mod if hasattr(model, '_orig_mod') and isinstance(model._orig_mod, nn.Module) else model
            
            if not is_frozen_phase_for_epoch and epoch_1_based == cfg["freeze_backbone_epochs"] + 1:
                logger.info(f"Epoch {epoch_1_based}: Setting backbone requires_grad=True for unfreezing.")
                unfrozen_count = 0
                for name, param in model_for_grads.named_parameters():
                    if not name.startswith(head_name + "."):
                        if not param.requires_grad: param.requires_grad = True; unfrozen_count += 1
                if unfrozen_count > 0: logger.info(f"{unfrozen_count} backbone params requires_grad set True.")
            
            for pg in optimizer.param_groups:
                group_name = pg.get('name'); current_pg_lr = pg['lr']
                target_lr_for_group = cfg['lr_head_frozen_phase'] if group_name == 'head' and is_frozen_phase_for_epoch else \
                                     cfg['lr_head_unfrozen_phase'] if group_name == 'head' and not is_frozen_phase_for_epoch else \
                                     0.0 if group_name == 'backbone' and is_frozen_phase_for_epoch else \
                                     cfg['lr_backbone_unfrozen_phase'] if group_name == 'backbone' and not is_frozen_phase_for_epoch else current_pg_lr # Should not happen
                if abs(current_pg_lr - target_lr_for_group) > 1e-9: pg['lr'] = target_lr_for_group; logger.info(f"E{epoch_1_based}: Optim group '{group_name}' base LR set to {target_lr_for_group:.2e}")
            
            if not is_frozen_phase_for_epoch and epoch_1_based == cfg["freeze_backbone_epochs"] + 1: logger.info(f"Trainable params after unfreeze: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

            train_loss, nan_flag = finetuner.train_one_epoch(train_loader, epoch_1_based, cfg["epochs"])
            history['train_loss'].append(train_loss); history['lr'].append(optimizer.param_groups[0]['lr']) # Log LR of first group
            if nan_flag: logger.error(f"NaNs in E{epoch_1_based}, stopping."); break
            
            val_metric_current = None
            if epoch_1_based % cfg.get("evaluate_every_n_epochs", 1) == 0 or epoch_1_based == cfg["epochs"]:
                _, val_metrics = finetuner.validate_one_epoch(val_loader, class_names)
                for vm_k, vm_v in val_metrics.items(): history[f"val_{vm_k}"].append(vm_v)
                val_metric_current = val_metrics.get(metric_to_watch)
                if finetuner.scheduler and not finetuner.lr_scheduler_on_batch: # Epoch schedulers
                    if isinstance(finetuner.scheduler, ReduceLROnPlateau): finetuner.scheduler.step(val_metric_current if val_metric_current is not None else float('-inf'))
                    else: finetuner.scheduler.step()
                if val_metric_current is not None:
                    is_better = (val_metric_current > best_val_metric) if metric_to_watch!="val_loss" else (val_metric_current < best_val_metric)
                    if is_better:
                        best_val_metric = val_metric_current
                        savename_best = f"{cfg['best_model_filename_base']}.pth"
                        finetuner.save_checkpoint(os.path.join(abs_ckpt_save_dir, savename_best), epoch_1_based, best_val_metric, metric_to_watch)
                        logger.info(f"E{epoch_1_based}: New best! Val {metric_to_watch}: {best_val_metric:.4f}"); patience_counter = 0
                    else: patience_counter += 1; logger.info(f"E{epoch_1_based}: Val {metric_to_watch} ({val_metric_current:.4f}) not better than {best_val_metric:.4f}. Patience: {patience_counter}/{patience}")
            
            save_freq = cfg.get('save_checkpoint_every_n_epochs',0)
            if save_freq > 0 and epoch_1_based % save_freq == 0 and epoch_1_based != cfg["epochs"]: # Avoid double save on last epoch
                 finetuner.save_checkpoint(os.path.join(abs_ckpt_save_dir,f"checkpoint_epoch_{epoch_1_based}.pth"), epoch_1_based, val_metric_current if val_metric_current is not None else best_val_metric, metric_to_watch)
            if patience_counter >= patience: logger.info(f"Early stopping at E{epoch_1_based}."); break
            epochs_this_run +=1; logger.info(f"Epoch {epoch_1_based} completed in {(time.time() - epoch_time_start):.2f}s.")
            if device == 'cuda': logger.debug(f"CUDA Mem E{epoch_1_based} End: Alloc {torch.cuda.memory_allocated(0)/1024**2:.1f}MB")
            
    except KeyboardInterrupt: logger.warning(f"FT interrupted. Epochs this run: {epochs_this_run}. Last completed (abs): {start_epoch + epochs_this_run -1 if epochs_this_run > 0 else start_epoch -1}.")
    except Exception as e_fatal: logger.critical(f"FT error at E{start_epoch + epochs_this_run}: {e_fatal}", exc_info=True); sys.exit(1)
    finally:
        final_abs_epoch = start_epoch + epochs_this_run -1 if epochs_this_run > 0 else start_epoch -1
        if epochs_this_run == 0 and start_epoch == 1 : final_abs_epoch = 0
        logger.info(f"FT ended. Absolute epochs completed: {final_abs_epoch}.")
        if 'finetuner' in locals() and finetuner is not None and cfg.get("final_model_filename_base"):
            final_savename = f"{cfg['final_model_filename_base']}.pth"
            finetuner.save_model_checkpoint(os.path.join(abs_ckpt_save_dir, final_savename), final_abs_epoch, best_val_metric, metric_to_watch)
        logger.info(f"FT summary: Best validation '{metric_to_watch}': {best_val_metric:.4f}")
        if history['train_loss']: plot_training_curves(history, run_specific_log_dir, metric_to_watch)

if __name__ == "__main__":
    try:
        main_execution_logic()
    except SystemExit as se:
        final_logger_exit = logging.getLogger(__name__) if logger_main_global and logger_main_global.hasHandlers() else logging.getLogger("main_fallback_exit")
        if not final_logger_exit.hasHandlers() and not (final_logger_exit.parent and final_logger_exit.parent.hasHandlers()): logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        if se.code != 0 and se.code is not None: final_logger_exit.info(f"Application exited with code {se.code}.")
        sys.exit(se.code if se.code is not None else 0)
    except Exception as e_unhandled:
        final_logger_unhandled = logging.getLogger(__name__) if logger_main_global and logger_main_global.hasHandlers() else logging.getLogger("main_fallback_unhandled")
        if not final_logger_unhandled.hasHandlers() and not (final_logger_unhandled.parent and final_logger_unhandled.parent.hasHandlers()): logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
        final_logger_unhandled.critical(f"Unhandled CRITICAL exception in __main__ execution: {e_unhandled}", exc_info=True)
        sys.exit(1)