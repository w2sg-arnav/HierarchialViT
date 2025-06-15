# phase5_analysis_and_ablation/analyze_efficiency.py

import torch
import timm
import time
import pandas as pd
import yaml
import os
import sys
from fvcore.nn import FlopCountAnalysis

# --- Path Setup ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import your HVT factory from the project
from phase2_model.models.hvt import create_disease_aware_hvt

# --- Configuration ---
# Point this to the config file of your BEST run
# It's mainly used to get the hvt_params for model instantiation
CONFIG_PATH = "/teamspace/studios/this_studio/cvpr25/phase5_analysis_and_ablation/temp_configs/03_ablation_no_advanced_augs.yaml"

def analyze_model(model, model_name, device, input_res=(448, 448)):
    """Analyzes a single model for params, FLOPs, and throughput."""
    print(f"\n--- Analyzing {model_name} at resolution {input_res} ---")
    model.to(device).eval()
    
    # Create a dummy input tensor with batch size 1 for standard analysis
    dummy_input = torch.randn(1, 3, input_res[0], input_res[1]).to(device)
    
    # 1. Calculate Parameters
    params = sum(p.numel() for p in model.parameters()) / 1e6  # In millions
    
    # 2. Calculate FLOPs
    flops = "N/A"
    try:
        # Use a try-except block as some models might have unsupported ops
        flops_analyzer = FlopCountAnalysis(model, dummy_input)
        flops = flops_analyzer.total() / 1e9 # In GFLOPs
    except Exception as e:
        print(f"  Warning: Could not calculate FLOPs for {model_name}. Reason: {e}")

    # 3. Calculate Throughput
    throughput = "N/A"
    if device == 'cuda':
        # Warmup
        with torch.no_grad():
            for _ in range(20):
                _ = model(dummy_input)
        
        # Timing
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(dummy_input)
        torch.cuda.synchronize()
        end_time = time.time()
        throughput = 100 / (end_time - start_time) # images per second

    print(f"  > Results: {params:.2f}M params, {flops if isinstance(flops, str) else f'{flops:.2f}'} GFLOPs, {throughput if isinstance(throughput, str) else f'{throughput:.2f}'} img/s")
    return {"Model": model_name, "Params (M)": f"{params:.2f}", "GFLOPs (G)": f"{flops:.2f}" if isinstance(flops, float) else flops, "Throughput (img/s)": f"{throughput:.2f}" if isinstance(throughput, float) else throughput}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: Running on CPU. Throughput analysis will not be representative.")

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}. Please update the path.")
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # The resolution used for fine-tuning all models
    IMG_SIZE = tuple(cfg['data']['img_size'])
    print(f"Using standard image size for analysis: {IMG_SIZE}")

    # --- Instantiate Models ---
    print("\nInstantiating models for analysis...")
    # 1. Our HVT-Leaf Model
    hvt_model = create_disease_aware_hvt(
        current_img_size=IMG_SIZE,
        num_classes=7,
        model_params_dict=cfg['model']['hvt_params']
    )
    
    # 2. ResNet Baseline
    resnet_model = timm.create_model('resnet101', pretrained=False, num_classes=7)

    # 3. ViT Baseline
    vit_model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=False, num_classes=7, img_size=IMG_SIZE)

    models_to_analyze = {
        "HVT-Leaf (Ours)": hvt_model,
        "ResNet-101": resnet_model,
        "ViT-Base-16": vit_model
    }
    
    results_list = []
    for name, model in models_to_analyze.items():
        try:
            # FIX: All models are analyzed at the same fine-tuning resolution for a fair comparison.
            results_list.append(analyze_model(model, name, device, input_res=IMG_SIZE))
        except Exception as e:
            print(f"FATAL: Could not analyze {name}. Error: {e}")
            
    # Create and print a clean pandas DataFrame
    df = pd.DataFrame(results_list)
    print("\n\n--- Efficiency Analysis Results ---")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()