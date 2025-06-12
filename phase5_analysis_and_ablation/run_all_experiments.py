# phase5_analysis_and_ablation/run_all_experiments.py
import yaml
import subprocess
import os
import sys
import copy
import logging
import time
from datetime import timedelta

# --- Path Setup ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path: sys.path.insert(0, _project_root)

# --- Configuration ---
EXPERIMENTS_CONFIG_PATH = os.path.join(_current_dir, "config_experiments.yaml")
PYTHON_EXECUTABLE = sys.executable
MAIN_TRAINER_SCRIPT = os.path.join(_project_root, "phase4_finetuning/main.py")
TEMP_CONFIG_DIR = os.path.join(_current_dir, "temp_configs")
os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s - Orchestrator - %(levelname)s] %(message)s')

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def format_duration(seconds):
    """Formats a duration in seconds into a human-readable string H:M:S."""
    return str(timedelta(seconds=int(seconds)))

def main():
    logging.info("======== Starting Experiment Orchestrator ========")
    if not os.path.exists(MAIN_TRAINER_SCRIPT):
        logging.error(f"Main trainer script not found at {MAIN_TRAINER_SCRIPT}")
        return

    with open(EXPERIMENTS_CONFIG_PATH, 'r') as f:
        config_data = yaml.safe_load(f)

    base_config = config_data['base_config']
    experiments = config_data['experiments']
    
    overall_start_time = time.time()
    num_experiments = len(experiments)

    for i, exp in enumerate(experiments):
        exp_name = exp['name']
        logging.info(f"\n{'='*25} Preparing Experiment {i+1}/{num_experiments}: {exp_name} {'='*25}")

        exp_config = copy.deepcopy(base_config)
        if 'changes' in exp:
            exp_config = deep_update(exp_config, exp['changes'])
        if 'model_override' in exp:
            exp_config['model_override'] = exp['model_override']

        exp_config['run_name_prefix'] = exp_name
        
        temp_config_path = os.path.join(TEMP_CONFIG_DIR, f"{exp_name}.yaml")
        with open(temp_config_path, 'w') as f:
            yaml.dump(exp_config, f)
        
        logging.info(f"Generated temp config: {temp_config_path}")
        
        command = [PYTHON_EXECUTABLE, MAIN_TRAINER_SCRIPT, "--config", temp_config_path]
        
        logging.info(f"Executing: {' '.join(command)}")
        logging.info("Subprocess output will be displayed below. This may take a long time.")
        
        exp_start_time = time.time()
        
        # --- CHANGE: Use subprocess.run to let the child process control the terminal ---
        # This fixes the tqdm repetitive logging issue.
        # We set capture_output=True to get stdout/stderr after completion for logging.
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=False # Set to False to handle non-zero exit codes manually
        )
        
        exp_end_time = time.time()
        exp_duration = exp_end_time - exp_start_time

        if result.returncode == 0:
            logging.info(f"--- Experiment {exp_name} finished successfully in {format_duration(exp_duration)} ---")
        else:
            logging.error(f"!!! Experiment {exp_name} failed with return code {result.returncode} after {format_duration(exp_duration)} !!!")
            logging.error("--- STDOUT ---")
            logging.error(result.stdout)
            logging.error("--- STDERR ---")
            logging.error(result.stderr)
            logging.error("Stopping orchestrator due to failure.")
            break # Stop on first error
            
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    logging.info(f"\n======== Experiment Orchestrator Finished ========")
    logging.info(f"Total run time for all experiments: {format_duration(total_duration)}")

if __name__ == "__main__":
    main()