# phase5_analysis_and_ablation/run_all_experiments.py
import yaml
import subprocess
import os
import sys
import copy
import logging

# --- Path Setup ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path: sys.path.insert(0, _project_root)

# --- Configuration ---
EXPERIMENTS_CONFIG_PATH = os.path.join(_current_dir, "config_experiments.yaml")
PYTHON_EXECUTABLE = sys.executable # Use the same python that runs this script
# Point to the main trainer script from the previous phase
MAIN_TRAINER_SCRIPT = os.path.join(_project_root, "phase4_finetuning/main.py")
TEMP_CONFIG_DIR = os.path.join(_current_dir, "temp_configs")
os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s')

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def main():
    logging.info("======== Starting Experiment Orchestrator ========")
    if not os.path.exists(MAIN_TRAINER_SCRIPT):
        logging.error(f"Main trainer script not found at {MAIN_TRAINER_SCRIPT}")
        return

    with open(EXPERIMENTS_CONFIG_PATH, 'r') as f:
        config_data = yaml.safe_load(f)

    base_config = config_data['base_config']
    experiments = config_data['experiments']

    for exp in experiments:
        exp_name = exp['name']
        logging.info(f"\n{'='*20} Preparing Experiment: {exp_name} {'='*20}")

        exp_config = copy.deepcopy(base_config)
        if 'changes' in exp:
            exp_config = deep_update(exp_config, exp['changes'])
        if 'model_override' in exp: # Handle SOTA model swaps
            exp_config['model_override'] = exp['model_override']

        exp_config['run_name_prefix'] = exp_name
        
        temp_config_path = os.path.join(TEMP_CONFIG_DIR, f"{exp_name}.yaml")
        with open(temp_config_path, 'w') as f:
            yaml.dump(exp_config, f)
        
        logging.info(f"Generated temp config: {temp_config_path}")
        
        command = [PYTHON_EXECUTABLE, MAIN_TRAINER_SCRIPT, "--config", temp_config_path]
        
        logging.info(f"Executing: {' '.join(command)}")
        try:
            # Note: For long runs, you might want to run this in the background
            # or use a more robust job management system.
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line) # Print live output
            process.wait()
            if process.returncode != 0:
                logging.error(f"Experiment {exp_name} failed with return code {process.returncode}")
                break
        except Exception as e:
            logging.error(f"Failed to execute experiment {exp_name}: {e}", exc_info=True)
            break

    logging.info("======== Experiment Orchestrator Finished ========")

if __name__ == "__main__":
    main()