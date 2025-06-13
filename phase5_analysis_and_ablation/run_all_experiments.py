# phase5_analysis_and_ablation/run_all_experiments.py
import yaml
import subprocess
import os
import sys
import copy
import logging
import time
from datetime import timedelta
import argparse # <--- CHANGE: Import argparse

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
    # --- CHANGE: Add argument parsing to specify a starting point ---
    parser = argparse.ArgumentParser(description="Run a sequence of fine-tuning experiments.")
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="The name of the experiment to start from. Skips all preceding experiments in the config file."
    )
    args = parser.parse_args()
    start_experiment_name = args.start_from
    # --- END OF CHANGE ---

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

    # --- CHANGE: Logic to handle skipping experiments ---
    processing_started = start_experiment_name is None
    if not processing_started:
        logging.info(f"Attempting to resume from experiment: '{start_experiment_name}'")
    # --- END OF CHANGE ---

    for i, exp in enumerate(experiments):
        exp_name = exp['name']

        # --- CHANGE: Check if we should skip this experiment ---
        if not processing_started:
            if exp_name == start_experiment_name:
                processing_started = True
                logging.info(f"Found start experiment '{exp_name}'. Resuming execution...")
            else:
                logging.info(f"Skipping completed experiment ({i+1}/{num_experiments}): {exp_name}")
                continue
        # --- END OF CHANGE ---
        
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
        logging.info("--- Subprocess Live Output Starts Below ---")
        
        exp_start_time = time.time()
        
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )

        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            sys.stdout.flush()

        process.wait()
        
        exp_end_time = time.time()
        exp_duration = exp_end_time - exp_start_time

        logging.info("--- Subprocess Live Output Ended ---")
        if process.returncode == 0:
            logging.info(f"--- Experiment {exp_name} finished successfully in {format_duration(exp_duration)} ---")
        else:
            logging.error(f"!!! Experiment {exp_name} failed with return code {process.returncode} after {format_duration(exp_duration)} !!!")
            logging.error("Stopping orchestrator due to failure.")
            break
            
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    logging.info(f"\n======== Experiment Orchestrator Finished ========")
    logging.info(f"Total run time for all experiments: {format_duration(total_duration)}")

if __name__ == "__main__":
    main()