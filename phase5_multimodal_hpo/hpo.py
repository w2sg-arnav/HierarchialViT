# phase5_multimodal_hpo/hpo.py
import optuna
import logging
import os
import sys
import torch # For device check

# --- Path Setup ---
# Ensure project root is in path to allow imports from phase2_model, phase3_pretraining etc.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"DEBUG (hpo.py): Added project root to sys.path: {project_root}")

# --- Project Imports ---
from phase5_multimodal_hpo.config import config as base_config # Base config
from phase5_multimodal_hpo.main import run_training_session # Import the training function
from phase5_multimodal_hpo.utils.logging_setup import setup_logging

# --- HPO Objective Function ---
def objective(trial: optuna.Trial) -> float:
    """ Optuna objective function wraps the training session. """
    logger = logging.getLogger("optuna_hpo") # Use a specific logger for HPO
    
    # --- Create Config for this Trial ---
    trial_config = base_config.copy()
    
    # --- Suggest Hyperparameters ---
    # Optimizer Params
    trial_config['learning_rate'] = trial.suggest_float(
        "learning_rate", 
        base_config['HPO_LR_LOW'], 
        base_config['HPO_LR_HIGH'], 
        log=True
    )
    trial_config['weight_decay'] = trial.suggest_float(
        "weight_decay", 
        base_config['HPO_WD_LOW'], 
        base_config['HPO_WD_HIGH'], 
        log=True
    )
    
    # Loss Params
    trial_config['loss_label_smoothing'] = trial.suggest_float(
        "label_smoothing", 
        base_config['HPO_LABEL_SMOOTHING_LOW'], 
        base_config['HPO_LABEL_SMOOTHING_HIGH']
    )

    # Scheduler Params (Example: T_0 for Cosine)
    # trial_config['cosine_t_0'] = trial.suggest_int("cosine_t_0", 5, 15) 
    
    # Model Params (Example: Drop path rate)
    # trial_config['hvt_drop_path_rate'] = trial.suggest_float("hvt_drop_path_rate", 0.0, 0.3)

    # Log suggested parameters for this trial
    logger.info(f"--- Optuna Trial {trial.number} ---")
    logger.info(f"  Params: {trial.params}")
    # Update the config dict passed to the training function
    # Directly modify the trial_config dictionary which is a copy
    
    # --- Run Training ---
    try:
        # Ensure seed is different per trial *if desired*, or keep fixed for reproducibility check
        # trial_config['seed'] = trial.suggest_int("seed", 1, 10000) # Optional: vary seed
        
        # Redirect logging for the training session temporarily if needed, 
        # or let the main setup handle it.
        
        # Clear CUDA cache before starting trial (optional)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        best_metric = run_training_session(trial_config)
        
        logger.info(f"--- Trial {trial.number} Finished ---")
        logger.info(f"  Best Validation Metric ({trial_config['metric_to_monitor']}): {best_metric:.6f}")
        
        # Return the metric Optuna should maximize/minimize
        return best_metric # Optuna maximizes by default

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        # Tell Optuna this trial failed (it will be pruned or marked failed)
        # Returning a value indicating failure (e.g., 0.0 or -1.0 if maximizing accuracy/F1)
        return 0.0 # Or use optuna.exceptions.TrialPruned() if applicable


# --- Main HPO Execution ---
if __name__ == "__main__":
    # Setup logging specifically for the HPO script maybe? Or use root logger.
    setup_logging(log_file_name="hpo_main.log", log_dir=base_config["log_dir"], log_level=logging.INFO, logger_name=None)
    hpo_logger = logging.getLogger("optuna_hpo") # Get the specific logger
    hpo_logger.setLevel(logging.INFO) # Set level for HPO logger

    study_name = base_config['HPO_STUDY_NAME']
    storage_name = base_config['HPO_STORAGE_DB']
    n_trials = base_config['HPO_N_TRIALS']
    
    hpo_logger.info(f"Starting Optuna HPO Study: '{study_name}'")
    hpo_logger.info(f"Storage: {storage_name}")
    hpo_logger.info(f"Number of trials: {n_trials}")

    # Create or load the study
    # Using JournalStorage for file-based persistence, JournalFileStorage for simpler locking
    # For SQLite: storage=f"sqlite:///{storage_name}" # Use relative path
    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(storage_name))
    
    study = optuna.create_study(
        storage=storage, 
        study_name=study_name, 
        direction="maximize", # Maximize validation metric (e.g., accuracy or F1)
        load_if_exists=True # Resume study if it already exists
    )

    # Enqueue trials if study is new or interrupted (optional)
    # study.enqueue_trial({...}) # Can specify parameters for first few trials

    # Start optimization
    try:
        study.optimize(objective, n_trials=n_trials, timeout=None) # No timeout
    except KeyboardInterrupt:
         hpo_logger.warning("HPO interrupted by user.")
    except Exception as e:
         hpo_logger.exception(f"An error occurred during HPO: {e}")

    # --- Print Results ---
    hpo_logger.info("--- HPO Finished ---")
    hpo_logger.info(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    hpo_logger.info(f"Best trial value ({base_config['metric_to_monitor']}): {best_trial.value:.6f}")
    hpo_logger.info("Best parameters found:")
    for key, value in best_trial.params.items():
        hpo_logger.info(f"  {key}: {value}")

    # You can also save results to a file or visualize them
    # df = study.trials_dataframe()
    # df.to_csv("hpo_results.csv")
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_param_importances(study).show()