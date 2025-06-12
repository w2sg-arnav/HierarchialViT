# Phase 5: Analysis & Experimentation

This phase focuses on analyzing the best model from Phase 4 and running a comprehensive suite of ablation and comparison studies.

## 1. Analyze Best Model

To generate a confusion matrix and performance statistics for your single best fine-tuned model:

1.  Update the `CHECKPOINT_PATH` in `analyze_best_model.py` to point to your `best_model.pth` from the Phase 4 run.
2.  Run the script:
    ```bash
    python phase5_analysis_and_ablation/analyze_best_model.py
    ```

## 2. Run All Experiments

This script orchestrates all experiments defined in `config_experiments.yaml`.

**Prerequisite:** You must modify `phase4_finetuning/main.py` to handle model overrides as described in the comments of `run_all_experiments.py`.

To run all experiments:
```bash
python phase5_analysis_and_ablation/run_all_experiments.py