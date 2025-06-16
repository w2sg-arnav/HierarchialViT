# phase5_analysis_and_ablation/test_significance.py
import numpy as np
from scipy import stats

def run_t_test(group1_scores, group2_scores, group1_name="Group 1", group2_name="Group 2"):
    """
    Performs an independent t-test and prints the results.
    """
    if len(group1_scores) < 2 or len(group2_scores) < 2:
        print("Need at least two data points in each group to perform a t-test.")
        return

    # Perform Welch's t-test, which does not assume equal variances
    t_stat, p_value = stats.ttest_ind(group1_scores, group2_scores, equal_var=False)
    
    print(f"\n--- T-Test: {group1_name} vs. {group2_name} ---")
    print(f"Scores for {group1_name}: {np.round(group1_scores, 2)}")
    print(f"Scores for {group2_name}: {np.round(group2_scores, 2)}")
    print(f"Mean {group1_name}: {np.mean(group1_scores):.2f} | Mean {group2_name}: {np.mean(group2_scores):.2f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"Conclusion: The difference is statistically significant (p < {alpha}).")
    else:
        print(f"Conclusion: The difference is not statistically significant (p >= {alpha}).")
    print("-" * (13 + len(group1_name) + len(group2_name)))

if __name__ == "__main__":
    # Example using your synthetic data (replace with your actual 5-run scores)
    
    # Main Comparison: Our Best System vs. Strongest Baseline
    hvt_ssl_scores = [96.71, 96.60, 96.85, 96.55, 96.82] # Centered around 96.71
    vit_base_scores = [95.18, 95.30, 95.05, 95.25, 95.12] # Centered around 95.18
    run_t_test(hvt_ssl_scores, vit_base_scores, "HVT-Leaf (SSL, Best Config)", "ViT-Base")

    # Ablation: Our Best System vs. Training from Scratch
    hvt_scratch_scores = [91.52, 91.20, 91.80, 91.35, 91.75] # Centered around 91.52
    run_t_test(hvt_ssl_scores, hvt_scratch_scores, "HVT-Leaf (SSL, Best Config)", "HVT-Leaf (From Scratch)")

    # Ablation: Our Best System vs. No EMA/TTA
    no_ema_tta_scores = [92.35, 92.10, 92.55, 92.20, 92.50] # Centered around 92.35
    run_t_test(hvt_ssl_scores, no_ema_tta_scores, "HVT-Leaf (SSL, Best Config)", "Ablation (No EMA/TTA)")