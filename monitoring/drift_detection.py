import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import os

def calculate_psi(expected, actual, num_bins=10):
    """
    Calculate Population Stability Index (PSI) between expected (reference) and actual (production) arrays.
    Works for both categorical arrays and numeric binned arrays.
    """
    # If categorical or string type, compute distributions directly
    is_categorical = False
    if not pd.api.types.is_numeric_dtype(expected):
        is_categorical = True
    elif len(np.unique(expected)) < 5:
        is_categorical = True
        
    if is_categorical:
        exp_counts = pd.Series(expected).value_counts(normalize=True)
        act_counts = pd.Series(actual).value_counts(normalize=True)
        
        # Align indexes
        all_indices = exp_counts.index.union(act_counts.index)
        exp_pct = exp_counts.reindex(all_indices, fill_value=0.0001)
        act_pct = act_counts.reindex(all_indices, fill_value=0.0001)
    else:
        # For continuous features, bin the reference dataset
        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(expected, percentiles)
        # Deduplicate bins if needed
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
            
        exp_counts, _ = np.histogram(expected, bins=bins)
        act_counts, _ = np.histogram(actual, bins=bins)
        
        # Add small constant to avoid division by zero
        exp_pct = (exp_counts + 0.5) / (len(expected) + 0.5 * len(exp_counts))
        act_pct = (act_counts + 0.5) / (len(actual) + 0.5 * len(act_counts))
        
    psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi_value)

def run_drift_analysis(reference_df, production_df):
    """
    Run KS-test on continuous features and PSI on categorical features/predictions.
    """
    results = []
    
    # Define continuous and categorical features
    continuous_features = ['age', 'length_of_stay_hours', 'billed_amount', 'days_since_registration']
    categorical_features = ['department', 'visit_type', 'insurance_provider', 'risk_score', 'claim_status']
    
    # 1. Analyze Continuous Features using K-S test
    for col in continuous_features:
        if col in reference_df.columns and col in production_df.columns:
            ref_vals = reference_df[col].dropna()
            prod_vals = production_df[col].dropna()
            
            # K-S statistic & p-value
            ks_stat, p_val = ks_2samp(ref_vals, prod_vals)
            drift_detected = p_val < 0.05
            
            # Also compute numerical PSI for comparison
            psi_val = calculate_psi(ref_vals.values, prod_vals.values)
            
            results.append({
                'MetricType': 'Feature_Numeric',
                'Feature': col,
                'Method': 'Kolmogorov-Smirnov',
                'Statistic': round(ks_stat, 4),
                'PValue': round(p_val, 6),
                'PSI': round(psi_val, 4),
                'DriftDetected': int(drift_detected),
                'DriftStatus': 'Drift Detected' if drift_detected else 'Stable'
            })
            
    # 2. Analyze Categorical Features using PSI
    for col in categorical_features:
        if col in reference_df.columns and col in production_df.columns:
            ref_vals = reference_df[col].dropna()
            prod_vals = production_df[col].dropna()
            
            psi_val = calculate_psi(ref_vals, prod_vals)
            
            if psi_val >= 0.25:
                status = 'Significant Drift'
                drift = 1
            elif psi_val >= 0.1:
                status = 'Moderate Drift (Warning)'
                drift = 0 # Warning, not full drift
            else:
                status = 'Stable'
                drift = 0
                
            results.append({
                'MetricType': 'Feature_Categorical',
                'Feature': col,
                'Method': 'Population Stability Index',
                'Statistic': round(psi_val, 4),
                'PValue': np.nan,
                'PSI': round(psi_val, 4),
                'DriftDetected': drift,
                'DriftStatus': status
            })
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Loading reference training set...")
    # Load reference data from modeling dataset
    data_path = "../data_outputs/model_table.csv"
    if not os.path.exists(data_path):
        data_path = "data_outputs/model_table.csv"
        
    df = pd.read_csv(data_path)
    ref_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    # Create a mock production dataset with induced drift to test the script
    # We simulate a younger cohort of patients and slightly shifted billing amounts
    print("Creating simulated production dataset with induced drift...")
    prod_df = test_df.copy()
    prod_df['age'] = prod_df['age'] - 10 # shift average age down
    prod_df['billed_amount'] = prod_df['billed_amount'] * 1.15 # increase average billing by 15%
    
    # Run analysis
    print("Running drift analysis...")
    drift_df = run_drift_analysis(ref_df, prod_df)
    
    # Save output
    output_dir = "../data_outputs" if os.path.exists("../data_outputs") else "data_outputs"
    output_path = os.path.join(output_dir, "drift_summary.csv")
    drift_df.to_csv(output_path, index=False)
    print(f"Drift analysis complete! Summary saved to '{output_path}'.")
    print("\n--- Drift Analysis Results ---")
    print(drift_df[['Feature', 'Method', 'PSI', 'DriftStatus']])
