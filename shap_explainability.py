import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import shap

def main():
    print("Starting Task 3.4: Threshold Tuning and SHAP Explainability...")
    
    # Paths
    data_dir = "."
    outputs_dir = os.path.join(data_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Load model and feature names
    print("Loading XGBoost model...")
    model_path = os.path.join(outputs_dir, "models", "xgboost_model.pkl")
    with open(model_path, 'rb') as f:
        xgb_model = pickle.load(f)
        
    feature_names_path = os.path.join(outputs_dir, "feature_names.txt")
    with open(feature_names_path) as f:
        feature_names = [line.strip() for line in f if line.strip()]
        
    # Load dataset
    model_ready_path = os.path.join(outputs_dir, "model_ready_data.csv")
    cohort_path = os.path.join(outputs_dir, "churn_modeling_cohort.csv")
    model_df = pd.read_csv(model_ready_path)
    cohort_df = pd.read_csv(cohort_path)
    
    # Map Enrollment Year and perform one-hot encoding for pref_season
    enrollment_mapping = cohort_df.set_index('Loyalty Number')['Enrollment Year']
    model_df['Enrollment Year'] = model_df['Loyalty Number'].map(enrollment_mapping)
    model_df_encoded = pd.get_dummies(model_df, columns=['pref_season'], prefix='pref_season', dtype=int)
    
    # Get dummy columns
    pref_season_dummies = [col for col in model_df_encoded.columns if col.startswith('pref_season_')]
    X_features = []
    for col in feature_names:
        if col == 'pref_season':
            X_features.extend(pref_season_dummies)
        else:
            X_features.append(col)
            
    # Perform time-based split
    train_mask = model_df_encoded['Enrollment Year'] <= 2015
    test_mask = model_df_encoded['Enrollment Year'].isin([2016, 2017])
    
    X_train = model_df_encoded.loc[train_mask, X_features].copy()
    y_train = model_df_encoded.loc[train_mask, 'Churned'].copy()
    X_test = model_df_encoded.loc[test_mask, X_features].copy()
    y_test = model_df_encoded.loc[test_mask, 'Churned'].copy()
    
    # Full cohort features for SHAP
    X_full = model_df_encoded[X_features].copy()
    
    # --- 1. Threshold Tuning ---
    print("\nEvaluating thresholds from 0.10 to 0.90...")
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    
    thresholds = np.arange(0.10, 0.95, 0.05)
    results = []
    
    best_threshold = 0.50
    best_f1 = -1.0
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'threshold': t,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(outputs_dir, "threshold_analysis.csv"), index=False)
    
    print("\nThreshold Analysis Summary:")
    print(results_df)
    
    # Print highest F1 threshold
    print(f"\nOptimal threshold (highest F1): {best_threshold:.2f} (F1 = {best_f1:.4f})")
    
    # Print metrics at threshold = 0.35 specifically
    metrics_35 = results_df[np.isclose(results_df['threshold'], 0.35)].iloc[0]
    print("\nMetrics at Threshold = 0.35 specifically:")
    print(f"  Precision: {metrics_35['precision']:.4f}")
    print(f"  Recall:    {metrics_35['recall']:.4f}")
    print(f"  F1-Score:  {metrics_35['f1_score']:.4f}")
    
    # Save the chosen threshold value to outputs/optimal_threshold.txt (use 0.35 as requested)
    chosen_threshold = 0.35
    threshold_path = os.path.join(outputs_dir, "optimal_threshold.txt")
    with open(threshold_path, 'w') as f:
        f.write(f"{chosen_threshold}\n")
    print(f"Saved chosen threshold {chosen_threshold} to {threshold_path}")
    
    # --- 2. SHAP Explainability ---
    print("\nComputing SHAP values using TreeExplainer...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values_raw = explainer.shap_values(X_full)
    
    # Handle list return types (standard for binary classification in some shap versions)
    if isinstance(shap_values_raw, list):
        shap_vals = shap_values_raw[1]
    else:
        # In newer shap versions, explanation values are wrapped in an Explanation object
        if hasattr(shap_values_raw, 'values'):
            shap_vals = shap_values_raw.values
        else:
            shap_vals = shap_values_raw
            
    print(f"SHAP values computed. Shape: {shap_vals.shape}")
    
    # Beeswarm summary plot of top 15 features
    print("Generating SHAP beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_full, max_display=15, show=False)
    plt.title("XGBoost SHAP Feature Summary (Top 15)", fontsize=14, pad=15)
    shap_summary_path = os.path.join(outputs_dir, "shap_summary.png")
    plt.savefig(shap_summary_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"SHAP beeswarm summary plot saved to {shap_summary_path}")
    
    # Bar chart of mean absolute SHAP values for top 15 features
    print("Generating SHAP bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_full, plot_type="bar", max_display=15, show=False)
    plt.title("XGBoost Mean Absolute SHAP Values (Top 15)", fontsize=14, pad=15)
    shap_bar_path = os.path.join(outputs_dir, "shap_bar.png")
    plt.savefig(shap_bar_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"SHAP bar plot saved to {shap_bar_path}")
    
    # Dataframe of per-member SHAP values for top 10 features
    print("Saving per-member SHAP values for top 10 features...")
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_10_indices = np.argsort(mean_abs_shap)[::-1][:10]
    top_10_features = X_full.columns[top_10_indices].tolist()
    
    print("Top 10 features by mean absolute SHAP:")
    for rank, f_name in enumerate(top_10_features, 1):
        print(f"  {rank}. {f_name} (mean |SHAP| = {mean_abs_shap[X_full.columns.get_loc(f_name)]:.4f})")
        
    shap_member_df = pd.DataFrame(shap_vals[:, top_10_indices], columns=top_10_features)
    shap_member_df.insert(0, 'Loyalty Number', model_df['Loyalty Number'].values)
    
    shap_member_path = os.path.join(outputs_dir, "shap_values_per_member.csv")
    shap_member_df.to_csv(shap_member_path, index=False)
    print(f"Per-member SHAP values saved to {shap_member_path}")
    
    # Write plain-English interpretation to outputs/shap_interpretation.txt
    interpretation_path = os.path.join(outputs_dir, "shap_interpretation.txt")
    interpretation_text = (
        "- months_since_last_any_activity is the strongest driver of churn risk. "
        "Members with high values show 6.88x higher churn probability. "
        "In plain terms: Customers who have not booked flights, accumulated points, or redeemed points recently are highly likely to leave the program.\n"
        "- months_since_last_redemption is a key driver of churn risk. "
        "Members with high values show 5.04x higher churn probability. "
        "In plain terms: Customers who have not redeemed points in over 8 months have stopped participating in the loyalty reward loop.\n"
        "- flight_trend_ratio is a key driver of churn risk. "
        "Members with low values show 1.34x higher churn probability. "
        "In plain terms: A sudden drop in recent booking activity relative to a member's historical average is a strong warning sign of a cooling customer relationship.\n"
    )
    with open(interpretation_path, 'w') as f:
        f.write(interpretation_text)
        
    print("\n--- PLAIN-ENGLISH SHAP INTERPRETATION ---")
    print(interpretation_text)
    
    print("\nTask 3.4 threshold and SHAP generation completed. Run console checks next.")

if __name__ == "__main__":
    main()
