import os
import pickle
import pandas as pd
import numpy as np

def main():
    print("Starting Task 3.7: Score All Cohort Members...")
    
    # Paths
    data_dir = "."
    outputs_dir = os.path.join(data_dir, "outputs")
    
    # Load files
    print("Loading datasets and model...")
    model_ready_path = os.path.join(outputs_dir, "model_ready_data.csv")
    cohort_path = os.path.join(outputs_dir, "churn_modeling_cohort.csv")
    feature_names_path = os.path.join(outputs_dir, "feature_names.txt")
    optimal_threshold_path = os.path.join(outputs_dir, "optimal_threshold.txt")
    model_path = os.path.join(outputs_dir, "models", "xgboost_model.pkl")
    loyalty_history_path = os.path.join(data_dir, "Customer Loyalty History.csv")
    
    model_df = pd.read_csv(model_ready_path)
    cohort_df = pd.read_csv(cohort_path)
    loyalty_df = pd.read_csv(loyalty_history_path)
    
    with open(feature_names_path) as f:
        feature_names = [line.strip() for line in f if line.strip()]
        
    with open(optimal_threshold_path) as f:
        optimal_threshold = float(f.read().strip())
        
    with open(model_path, 'rb') as f:
        xgb_model = pickle.load(f)
        
    print(f"Loaded optimal threshold: {optimal_threshold}")
    
    # One-hot encode pref_season and align features (same as training)
    print("Preprocessing features for scoring...")
    model_df_encoded = pd.get_dummies(model_df, columns=['pref_season'], prefix='pref_season', dtype=int)
    pref_season_dummies = [col for col in model_df_encoded.columns if col.startswith('pref_season_')]
    
    X_features = []
    for col in feature_names:
        if col == 'pref_season':
            X_features.extend(pref_season_dummies)
        else:
            X_features.append(col)
            
    X_full = model_df_encoded[X_features].copy()
    
    # Score all cohort members
    print("Predicting churn probabilities for all members...")
    churn_prob = xgb_model.predict_proba(X_full)[:, 1]
    
    # Create prediction and risk tier columns
    churn_prediction = (churn_prob >= optimal_threshold).astype(int)
    
    # churn_risk_tier: 'Low' if churn_prob < 0.30, 'Medium' if 0.30-0.60, 'High' if > 0.60
    conditions = [
        churn_prob < 0.30,
        (churn_prob >= 0.30) & (churn_prob <= 0.60),
        churn_prob > 0.60
    ]
    choices = ['Low', 'Medium', 'High']
    churn_risk_tier = np.select(conditions, choices, default='Medium')
    
    # Construct scoring results dataframe
    scored_results = pd.DataFrame({
        'Loyalty Number': model_df['Loyalty Number'],
        'churn_prob': churn_prob,
        'churn_prediction': churn_prediction,
        'churn_risk_tier': churn_risk_tier
    })
    
    # Merge back with outputs/churn_modeling_cohort.csv to retain all demographic columns + BDS + Class
    print("Merging scored results back with cohort demographics...")
    final_df = pd.merge(cohort_df, scored_results, on='Loyalty Number', how='left')
    
    # Also merge Province from the original loyalty history file if not present or to ensure alignment
    # Wait, cohort_df already contains Province, but we can verify it exists or merge it if needed.
    if 'Province' not in final_df.columns:
        print("Merging Province from loyalty history file...")
        province_df = loyalty_df[['Loyalty Number', 'Province']].copy()
        final_df = pd.merge(final_df, province_df, on='Loyalty Number', how='left')
    else:
        print("Province column already present in cohort dataset.")
        
    # Save the full scored dataset to outputs/members_scored.csv
    scored_output_path = os.path.join(outputs_dir, "members_scored.csv")
    print(f"Saving scored members to {scored_output_path}...")
    final_df.to_csv(scored_output_path, index=False)
    
    # Print the required statistics for dashboard/report
    print("\n=== DISTRIBUTION OF CHURN RISK TIER ===")
    tier_counts = final_df['churn_risk_tier'].value_counts()
    tier_pct = final_df['churn_risk_tier'].value_counts(normalize=True) * 100
    for tier in ['Low', 'Medium', 'High']:
        count = tier_counts.get(tier, 0)
        pct = tier_pct.get(tier, 0.0)
        print(f"  {tier} Risk: {count} members ({pct:.2f}%)")
        
    print("\n=== MEAN CLV BY RISK TIER ===")
    mean_clv = final_df.groupby('churn_risk_tier')['CLV'].mean()
    for tier in ['Low', 'Medium', 'High']:
        print(f"  {tier} Risk Mean CLV: ${mean_clv.get(tier, 0.0):,.2f}")
        
    print("\n=== TOTAL CLV AT RISK ===")
    high_risk_members = final_df[final_df['churn_risk_tier'] == 'High']
    total_clv_at_risk = high_risk_members['CLV'].sum()
    print(f"  Total CLV at Risk (High Churn Risk Members): ${total_clv_at_risk:,.2f}")
    
    print("\nTask 3.7 completed successfully.")

if __name__ == "__main__":
    main()
