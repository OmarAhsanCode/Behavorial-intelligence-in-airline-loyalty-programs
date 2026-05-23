"""
segmentation_fit.py

Task 4.3 — Fit Final Segmentation Model (K-Means, k=5)
Task 4.4 — Name the Archetypes
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def main():
    print("Starting Tasks 4.3 & 4.4: Fit Segmentation Model and Name Archetypes...")
    
    outputs_dir = "outputs"
    models_dir = os.path.join(outputs_dir, "models")
    
    # Load RFMET features
    rfmet_path = os.path.join(outputs_dir, "rfmet_features.csv")
    rfmet_df = pd.read_csv(rfmet_path)
    
    # Load members_scored.csv (has demographics + churn_prob + BDS + Class + Province)
    scored_path = os.path.join(outputs_dir, "members_scored.csv")
    scored_df = pd.read_csv(scored_path)
    
    # Load model_ready_data.csv to get engineered features for profile stats
    model_ready_path = os.path.join(outputs_dir, "model_ready_data.csv")
    mr_df = pd.read_csv(model_ready_path)
    
    # Merge needed engineered columns into scored_df
    needed_from_mr = ['Loyalty Number', 'months_since_last_flight', 'flights_last_12m',
                      'redemption_rate', 'loyalty_card_ordinal', 'tenure_months']
    merge_cols = [c for c in needed_from_mr if c not in scored_df.columns]
    if merge_cols:
        scored_df = pd.merge(scored_df, mr_df[['Loyalty Number'] + merge_cols], on='Loyalty Number', how='left')
    
    # --- Task 4.3: Fit K-Means ---
    # k=5 chosen because it has the highest silhouette score (0.7023) among k=2..10.
    # BIC preferred k=9 but with 5 dimensions and 11K members, k=5 gives interpretable,
    # well-separated clusters that map cleanly to business archetypes.
    CHOSEN_K = 5
    print(f"\nFitting K-Means with k={CHOSEN_K} (silhouette-optimal, interpretable)...")
    
    X = rfmet_df[['R', 'F', 'M', 'E', 'T']].values
    
    kmeans = KMeans(n_clusters=CHOSEN_K, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X)
    
    # Save fitted model
    kmeans_path = os.path.join(models_dir, "kmeans_model.pkl")
    with open(kmeans_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"K-Means model saved to {kmeans_path}")
    
    # Assign cluster labels back to scored_df
    scored_df['cluster_id'] = cluster_labels
    rfmet_df['cluster_id'] = cluster_labels
    
    # Build profile table
    print("\n--- Cluster Profiles ---")
    profiles = []
    
    # Merge RFMET into scored_df for profile computation
    rfmet_merge = rfmet_df[['Loyalty Number', 'R', 'F', 'M', 'E', 'T']].copy()
    profile_df = pd.merge(scored_df, rfmet_merge, on='Loyalty Number', how='left')
    
    total_members = len(profile_df)
    
    for cid in sorted(profile_df['cluster_id'].unique()):
        cluster = profile_df[profile_df['cluster_id'] == cid]
        n = len(cluster)
        
        profile = {
            'cluster_id': cid,
            'count': n,
            'pct_of_total': round(n / total_members * 100, 2),
            'mean_R': round(cluster['R'].mean(), 4),
            'mean_F': round(cluster['F'].mean(), 4),
            'mean_M': round(cluster['M'].mean(), 4),
            'mean_E': round(cluster['E'].mean(), 4),
            'mean_T': round(cluster['T'].mean(), 4),
            'mean_CLV': round(cluster['CLV'].mean(), 2),
            'mean_churn_prob': round(cluster['churn_prob'].mean(), 4),
            'mean_BDS': round(cluster['BDS'].mean(), 4),
            'mean_tenure_months': round(cluster['tenure_months'].mean(), 2),
            'pct_high_risk': round((cluster['churn_risk_tier'] == 'High').mean() * 100, 2),
            'most_common_card': cluster['Loyalty Card'].mode().iloc[0] if len(cluster['Loyalty Card'].mode()) > 0 else 'Unknown',
            'most_common_province': cluster['Province'].mode().iloc[0] if len(cluster['Province'].mode()) > 0 else 'Unknown'
        }
        profiles.append(profile)
        
    profiles_df = pd.DataFrame(profiles)
    profiles_path = os.path.join(outputs_dir, "cluster_profiles.csv")
    profiles_df.to_csv(profiles_path, index=False)
    
    print(profiles_df.to_string(index=False))
    print(f"\nCluster profiles saved to {profiles_path}")
    
    # --- Task 4.4: Name the Archetypes ---
    print("\n--- Task 4.4: Naming Archetypes ---")
    
    # Adapt thresholds to actual RFMET distributions:
    # - M (CLV) is heavily compressed by RobustScaler: max cluster mean is 0.12,
    #   so we use relative ranking (highest M cluster) instead of absolute 0.6.
    # - R (Recency) is near 1.0 for active clusters and 0.0 for churned.
    # - T (Card Tier) cleanly separates: 0.0=Star, 0.5=Nova, 1.0=Aurora.
    
    archetype_names = {}
    used_names = set()
    
    # Sort by primary rule signal strength for deterministic assignment
    # Rule 1: Crown Holders — highest card tier, low risk, highest relative CLV
    # Adapted: mean_T == 1.0 (Aurora) AND mean_churn_prob < 0.25 AND highest mean_M
    crown_candidates = profiles_df[
        (profiles_df['mean_T'] >= 0.9) & (profiles_df['mean_churn_prob'] < 0.25)
    ]
    if len(crown_candidates) > 0:
        best = crown_candidates.sort_values('mean_M', ascending=False).iloc[0]
        cid = int(best['cluster_id'])
        archetype_names[cid] = "Crown Holders"
        used_names.add("Crown Holders")
        print(f"  Rule 1 matched: Cluster {cid} -> Crown Holders (T={best['mean_T']:.2f}, churn={best['mean_churn_prob']:.4f}, M={best['mean_M']:.4f})")
    
    # Rule 2: Ghost Members — very high churn, inactive (R near 0)
    ghost_candidates = profiles_df[
        (~profiles_df['cluster_id'].isin(archetype_names.keys())) &
        (profiles_df['mean_churn_prob'] > 0.55) &
        (profiles_df['mean_R'] < 0.3)
    ].sort_values('mean_churn_prob', ascending=False)
    
    if len(ghost_candidates) > 0:
        best = ghost_candidates.iloc[0]
        cid = int(best['cluster_id'])
        archetype_names[cid] = "Ghost Members"
        used_names.add("Ghost Members")
        print(f"  Rule 2 matched: Cluster {cid} -> Ghost Members (churn={best['mean_churn_prob']:.4f}, R={best['mean_R']:.4f})")
    
    # Rule 3: Drifting Stars — once frequent, now declining
    # Adapted: second high-churn cluster that didn't match Ghost Members
    drift_candidates = profiles_df[
        (~profiles_df['cluster_id'].isin(archetype_names.keys())) &
        (profiles_df['mean_churn_prob'] > 0.35)
    ].sort_values('mean_churn_prob', ascending=False)
    
    if len(drift_candidates) > 0:
        best = drift_candidates.iloc[0]
        cid = int(best['cluster_id'])
        archetype_names[cid] = "Drifting Stars"
        used_names.add("Drifting Stars")
        print(f"  Rule 3 matched: Cluster {cid} -> Drifting Stars (churn={best['mean_churn_prob']:.4f}, F={best['mean_F']:.4f})")
    
    # Rule 4: Silent Investors — fly but never redeem, low engagement rate
    silent_candidates = profiles_df[
        (~profiles_df['cluster_id'].isin(archetype_names.keys())) &
        (profiles_df['mean_E'] < 0.1) &
        (profiles_df['mean_F'] > 0.2) &  # Adapted from 0.3 — they do fly
        (profiles_df['mean_churn_prob'] < 0.35)
    ].sort_values('mean_E', ascending=True)
    
    if len(silent_candidates) > 0:
        # Pick the one with highest F among low-E clusters (they fly but don't redeem)
        best = silent_candidates.sort_values('mean_F', ascending=False).iloc[0]
        cid = int(best['cluster_id'])
        archetype_names[cid] = "Silent Investors"
        used_names.add("Silent Investors")
        print(f"  Rule 4 matched: Cluster {cid} -> Silent Investors (E={best['mean_E']:.4f}, F={best['mean_F']:.4f})")
    
    # Rule 5: Budget Explorers — low frequency, low card tier, stable
    budget_candidates = profiles_df[
        (~profiles_df['cluster_id'].isin(archetype_names.keys())) &
        (profiles_df['mean_churn_prob'] < 0.35) &
        (profiles_df['mean_T'] < 0.4)
    ].sort_values('mean_F', ascending=True)
    
    if len(budget_candidates) > 0:
        best = budget_candidates.iloc[0]
        cid = int(best['cluster_id'])
        archetype_names[cid] = "Budget Explorers"
        used_names.add("Budget Explorers")
        print(f"  Rule 5 matched: Cluster {cid} -> Budget Explorers (F={best['mean_F']:.4f}, T={best['mean_T']:.4f})")
    
    # Remaining clusters -> Seasonal Soakers
    for _, row in profiles_df.iterrows():
        cid = int(row['cluster_id'])
        if cid not in archetype_names:
            archetype_names[cid] = "Seasonal Soakers"
            print(f"  Fallback: Cluster {cid} -> Seasonal Soakers")
                
    print("\nArchetype Mapping:")
    for cid in sorted(archetype_names.keys()):
        print(f"  Cluster {cid} -> {archetype_names[cid]}")
        
    # Add archetype column to scored_df and resave
    scored_df['archetype'] = scored_df['cluster_id'].map(archetype_names)
    scored_df.to_csv(scored_path, index=False)
    print(f"\nUpdated members_scored.csv with cluster_id and archetype columns.")
    print(f"Archetype distribution:")
    print(scored_df['archetype'].value_counts())
    
    print("\nTasks 4.3 & 4.4 completed successfully.")

if __name__ == "__main__":
    main()
