"""
segmentation_rfmet.py

Task 4.1 — RFMET Feature Construction
Task 4.2 — Optimal Cluster Count (K-Means + GMM)

Phase 3 fix applied first: adds churn_prediction_conservative column to members_scored.csv
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    print("Starting Phase 4: Segmentation — Tasks 4.1 & 4.2")
    
    # Paths
    outputs_dir = "outputs"
    models_dir = os.path.join(outputs_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    scored_path = os.path.join(outputs_dir, "members_scored.csv")
    df = pd.read_csv(scored_path)
    
    # --- Phase 3 fix: add churn_prediction_conservative ---
    print("Applying Phase 3 fix: adding churn_prediction_conservative (threshold=0.65)...")
    df['churn_prediction_conservative'] = (df['churn_prob'] >= 0.65).astype(int)
    df.to_csv(scored_path, index=False)
    print(f"  churn_prediction_conservative distribution:")
    print(f"    0: {(df['churn_prediction_conservative'] == 0).sum()}")
    print(f"    1: {(df['churn_prediction_conservative'] == 1).sum()}")
    
    # Load model_ready_data.csv to get the engineered features not present in members_scored
    model_ready_path = os.path.join(outputs_dir, "model_ready_data.csv")
    mr_df = pd.read_csv(model_ready_path)
    
    # Merge needed engineered columns into df
    needed_cols = ['Loyalty Number', 'months_since_last_flight', 'flights_last_12m',
                   'redemption_rate', 'loyalty_card_ordinal', 'tenure_months']
    merge_cols = [c for c in needed_cols if c not in df.columns]
    if merge_cols:
        df = pd.merge(df, mr_df[['Loyalty Number'] + merge_cols], on='Loyalty Number', how='left')
    
    # --- Task 4.1: RFMET Feature Construction ---
    print("\n--- Task 4.1: RFMET Feature Construction ---")
    
    # R_raw: months_since_last_flight (lower is better → invert after scaling)
    R_raw = df['months_since_last_flight'].values.reshape(-1, 1)
    
    # F_raw: flights_last_12m (higher is better)
    F_raw = df['flights_last_12m'].values.reshape(-1, 1)
    
    # M_raw: CLV — RobustScaler first (due to 660 outliers), then MinMaxScaler on top
    clv_robust_scaler = RobustScaler()
    M_robust = clv_robust_scaler.fit_transform(df['CLV'].values.reshape(-1, 1))
    
    # Save RobustScaler
    clv_robust_path = os.path.join(models_dir, "clv_robust_scaler.pkl")
    with open(clv_robust_path, 'wb') as f:
        pickle.dump(clv_robust_scaler, f)
    print(f"Saved CLV RobustScaler to {clv_robust_path}")
    
    # E_raw: redemption_rate — cap at 1.0
    E_raw = df['redemption_rate'].clip(upper=1.0).values.reshape(-1, 1)
    
    # T_raw: loyalty_card_ordinal — already 1–3
    T_raw = df['loyalty_card_ordinal'].values.reshape(-1, 1)
    
    # Stack R, F, E, T for joint MinMaxScaler
    raw_rfet = np.hstack([R_raw, F_raw, E_raw, T_raw])
    
    # Also stack M_robust as its own column for separate MinMax
    rfmet_scaler = MinMaxScaler()
    scaled_rfet = rfmet_scaler.fit_transform(raw_rfet)
    
    # MinMax on M_robust separately
    m_minmax = MinMaxScaler()
    M_scaled = m_minmax.fit_transform(M_robust)
    
    # Save combined scaler (for R, F, E, T) to outputs/models/rfmet_scaler.pkl
    rfmet_scaler_path = os.path.join(models_dir, "rfmet_scaler.pkl")
    with open(rfmet_scaler_path, 'wb') as f:
        pickle.dump({'rfet_scaler': rfmet_scaler, 'm_minmax_scaler': m_minmax}, f)
    print(f"Saved RFMET scaler to {rfmet_scaler_path}")
    
    # Build final RFMET columns
    R = 1.0 - scaled_rfet[:, 0]  # Invert: high R = recently active
    F = scaled_rfet[:, 1]
    M = M_scaled.flatten()
    E = scaled_rfet[:, 2]
    T = scaled_rfet[:, 3]
    
    rfmet_df = pd.DataFrame({
        'Loyalty Number': df['Loyalty Number'],
        'R': R,
        'F': F,
        'M': M,
        'E': E,
        'T': T
    })
    
    rfmet_path = os.path.join(outputs_dir, "rfmet_features.csv")
    rfmet_df.to_csv(rfmet_path, index=False)
    print(f"Saved RFMET features to {rfmet_path}")
    
    print("\nRFMET Descriptive Statistics:")
    print(rfmet_df[['R', 'F', 'M', 'E', 'T']].describe())
    
    # --- Task 4.2: Optimal Cluster Count ---
    print("\n--- Task 4.2: Optimal Cluster Count ---")
    
    X = rfmet_df[['R', 'F', 'M', 'E', 'T']].values
    
    k_range = range(2, 11)
    results = []
    
    for k in k_range:
        print(f"  Fitting k={k}...")
        
        # K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_km = kmeans.fit_predict(X)
        inertia = kmeans.inertia_
        sil = silhouette_score(X, labels_km)
        
        # GMM
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
        gmm.fit(X)
        bic = gmm.bic(X)
        
        results.append({
            'k': k,
            'kmeans_inertia': inertia,
            'kmeans_silhouette': sil,
            'gmm_bic': bic
        })
        
    results_df = pd.DataFrame(results)
    results_path = os.path.join(outputs_dir, "cluster_selection_metrics.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nCluster selection metrics saved to {results_path}")
    print(results_df)
    
    # Best k by silhouette and BIC
    best_sil_k = results_df.loc[results_df['kmeans_silhouette'].idxmax(), 'k']
    best_bic_k = results_df.loc[results_df['gmm_bic'].idxmin(), 'k']
    
    print(f"\nK with highest silhouette score: k={int(best_sil_k)} (silhouette={results_df.loc[results_df['kmeans_silhouette'].idxmax(), 'kmeans_silhouette']:.4f})")
    print(f"K with lowest GMM BIC: k={int(best_bic_k)} (BIC={results_df.loc[results_df['gmm_bic'].idxmin(), 'gmm_bic']:.2f})")
    
    # Recommendation
    if best_sil_k == best_bic_k:
        rec_k = int(best_sil_k)
        print(f"Recommended k: {rec_k} — both silhouette and BIC agree on this value.")
    else:
        # Default to silhouette unless BIC strongly disagrees; prefer 5 or 6
        rec_k = int(best_sil_k)
        print(f"Recommended k: {rec_k} — using silhouette-optimal value; BIC preferred k={int(best_bic_k)}.")
    
    # Plot
    print("\nGenerating cluster selection plot...")
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Elbow Curve (Inertia)", "Silhouette Score", "GMM BIC"),
        horizontal_spacing=0.08
    )
    
    ks = results_df['k'].tolist()
    
    fig.add_trace(go.Scatter(
        x=ks, y=results_df['kmeans_inertia'].tolist(),
        mode='lines+markers',
        line=dict(color='#1e3a8a', width=2.5),
        marker=dict(size=8),
        name='Inertia'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=ks, y=results_df['kmeans_silhouette'].tolist(),
        mode='lines+markers',
        line=dict(color='#047857', width=2.5),
        marker=dict(size=8),
        name='Silhouette'
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=ks, y=results_df['gmm_bic'].tolist(),
        mode='lines+markers',
        line=dict(color='#b91c1c', width=2.5),
        marker=dict(size=8),
        name='BIC'
    ), row=1, col=3)
    
    fig.update_layout(
        title=dict(
            text='<b>Cluster Count Selection</b>',
            font=dict(family="Outfit, Inter, sans-serif", size=20, color="#1e293b")
        ),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1100,
        height=400,
        margin=dict(t=80, b=50, l=50, r=30)
    )
    for i in range(1, 4):
        fig.update_xaxes(title_text="k (Number of Clusters)", row=1, col=i, gridcolor='#f1f5f9')
        fig.update_yaxes(gridcolor='#f1f5f9', row=1, col=i)
        
    plot_path = os.path.join(outputs_dir, "cluster_selection_plot.html")
    fig.write_html(plot_path)
    print(f"Cluster selection plot saved to {plot_path}")
    
    print(f"\nTasks 4.1 & 4.2 completed. Recommended k = {rec_k}")

if __name__ == "__main__":
    main()
