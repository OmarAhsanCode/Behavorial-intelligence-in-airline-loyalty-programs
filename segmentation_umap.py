"""
segmentation_umap.py

Task 4.5 — UMAP Visualization of Loyalty Archetypes
"""

import os
import pandas as pd
import numpy as np
import umap
import plotly.express as px

def main():
    print("Starting Task 4.5: UMAP Visualization...")
    
    outputs_dir = "outputs"
    
    # Load RFMET features
    rfmet_df = pd.read_csv(os.path.join(outputs_dir, "rfmet_features.csv"))
    X = rfmet_df[['R', 'F', 'M', 'E', 'T']].values
    
    # Load members_scored.csv (has archetype, CLV, churn_prob, etc.)
    scored_df = pd.read_csv(os.path.join(outputs_dir, "members_scored.csv"))
    
    # Fit UMAP
    print("Fitting UMAP (n_components=2, n_neighbors=30, min_dist=0.1)...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        random_state=42
    )
    embedding = reducer.fit_transform(X)
    
    scored_df['umap_x'] = embedding[:, 0]
    scored_df['umap_y'] = embedding[:, 1]
    
    # Save updated members_scored.csv with UMAP coordinates
    scored_path = os.path.join(outputs_dir, "members_scored.csv")
    scored_df.to_csv(scored_path, index=False)
    print(f"Updated members_scored.csv with umap_x, umap_y columns.")
    
    # Save coordinates separately
    umap_coords = scored_df[['Loyalty Number', 'umap_x', 'umap_y', 'archetype']].copy()
    umap_coords_path = os.path.join(outputs_dir, "umap_coordinates.csv")
    umap_coords.to_csv(umap_coords_path, index=False)
    print(f"UMAP coordinates saved to {umap_coords_path}")
    
    # --- Create Plotly scatter ---
    print("Generating UMAP scatter plot...")
    
    # Color map
    color_map = {
        "Crown Holders": "#1D9E75",
        "Ghost Members": "#D85A30",
        "Drifting Stars": "#BA7517",
        "Silent Investors": "#534AB7",
        "Budget Explorers": "#888780",
        "Seasonal Soakers": "#185FA5"
    }
    
    # Normalize CLV to marker size range 4-14
    clv_min = scored_df['CLV'].min()
    clv_max = scored_df['CLV'].max()
    if clv_max > clv_min:
        scored_df['marker_size'] = 4 + 10 * (scored_df['CLV'] - clv_min) / (clv_max - clv_min)
    else:
        scored_df['marker_size'] = 9  # midpoint
    
    # Opacity = churn_prob (but floor at 0.15 so low-risk are still visible)
    scored_df['opacity'] = scored_df['churn_prob'].clip(lower=0.15)
    
    fig = px.scatter(
        scored_df,
        x='umap_x',
        y='umap_y',
        color='archetype',
        color_discrete_map=color_map,
        size='marker_size',
        opacity=0.7,
        hover_data={
            'Loyalty Number': True,
            'archetype': True,
            'CLV': ':.2f',
            'churn_prob': ':.4f',
            'churn_risk_tier': True,
            'Province': True,
            'umap_x': False,
            'umap_y': False,
            'marker_size': False
        },
        title='<b>Loyalty Archetype Map - 11,076 Members</b>'
    )
    
    # Apply per-point opacity based on churn_prob
    for trace in fig.data:
        archetype_name = trace.name
        mask = scored_df['archetype'] == archetype_name
        if mask.any():
            trace.marker.opacity = scored_df.loc[mask, 'opacity'].values
    
    fig.update_layout(
        plot_bgcolor='#0f172a',
        paper_bgcolor='#0f172a',
        font=dict(
            family="Outfit, Inter, sans-serif",
            color='#e2e8f0'
        ),
        title=dict(
            font=dict(size=22, color='#f8fafc'),
            x=0.5
        ),
        legend=dict(
            title=dict(text='Archetype', font=dict(size=14, color='#94a3b8')),
            bgcolor='rgba(15, 23, 42, 0.8)',
            bordercolor='#334155',
            borderwidth=1,
            font=dict(size=12, color='#cbd5e1')
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            title='UMAP 1',
            color='#64748b'
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            title='UMAP 2',
            color='#64748b'
        ),
        width=1000,
        height=700,
        margin=dict(t=80, b=50, l=50, r=50)
    )
    
    fig.update_traces(
        marker=dict(line=dict(width=0.3, color='rgba(255,255,255,0.2)'))
    )
    
    umap_plot_path = os.path.join(outputs_dir, "segment_umap.html")
    fig.write_html(umap_plot_path)
    print(f"UMAP plot saved to {umap_plot_path}")
    
    print("\nTask 4.5 completed successfully.")

if __name__ == "__main__":
    main()
