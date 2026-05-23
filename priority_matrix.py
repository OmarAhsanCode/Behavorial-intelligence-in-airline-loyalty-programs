"""
priority_matrix.py

Task 5.3 — Priority Matrix Scatter Plot
Loads segment summary data and constructs a 4-quadrant priority matrix.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def main():
    print("Starting Task 5.3: Priority Matrix...")
    
    outputs_dir = "outputs"
    summary_path = os.path.join(outputs_dir, "segment_summary.csv")
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Segment summary file not found: {summary_path}")
        
    df = pd.read_csv(summary_path)
    
    # Calculate quadrant splits
    x_split = df['mean_churn_prob'].mean()
    y_split = df['mean_clv'].mean()
    
    print(f"Quadrant boundaries: x_split (mean_churn_prob) = {x_split:.4f}, y_split (mean_clv) = {y_split:.2f}")
    
    # Color map per archetype
    color_map = {
        "Crown Holders": "#1D9E75",
        "Ghost Members": "#D85A30",
        "Drifting Stars": "#BA7517",
        "Silent Investors": "#534AB7",
        "Budget Explorers": "#888780"
    }
    
    # Normalize bubble sizes to 20-80px range
    # Formula: size = min_size + (max_size - min_size) * (val - val_min) / (val_max - val_min)
    min_size = 20
    max_size = 80
    c_min = df['member_count'].min()
    c_max = df['member_count'].max()
    
    sizes = []
    for count in df['member_count']:
        if c_max > c_min:
            sizes.append(min_size + (max_size - min_size) * (count - c_min) / (c_max - c_min))
        else:
            sizes.append((min_size + max_size) / 2)
            
    fig = go.Figure()
    
    # Add quadrant background rectangles
    # We want them to span from slightly below/above the limits
    x_min, x_max = -0.05, 1.05
    y_min, y_max = 0.0, df['mean_clv'].max() * 1.2
    
    # Quadrant 1: Top-Right (Act Now) - Red
    fig.add_shape(
        type="rect",
        x0=x_split, y0=y_split, x1=x_max, y1=y_max,
        fillcolor="rgba(239, 68, 68, 0.07)",
        line_width=0,
        layer="below"
    )
    
    # Quadrant 2: Top-Left (Nurture) - Green
    fig.add_shape(
        type="rect",
        x0=x_min, y0=y_split, x1=x_split, y1=y_max,
        fillcolor="rgba(34, 197, 94, 0.07)",
        line_width=0,
        layer="below"
    )
    
    # Quadrant 3: Bottom-Right (Triage) - Orange
    fig.add_shape(
        type="rect",
        x0=x_split, y0=y_min, x1=x_max, y1=y_split,
        fillcolor="rgba(249, 115, 22, 0.07)",
        line_width=0,
        layer="below"
    )
    
    # Quadrant 4: Bottom-Left (Monitor) - Gray
    fig.add_shape(
        type="rect",
        x0=x_min, y0=y_min, x1=x_split, y1=y_split,
        fillcolor="rgba(148, 163, 184, 0.07)",
        line_width=0,
        layer="below"
    )
    
    # Add text labels for the quadrants
    fig.add_annotation(
        x=(x_split + x_max)/2, y=y_max - (y_max-y_split)*0.15,
        text="<b>🔴 Act Now</b>", showarrow=False,
        font=dict(size=14, color="#b91c1c", family="Outfit, Inter, sans-serif")
    )
    fig.add_annotation(
        x=(x_min + x_split)/2, y=y_max - (y_max-y_split)*0.15,
        text="<b>🟢 Nurture</b>", showarrow=False,
        font=dict(size=14, color="#15803d", family="Outfit, Inter, sans-serif")
    )
    fig.add_annotation(
        x=(x_split + x_max)/2, y=y_min + (y_split-y_min)*0.15,
        text="<b>🟡 Triage</b>", showarrow=False,
        font=dict(size=14, color="#c2410c", family="Outfit, Inter, sans-serif")
    )
    fig.add_annotation(
        x=(x_min + x_split)/2, y=y_min + (y_split-y_min)*0.15,
        text="<b>⬜ Monitor</b>", showarrow=False,
        font=dict(size=14, color="#475569", family="Outfit, Inter, sans-serif")
    )
    
    # Add bubble markers
    for idx, row in df.iterrows():
        name = row['archetype']
        color = color_map.get(name, "#3b82f6")
        size = sizes[idx]
        
        # Add bubble trace
        fig.add_trace(
            go.Scatter(
                x=[row['mean_churn_prob']],
                y=[row['mean_clv']],
                mode='markers+text',
                marker=dict(
                    size=[size],
                    color=color,
                    line=dict(width=2, color='white')
                ),
                text=[f"<b>{name}</b><br>({row['member_count']:,} members)"],
                textposition="top center",
                textfont=dict(family="Outfit, Inter, sans-serif", size=11, color="#0f172a"),
                name=name,
                hoverinfo='text',
                hovertext=(
                    f"<b>{name}</b><br>"
                    f"Members: {row['member_count']:,} ({row['pct_of_total']:.1f}%)<br>"
                    f"Mean Churn Prob: {row['mean_churn_prob']*100:.2f}%<br>"
                    f"Mean CLV: ${row['mean_clv']:,.2f}<br>"
                    f"Total CLV: ${row['total_clv']:,.0f}"
                )
            )
        )
        
    # Layout configuration
    fig.update_layout(
        title=dict(
            text='<b>Retention Priority Matrix — Where to Act First</b>',
            font=dict(family="Outfit, Inter, sans-serif", size=20, color="#0f172a"),
            x=0.5
        ),
        xaxis=dict(
            title="Churn Probability →",
            range=[x_min, x_max],
            gridcolor='#f1f5f9',
            zeroline=False,
            tickformat='.0%'
        ),
        yaxis=dict(
            title="Customer Lifetime Value →",
            range=[y_min, y_max],
            gridcolor='#f1f5f9',
            zeroline=False,
            tickprefix='$'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        width=900,
        height=700,
        margin=dict(t=80, b=50, l=80, r=50)
    )
    
    # Save chart
    chart_path = os.path.join(outputs_dir, "priority_matrix.html")
    fig.write_html(chart_path)
    print(f"Priority matrix chart saved to {chart_path}")

if __name__ == "__main__":
    main()
