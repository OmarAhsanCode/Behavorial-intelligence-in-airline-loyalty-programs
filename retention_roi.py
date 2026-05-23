"""
retention_roi.py

Task 5.2 — CLV at Risk and Recovery Calculation
Loads data from outputs/segment_summary.csv, outputs/intervention_playbook.json, and outputs/members_scored.csv.
Computes costs, recovery, ROI, and plots the grouped bar chart with secondary y-axis.
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    print("Starting Task 5.2: CLV at Risk and Recovery Calculation...")
    
    outputs_dir = "outputs"
    scored_path = os.path.join(outputs_dir, "members_scored.csv")
    playbook_path = os.path.join(outputs_dir, "intervention_playbook.json")
    summary_path = os.path.join(outputs_dir, "segment_summary.csv")
    
    # Load files
    df = pd.read_csv(scored_path)
    summary_df = pd.read_csv(summary_path)
    with open(playbook_path, 'r') as f:
        playbook = json.load(f)
        
    roi_rows = []
    
    for _, row in summary_df.iterrows():
        arch = row['archetype']
        mean_clv = float(row['mean_clv'])
        
        # Load details from playbook
        pb_details = playbook.get(arch, {})
        cost_per_member = pb_details.get('estimated_cost_per_member', 0.0)
        reengage_rate = pb_details.get('expected_reengagement_rate', 0.0)
        
        # Filter high risk members in scored df
        seg_df = df[df['archetype'] == arch]
        members_high_risk = int((seg_df['churn_risk_tier'] == 'High').sum())
        
        # Calculations
        clv_at_risk = round(members_high_risk * mean_clv, 2)
        campaign_cost = round(members_high_risk * cost_per_member, 2)
        expected_reengaged = int(round(members_high_risk * reengage_rate))
        expected_clv_recovered = round(expected_reengaged * mean_clv, 2)
        
        if campaign_cost > 0:
            roi_multiple = round(expected_clv_recovered / campaign_cost, 1)
        else:
            roi_multiple = 0.0
            
        roi_rows.append({
            'archetype': arch,
            'members_high_risk': members_high_risk,
            'clv_at_risk': clv_at_risk,
            'campaign_cost': campaign_cost,
            'expected_reengaged': expected_reengaged,
            'expected_clv_recovered': expected_clv_recovered,
            'roi_multiple': roi_multiple,
            'mean_clv': mean_clv
        })
        
    roi_df = pd.DataFrame(roi_rows)
    roi_path = os.path.join(outputs_dir, "retention_roi_table.csv")
    roi_df.to_csv(roi_path, index=False)
    print(f"Saved ROI table to {roi_path}")
    print(roi_df.to_string(index=False))
    
    # Summary stats calculations
    total_clv_at_risk = roi_df['clv_at_risk'].sum()
    total_campaign_cost = roi_df['campaign_cost'].sum()
    total_clv_recovered = roi_df['expected_clv_recovered'].sum()
    
    if total_campaign_cost > 0:
        overall_roi = round(total_clv_recovered / total_campaign_cost, 2)
    else:
        overall_roi = 0.0
        
    highest_roi_row = roi_df.loc[roi_df['roi_multiple'].idxmax()]
    highest_risk_row = roi_df.loc[roi_df['clv_at_risk'].idxmax()]
    
    # Print exactly as required
    print("\n=== RETENTION ROI SUMMARY ===")
    print(f"Total CLV at risk (all segments):     ${total_clv_at_risk:,.0f}")
    print(f"Total campaign cost (all segments):   ${total_campaign_cost:,.0f}")
    print(f"Total expected CLV recovered:         ${total_clv_recovered:,.0f}")
    print(f"Overall portfolio ROI multiple:        {overall_roi:.2f}x")
    print(f"Highest ROI segment:                  {highest_roi_row['archetype']} ({highest_roi_row['roi_multiple']:.1f}x ROI)")
    print(f"Priority action segment:              {highest_risk_row['archetype']} (${highest_risk_row['clv_at_risk']:,.0f} at risk)")
    
    # Write chart
    print("\nGenerating grouped bar chart with secondary y-axis...")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    archetypes = roi_df['archetype'].tolist()
    
    # CLV at Risk Bar
    fig.add_trace(
        go.Bar(
            x=archetypes,
            y=roi_df['clv_at_risk'].tolist(),
            name='CLV at Risk',
            marker_color='#D85A30',
            offsetgroup=1
        ),
        secondary_y=False
    )
    
    # Expected CLV Recovered Bar
    fig.add_trace(
        go.Bar(
            x=archetypes,
            y=roi_df['expected_clv_recovered'].tolist(),
            name='Expected CLV Recovered',
            marker_color='#1D9E75',
            offsetgroup=2
        ),
        secondary_y=False
    )
    
    # ROI Line
    fig.add_trace(
        go.Scatter(
            x=archetypes,
            y=roi_df['roi_multiple'].tolist(),
            name='ROI Multiple (x)',
            mode='lines+markers',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8, symbol='circle')
        ),
        secondary_y=True
    )
    
    # Layout styling
    fig.update_layout(
        title=dict(
            text='<b>Retention ROI Analysis - Value at Risk vs. Recovered</b>',
            font=dict(family="Outfit, Inter, sans-serif", size=20, color="#1e293b"),
            x=0.5
        ),
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        width=1000,
        height=600,
        margin=dict(t=120, b=50, l=80, r=80)
    )
    
    fig.update_xaxes(title_text="Loyalty Archetype", gridcolor='#f1f5f9')
    fig.update_yaxes(title_text="Financial Value ($)", secondary_y=False, gridcolor='#f1f5f9')
    fig.update_yaxes(title_text="ROI Multiple (x)", secondary_y=True, showgrid=False)
    
    chart_path = os.path.join(outputs_dir, "clv_recovery_chart.html")
    fig.write_html(chart_path)
    print(f"Recovery chart saved to {chart_path}")
    
if __name__ == "__main__":
    main()
