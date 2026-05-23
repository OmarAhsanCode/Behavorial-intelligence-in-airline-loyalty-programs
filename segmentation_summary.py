"""
segmentation_summary.py

Task 4.6 — Segment Summary Table (for Streamlit dashboard & executive report)
"""

import os
import pandas as pd

def main():
    print("Starting Task 4.6: Segment Summary...")
    outputs_dir = "outputs"
    scored_path = os.path.join(outputs_dir, "members_scored.csv")
    df = pd.read_csv(scored_path)
    
    # Ensure required columns exist
    required = [
        'archetype', 'Loyalty Number', 'CLV', 'churn_prob', 'churn_risk_tier',
        'Province', 'loyalty_card_ordinal', 'flights_last_12m', 'redemption_rate'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in members_scored.csv: {missing}")
    
    # Helper to compute dominant value
    def dominant(series):
        return series.mode().iloc[0] if not series.mode().empty else None
    
    summary_rows = []
    total_clv_at_risk = 0.0
    highest_churn = (None, -1.0)
    highest_clv_at_risk = (None, -1.0)
    
    for arch, group in df.groupby('archetype'):
        member_count = len(group)
        pct_of_total = round(100 * member_count / len(df), 2)
        mean_clv = round(group['CLV'].mean(), 2)
        total_clv = round(group['CLV'].sum(), 2)
        mean_churn_prob = round(group['churn_prob'].mean(), 4)
        pct_high_risk = round((group['churn_risk_tier'] == 'High').mean() * 100, 2)
        clv_at_risk = round(mean_clv * (group['churn_risk_tier'] == 'High').sum(), 2)
        mean_tenure = round(group['tenure_months'].mean(), 2)
        mean_flights = round(group['flights_last_12m'].mean(), 2)
        mean_redemption = round(group['redemption_rate'].mean(), 4)
        dominant_card = dominant(group['Loyalty Card'])
        dominant_province = dominant(group['Province'])
        
        summary_rows.append({
            'archetype': arch,
            'member_count': member_count,
            'pct_of_total': pct_of_total,
            'mean_clv': mean_clv,
            'total_clv': total_clv,
            'mean_churn_prob': mean_churn_prob,
            'pct_high_risk': pct_high_risk,
            'clv_at_risk': clv_at_risk,
            'mean_tenure_months': mean_tenure,
            'mean_flights_last_12m': mean_flights,
            'mean_redemption_rate': mean_redemption,
            'dominant_card_tier': dominant_card,
            'dominant_province': dominant_province
        })
        
        total_clv_at_risk += clv_at_risk
        if mean_churn_prob > highest_churn[1]:
            highest_churn = (arch, mean_churn_prob)
        if clv_at_risk > highest_clv_at_risk[1]:
            highest_clv_at_risk = (arch, clv_at_risk)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(outputs_dir, "segment_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("Segment summary table saved to", summary_path)
    print(summary_df.to_string(index=False))
    
    # Headline stats
    print("\nHEADLINE STATS FOR REPORT:")
    print(f"Total CLV at risk across all segments: ${total_clv_at_risk:,.2f}")
    print(f"Segment with highest churn probability: {highest_churn[0]} at {highest_churn[1]*100:.2f}%")
    print(f"Segment with highest CLV at risk: {highest_clv_at_risk[0]} at ${highest_clv_at_risk[1]:,.2f}")

if __name__ == "__main__":
    main()
