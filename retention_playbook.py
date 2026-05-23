"""
retention_playbook.py

Task 5.1 — Build the Intervention Playbook
Loads data from outputs/segment_summary.csv to dynamically fill segment_size and clv_at_risk_per_member.
"""

import os
import json
import pandas as pd

def main():
    print("Starting Task 5.1: Build the Intervention Playbook...")
    
    outputs_dir = "outputs"
    summary_path = os.path.join(outputs_dir, "segment_summary.csv")
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Segment summary file not found: {summary_path}. Run segmentation_summary.py first.")
        
    summary_df = pd.read_csv(summary_path)
    
    # Create helper dictionary from summary data
    summary_dict = {}
    for _, row in summary_df.iterrows():
        summary_dict[row['archetype']] = {
            'member_count': int(row['member_count']),
            'mean_clv': float(row['mean_clv'])
        }
        
    # Define PLAYBOOK
    PLAYBOOK = {
        "Ghost Members": {
            "risk_level": "Critical",
            "trigger_condition": "No flight, redemption, or point accumulation activity in past 90+ days",
            "channel": "Email + Direct Mail",
            "offer_type": "Reactivation bonus — 2,000 bonus points on next flight booked within 30 days",
            "timing": "Send immediately upon trigger; follow up at day 14 if no booking",
            "success_metric": "Flight booked within 30 days of campaign send",
            "estimated_cost_per_member": 3.50,
            "expected_reengagement_rate": 0.15,
            "segment_size": summary_dict.get("Ghost Members", {}).get('member_count', 0),
            "clv_at_risk_per_member": summary_dict.get("Ghost Members", {}).get('mean_clv', 0.0)
        },
        "Crown Holders": {
            "risk_level": "Low",
            "trigger_condition": "High-value Aurora member with low engagement in last 6 months",
            "channel": "Personal Email",
            "offer_type": "Upgrade experience offer (lounge access, priority boarding)",
            "timing": "Send within 7 days of activity decline; quarterly touchpoint",
            "success_metric": "Positive response or booking within 90 days of outreach",
            "estimated_cost_per_member": 15.00,
            "expected_reengagement_rate": 0.35,
            "segment_size": summary_dict.get("Crown Holders", {}).get('member_count', 0),
            "clv_at_risk_per_member": summary_dict.get("Crown Holders", {}).get('mean_clv', 0.0)
        },
        "Drifting Stars": {
            "risk_level": "High",
            "trigger_condition": "Once frequent flyer with >50% decline in flight frequency in past 6 months",
            "channel": "Email + Push",
            "offer_type": "We miss you campaign with a time-limited double-miles offer on Fall season bookings",
            "timing": "Send at start of historically preferred season (Fall)",
            "success_metric": "Flight booked for Fall season within 45 days of campaign send",
            "estimated_cost_per_member": 5.00,
            "expected_reengagement_rate": 0.22,
            "segment_size": summary_dict.get("Drifting Stars", {}).get('member_count', 0),
            "clv_at_risk_per_member": summary_dict.get("Drifting Stars", {}).get('mean_clv', 0.0)
        },
        "Silent Investors": {
            "risk_level": "Low",
            "trigger_condition": "Accumulated points balance above 25,000 with zero redemptions in past 12 months",
            "channel": "Email",
            "offer_type": "Redemption nudge — personalized point balance email with curated redemption options",
            "timing": "Send monthly statement email with clear call-to-action to redeem points",
            "success_metric": "Points redemption within 60 days of email send",
            "estimated_cost_per_member": 2.50,
            "expected_reengagement_rate": 0.28,
            "segment_size": summary_dict.get("Silent Investors", {}).get('member_count', 0),
            "clv_at_risk_per_member": summary_dict.get("Silent Investors", {}).get('mean_clv', 0.0)
        },
        "Budget Explorers": {
            "risk_level": "Low",
            "trigger_condition": "Nova/Star tier member with stable low-frequency booking pattern",
            "channel": "Email only",
            "offer_type": "Partner reward offer (hotel, car rental discount)",
            "timing": "Send during off-peak periods (January/February) to stimulate partner activity",
            "success_metric": "Partner offer redemption within 60 days of send",
            "estimated_cost_per_member": 1.50,
            "expected_reengagement_rate": 0.18,
            "segment_size": summary_dict.get("Budget Explorers", {}).get('member_count', 0),
            "clv_at_risk_per_member": summary_dict.get("Budget Explorers", {}).get('mean_clv', 0.0)
        }
    }
    
    # Save to JSON
    json_path = os.path.join(outputs_dir, "intervention_playbook.json")
    with open(json_path, 'w') as f:
        json.dump(PLAYBOOK, f, indent=2)
    print(f"Saved playbook to {json_path}")
    
    # Save to CSV
    playbook_rows = []
    for archetype, details in PLAYBOOK.items():
        row = {'archetype': archetype}
        row.update(details)
        playbook_rows.append(row)
        
    playbook_df = pd.DataFrame(playbook_rows)
    csv_path = os.path.join(outputs_dir, "intervention_playbook.csv")
    playbook_df.to_csv(csv_path, index=False)
    print(f"Saved playbook to {csv_path}")
    
    # Print playbook
    print("\n================== INTERVENTION PLAYBOOK ==================")
    for arch, details in PLAYBOOK.items():
        print(f"\nArchetype: {arch}")
        print(f"  Risk Level:        {details['risk_level']}")
        print(f"  Trigger Condition: {details['trigger_condition']}")
        print(f"  Channel:           {details['channel']}")
        print(f"  Offer Type:        {details['offer_type']}")
        print(f"  Timing:            {details['timing']}")
        print(f"  Success Metric:    {details['success_metric']}")
        print(f"  Cost Per Member:   ${details['estimated_cost_per_member']:.2f}")
        print(f"  Reengagement Rate: {details['expected_reengagement_rate']*100:.1f}%")
        print(f"  Segment Size:      {details['segment_size']}")
        print(f"  CLV at Risk/Member:${details['clv_at_risk_per_member']:,.2f}")
    print("===========================================================")

if __name__ == "__main__":
    main()
