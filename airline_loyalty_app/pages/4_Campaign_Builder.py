import os
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from utils.data_loader import load_all_data, add_sidebar_info
from utils.chart_helpers import apply_chart_style

def main():
    st.set_page_config(
        page_title="Campaign Builder",
        page_icon="calculator",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    css_path = os.path.join(base_dir, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            
    # Load data
    data = load_all_data()
    df_scored = data['members_scored']
    df_summary = data['segment_summary']
    playbook = data['playbook']
    
    # persistent sidebar
    add_sidebar_info()
    
    st.title("Retention Campaign Builder")
    st.subheader("Build and price a targeted retention campaign in real time")
    st.markdown("---")
    
    # Archetype list + All High-Risk
    archetypes = list(df_summary['archetype'].unique())
    target_options = archetypes + ["All High-Risk Members"]
    
    # Input panel (left column) & Output panel (right column)
    in_col, out_col = st.columns([0.35, 0.65])
    
    with in_col:
        st.subheader("Campaign Configurations")
        
        target_seg = st.selectbox("Target segment", options=target_options, index=0)
        offer_type = st.radio("Offer type", options=["Bonus Miles", "Upgrade Voucher", "Partner Reward", "Personalized Points Nudge"])
        budget = st.slider("Campaign budget ($)", min_value=500, max_value=50000, value=5000, step=500)
        
        # Pull default re-engagement rate from playbook if available
        default_rate = 20
        if target_seg in playbook:
            default_rate = int(playbook[target_seg]['expected_reengagement_rate'] * 100)
            
        reengage_rate = st.slider("Assumed re-engagement rate (%)", min_value=5, max_value=50, value=default_rate, step=1)
        
        calculate_btn = st.button("Calculate ROI", use_container_width=True)
        
    with out_col:
        st.subheader("Campaign Forecast")
        
        # Calculate statistics
        if target_seg == "All High-Risk Members":
            members_targeted = int((df_scored['churn_risk_tier'] == 'High').sum())
            mean_clv = df_scored.loc[df_scored['churn_risk_tier'] == 'High', 'CLV'].mean()
        else:
            seg_members = df_scored[df_scored['archetype'] == target_seg]
            members_targeted = int((seg_members['churn_risk_tier'] == 'High').sum())
            mean_clv = seg_members['CLV'].mean() if len(seg_members) > 0 else 0.0
            
        cost_per_member = budget / members_targeted if members_targeted > 0 else 0.0
        expected_reengaged = int(round(members_targeted * reengage_rate / 100.0))
        clv_recovered = expected_reengaged * mean_clv
        roi_multiple = clv_recovered / budget if budget > 0 else 0.0
        
        # Metrics row
        col_m1, col_m2 = st.columns(2)
        col_m3, col_m4 = st.columns(2)
        
        col_m1.metric("Members Targeted", f"{members_targeted:,}")
        col_m2.metric("Expected Re-engaged", f"{expected_reengaged:,}")
        col_m3.metric("CLV Recovered", f"${clv_recovered:,.2f}")
        col_m4.metric("ROI Multiple", f"{roi_multiple:.2f}x")
        
        # Boardroom ROI disclaimer note
        st.markdown(
            "*ROI assumes conservative re-engagement rates of 15–35% and direct campaign costs only. "
            "Excludes staff time and offer redemption costs.*"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Waterfall Chart
        fig_waterfall = go.Figure(go.Waterfall(
            name="Campaign Value Flow",
            orientation="v",
            measure=["relative", "relative", "relative", "relative"],
            x=["Campaign Budget", "Members Reached", "Re-engaged Members", "CLV Recovered"],
            textposition="outside",
            text=[f"-${budget:,}", "Connector", f"+{expected_reengaged}", f"+${clv_recovered:,.0f}"],
            y=[-budget, 0, expected_reengaged * cost_per_member, clv_recovered],
            connector={"line": {"color": "rgb(63, 63, 63)", "width": 1.5}},
            increasing={"marker": {"color": "#1D9E75"}},
            decreasing={"marker": {"color": "#D85A30"}},
            totals={"marker": {"color": "#3b82f6"}}
        ))
        
        fig_waterfall.update_layout(
            title="<b>Campaign Value Flow</b>",
            showlegend=False,
            height=350
        )
        apply_chart_style(fig_waterfall)
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # Disclaimer box
        st.info("ROI assumes direct campaign costs only. Excludes staff time and offer redemption costs. Re-engagement rates are model estimates based on industry benchmarks.")
        
        # Export Brief
        brief_data = {
            'segment': [target_seg],
            'offer_type': [offer_type],
            'budget': [budget],
            'members_targeted': [members_targeted],
            'cost_per_member': [round(cost_per_member, 2)],
            'reengagement_rate_assumed': [reengage_rate],
            'expected_reengaged': [expected_reengaged],
            'clv_recovered': [round(clv_recovered, 2)],
            'roi_multiple': [round(roi_multiple, 2)],
            'generated_at': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        brief_df = pd.DataFrame(brief_data)
        brief_csv = brief_df.to_csv(index=False)
        
        st.download_button(
            label="Download Campaign Brief",
            data=brief_csv,
            file_name="campaign_brief.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
