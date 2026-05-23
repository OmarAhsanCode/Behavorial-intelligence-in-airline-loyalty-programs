import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from utils.data_loader import load_all_data, add_sidebar_info
from utils.chart_helpers import ARCHETYPE_COLORS, apply_chart_style

def main():
    st.set_page_config(
        page_title="Segment Explorer",
        page_icon="pie-chart",
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
    rfmet_df = data['rfmet_features']
    playbook = data['playbook']
    
    # persistent sidebar
    add_sidebar_info()
    
    st.title("Segment Explorer")
    st.subheader("Deep-Dive Analysis of Behavioral Archetypes")
    st.markdown("---")
    
    # Archetype Options
    archetypes = list(df_summary['archetype'].unique())
    
    # Top Control
    col_sel1, col_toggle = st.columns([0.6, 0.4])
    with col_sel1:
        arch_selected = st.selectbox("Select archetype to explore", options=archetypes, index=0)
    with col_toggle:
        compare_mode = st.toggle("Compare with another segment")
        
    arch_selected_2 = None
    if compare_mode:
        # Choose a different default index if possible
        default_idx = 1 if len(archetypes) > 1 else 0
        arch_selected_2 = st.selectbox("Select second archetype to compare", options=archetypes, index=default_idx)
        
    st.markdown("---")
    
    # Prepare RFMET mapping for Radar Chart
    merged_rfmet = pd.merge(rfmet_df, df_scored[['Loyalty Number', 'archetype']], on='Loyalty Number', how='left')
    
    # Overall averages
    overall_rfmet_means = [
        rfmet_df['R'].mean(),
        rfmet_df['F'].mean(),
        rfmet_df['M'].mean(),
        rfmet_df['E'].mean(),
        rfmet_df['T'].mean()
    ]
    
    categories = ['Recency', 'Frequency', 'Monetary', 'Engagement', 'Tier']
    
    # Helper to calculate segment RFMET
    def get_segment_rfmet(arch_name):
        seg_data = merged_rfmet[merged_rfmet['archetype'] == arch_name]
        if seg_data.empty:
            return [0.0] * 5
        return [
            seg_data['R'].mean(),
            seg_data['F'].mean(),
            seg_data['M'].mean(),
            seg_data['E'].mean(),
            seg_data['T'].mean()
        ]
        
    # Helper to compute Salary Quartiles
    # compute overall boundaries
    df_scored['Salary_Quartile'] = pd.qcut(df_scored['Salary'], 4, labels=['Q1 (Low)', 'Q2 (Mid-Low)', 'Q3 (Mid-High)', 'Q4 (High)'])
    
    if not compare_mode:
        # ================= SINGLE SEGMENT VIEW =================
        # Row 1 — Metric Columns
        sum_row = df_summary[df_summary['archetype'] == arch_selected].iloc[0]
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Member Count", f"{int(sum_row['member_count']):,}")
        m_col2.metric("Mean CLV", f"${float(sum_row['mean_clv']):,.2f}")
        m_col3.metric("Mean Churn Probability", f"{float(sum_row['mean_churn_prob']):.1%}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 2 — Radar & Demographic Charts
        col_radar, col_demo = st.columns([0.45, 0.55])
        
        with col_radar:
            st.subheader("RFMET Behavioral Radar")
            seg_rfmet = get_segment_rfmet(arch_selected)
            
            fig_radar = go.Figure()
            
            # Segment trace
            fig_radar.add_trace(go.Scatterpolar(
                r=seg_rfmet + [seg_rfmet[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=arch_selected,
                line_color=ARCHETYPE_COLORS.get(arch_selected, '#3b82f6')
            ))
            
            # Overall trace
            fig_radar.add_trace(go.Scatterpolar(
                r=overall_rfmet_means + [overall_rfmet_means[0]],
                theta=categories + [categories[0]],
                name='All Members (Average)',
                line=dict(color='gray', dash='dash')
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=True,
                height=450
            )
            apply_chart_style(fig_radar)
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col_demo:
            st.subheader("Demographic Distributions")
            
            # We construct three subplots stacked vertically: Education, Loyalty Card, Salary Quartile
            fig_demo = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Education Distribution", "Loyalty Card Tier", "Salary Quartile Distribution"),
                vertical_spacing=0.1
            )
            
            seg_df = df_scored[df_scored['archetype'] == arch_selected]
            color = ARCHETYPE_COLORS.get(arch_selected, '#3b82f6')
            
            # 1. Education
            edu_counts = seg_df['Education'].value_counts().reset_index()
            fig_demo.add_trace(
                go.Bar(x=edu_counts['Education'], y=edu_counts['count'], marker_color=color, name="Education"),
                row=1, col=1
            )
            
            # 2. Loyalty Card
            card_counts = seg_df['Loyalty Card'].value_counts().reset_index()
            fig_demo.add_trace(
                go.Bar(x=card_counts['Loyalty Card'], y=card_counts['count'], marker_color=color, name="Card Tier"),
                row=2, col=1
            )
            
            # 3. Salary Quartile
            sal_counts = seg_df['Salary_Quartile'].value_counts().reset_index()
            # Order them correctly
            sal_counts['Salary_Quartile'] = pd.Categorical(sal_counts['Salary_Quartile'], categories=['Q1 (Low)', 'Q2 (Mid-Low)', 'Q3 (Mid-High)', 'Q4 (High)'], ordered=True)
            sal_counts = sal_counts.sort_values('Salary_Quartile')
            fig_demo.add_trace(
                go.Bar(x=sal_counts['Salary_Quartile'], y=sal_counts['count'], marker_color=color, name="Salary"),
                row=3, col=1
            )
            
            fig_demo.update_layout(height=500, showlegend=False)
            apply_chart_style(fig_demo)
            st.plotly_chart(fig_demo, use_container_width=True)
            
        st.markdown("---")
        
        # Row 3 — Survival Analysis Curves (Full Width)
        st.subheader("Survival Analysis (Time to Churn)")
        
        km_option = st.selectbox(
            "Select Kaplan-Meier Survival Cut",
            options=["BDS Risk Tier", "Loyalty Card Tier", "Enrollment Type", "Salary Quartile"]
        )
        
        km_mapping = {
            "BDS Risk Tier": "km_by_bds",
            "Loyalty Card Tier": "km_by_card",
            "Enrollment Type": "km_by_enrollment",
            "Salary Quartile": "km_by_salary"
        }
        
        km_key = km_mapping[km_option]
        km_html = data.get(km_key, "")
        
        if km_html:
            st.components.v1.html(km_html, height=450, scrolling=True)
        else:
            st.warning("Survival curve HTML output not found.")
            
        st.markdown("---")
        
        # Row 4 — Playbook card
        st.subheader("Intervention Playbook Campaign Recommendation")
        if playbook and arch_selected in playbook:
            pb = playbook[arch_selected]
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; color: #0f172a; padding: 25px; border-radius: 8px; border-left: 5px solid {color};">
                    <h3 style="color: #0f172a; margin-top: 0; margin-bottom: 15px;">Playbook: {pb['offer_type']}</h3>
                    <p style="color: #0f172a;"><b>Targeting Trigger:</b> {pb['trigger_condition']}</p>
                    <p style="color: #0f172a;"><b>Outreach Channel:</b> {pb['channel']}</p>
                    <p style="color: #0f172a;"><b>Execution Timing:</b> {pb['timing']}</p>
                    <p style="color: #0f172a;"><b>KPI Success Metric:</b> {pb['success_metric']}</p>
                    <div style="margin-top: 15px;">
                        <span style="background-color: #e2e8f0; color: #0f172a; padding: 5px 12px; border-radius: 15px; font-size: 13px; font-weight: bold; margin-right: 10px;">
                            Cost: ${pb['estimated_cost_per_member']:.2f}/member
                        </span>
                        <span style="background-color: #e2e8f0; color: #0f172a; padding: 5px 12px; border-radius: 15px; font-size: 13px; font-weight: bold;">
                            Expected Response: {pb['expected_reengagement_rate']:.0%}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Playbook entry not found for this segment.")
            
    else:
        # ================= COMPARATIVE VIEW =================
        st.subheader(f"Comparing: {arch_selected} vs. {arch_selected_2}")
        
        # Combined Radar Chart at the top
        seg_rfmet1 = get_segment_rfmet(arch_selected)
        seg_rfmet2 = get_segment_rfmet(arch_selected_2)
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=seg_rfmet1 + [seg_rfmet1[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=arch_selected,
            line_color=ARCHETYPE_COLORS.get(arch_selected, '#3b82f6')
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=seg_rfmet2 + [seg_rfmet2[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=arch_selected_2,
            line_color=ARCHETYPE_COLORS.get(arch_selected_2, '#10b981')
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=overall_rfmet_means + [overall_rfmet_means[0]],
            theta=categories + [categories[0]],
            name='All Members (Average)',
            line=dict(color='gray', dash='dash')
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            height=450
        )
        apply_chart_style(fig_radar)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Two columns for side-by-side metrics and details
        col1, col2 = st.columns(2)
        
        # Segment 1 details
        with col1:
            st.markdown(f"### {arch_selected}")
            sum_row1 = df_summary[df_summary['archetype'] == arch_selected].iloc[0]
            st.metric("Member Count", f"{int(sum_row1['member_count']):,}")
            st.metric("Mean CLV", f"${float(sum_row1['mean_clv']):,.2f}")
            st.metric("Mean Churn Probability", f"{float(sum_row1['mean_churn_prob']):.1%}")
            
            st.markdown("#### Demographics")
            seg_df1 = df_scored[df_scored['archetype'] == arch_selected]
            card_counts1 = seg_df1['Loyalty Card'].value_counts()
            st.write("**Card Tiers:**")
            st.dataframe(card_counts1, use_container_width=True)
            
            # Playbook Card 1
            if playbook and arch_selected in playbook:
                pb1 = playbook[arch_selected]
                st.markdown(
                    f"""
                    <div style="background-color: #f8f9fa; color: #0f172a; padding: 15px; border-radius: 8px; border-left: 5px solid {ARCHETYPE_COLORS[arch_selected]};">
                        <h5 style="color: #0f172a; margin-top: 0; margin-bottom: 10px;">Campaign Action: {pb1['offer_type']}</h5>
                        <p style="color: #0f172a; font-size: 13px; margin-bottom: 0;"><b>Channel:</b> {pb1['channel']}<br>
                        <b>Timing:</b> {pb1['timing']}<br>
                        <b>Cost:</b> ${pb1['estimated_cost_per_member']:.2f}/member</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        # Segment 2 details
        with col2:
            st.markdown(f"### {arch_selected_2}")
            sum_row2 = df_summary[df_summary['archetype'] == arch_selected_2].iloc[0]
            st.metric("Member Count", f"{int(sum_row2['member_count']):,}")
            st.metric("Mean CLV", f"${float(sum_row2['mean_clv']):,.2f}")
            st.metric("Mean Churn Probability", f"{float(sum_row2['mean_churn_prob']):.1%}")
            
            st.markdown("#### Demographics")
            seg_df2 = df_scored[df_scored['archetype'] == arch_selected_2]
            card_counts2 = seg_df2['Loyalty Card'].value_counts()
            st.write("**Card Tiers:**")
            st.dataframe(card_counts2, use_container_width=True)
            
            # Playbook Card 2
            if playbook and arch_selected_2 in playbook:
                pb2 = playbook[arch_selected_2]
                st.markdown(
                    f"""
                    <div style="background-color: #f8f9fa; color: #0f172a; padding: 15px; border-radius: 8px; border-left: 5px solid {ARCHETYPE_COLORS[arch_selected_2]};">
                        <h5 style="color: #0f172a; margin-top: 0; margin-bottom: 10px;">Campaign Action: {pb2['offer_type']}</h5>
                        <p style="color: #0f172a; font-size: 13px; margin-bottom: 0;"><b>Channel:</b> {pb2['channel']}<br>
                        <b>Timing:</b> {pb2['timing']}<br>
                        <b>Cost:</b> ${pb2['estimated_cost_per_member']:.2f}/member</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()
