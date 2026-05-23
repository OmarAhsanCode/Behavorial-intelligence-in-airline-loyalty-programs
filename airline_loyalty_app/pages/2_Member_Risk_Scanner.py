import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from utils.data_loader import load_all_data, add_sidebar_info
from utils.chart_helpers import translate_feature, apply_chart_style

def main():
    st.set_page_config(
        page_title="Member Risk Scanner",
        page_icon="search",
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
    df_shap = data['shap_values_per_member']
    playbook = data['playbook']
    
    # Sidebar
    add_sidebar_info()
    
    # Initialize flagged members list in session state
    if "flagged_members" not in st.session_state:
        st.session_state["flagged_members"] = []
        
    st.title("Member Risk Scanner")
    st.subheader("Individual Member Churn Risk Diagnostic & SHAP Explainability")
    st.markdown("---")
    
    # Search input
    loyalty_num_input = st.text_input("Enter Loyalty Number (e.g. try searching for a member, or search 100018, etc.)")
    
    if loyalty_num_input:
        try:
            loyalty_num = int(loyalty_num_input.strip())
        except ValueError:
            st.error("Please enter a valid numeric Loyalty Number.")
            return
            
        # Retrieve member
        member_row = df_scored[df_scored['Loyalty Number'] == loyalty_num]
        
        if not member_row.empty:
            member = member_row.iloc[0]
            
            # Layout cols
            left_col, right_col = st.columns([0.4, 0.6])
            
            with left_col:
                st.subheader("Member Profile")
                
                # Risk tier determination for CSS class
                risk_tier = member['churn_risk_tier']
                if risk_tier == 'High':
                    badge_class = "risk-high"
                elif risk_tier == 'Medium':
                    badge_class = "risk-medium"
                else:
                    badge_class = "risk-low"
                    
                # Archetype Badge
                archetype = member['archetype']
                st.markdown(
                    f"<h3>{archetype} <span class='archetype-badge {badge_class}'>{risk_tier} Risk</span></h3>",
                    unsafe_allow_html=True
                )
                
                # Demographic information
                st.markdown(f"**Province:** {member['Province']}")
                st.markdown(f"**Card Tier:** {member['Loyalty Card']}")
                st.markdown(f"**Enrollment Year:** {member['Enrollment Year']}")
                st.markdown(f"**Tenure:** {member['tenure_months']} months")
                
                # BDS Score Progress Bar
                bds_val = float(member['BDS'])
                # clamp progress to [0.0, 1.0] just in case
                bds_val_clamped = max(0.0, min(1.0, bds_val))
                st.write(f"**Disengagement Score (BDS):** {bds_val:.2f}")
                st.progress(bds_val_clamped)
                
                # Churn probability indicator
                churn_prob = float(member['churn_prob'])
                if churn_prob > 0.6:
                    prob_color = "#ef4444"
                elif churn_prob >= 0.3:
                    prob_color = "#f59e0b"
                else:
                    prob_color = "#1d9e75"
                    
                st.markdown(
                    f"**Churn Probability:** <span style='font-size: 24px; font-weight: bold; color: {prob_color};'>{churn_prob:.1%}</span>",
                    unsafe_allow_html=True
                )
                
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=churn_prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    number={'valueformat': '.1%'},
                    title={'text': "Churn Probability", 'font': {'size': 14}},
                    gauge={
                        'axis': {'range': [0, 1], 'tickformat': '.0%'},
                        'bar': {'color': prob_color},
                        'steps': [
                            {'range': [0, 0.3], 'color': "rgba(29, 158, 117, 0.15)"},
                            {'range': [0.3, 0.6], 'color': "rgba(245, 158, 11, 0.15)"},
                            {'range': [0.6, 1.0], 'color': "rgba(239, 68, 68, 0.15)"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10))
                apply_chart_style(fig_gauge)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with right_col:
                # Section 2: SHAP Explainability
                st.subheader("Why is this member at risk?")
                
                # Find SHAP values for this member
                member_shap = df_shap[df_shap['Loyalty Number'] == loyalty_num]
                
                if not member_shap.empty:
                    # Drop Loyalty Number
                    shap_vals = member_shap.iloc[0].drop('Loyalty Number')
                    # Get top 3 features by absolute SHAP value
                    abs_shap = shap_vals.abs().sort_values(ascending=False)
                    top_3_features = abs_shap.index[:3]
                    
                    top_3_vals = shap_vals[top_3_features]
                    
                    # Create plot dataframe
                    plot_data = pd.DataFrame({
                        'feature': [translate_feature(f) for f in top_3_features],
                        'SHAP Value': top_3_vals.values,
                        'Impact': ['Increases Risk' if v >= 0 else 'Reduces Risk' for v in top_3_vals.values]
                    })
                    
                    # Horizontal bar plot of SHAP impact
                    fig_shap = px.bar(
                        plot_data,
                        x='SHAP Value',
                        y='feature',
                        color='Impact',
                        color_discrete_map={
                            'Increases Risk': '#EF4444',
                            'Reduces Risk': '#1D9E75'
                        },
                        orientation='h',
                        labels={'feature': 'Feature Diagnostic', 'SHAP Value': 'SHAP Contribution to Risk'}
                    )
                    fig_shap.update_layout(
                        height=250,
                        yaxis=dict(autorange="reversed"),
                        showlegend=True,
                        legend=dict(
                            title=dict(text="Direction"),
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    apply_chart_style(fig_shap)
                    st.plotly_chart(fig_shap, use_container_width=True)
                else:
                    st.info("SHAP values not available for this member.")
                    
                # Section 3: Recommended Action
                st.subheader("Recommended Action")
                
                # Fetch details from playbook
                if playbook and archetype in playbook:
                    pb = playbook[archetype]
                    
                    # Card UI with dark text color to be visible on #f8f9fa
                    st.markdown(
                        f"""
                        <div style="background-color: #f8f9fa; color: #0f172a; padding: 20px; border-radius: 8px; border-left: 5px solid {prob_color};">
                            <h4 style="color: #0f172a; margin-top: 0; margin-bottom: 10px;">Playbook Action: {pb['offer_type']}</h4>
                            <p style="color: #0f172a; margin-bottom: 8px;"><b>Campaign Trigger:</b> {pb['trigger_condition']}</p>
                            <p style="color: #0f172a; margin-bottom: 8px;"><b>Channel:</b> {pb['channel']}</p>
                            <p style="color: #0f172a; margin-bottom: 8px;"><b>Outreach Timing:</b> {pb['timing']}</p>
                            <p style="color: #0f172a; margin-bottom: 0;"><b>Expected Re-engagement Rate:</b> {pb['expected_reengagement_rate']:.0%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("Intervention playbook details not found for this segment.")
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Flag for Retention Campaign Button
                if st.button("Flag for Retention Campaign", key="flag_btn"):
                    # Avoid duplicates
                    existing_loyalty_nums = [m['Loyalty Number'] for m in st.session_state["flagged_members"]]
                    if loyalty_num not in existing_loyalty_nums:
                        st.session_state["flagged_members"].append({
                            'Loyalty Number': loyalty_num,
                            'Archetype': archetype,
                            'Churn Probability': f"{churn_prob:.1%}",
                            'Risk Tier': risk_tier
                        })
                        st.success(f"Added member {loyalty_num} to campaign list.")
                    else:
                        st.warning("This member is already flagged in the campaign list.")
                        
            st.markdown("---")
            
        else:
            st.warning("Member not found. Check the loyalty number and try again.")
            
    # Display Flagged Campaign List at bottom
    if st.session_state["flagged_members"]:
        st.subheader("Flagged Retention Campaign List")
        flagged_df = pd.DataFrame(st.session_state["flagged_members"])
        st.dataframe(flagged_df, use_container_width=True)
        
        csv = flagged_df.to_csv(index=False)
        st.download_button(
            label="Download Campaign List",
            data=csv,
            file_name="flagged_retention_campaign.csv",
            mime="text/csv"
        )
        
        if st.button("Clear Flagged List"):
            st.session_state["flagged_members"] = []
            st.rerun()

if __name__ == "__main__":
    main()
