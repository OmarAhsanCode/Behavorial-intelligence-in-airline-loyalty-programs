import os
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_all_data, add_sidebar_info
from utils.chart_helpers import ARCHETYPE_COLORS, apply_chart_style

def main():
    st.set_page_config(
        page_title="Loyalty Intelligence Hub",
        page_icon="plane",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            
    # Load data
    data = load_all_data()
    df_scored = data['members_scored']
    df_umap = data['umap_coordinates']
    df_summary = data['segment_summary']
    threshold = data['optimal_threshold']
    
    # persistent sidebar
    add_sidebar_info()
    
    # Title & Subtitle
    st.title("Loyalty Intelligence Hub")
    st.subheader("Behavioral Analytics for Canadian Airline Loyalty Program · 2012–2018 Cohort")
    st.markdown("---")
    
    # Metric counts
    total_members = len(df_scored)
    high_risk_members = (df_scored['churn_risk_tier'] == 'High').sum()
    clv_at_risk_val = df_scored.loc[df_scored['churn_risk_tier'] == 'High', 'CLV'].sum()
    clv_at_risk_str = f"${clv_at_risk_val/1e6:.2f}M"
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Members", f"{total_members:,}")
    col2.metric("High-Risk Members", f"{high_risk_members:,}")
    col3.metric("CLV at Risk", clv_at_risk_str)
    col4.metric("Optimal Threshold", f"{threshold:.2f}")
    
    st.info(
        "This tool identifies members likely to disengage, segments them into five behavioral archetypes, "
        "and recommends specific retention actions. A first-time user should be able to identify who needs "
        "attention and what to do about it within 60 seconds."
    )
    
    # Charts layout
    chart_col1, chart_col2 = st.columns([1.1, 0.9])
    
    with chart_col1:
        st.subheader("Member Archetype Map")
        if not df_umap.empty:
            fig_umap = px.scatter(
                df_umap,
                x='umap_x',
                y='umap_y',
                color='archetype',
                color_discrete_map=ARCHETYPE_COLORS,
                opacity=0.6,
                hover_data={'Loyalty Number': True, 'archetype': True, 'umap_x': False, 'umap_y': False}
            )
            fig_umap.update_traces(marker=dict(size=4))
            fig_umap.update_layout(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                legend=dict(
                    title=dict(text="Archetype"),
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=500
            )
            apply_chart_style(fig_umap)
            st.plotly_chart(fig_umap, use_container_width=True)
        else:
            st.warning("UMAP coordinates not available.")
            
    with chart_col2:
        st.subheader("Member Count per Archetype")
        if not df_summary.empty:
            summary_sorted = df_summary.sort_values(by='member_count', ascending=False)
            fig_bar = px.bar(
                summary_sorted,
                x='member_count',
                y='archetype',
                color='archetype',
                color_discrete_map=ARCHETYPE_COLORS,
                orientation='h',
                category_orders={'archetype': list(summary_sorted['archetype'])},
                labels={'member_count': 'Member Count', 'archetype': 'Archetype'}
            )
            fig_bar.update_layout(
                showlegend=False,
                height=500,
                xaxis_title="Member Count",
                yaxis_title=""
            )
            apply_chart_style(fig_bar)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Segment summary not available.")
            
if __name__ == "__main__":
    main()
