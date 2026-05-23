import os
import urllib.request
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_all_data, add_sidebar_info
from utils.chart_helpers import ARCHETYPE_COLORS, apply_chart_style

def get_geojson_provinces():
    """
    Downloads and caches the Canadian provinces GeoJSON.
    """
    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson"
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        st.error(f"Failed to load Canada GeoJSON: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Mission Control",
        page_icon="bar-chart",
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
    
    # Sidebar
    add_sidebar_info()
    
    st.title("Mission Control")
    st.subheader("Loyalty Portfolio Performance & Risk Monitoring")
    st.markdown("---")
    
    # --- Section 1: KPI Row ---
    active_count = (df_scored['churn_risk_tier'] == 'Low').sum()
    high_risk_count = (df_scored['churn_risk_tier'] == 'High').sum()
    medium_risk_count = (df_scored['churn_risk_tier'] == 'Medium').sum()
    avg_clv = df_scored['CLV'].mean()
    avg_tenure_years = df_scored['tenure_months'].mean() / 12.0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Members (Low Risk)", f"{active_count:,}")
    col2.metric("Members Flagged Today (High Risk)", f"{high_risk_count:,}", delta=f"{medium_risk_count} Medium Risk", delta_color="off")
    col3.metric("Average CLV", f"${avg_clv:,.2f}")
    col4.metric("Average Tenure", f"{avg_tenure_years:.1f} years")
    
    st.markdown("---")
    
    # --- Section 2: Risk Distribution Across Archetypes ---
    st.subheader("Risk Distribution Across Loyalty Archetypes")
    
    # Compute stacked bar data
    risk_dist = df_scored.groupby(['archetype', 'churn_risk_tier']).size().reset_index(name='count')
    
    # Pivot to display as a table under the chart
    risk_pivot = risk_dist.pivot(index='archetype', columns='churn_risk_tier', values='count').fillna(0).astype(int)
    # Ensure correct columns order
    for col in ['Low', 'Medium', 'High']:
        if col not in risk_pivot.columns:
            risk_pivot[col] = 0
    risk_pivot = risk_pivot[['Low', 'Medium', 'High']]
    
    fig_risk = px.bar(
        risk_dist,
        x='archetype',
        y='count',
        color='churn_risk_tier',
        color_discrete_map={
            'Low': '#1D9E75',
            'Medium': '#F59E0B',
            'High': '#EF4444'
        },
        category_orders={'churn_risk_tier': ['Low', 'Medium', 'High']},
        labels={'count': 'Members Count', 'archetype': 'Archetype', 'churn_risk_tier': 'Risk Tier'}
    )
    fig_risk.update_layout(height=400, barmode='stack')
    apply_chart_style(fig_risk)
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Download underlying data
    risk_csv = risk_pivot.to_csv(index=True)
    st.download_button(
        label="Download Risk Distribution Data",
        data=risk_csv,
        file_name="archetype_risk_distribution.csv",
        mime="text/csv"
    )
    
    with st.expander("View raw data"):
        st.dataframe(risk_pivot)
        
    st.markdown("---")
    
    # --- Section 3: Canadian Province Choropleth ---
    st.subheader("High-Risk Member Concentration by Province")
    
    # GeoJSON Province mapping
    PROVINCE_MAPPING = {
        'Newfoundland': 'Newfoundland and Labrador',
        'Yukon': 'Yukon Territory'
    }
    
    df_scored['geojson_province'] = df_scored['Province'].map(PROVINCE_MAPPING).fillna(df_scored['Province'])
    
    # Group by Province
    prov_stats = df_scored.groupby('geojson_province').agg(
        high_risk_count=('churn_risk_tier', lambda x: (x == 'High').sum()),
        total_members=('Loyalty Number', 'count')
    ).reset_index()
    
    prov_stats['pct_high_risk'] = (prov_stats['high_risk_count'] / prov_stats['total_members'] * 100).round(2)
    
    geojson = get_geojson_provinces()
    if geojson:
        # Check for mismatches
        geojson_names = [f['properties']['name'] for f in geojson['features']]
        mismatches = []
        for p in prov_stats['geojson_province']:
            if p not in geojson_names:
                mismatches.append(p)
                
        if mismatches:
            st.warning(f"Warning: The following provinces did not match the GeoJSON names: {mismatches}")
            
        fig_map = px.choropleth(
            prov_stats,
            geojson=geojson,
            locations='geojson_province',
            featureidkey='properties.name',
            color='high_risk_count',
            color_continuous_scale='Reds',
            labels={'high_risk_count': 'High-Risk Members', 'geojson_province': 'Province'},
            hover_data={
                'geojson_province': True,
                'high_risk_count': True,
                'total_members': True,
                'pct_high_risk': ':.2f%'
            }
        )
        fig_map.update_geos(
            fitbounds="locations",
            visible=False
        )
        fig_map.update_layout(height=500)
        apply_chart_style(fig_map)
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Map could not be rendered because GeoJSON failed to load.")
        
    with st.expander("View raw data"):
        st.dataframe(prov_stats.rename(columns={'geojson_province': 'Province'}))
        
    st.markdown("---")
    
    # --- Section 4: Archetype Shares and Mean CLV ---
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Archetype Share")
        fig_pie = px.pie(
            df_summary,
            names='archetype',
            values='member_count',
            color='archetype',
            color_discrete_map=ARCHETYPE_COLORS,
            hole=0.4
        )
        fig_pie.update_layout(height=400)
        apply_chart_style(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with chart_col2:
        st.subheader("Mean CLV per Archetype")
        summary_sorted_clv = df_summary.sort_values(by='mean_clv', ascending=False)
        fig_clv = px.bar(
            summary_sorted_clv,
            x='mean_clv',
            y='archetype',
            color='archetype',
            color_discrete_map=ARCHETYPE_COLORS,
            orientation='h',
            category_orders={'archetype': list(summary_sorted_clv['archetype'])},
            labels={'mean_clv': 'Mean CLV ($)', 'archetype': 'Archetype'}
        )
        fig_clv.update_layout(height=400, showlegend=False)
        apply_chart_style(fig_clv)
        st.plotly_chart(fig_clv, use_container_width=True)
        
    with st.expander("View raw data"):
        st.dataframe(df_summary[['archetype', 'member_count', 'mean_clv']])

if __name__ == "__main__":
    main()
