import os
import json
import pandas as pd
import streamlit as st

@st.cache_data
def load_all_data():
    """
    Loads all datasets from the data/ folder and caches them.
    Returns a dictionary of dataframes and configurations.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    data = {}
    
    # Load CSVs
    csv_files = {
        'members_scored': 'members_scored.csv',
        'segment_summary': 'segment_summary.csv',
        'retention_roi_table': 'retention_roi_table.csv',
        'shap_values_per_member': 'shap_values_per_member.csv',
        'cluster_profiles': 'cluster_profiles.csv',
        'umap_coordinates': 'umap_coordinates.csv',
        'cox_hazard_ratios': 'cox_hazard_ratios.csv',
        'threshold_analysis': 'threshold_analysis.csv',
        'rfmet_features': 'rfmet_features.csv'
    }
    
    for key, filename in csv_files.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            data[key] = pd.DataFrame()
            
    # Load Playbook JSON
    playbook_path = os.path.join(data_dir, "intervention_playbook.json")
    if os.path.exists(playbook_path):
        with open(playbook_path, 'r') as f:
            data['playbook'] = json.load(f)
    else:
        data['playbook'] = {}
        
    # Load KM HTML files (cache their content)
    km_files = {
        'km_by_card': 'km_by_card.html',
        'km_by_enrollment': 'km_by_enrollment.html',
        'km_by_salary': 'km_by_salary.html',
        'km_by_bds': 'km_by_bds.html'
    }
    for key, filename in km_files.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data[key] = f.read()
        else:
            data[key] = ""
            
    # Load optimal threshold
    # Since optimal_threshold.txt might be in outputs/ or parent directory, we look for it
    # We copied optimal_threshold.txt during setup or we can load it from parent
    # Let's search in data_dir, if not found search in parent's outputs/
    opt_thresh_path = os.path.join(data_dir, "optimal_threshold.txt")
    if not os.path.exists(opt_thresh_path):
        # try parent outputs/
        opt_thresh_path = os.path.join(base_dir, "..", "outputs", "optimal_threshold.txt")
        
    if os.path.exists(opt_thresh_path):
        with open(opt_thresh_path, 'r') as f:
            data['optimal_threshold'] = float(f.read().strip())
    else:
        data['optimal_threshold'] = 0.35  # fallback
        
    return data

def add_sidebar_info():
    """
    Adds persistent sidebar metadata to the page.
    """
    st.sidebar.title("Loyalty Intelligence")
    st.sidebar.markdown("**Airline:** Canadian Airline Partner")
    st.sidebar.markdown("**Data Freshness:** November 2017")
    st.sidebar.markdown("**Model Version:** XGBoost (v1.0)")
    st.sidebar.markdown("**Model Metric:** AUC 0.73 · Threshold 0.35")
    st.sidebar.info("Designed for non-technical marketing managers.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Project:** Airline Loyalty Analytics")
    st.sidebar.markdown("**Model:** XGBoost · AUC 0.73")
    st.sidebar.markdown("**Cohort:** 11,076 Canadian members · 2012–2018")
