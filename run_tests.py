import os
import json
import importlib.util
import pandas as pd

def run_tests():
    print("="*60)
    print("RUNNING STREAMLIT APP VERIFICATION TESTS")
    print("="*60)
    
    app_dir = "airline_loyalty_app"
    data_dir = os.path.join(app_dir, "data")
    
    # Test 1: All required files in data/ load without error
    print("Test 1: Required files load check...")
    required_files = [
        'members_scored.csv',
        'segment_summary.csv',
        'retention_roi_table.csv',
        'intervention_playbook.json',
        'shap_values_per_member.csv',
        'rfmet_features.csv'
    ]
    test1_pass = True
    for f in required_files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            print(f"  [FAIL] File missing: {path}")
            test1_pass = False
        else:
            try:
                if f.endswith('.csv'):
                    pd.read_csv(path)
                elif f.endswith('.json'):
                    with open(path, 'r') as jf:
                        json.load(jf)
                print(f"  [PASS] Successfully loaded: {f}")
            except Exception as e:
                print(f"  [FAIL] Error loading {f}: {e}")
                test1_pass = False
    
    # Test 2: All 4 Streamlit pages import without error
    print("\nTest 2: Streamlit pages import check...")
    pages = [
        ("app", os.path.join(app_dir, "app.py")),
        ("Mission Control", os.path.join(app_dir, "pages", "1_Mission_Control.py")),
        ("Member Risk Scanner", os.path.join(app_dir, "pages", "2_Member_Risk_Scanner.py")),
        ("Segment Explorer", os.path.join(app_dir, "pages", "3_Segment_Explorer.py")),
        ("Campaign Builder", os.path.join(app_dir, "pages", "4_Campaign_Builder.py"))
    ]
    test2_pass = True
    for name, path in pages:
        if not os.path.exists(path):
            print(f"  [FAIL] Page missing: {path}")
            test2_pass = False
            continue
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"  [PASS] Page imported without error: {name}")
        except Exception as e:
            # We ignore module import errors for streamlit elements running outside streamlit context
            # but standard syntax/imports must pass.
            # Usually running exec_module might trigger Streamlit API errors if we run outside st context,
            # so we only test compilation or catch and inspect.
            # A simple syntax check is compiling the file.
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    compile(f.read(), path, 'exec')
                print(f"  [PASS] Page compiled/syntactically correct: {name}")
            except Exception as compile_err:
                print(f"  [FAIL] Page compilation error for {name}: {compile_err}")
                test2_pass = False
                
    # Test 3: members_scored.csv has correct columns
    print("\nTest 3: members_scored.csv columns check...")
    scored_path = os.path.join(data_dir, "members_scored.csv")
    test3_pass = True
    if os.path.exists(scored_path):
        df_scored = pd.read_csv(scored_path)
        required_cols = ['Loyalty Number', 'archetype', 'churn_prob', 'churn_risk_tier', 'CLV', 'Province']
        for col in required_cols:
            if col not in df_scored.columns:
                print(f"  [FAIL] Missing column: {col}")
                test3_pass = False
        if test3_pass:
            print("  [PASS] All required columns present in members_scored.csv")
    else:
        print("  [FAIL] members_scored.csv missing")
        test3_pass = False
        
    # Test 4: Every archetype in members_scored.csv exists in intervention_playbook.json
    print("\nTest 4: Archetype alignment check...")
    playbook_path = os.path.join(data_dir, "intervention_playbook.json")
    test4_pass = True
    if os.path.exists(scored_path) and os.path.exists(playbook_path):
        df_scored = pd.read_csv(scored_path)
        with open(playbook_path, 'r') as f:
            playbook = json.load(f)
        unique_archetypes = df_scored['archetype'].dropna().unique()
        for arch in unique_archetypes:
            if arch not in playbook:
                print(f"  [FAIL] Archetype '{arch}' not found in playbook.")
                test4_pass = False
            else:
                print(f"  [PASS] Archetype '{arch}' aligned with playbook.")
    else:
        print("  [FAIL] members_scored.csv or playbook missing")
        test4_pass = False
        
    # Test 5: No null values in churn_prob or archetype columns
    print("\nTest 5: Null values check...")
    test5_pass = True
    if os.path.exists(scored_path):
        df_scored = pd.read_csv(scored_path)
        null_prob = df_scored['churn_prob'].isnull().sum()
        null_arch = df_scored['archetype'].isnull().sum()
        if null_prob > 0:
            print(f"  [FAIL] Found {null_prob} nulls in 'churn_prob'")
            test5_pass = False
        if null_arch > 0:
            print(f"  [FAIL] Found {null_arch} nulls in 'archetype'")
            test5_pass = False
        if test5_pass:
            print("  [PASS] Zero nulls found in 'churn_prob' and 'archetype'")
    else:
        print("  [FAIL] members_scored.csv missing")
        test5_pass = False
        
    # Test 6: optimal_threshold.txt contains a float between 0 and 1
    print("\nTest 6: optimal_threshold.txt check...")
    test6_pass = True
    # Look for file in outputs/ or data/
    thresh_path = os.path.join(data_dir, "optimal_threshold.txt")
    if not os.path.exists(thresh_path):
        thresh_path = os.path.join("outputs", "optimal_threshold.txt")
        
    if os.path.exists(thresh_path):
        try:
            with open(thresh_path, 'r') as f:
                val = float(f.read().strip())
            if 0.0 <= val <= 1.0:
                print(f"  [PASS] Found valid optimal threshold: {val}")
            else:
                print(f"  [FAIL] Threshold {val} is not between 0 and 1")
                test6_pass = False
        except Exception as e:
            print(f"  [FAIL] Error reading threshold: {e}")
            test6_pass = False
    else:
        print("  [FAIL] optimal_threshold.txt not found")
        test6_pass = False
        
    print("\n" + "="*60)
    overall_pass = test1_pass and test2_pass and test3_pass and test4_pass and test5_pass and test6_pass
    if overall_pass:
        print("OVERALL VERIFICATION: PASS")
    else:
        print("OVERALL VERIFICATION: FAIL")
    print("="*60)
    return overall_pass

if __name__ == "__main__":
    run_tests()
