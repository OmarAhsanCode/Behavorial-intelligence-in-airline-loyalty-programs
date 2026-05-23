import os
import pandas as pd
from ydata_profiling import ProfileReport

def check_flags(df, name):
    print(f"\n--- DATA PROFILE FLAGS FOR: {name} ---")
    
    # 1. Missing value counts
    missing_counts = df.isnull().sum()
    print("\nMissing Value Counts (Columns with any missing values):")
    has_missing = False
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
            has_missing = True
    if not has_missing:
        print("  None")
        
    # 2. Columns with > 10% missing values
    print("\nColumns with >10% missing values:")
    has_high_missing = False
    for col, pct in (df.isnull().mean()).items():
        if pct > 0.10:
            print(f"  {col}: {pct*100:.2f}% missing ({df[col].isnull().sum()} rows)")
            has_high_missing = True
    if not has_high_missing:
        print("  None")
        
    # 3. Zero-variance columns (only 1 unique value)
    print("\nZero-variance (Constant) Columns:")
    zero_variance_cols = [col for col in df.columns if df[col].nunique() <= 1]
    for col in zero_variance_cols:
        val = df[col].iloc[0] if len(df) > 0 else "N/A"
        print(f"  {col}: Constant value = '{val}'")
    if not zero_variance_cols:
        print("  None")
        
    # 4. High-cardinality categoricals
    # We define categoricals as object/category dtype
    # Let's flag columns with > 10 unique values as high cardinality (especially object types)
    print("\nCategorical Column Cardinality (Object types):")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    has_high_card = False
    for col in cat_cols:
        n_unique = df[col].nunique()
        is_high = n_unique > 20
        flag_str = " [HIGH CARDINALITY > 20]" if is_high else ""
        print(f"  {col}: {n_unique} unique values{flag_str}")
        if is_high:
            has_high_card = True
    if len(cat_cols) == 0:
        print("  No categorical columns.")

def main():
    print("Starting Task 1.2: Running ydata-profiling...")
    
    # Define paths
    data_dir = "."
    outputs_dir = os.path.join(data_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Load loyalty history
    loyalty_path = os.path.join(data_dir, "Customer Loyalty History.csv")
    print(f"Loading loyalty history from {loyalty_path}...")
    loyalty_df = pd.read_csv(loyalty_path)
    
    # Load merged activity table
    merged_path = os.path.join(outputs_dir, "merged_loyalty_activity.csv")
    print(f"Loading merged activity table from {merged_path}...")
    merged_df = pd.read_csv(merged_path)
    
    # Run manual checks first to print flags clearly
    check_flags(loyalty_df, "Customer Loyalty History")
    check_flags(merged_df, "Merged Loyalty & Flight Activity")
    
    # Generate ydata-profiling report for Loyalty History
    print("\nGenerating ydata-profiling report for Customer Loyalty History...")
    loyalty_report_path = os.path.join(outputs_dir, "eda_loyalty_profile.html")
    # Using minimal=True for speed and compatibility, as it's sufficient for basic EDA
    loyalty_profile = ProfileReport(loyalty_df, title="Customer Loyalty History Profile Report", minimal=True)
    loyalty_profile.to_file(loyalty_report_path)
    print(f"Saved loyalty profile report to {loyalty_report_path}")
    
    # Generate ydata-profiling report for Merged Activity Table
    print("\nGenerating ydata-profiling report for Merged Loyalty & Flight Activity...")
    activity_report_path = os.path.join(outputs_dir, "eda_activity_profile.html")
    # Using minimal=True is highly recommended here due to large row count (~400k rows)
    activity_profile = ProfileReport(merged_df, title="Merged Loyalty & Flight Activity Profile Report", minimal=True)
    activity_profile.to_file(activity_report_path)
    print(f"Saved merged activity profile report to {activity_report_path}")
    
    print("\nTask 1.2 completed successfully.")

if __name__ == "__main__":
    main()
