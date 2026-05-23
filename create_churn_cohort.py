import os
import pandas as pd
import numpy as np

def main():
    print("Starting Task 2.1 & 2.2: Creating modeling cohort and churn target labels...")
    
    # Paths
    data_dir = "."
    outputs_dir = os.path.join(data_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Load Loyalty History and Cleaned Merged Activity
    print("Loading datasets...")
    loyalty_df = pd.read_csv(os.path.join(data_dir, "Customer Loyalty History.csv"))
    merged_path = os.path.join(outputs_dir, "merged_loyalty_activity.csv")
    df = pd.read_csv(merged_path)
    
    # Apply baseline cleaning to merged activity
    print("Applying baseline cleaning to merged activity data...")
    is_placeholder = (
        (df['Total Flights'] == 0) & 
        (df['Distance'] == 0) & 
        (df['Points Accumulated'] == 0) & 
        (df['Points Redeemed'] == 0) & 
        df['Cancellation Year'].notnull() & 
        (
            (df['Year'] > df['Cancellation Year']) | 
            ((df['Year'] == df['Cancellation Year']) & (df['Month'] > df['Cancellation Month']))
        )
    )
    df_clean = df[~is_placeholder].copy()
    if 'Country' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Country'])
    
    # Apply minimum tenure guard (enrolled on or before June 2017)
    print("Filtering cohort by enrollment date (min 6 months tenure as of Dec 2017)...")
    cond_enroll = (loyalty_df['Enrollment Year'] < 2017) | (
        (loyalty_df['Enrollment Year'] == 2017) & (loyalty_df['Enrollment Month'] <= 6)
    )
    # Exclude members who cancelled on or before December 2017
    cond_active_at_cutoff = (loyalty_df['Cancellation Year'].isnull()) | (loyalty_df['Cancellation Year'] > 2017)
    
    cohort_customers = loyalty_df[cond_enroll & cond_active_at_cutoff].copy()
    print(f"Cohort size (active as of Dec 2017 with >= 6 months tenure): {len(cohort_customers)}")
    
    # Filter 2017 activity for cohort, only up to November 2017 (Month <= 11)
    df_2017 = df_clean[(df_clean['Year'] == 2017) & (df_clean['Month'] <= 11)].copy()
    cohort_loyalty_nums = set(cohort_customers['Loyalty Number'])
    df_2017_cohort = df_2017[df_2017['Loyalty Number'].isin(cohort_loyalty_nums)].copy()
    
    # Group and sum any duplicate transaction/activity records within the same customer and month
    df_2017_cohort_agg = df_2017_cohort.groupby(['Loyalty Number', 'Month']).agg({
        'Total Flights': 'sum',
        'Points Accumulated': 'sum',
        'Points Redeemed': 'sum',
        'Distance': 'sum',
        'Dollar Cost Points Redeemed': 'sum'
    }).reset_index()
    
    # 1. Flight Inactivity
    print("Calculating Flight Inactivity scores...")
    last_flight = df_2017_cohort_agg[df_2017_cohort_agg['Total Flights'] > 0].groupby('Loyalty Number')['Month'].max()
    
    # 2. Redemption Dormancy
    print("Calculating Redemption Dormancy scores...")
    last_red = df_2017_cohort_agg[df_2017_cohort_agg['Points Redeemed'] > 0].groupby('Loyalty Number')['Month'].max()
    
    # 3. Point Momentum
    print("Calculating Point Momentum slopes...")
    # Pivot points to get monthly points for regression
    points_pivot = df_2017_cohort_agg.pivot(index='Loyalty Number', columns='Month', values='Points Accumulated').fillna(0)
    # Ensure all 11 columns exist
    for m in range(1, 12):
        if m not in points_pivot.columns:
            points_pivot[m] = 0.0
    points_pivot = points_pivot[list(range(1, 12))]
    
    # Slope of linear regression over months 1-11
    x = np.arange(1, 12)
    x_mean = 6.0
    x_diff = x - x_mean
    
    slopes = {}
    for loyalty_num, row in points_pivot.iterrows():
        y = row.values
        slope = np.dot(x_diff, y) / 110.0
        slopes[loyalty_num] = slope
        
    # Calculate min_slope and max_slope first for default fallback
    min_slope = min(slopes.values()) if slopes else 0.0
    max_slope = max(slopes.values()) if slopes else 0.0
    print(f"Raw Slope range: [{min_slope:.4f}, {max_slope:.4f}]")
    
    # Compute BDS for cohort members
    print("Combining sub-scores into Behavioral Disengagement Score (BDS)...")
    scores = []
    for loyalty_num in cohort_customers['Loyalty Number']:
        # Flight Inactivity Score (normalize 0-1 relative to Nov 2017)
        if loyalty_num in last_flight:
            months_since_flight = 11 - last_flight[loyalty_num]
        else:
            months_since_flight = 11
        flight_inactivity = months_since_flight / 11.0
        
        # Redemption Dormancy (use 11 if never redeemed relative to Nov 2017)
        if loyalty_num in last_red:
            months_since_red = 11 - last_red[loyalty_num]
        else:
            months_since_red = 11 # never redeemed -> fully dormant (score 1.0)
        redemption_dormancy = months_since_red / 11.0
        
        # Point Momentum slope lookup (default to min_slope if missing)
        slope = slopes.get(loyalty_num, min_slope)
        
        scores.append({
            'Loyalty Number': loyalty_num,
            'FlightInactivity': flight_inactivity,
            'RedemptionDormancy': redemption_dormancy,
            'RawSlope': slope
        })
        
    scores_df = pd.DataFrame(scores)
    
    # Normalize slope (min-max scaling)
    scores_df['NormalizedMomentum'] = (scores_df['RawSlope'] - min_slope) / (max_slope - min_slope)
    
    # Compute final BDS
    scores_df['BDS'] = (
        0.4 * scores_df['FlightInactivity'] + 
        0.3 * scores_df['RedemptionDormancy'] + 
        0.3 * (1.0 - scores_df['NormalizedMomentum'])
    )
    
    # Task 2.2: Create churn labels
    print("Creating churn target labels...")
    # Get cancellation data
    churn_info = cohort_customers[['Loyalty Number', 'Cancellation Year']].copy()
    scores_df = pd.merge(scores_df, churn_info, on='Loyalty Number', how='left')
    
    # Label Class:
    # 0 = Active
    # 1 = Soft Churn (BDS > 0.55 and not cancelled in 2018)
    # 2 = Hard Churn (cancelled in 2018)
    scores_df['Class'] = 0
    scores_df.loc[(scores_df['Cancellation Year'] == 2018), 'Class'] = 2
    scores_df.loc[(scores_df['BDS'] > 0.55) & (scores_df['Class'] == 0), 'Class'] = 1
    
    # Binary label
    scores_df['Churned'] = (scores_df['Class'] > 0).astype(int)
    
    # Merge demographics back to scores_df for modeling base
    print("Merging member demographics back to the cohort dataset...")
    # Drop cancellation year and raw scores from scores_df if duplicate, keep loyalty number
    cohort_modeling_df = pd.merge(cohort_customers, scores_df.drop(columns=['Cancellation Year']), on='Loyalty Number', how='left')
    
    # Save modeling cohort
    output_path = os.path.join(outputs_dir, "churn_modeling_cohort.csv")
    print(f"Saving cohort modeling dataset to {output_path}...")
    cohort_modeling_df.to_csv(output_path, index=False)
    
    # Print stats
    print("\n--- COHORT MODELING DATASET STATISTICS ---")
    print(f"Shape: {cohort_modeling_df.shape}")
    print("\nBDS Score Distribution:")
    print(cohort_modeling_df['BDS'].describe())
    
    print("\nClass Target Distribution:")
    class_counts = cohort_modeling_df['Class'].value_counts()
    class_pct = cohort_modeling_df['Class'].value_counts(normalize=True) * 100
    for idx in sorted(class_counts.index):
        label_name = {0: "Active", 1: "Soft Churn", 2: "Hard Churn"}[idx]
        print(f"  Class {idx} ({label_name}): {class_counts[idx]} ({class_pct[idx]:.2f}%)")
        
    print("\nBinary Churn Target Distribution:")
    churn_counts = cohort_modeling_df['Churned'].value_counts()
    churn_pct = cohort_modeling_df['Churned'].value_counts(normalize=True) * 100
    print(f"  Not Churned (0): {churn_counts[0]} ({churn_pct[0]:.2f}%)")
    print(f"  Churned (1): {churn_counts[1]} ({churn_pct[1]:.2f}%)")
    
    print("\nTask 2.1 & 2.2 completed successfully.")

if __name__ == "__main__":
    main()
