import os
import pandas as pd
import numpy as np

def main():
    print("Starting Task 2.3: Temporal Feature Engineering...")
    
    # Paths
    data_dir = "."
    outputs_dir = os.path.join(data_dir, "outputs")
    
    # Load cohort to get Loyalty Numbers (11,076 members)
    cohort_path = os.path.join(outputs_dir, "churn_modeling_cohort.csv")
    print(f"Loading cohort from {cohort_path}...")
    cohort_df = pd.read_csv(cohort_path)
    cohort_loyalty_nums = cohort_df['Loyalty Number'].unique()
    print(f"Loaded {len(cohort_loyalty_nums)} cohort members.")
    
    # Load activity dataset
    activity_path = os.path.join(outputs_dir, "merged_loyalty_activity.csv")
    print(f"Loading loyalty activity from {activity_path}...")
    df = pd.read_csv(activity_path)
    
    # Apply baseline cleaning
    print("Applying baseline cleaning (dropping placeholders and Country column)...")
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
        
    # Critical rule: all features must use only data where Year < 2017, OR where Year == 2017 AND Month <= 11.
    # No December 2017 data and no 2018 data may be used.
    # Apply temporal cutoff filter:
    df_filtered = df_clean[
        (df_clean['Year'] < 2017) | 
        ((df_clean['Year'] == 2017) & (df_clean['Month'] <= 11))
    ].copy()
    
    # Explicit assertion at the top of the script
    print("Verifying temporal cutoff assertion...")
    max_year = df_filtered['Year'].max()
    max_month_in_max_year = df_filtered[df_filtered['Year'] == max_year]['Month'].max()
    print(f"Max Year in filtered data: {max_year}, Max Month in that year: {max_month_in_max_year}")
    assert max_year < 2017 or (max_year == 2017 and max_month_in_max_year <= 11), \
        "Temporal leakage detected: data after November 2017 is present!"
    print("Assertion passed: No data from December 2017 or 2018 is present in the feature calculation set.")
    
    # Group and aggregate duplicate records in the activity history to prevent counting duplicates
    df_filtered = df_filtered.groupby(['Loyalty Number', 'Year', 'Month', 'Season']).agg({
        'Total Flights': 'sum',
        'Points Accumulated': 'sum',
        'Points Redeemed': 'sum',
        'Distance': 'sum',
        'Dollar Cost Points Redeemed': 'sum'
    }).reset_index()
    
    # Calculate months_since_cutoff (months between Nov 2017 and record month)
    # Nov 2017 is month 0, Oct 2017 is month 1, etc.
    df_filtered['months_since_cutoff'] = (2017 - df_filtered['Year']) * 12 + (11 - df_filtered['Month'])
    
    # Initialize the features DataFrame with the cohort Loyalty Numbers
    features_df = pd.DataFrame(index=pd.Index(cohort_loyalty_nums, name='Loyalty Number'))
    
    # --- 1. Recency features ---
    print("Computing Recency features...")
    # months_since_last_flight
    flight_records = df_filtered[df_filtered['Total Flights'] > 0]
    recency_flight = flight_records.groupby('Loyalty Number')['months_since_cutoff'].min()
    features_df['months_since_last_flight'] = recency_flight.reindex(features_df.index, fill_value=72)
    
    # months_since_last_redemption
    red_records = df_filtered[df_filtered['Points Redeemed'] > 0]
    recency_red = red_records.groupby('Loyalty Number')['months_since_cutoff'].min()
    features_df['months_since_last_redemption'] = recency_red.reindex(features_df.index, fill_value=72)
    
    # months_since_last_any_activity
    any_act_records = df_filtered[
        (df_filtered['Total Flights'] > 0) | 
        (df_filtered['Points Accumulated'] > 0) | 
        (df_filtered['Points Redeemed'] > 0)
    ]
    recency_any = any_act_records.groupby('Loyalty Number')['months_since_cutoff'].min()
    features_df['months_since_last_any_activity'] = recency_any.reindex(features_df.index, fill_value=72)
    
    # --- 2. Frequency features ---
    print("Computing Frequency features...")
    # flights_last_3m
    f3m = df_filtered[df_filtered['months_since_cutoff'] < 3].groupby('Loyalty Number')['Total Flights'].sum()
    features_df['flights_last_3m'] = f3m.reindex(features_df.index, fill_value=0)
    
    # flights_last_6m
    f6m = df_filtered[df_filtered['months_since_cutoff'] < 6].groupby('Loyalty Number')['Total Flights'].sum()
    features_df['flights_last_6m'] = f6m.reindex(features_df.index, fill_value=0)
    
    # flights_last_12m
    f12m = df_filtered[df_filtered['months_since_cutoff'] < 12].groupby('Loyalty Number')['Total Flights'].sum()
    features_df['flights_last_12m'] = f12m.reindex(features_df.index, fill_value=0)
    
    # active_months_last_12m
    act12m = df_filtered[(df_filtered['months_since_cutoff'] < 12) & (df_filtered['Total Flights'] > 0)].groupby('Loyalty Number')['Month'].nunique()
    features_df['active_months_last_12m'] = act12m.reindex(features_df.index, fill_value=0)
    
    # zero_flight_months_last_12m
    features_df['zero_flight_months_last_12m'] = 12 - features_df['active_months_last_12m']
    
    # flights_total_ever
    f_ever = df_filtered.groupby('Loyalty Number')['Total Flights'].sum()
    features_df['flights_total_ever'] = f_ever.reindex(features_df.index, fill_value=0)
    
    # --- 3. Monetary features ---
    print("Computing Monetary features...")
    # distance_last_12m
    d12m = df_filtered[df_filtered['months_since_cutoff'] < 12].groupby('Loyalty Number')['Distance'].sum()
    features_df['distance_last_12m'] = d12m.reindex(features_df.index, fill_value=0)
    
    # points_accumulated_last_12m
    p_acc12m = df_filtered[df_filtered['months_since_cutoff'] < 12].groupby('Loyalty Number')['Points Accumulated'].sum()
    features_df['points_accumulated_last_12m'] = p_acc12m.reindex(features_df.index, fill_value=0)
    
    # points_redeemed_last_12m
    p_red12m = df_filtered[df_filtered['months_since_cutoff'] < 12].groupby('Loyalty Number')['Points Redeemed'].sum()
    features_df['points_redeemed_last_12m'] = p_red12m.reindex(features_df.index, fill_value=0)
    
    # dollar_redeemed_last_12m
    dlr12m = df_filtered[df_filtered['months_since_cutoff'] < 12].groupby('Loyalty Number')['Dollar Cost Points Redeemed'].sum()
    features_df['dollar_redeemed_last_12m'] = dlr12m.reindex(features_df.index, fill_value=0)
    
    # points_accumulated_total
    p_acc_tot = df_filtered.groupby('Loyalty Number')['Points Accumulated'].sum()
    features_df['points_accumulated_total'] = p_acc_tot.reindex(features_df.index, fill_value=0)
    
    # points_redeemed_total
    p_red_tot = df_filtered.groupby('Loyalty Number')['Points Redeemed'].sum()
    features_df['points_redeemed_total'] = p_red_tot.reindex(features_df.index, fill_value=0)
    
    # redemption_rate
    features_df['redemption_rate'] = np.where(
        features_df['points_accumulated_total'] == 0,
        0.0,
        features_df['points_redeemed_total'] / features_df['points_accumulated_total']
    )
    
    # --- 4. Trend features ---
    print("Computing Trend features...")
    # flight_trend_ratio: flights_last_3m / (flights_last_12m / 4.0)
    denom_flight = features_df['flights_last_12m'] / 4.0
    features_df['flight_trend_ratio'] = np.where(
        denom_flight == 0.0,
        1.0,
        features_df['flights_last_3m'] / denom_flight
    )
    
    # points_trend_ratio: points_accumulated_last_3m / (points_accumulated_last_12m / 4.0)
    p_acc3m = df_filtered[df_filtered['months_since_cutoff'] < 3].groupby('Loyalty Number')['Points Accumulated'].sum().reindex(features_df.index, fill_value=0)
    denom_pts = features_df['points_accumulated_last_12m'] / 4.0
    features_df['points_trend_ratio'] = np.where(
        denom_pts == 0.0,
        1.0,
        p_acc3m / denom_pts
    )
    
    # distance_per_flight
    features_df['distance_per_flight'] = np.where(
        features_df['flights_last_12m'] == 0,
        0.0,
        features_df['distance_last_12m'] / features_df['flights_last_12m']
    )
    
    # --- 5. Seasonal features ---
    print("Computing Seasonal features...")
    # pref_season: mode of Season across months where Total Flights > 0
    df_flights = df_filtered[df_filtered['Total Flights'] > 0]
    season_counts = df_flights.groupby(['Loyalty Number', 'Season']).size().reset_index(name='count')
    # Sort by Loyalty Number ascending, and count descending
    season_counts = season_counts.sort_values(by=['Loyalty Number', 'count'], ascending=[True, False])
    # Drop duplicates keeping the highest count season
    pref_season_df = season_counts.drop_duplicates(subset=['Loyalty Number'], keep='first')
    pref_season_map = pref_season_df.set_index('Loyalty Number')['Season']
    features_df['pref_season'] = pref_season_map.reindex(features_df.index, fill_value='None')
    
    # flew_in_winter / summer / spring / fall in the last 24 months (months_since_cutoff < 24)
    df_24m = df_filtered[(df_filtered['months_since_cutoff'] < 24) & (df_filtered['Total Flights'] > 0)]
    flew_winter = set(df_24m[df_24m['Season'] == 'Winter']['Loyalty Number'])
    flew_summer = set(df_24m[df_24m['Season'] == 'Summer']['Loyalty Number'])
    flew_spring = set(df_24m[df_24m['Season'] == 'Spring']['Loyalty Number'])
    flew_fall = set(df_24m[df_24m['Season'] == 'Fall']['Loyalty Number'])
    
    features_df['flew_in_winter'] = features_df.index.map(lambda ln: 1 if ln in flew_winter else 0)
    features_df['flew_in_summer'] = features_df.index.map(lambda ln: 1 if ln in flew_summer else 0)
    features_df['flew_in_spring'] = features_df.index.map(lambda ln: 1 if ln in flew_spring else 0)
    features_df['flew_in_fall'] = features_df.index.map(lambda ln: 1 if ln in flew_fall else 0)
    
    # --- Audit & Save ---
    print("\nChecking for null values in features...")
    null_counts = features_df.isnull().sum()
    print("Null counts per column:")
    print(null_counts)
    
    # Fill any remaining nulls with 0 (pref_season with 'None' if somehow null, though reindex with fill_value should prevent this)
    if null_counts.sum() > 0:
        print("WARNING: Nulls detected. Printing affected loyalty numbers and filling them:")
        for col in features_df.columns:
            null_members = features_df[features_df[col].isnull()].index.tolist()
            if null_members:
                print(f"  Column '{col}' has nulls for members: {null_members}")
                if col == 'pref_season':
                    features_df[col] = features_df[col].fillna('None')
                else:
                    features_df[col] = features_df[col].fillna(0)
                    
    # Save the output
    output_path = os.path.join(outputs_dir, "temporal_features.csv")
    print(f"Saving temporal features to {output_path}...")
    features_df.to_csv(output_path)
    
    # Print shape and first 5 rows
    print("\n--- TEMPORAL FEATURES SUMMARY ---")
    print(f"Shape: {features_df.shape}")
    print("\nFirst 5 rows:")
    print(features_df.head())
    
    print("\nTask 2.3 completed successfully.")

if __name__ == "__main__":
    main()
