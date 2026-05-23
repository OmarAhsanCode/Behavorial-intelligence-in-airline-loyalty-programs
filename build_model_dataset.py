import os
import pandas as pd
import numpy as np

def main():
    print("Starting Task 2.6: Merge All Features and Save Final Modeling Dataset...")
    
    # Paths
    data_dir = "."
    outputs_dir = os.path.join(data_dir, "outputs")
    
    # Load the three feature files
    cohort_path = os.path.join(outputs_dir, "churn_modeling_cohort.csv")
    temp_path = os.path.join(outputs_dir, "temporal_features.csv")
    dem_path = os.path.join(outputs_dir, "demographic_features.csv")
    
    print(f"Loading {cohort_path}...")
    cohort_df = pd.read_csv(cohort_path)
    
    print(f"Loading {temp_path}...")
    temp_df = pd.read_csv(temp_path)
    # Restore 'None' string for pref_season if parsed as NaN by pandas
    if 'pref_season' in temp_df.columns:
        temp_df['pref_season'] = temp_df['pref_season'].fillna('None')
    
    print(f"Loading {dem_path}...")
    dem_df = pd.read_csv(dem_path)
    
    # Merge all three on Loyalty Number using left joins anchored to the cohort file
    print("Merging features using left joins on Loyalty Number...")
    merged_df = pd.merge(cohort_df, temp_df, on='Loyalty Number', how='left')
    merged_df = pd.merge(merged_df, dem_df, on='Loyalty Number', how='left')
    
    # Drop raw columns from final dataset if present
    raw_cols_to_drop = [
        'Country', 'City', 'Postal Code', 'Enrollment Type', 
        'Enrollment Year', 'Enrollment Month', 
        'Cancellation Year', 'Cancellation Month', 
        'Gender', 'Education', 'Marital Status', 'Salary', 
        'Loyalty Card', 'Province'
    ]
    
    existing_drops = [col for col in raw_cols_to_drop if col in merged_df.columns]
    print(f"Dropping raw columns: {existing_drops}")
    merged_df = merged_df.drop(columns=existing_drops)
    
    # Assert zero null values in final dataframe. If nulls exist, fill and print affected members
    print("Checking for null values in the merged dataset...")
    null_counts = merged_df.isnull().sum()
    if null_counts.sum() > 0:
        print("WARNING: Null values found in the merged dataset!")
        for col in merged_df.columns:
            col_nulls = merged_df[col].isnull().sum()
            if col_nulls > 0:
                print(f"  Column '{col}' has {col_nulls} null values.")
                # Print affected members (up to 10)
                affected_members = merged_df[merged_df[col].isnull()]['Loyalty Number'].head(10).tolist()
                print(f"  Affected Loyalty Numbers (sample): {affected_members}")
                # Fill nulls
                if pd.api.types.is_numeric_dtype(merged_df[col]):
                    merged_df[col] = merged_df[col].fillna(0)
                else:
                    merged_df[col] = merged_df[col].fillna('Unknown')
    else:
        print("No null values found in the merged dataset.")
        
    assert merged_df.isnull().sum().sum() == 0, "Assertion failed: Null values remain in the final dataset!"
    
    # Print the final shape
    print(f"\nFinal dataset shape: {merged_df.shape}")
    
    # Print correlation table of the top 15 features most correlated with Churned binary column
    print("\nCalculating feature correlations with Churned...")
    numeric_df = merged_df.select_dtypes(include=[np.number])
    correlations = numeric_df.corrwith(numeric_df['Churned']).abs().sort_values(ascending=False)
    
    # Drop Churned itself and other target variables/IDs from correlation list
    exclude_corr = ['Churned', 'Class', 'BDS', 'Loyalty Number', 'RawSlope', 'NormalizedMomentum', 'FlightInactivity', 'RedemptionDormancy', 'HardChurn']
    correlations_features = correlations.drop(index=[c for c in exclude_corr if c in correlations.index], errors='ignore')
    
    print("\nTop 15 features most correlated with Churned:")
    print(correlations_features.head(15))
    
    # Save the final dataset to outputs/model_ready_data.csv
    output_path = os.path.join(outputs_dir, "model_ready_data.csv")
    print(f"\nSaving final model ready dataset to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    
    # Save feature names to outputs/feature_names.txt
    # Exclude Loyalty Number, BDS, Class, Churned, RawSlope, NormalizedMomentum, FlightInactivity, RedemptionDormancy, HardChurn
    exclude_from_features = {
        'Loyalty Number', 'BDS', 'Class', 'Churned', 'RawSlope', 
        'NormalizedMomentum', 'FlightInactivity', 'RedemptionDormancy', 'HardChurn'
    }
    
    feature_names = [col for col in merged_df.columns if col not in exclude_from_features]
    
    feature_names_path = os.path.join(outputs_dir, "feature_names.txt")
    print(f"Saving feature names list to {feature_names_path}...")
    with open(feature_names_path, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
            
    print(f"Saved {len(feature_names)} feature names.")
    print("\nTask 2.6 completed successfully.")

if __name__ == "__main__":
    main()
