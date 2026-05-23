import os
import pandas as pd
import numpy as np

def main():
    print("Starting Task 2.4: Demographic Feature Encoding...")
    
    # Paths
    data_dir = "."
    outputs_dir = os.path.join(data_dir, "outputs")
    
    # Load cohort data (outputs/churn_modeling_cohort.csv)
    cohort_path = os.path.join(outputs_dir, "churn_modeling_cohort.csv")
    print(f"Loading cohort from {cohort_path}...")
    df = pd.read_csv(cohort_path)
    
    # Initialize demographics features dataframe
    dem_df = pd.DataFrame()
    dem_df['Loyalty Number'] = df['Loyalty Number']
    
    # 1. education_ordinal
    # High School or Below=1, College=2, Bachelor=3, Master=4, Doctor=5. If missing, impute with 2 (College)
    education_map = {
        'High School or Below': 1,
        'College': 2,
        'Bachelor': 3,
        'Master': 4,
        'Doctor': 5
    }
    print("Encoding Education to ordinal...")
    dem_df['education_ordinal'] = df['Education'].map(education_map).fillna(2).astype(int)
    
    # 2. loyalty_card_ordinal
    # Star=1, Nova=2, Aurora=3
    card_map = {
        'Star': 1,
        'Nova': 2,
        'Aurora': 3
    }
    print("Encoding Loyalty Card to ordinal...")
    dem_df['loyalty_card_ordinal'] = df['Loyalty Card'].map(card_map).fillna(1).astype(int) # Star as default if missing
    
    # 3. is_married
    # 1 if Marital Status == 'Married', else 0
    print("Encoding Marital Status...")
    dem_df['is_married'] = (df['Marital Status'] == 'Married').astype(int)
    
    # 4. is_single
    # 1 if Marital Status == 'Single', else 0
    dem_df['is_single'] = (df['Marital Status'] == 'Single').astype(int)
    
    # 5. is_male
    # 1 if Gender == 'Male', else 0
    print("Encoding Gender...")
    dem_df['is_male'] = (df['Gender'] == 'Male').astype(int)
    
    # 6. is_promo_enrollment
    # 1 if Enrollment Type == '2018 Promotion', else 0
    # Note: check if this column exists and what its values are — print unique values before encoding
    print("Unique values of Enrollment Type before encoding:")
    enrollment_types = df['Enrollment Type'].unique()
    print(enrollment_types)
    dem_df['is_promo_enrollment'] = (df['Enrollment Type'] == '2018 Promotion').astype(int)
    
    # 7. tenure_months
    # number of months from (Enrollment Year, Enrollment Month) to November 2017.
    # Compute as (2017 - Enrollment Year) * 12 + (11 - Enrollment Month)
    print("Calculating tenure in months as of November 2017...")
    dem_df['tenure_months'] = (2017 - df['Enrollment Year']) * 12 + (11 - df['Enrollment Month'])
    
    # 8. salary_imputed
    # Salary column with missing values filled using the median of non-missing Salary values. Print the median value used
    median_salary = df['Salary'].median()
    print(f"Median salary calculated from cohort: {median_salary}")
    dem_df['salary_imputed'] = df['Salary'].fillna(median_salary)
    
    # 9. province_encoded
    # one-hot encode the Province column. Print the list of unique provinces before encoding
    print("Unique values of Province before one-hot encoding:")
    unique_provinces = sorted(df['Province'].unique())
    print(unique_provinces)
    
    province_dummies = pd.get_dummies(df['Province'], prefix='province').astype(int)
    # Concatenate the one-hot encoded columns
    dem_df = pd.concat([dem_df, province_dummies], axis=1)
    
    # Save the output
    output_path = os.path.join(outputs_dir, "demographic_features.csv")
    print(f"Saving demographic features to {output_path}...")
    dem_df.to_csv(output_path, index=False)
    
    # Print shape and null counts
    print("\n--- DEMOGRAPHIC FEATURES SUMMARY ---")
    print(f"Shape: {dem_df.shape}")
    print("\nNull counts per column:")
    print(dem_df.isnull().sum())
    
    print("\nTask 2.4 completed successfully.")

if __name__ == "__main__":
    main()
