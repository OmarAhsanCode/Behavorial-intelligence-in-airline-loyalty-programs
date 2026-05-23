import os
import pandas as pd

def main():
    print("Starting Task 1.1: Loading and merging datasets...")
    
    # Define paths
    data_dir = "."
    outputs_dir = os.path.join(data_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Load all four datasets
    print("Loading datasets...")
    loyalty_history = pd.read_csv(os.path.join(data_dir, "Customer Loyalty History.csv"))
    flight_activity = pd.read_csv(os.path.join(data_dir, "Customer Flight Activity.csv"))
    calendar = pd.read_csv(os.path.join(data_dir, "Calendar.csv"))
    data_dict = pd.read_csv(os.path.join(data_dir, "Airline Loyalty Data Dictionary.csv"))
    
    print(f"Customer Loyalty History shape: {loyalty_history.shape}")
    print(f"Customer Flight Activity shape: {flight_activity.shape}")
    print(f"Calendar shape: {calendar.shape}")
    print(f"Data Dictionary shape: {data_dict.shape}")
    
    # Process Calendar.csv to extract Quarter and Season
    print("Processing Calendar dataset to extract Quarter and Season...")
    calendar['Date'] = pd.to_datetime(calendar['Date'])
    calendar['Year'] = calendar['Date'].dt.year
    calendar['Month'] = calendar['Date'].dt.month
    
    # Drop duplicates to get unique Year-Month mapping
    calendar_monthly = calendar.drop_duplicates(subset=['Year', 'Month']).copy()
    
    # Extract Quarter from 'Start of Quarter' column
    calendar_monthly['Start of Quarter'] = pd.to_datetime(calendar_monthly['Start of Quarter'])
    calendar_monthly['Quarter'] = calendar_monthly['Start of Quarter'].dt.quarter
    
    # Map months to Season
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    calendar_monthly['Season'] = calendar_monthly['Month'].map(season_map)
    
    # Keep only the columns needed for joining
    calendar_features = calendar_monthly[['Year', 'Month', 'Quarter', 'Season']]
    
    # Merge loyalty history with flight activity on 'Loyalty Number'
    print("Merging Loyalty History and Flight Activity...")
    # Flight activity has multiple rows per Loyalty Number (monthly)
    # Loyalty history has unique rows per Loyalty Number
    merged_df = pd.merge(flight_activity, loyalty_history, on="Loyalty Number", how="left")
    
    # Join processed Calendar to add Quarter and Season
    print("Joining processed Calendar data...")
    merged_df = pd.merge(merged_df, calendar_features, on=["Year", "Month"], how="left")
    
    # Output merged dataframe
    output_path = os.path.join(outputs_dir, "merged_loyalty_activity.csv")
    print(f"Saving merged dataframe to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    
    # Print shape, dtypes, and first 5 rows
    print("\n--- MERGED DATAFRAME INFO ---")
    print(f"Shape: {merged_df.shape}")
    print("\nData Types:")
    print(merged_df.dtypes)
    print("\nFirst 5 Rows:")
    print(merged_df.head())
    
    print("\nTask 1.1 completed successfully.")

if __name__ == "__main__":
    main()
