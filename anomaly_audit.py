import os
import pandas as pd
import numpy as np

def main():
    print("Starting Task 1.3: Manual Anomaly Audit...")
    
    # Paths
    data_dir = "."
    outputs_dir = os.path.join(data_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Load datasets
    loyalty_df = pd.read_csv(os.path.join(data_dir, "Customer Loyalty History.csv"))
    merged_df = pd.read_csv(os.path.join(outputs_dir, "merged_loyalty_activity.csv"))
    
    anomalies = []
    
    # (a) members with Cancellation Year before Enrollment Year
    print("Checking for Cancellation Year before Enrollment Year...")
    cancel_before_enroll = loyalty_df[
        (loyalty_df['Cancellation Year'].notnull()) & 
        (loyalty_df['Cancellation Year'] < loyalty_df['Enrollment Year'])
    ]
    for _, row in cancel_before_enroll.iterrows():
        anomalies.append({
            'Loyalty Number': int(row['Loyalty Number']),
            'Table': 'Customer Loyalty History',
            'Anomaly Type': 'Cancellation before Enrollment',
            'Details': f"Enrollment Year: {int(row['Enrollment Year'])}, Cancellation Year: {int(row['Cancellation Year'])}",
            'Recommended Action': 'Investigate date entry; correct cancellation year or set to null if invalid'
        })
    print(f"Found {len(cancel_before_enroll)} members with Cancellation before Enrollment.")
    
    # (b) members with flight activity after their cancellation date
    print("Checking for flight activity after cancellation date...")
    # Flight activity is after cancellation if:
    # Activity Year > Cancellation Year OR (Activity Year == Cancellation Year AND Activity Month > Cancellation Month)
    flight_after_cancel = merged_df[
        merged_df['Cancellation Year'].notnull() & 
        ((merged_df['Year'] > merged_df['Cancellation Year']) | 
         ((merged_df['Year'] == merged_df['Cancellation Year']) & (merged_df['Month'] > merged_df['Cancellation Month'])))
    ]
    print(f"Found {len(flight_after_cancel)} flight activity records after cancellation date.")
    
    # Group by Loyalty Number to summarize per member
    for member_id, group in flight_after_cancel.groupby('Loyalty Number'):
        cancel_y = int(group['Cancellation Year'].iloc[0])
        cancel_m = int(group['Cancellation Month'].iloc[0])
        max_activity_y = int(group['Year'].max())
        max_activity_m = int(group[group['Year'] == max_activity_y]['Month'].max())
        total_flights = int(group['Total Flights'].sum())
        anomalies.append({
            'Loyalty Number': int(member_id),
            'Table': 'Customer Flight Activity',
            'Anomaly Type': 'Flight Activity After Cancellation',
            'Details': f"Cancelled: {cancel_y}-{cancel_m:02d}, Flight activity up to {max_activity_y}-{max_activity_m:02d} ({len(group)} months, {total_flights} total flights)",
            'Recommended Action': 'Investigate loyalty cancellation processing; drop post-cancellation activity or correct cancellation date'
        })
        
    # (c) members with Total Flights > 0 but Distance == 0
    print("Checking for Total Flights > 0 but Distance == 0...")
    flights_zero_distance = merged_df[
        (merged_df['Total Flights'] > 0) & (merged_df['Distance'] == 0)
    ]
    print(f"Found {len(flights_zero_distance)} records with Total Flights > 0 but Distance == 0.")
    
    # Group by Loyalty Number to summarize
    for member_id, group in flights_zero_distance.groupby('Loyalty Number'):
        total_flights = int(group['Total Flights'].sum())
        months = len(group)
        anomalies.append({
            'Loyalty Number': int(member_id),
            'Table': 'Customer Flight Activity',
            'Anomaly Type': 'Total Flights > 0 with 0 Distance',
            'Details': f"Found in {months} monthly record(s), totaling {total_flights} flights with 0 distance",
            'Recommended Action': 'Recalculate flight distance based on flight routes or check booking system logs'
        })
        
    # (d) negative Points Redeemed
    print("Checking for negative Points Redeemed...")
    negative_points = merged_df[merged_df['Points Redeemed'] < 0]
    print(f"Found {len(negative_points)} records with negative Points Redeemed.")
    for member_id, group in negative_points.groupby('Loyalty Number'):
        min_points = int(group['Points Redeemed'].min())
        anomalies.append({
            'Loyalty Number': int(member_id),
            'Table': 'Customer Flight Activity',
            'Anomaly Type': 'Negative Points Redeemed',
            'Details': f"Minimum Points Redeemed value: {min_points}",
            'Recommended Action': 'Correct negative points values; investigate if this represents a refund/reversal or data entry error'
        })
        
    # (e) CLV outliers beyond 3 IQR
    print("Checking for CLV outliers beyond 3 IQR...")
    q1 = loyalty_df['CLV'].quantile(0.25)
    q3 = loyalty_df['CLV'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    print(f"CLV Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    print(f"CLV Bounds for 3 IQR: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    clv_outliers = loyalty_df[
        (loyalty_df['CLV'] < lower_bound) | (loyalty_df['CLV'] > upper_bound)
    ]
    print(f"Found {len(clv_outliers)} CLV outliers.")
    for _, row in clv_outliers.iterrows():
        anomalies.append({
            'Loyalty Number': int(row['Loyalty Number']),
            'Table': 'Customer Loyalty History',
            'Anomaly Type': 'CLV Outlier (>3 IQR)',
            'Details': f"CLV: {row['CLV']:.2f} (Upper Bound: {upper_bound:.2f})",
            'Recommended Action': 'High-value customer outlier. Treat as a separate segment or evaluate for model truncation'
        })
        
    # Write to outputs/anomaly_log.csv
    if anomalies:
        anomalies_df = pd.DataFrame(anomalies)
    else:
        anomalies_df = pd.DataFrame(columns=['Loyalty Number', 'Table', 'Anomaly Type', 'Details', 'Recommended Action'])
        
    output_path = os.path.join(outputs_dir, "anomaly_log.csv")
    anomalies_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(anomalies_df)} total anomalies to {output_path}")

if __name__ == "__main__":
    main()
