import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    print("Starting Task 1.4: Generating temporal enrollment and churn trends...")
    
    # Paths
    data_dir = "."
    outputs_dir = os.path.join(data_dir, "outputs")
    
    # Load merged dataset
    merged_path = os.path.join(outputs_dir, "merged_loyalty_activity.csv")
    print(f"Loading merged dataset from {merged_path}...")
    df = pd.read_csv(merged_path)
    
    # Drop Country column immediately
    if 'Country' in df.columns:
        print("Dropping Country column (zero-variance)...")
        df = df.drop(columns=['Country'])
        
    # Filter out the 34,302 post-cancellation placeholder rows
    print("Filtering out post-cancellation placeholder rows...")
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
    print(f"Cleaned dataset shape: {df_clean.shape} (dropped {len(df) - len(df_clean)} placeholder rows)")
    
    # Get unique customers based on Loyalty Number
    print("Extracting unique customers...")
    customers = df_clean.drop_duplicates(subset=['Loyalty Number']).copy()
    print(f"Unique customers in cleaned dataset: {len(customers)}")
    
    # Create enrollment date column (first day of the month)
    customers['Enrollment Date'] = pd.to_datetime(
        customers['Enrollment Year'].astype(str) + '-' + 
        customers['Enrollment Month'].astype(str) + '-01'
    )
    
    # Create cancellation date column (first day of the month)
    # Filter out NaNs for cancellation and map to datetime
    cancelled_mask = customers['Cancellation Year'].notnull()
    customers.loc[cancelled_mask, 'Cancellation Date'] = pd.to_datetime(
        customers.loc[cancelled_mask, 'Cancellation Year'].astype(int).astype(str) + '-' + 
        customers.loc[cancelled_mask, 'Cancellation Month'].astype(int).astype(str) + '-01'
    )
    
    # Create a full date range from Jan 2012 to Dec 2018
    all_months = pd.date_range(start='2012-01-01', end='2018-12-01', freq='MS')
    
    # Group enrollments by type
    print("Computing monthly enrollment counts...")
    standard_enroll = customers[customers['Enrollment Type'] == 'Standard'].groupby('Enrollment Date').size()
    standard_enroll = standard_enroll.reindex(all_months, fill_value=0)
    
    promo_enroll = customers[customers['Enrollment Type'] == '2018 Promotion'].groupby('Enrollment Date').size()
    promo_enroll = promo_enroll.reindex(all_months, fill_value=0)
    
    # Group cancellations
    print("Computing monthly cancellation counts...")
    cancellations = customers[customers['Cancellation Date'].notnull()].groupby('Cancellation Date').size()
    cancellations = cancellations.reindex(all_months, fill_value=0)
    
    # Create plot data
    plot_df = pd.DataFrame({
        'Date': all_months,
        'Standard Enrollment': standard_enroll.values,
        '2018 Promotion Enrollment': promo_enroll.values,
        'Cancellations': cancellations.values
    })
    
    # Generate dual-axis Plotly figure
    print("Generating dual-axis Plotly chart...")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add Standard Enrollment trace (stacked bar)
    fig.add_trace(
        go.Bar(
            x=plot_df['Date'],
            y=plot_df['Standard Enrollment'],
            name="Standard Enrollment",
            marker=dict(color='#3A86C8', line=dict(width=0)),
            hovertemplate="Standard Enrollment: %{y}<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Add 2018 Promotion Enrollment trace (stacked bar)
    fig.add_trace(
        go.Bar(
            x=plot_df['Date'],
            y=plot_df['2018 Promotion Enrollment'],
            name="2018 Promotion Enrollment",
            marker=dict(color='#8A4FC4', line=dict(width=0)), # vibrant purple
            hovertemplate="2018 Promotion Enrollment: %{y}<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Add Cancellation trace (line)
    fig.add_trace(
        go.Scatter(
            x=plot_df['Date'],
            y=plot_df['Cancellations'],
            name="Cancellations (Churn)",
            line=dict(color='#E63946', width=3), # vibrant red/rose
            mode='lines',
            hovertemplate="Cancellations: %{y}<extra></extra>"
        ),
        secondary_y=True
    )
    
    # Style the layout
    fig.update_layout(
        title=dict(
            text="Monthly Member Enrollments vs. Cancellations (2012 - 2018)",
            font=dict(size=20, family="Outfit, Inter, sans-serif", color="#2B2D42"),
            x=0.5,
            xanchor='center'
        ),
        barmode='stack',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(family="Inter, sans-serif", size=12)
        ),
        margin=dict(l=60, r=60, t=100, b=60),
        plot_bgcolor='#F8F9FA',
        paper_bgcolor='#FFFFFF',
        hovermode='x unified',
    )
    
    # Style Axes
    fig.update_xaxes(
        title=dict(text="Timeline", font=dict(family="Inter, sans-serif", size=14)),
        gridcolor='#EAEAEA',
        tickformat="%b %Y",
        dtick="M6" # ticks every 6 months
    )
    
    fig.update_yaxes(
        title=dict(text="Enrollment Count (Bars)", font=dict(family="Inter, sans-serif", size=14, color="#3A86C8")),
        gridcolor='#EAEAEA',
        secondary_y=False,
        showgrid=True
    )
    
    fig.update_yaxes(
        title=dict(text="Cancellation Count (Line)", font=dict(family="Inter, sans-serif", size=14, color="#E63946")),
        secondary_y=True,
        showgrid=False
    )
    
    # Save chart to HTML
    output_html_path = os.path.join(outputs_dir, "enrollment_churn_trends.html")
    print(f"Saving chart to {output_html_path}...")
    fig.write_html(output_html_path, include_plotlyjs='cdn')
    print("Task 1.4 completed successfully.")

if __name__ == "__main__":
    main()
