import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def main():
    print("Starting Task 1.5: Generating flight activity heatmap...")
    
    # Paths
    data_dir = "."
    outputs_dir = os.path.join(data_dir, "outputs")
    
    # Load merged dataset
    merged_path = os.path.join(outputs_dir, "merged_loyalty_activity.csv")
    print(f"Loading merged dataset from {merged_path}...")
    df = pd.read_csv(merged_path)
    
    # Filter out post-cancellation placeholder rows
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
    print(f"Cleaned dataset shape: {df_clean.shape}")
    
    # Group by Year and Month to calculate average Total Flights per member
    print("Calculating average Total Flights per member by Year x Month...")
    monthly_avg = df_clean.groupby(['Year', 'Month'])['Total Flights'].mean().reset_index()
    
    # Create the pivot table (Year as row index, Month as column index)
    pivot_df = monthly_avg.pivot(index='Year', columns='Month', values='Total Flights')
    
    # Format Year and Month labels
    years = [str(y) for y in pivot_df.index]
    months_num = list(pivot_df.columns)
    
    # Season groupings map
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    
    # Create x-axis labels with month names and seasons as HTML sub-labels for Plotly
    x_labels = [f"{month_names[m]}<br><sup>{season_map[m]}</sup>" for m in months_num]
    
    # Convert z values to a 2D numpy array/list for Plotly
    z_values = pivot_df.values
    
    # Print the pivot table to console for debugging/walkthrough
    print("\n--- PIVOT TABLE: AVERAGE TOTAL FLIGHTS PER MEMBER ---")
    print(pivot_df.round(4))
    
    # Plotly heatmap
    print("\nGenerating Plotly heatmap...")
    
    # Premium color scale: a smooth, sequential sunset gradient
    # Deep indigo/blue -> orange -> light yellow
    colorscale = [
        [0.0, 'rgb(30, 34, 170)'],   # Deep blue
        [0.35, 'rgb(106, 51, 163)'],  # Deep purple
        [0.7, 'rgb(220, 80, 100)'],   # Coral/orange
        [1.0, 'rgb(254, 217, 118)']   # Bright warm gold
    ]
    
    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=years,
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text="Average Flights", font=dict(family="Inter", size=12)),
                thickness=15,
                len=0.7
            ),
            hoverongaps=False,
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Avg Flights: %{z:.4f}<extra></extra>"
        )
    )
    
    # Add text annotations in each cell
    annotations = []
    # Find min/max for color threshold
    z_min = z_values.min()
    z_max = z_values.max()
    z_mid = (z_min + z_max) / 2.0
    
    for y_idx, year in enumerate(years):
        for x_idx, x_lbl in enumerate(x_labels):
            val = z_values[y_idx][x_idx]
            # Use light text for dark cells and dark text for light cells
            text_color = "white" if val < z_mid else "rgb(30, 30, 45)"
            annotations.append(
                dict(
                    x=x_lbl,
                    y=year,
                    text=f"{val:.3f}",
                    showarrow=False,
                    font=dict(family="Inter, sans-serif", size=14, color=text_color, weight="bold")
                )
            )
            
    fig.update_layout(
        title=dict(
            text="Flight Activity Heatmap (Average Flights per Member by Year & Month)",
            font=dict(size=18, family="Outfit, Inter, sans-serif", color="#2B2D42"),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text="Month & Season", font=dict(family="Inter, sans-serif", size=14)),
            tickmode="array",
            tickvals=x_labels,
            ticktext=x_labels,
            constrain="domain"
        ),
        yaxis=dict(
            title=dict(text="Year", font=dict(family="Inter, sans-serif", size=14)),
            tickmode="array",
            tickvals=years,
            ticktext=years,
            autorange="reversed" # put 2017 on top, 2018 below (or vice versa, reversed places 2017 at the bottom or top depending on sorting)
        ),
        annotations=annotations,
        margin=dict(l=60, r=60, t=80, b=80),
        plot_bgcolor='white',
        paper_bgcolor='#FFFFFF',
    )
    
    # Save to outputs/flight_activity_heatmap.html
    output_html_path = os.path.join(outputs_dir, "flight_activity_heatmap.html")
    print(f"Saving heatmap to {output_html_path}...")
    fig.write_html(output_html_path, include_plotlyjs='cdn')
    print("Task 1.5 completed successfully.")

if __name__ == "__main__":
    main()
