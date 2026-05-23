Here is how to download, set up, and run the application starting from GitHub:

Step 1: Install Prerequisites (One-time only)
Before starting, make sure you have these two free tools installed on your computer:

Git: Download and install from git-scm.com.
Python (version 3.10 or higher): Download and install from python.org. Important: During installation, check the box that says "Add Python to PATH".
Step 2: Clone (Download) the Code from GitHub
Open Command Prompt or PowerShell on your computer.
Navigate to your Desktop or folder of choice:
bash


cd Desktop
Run the following command to download the project (replace the URL below with your actual GitHub repository URL):
bash


git clone https://github.com/your-username/your-repository-name.git
Move into the newly downloaded project directory:
bash


cd your-repository-name
Step 3: Set up the Python Environment
This ensures the app has its own isolated space with the correct package versions.

Create the virtual environment:
bash


python -m venv venv
Activate the virtual environment:
If using PowerShell:
powershell


.\venv\Scripts\Activate.ps1
If using Command Prompt:
cmd


.\venv\Scripts\activate.bat
(You should see (venv) appear at the front of your terminal line).
Step 4: Install the Required Packages
Install all libraries (Streamlit, Pandas, Plotly, etc.) by running:

bash


pip install -r airline_loyalty_app/requirements.txt
(This may take a minute or two to download and install everything).

Step 5: Launch the Application
Start the dashboard server by running:

bash


python -m streamlit run airline_loyalty_app/app.py
Your web browser will automatically open to http://localhost:8501 showing the dashboard.

To close the app later, go to your terminal window and press Ctrl + C.


User Manual: Loyalty Intelligence Hub
A Step-by-Step Guide for Marketing Managers & Business Analysts

Welcome to the Loyalty Intelligence Hub. This application is designed to help you monitor customer churn risk, understand segment behavior, and build high-ROI retention campaigns—all without needing any coding or technical knowledge.

Here is a simple, step-by-step guide on how to navigate and use the tool.

Step 1: Access the Hub (Landing Page)
When you first open the application, you will arrive at the main Landing Page. This page gives you a 60-second health check of your customer loyalty roster.

Review the Top Metrics: Check the four main cards:
Total Members: The total size of your loyalty cohort (11,076).
High-Risk Members: The number of members actively flagged as likely to churn.
CLV at Risk: The total financial Customer Lifetime Value (CLV) held by high-risk members.
Optimal Threshold: The risk score cutoff (0.35) used to catch at-risk members.
Examine the Archetype Map: Use the interactive scatter plot on the left to see how members group into different behavioral archetypes (like Crown Holders or Budget Explorers).
Compare Archetype Sizes: Look at the bar chart on the right to see which customer groups are the largest.
Step 2: Monitor Performance (Mission Control)
Click Mission Control in the left-hand sidebar to see where risk is concentrated.

Check Portfolio Health: Review the active vs. high-risk member counts in the top cards.
Inspect Risk by Segment: View the Risk Distribution Across Loyalty Archetypes stacked bar chart. This tells you which customer archetypes are carrying the most high-risk members.
Action: Click the Download Risk Distribution Data button under the chart to save this data for reports.
Explore the Risk Heatmap: Look at the map of Canada. Darker red provinces have a higher concentration of at-risk members. Hover over any province to see the exact numbers.
View Demographics: Scroll down to see the pie chart of customer types and a bar chart showing which segments hold the highest Customer Lifetime Value on average.
Step 3: Run an Individual Diagnostic (Member Risk Scanner)
Click Member Risk Scanner in the sidebar. This is where you can look up individual customers to understand exactly why they are disengaging.

Search a Member: Type a member's numeric Loyalty Number into the search bar at the top (e.g., search 100018 or 100194) and press Enter.
Review the Risk Card (Left): See their specific segment (e.g., Drifting Stars), their disengagement score progress bar, and their colored churn probability gauge.
Diagnose Churn Reasons (Right): Look at the horizontal bar chart under "Why is this member at risk?"
Red bars indicate behaviors that are pushing them to leave (e.g., not flying or redeeming points in months).
Green bars indicate protective behaviors keeping them loyal (e.g., long membership tenure).
Read the Recommended Playbook Action: Review the tailor-made offer, channel (email, push, direct mail), and cost per member suggested for this customer.
Flag for Action: Click the green Flag for Retention Campaign button to add them to your outreach list.
Action: Scroll to the bottom of the page to view your flagged list and click Download Campaign List to export a ready-to-use CSV for your email marketing tool.
Step 4: Build and Budget a Campaign (Campaign Builder)
Click Campaign Builder in the sidebar. This page is an interactive playground where you can price out retention campaigns and forecast your return on investment (ROI).

Select a Target Segment: Choose a specific customer archetype (or select "All High-Risk Members") in the dropdown on the left.
Select an Offer Type: Choose the offer (e.g., Bonus Miles, Upgrade Vouchers, Partner Rewards).
Set Your Budget: Adjust the Campaign Budget slider to allocate funds.
Set Expected Re-engagement: Slide the Assumed re-engagement rate to simulate response rates (e.g., set to 20% to assume 1 in 5 targeted members will respond).
Review the Forecast (Right):
Members Targeted: The number of high-risk customers who will receive the offer.
CLV Recovered: The total lifetime value of the customers you are expected to save.
ROI Multiple: The return ratio (e.g., a 5.0x multiple means returning 5forevery1 spent).
Examine the Value Flow: Look at the waterfall chart showing the financial transition from campaign cost to re-engaged member values.
Download Your Campaign Brief: Click Download Campaign Brief at the bottom to export your budget configurations and ROI metrics as a CSV summary sheet for leadership approval.
Pro-Tips for Business Leaders
Boardroom Disclaimer: When presenting high ROI numbers (e.g., 100x+), note that ROI figures assume direct campaign outreach costs only, and do not include offer redemption costs (like the cost of free flights) or staff execution hours.
Ghost Members Strategy: If you pull up members classified as Ghost Members, you will notice they have had zero activity for 6 years. Do not waste campaign budget on them. Instead, formally remove them from active loyalty rosters and redirect those outreach dollars toward Drifting Stars (once-frequent flyers who are currently slipping away).
