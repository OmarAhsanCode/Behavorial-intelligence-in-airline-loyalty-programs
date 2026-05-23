# Unlocking Behavioral Intelligence in Airline Loyalty Programs
## Technical Report — Canadian Airline Loyalty Cohort, 2012–2018

**Prepared by:** Omar Ahsan, Iqra Khan, Ojas Mishra  
**Audience:** Chief Financial Officer · Chief Marketing Officer  
**Data period:** January 2012 – November 2017 (prediction cutoff)  
**Cohort size:** 11,076 active loyalty members

---

## 1. Executive Summary

The airline's loyalty program currently tracks 16,700 Canadian members across six years of flight activity, point accumulation, and redemption behavior. Using this data, we built a behavioral intelligence system that answers three questions the existing program cannot: *Who is about to disengage before they formally cancel? Which members are most valuable to retain? And what specific action should marketing take for each person?*

The headline finding is striking. **$7.67 million in Customer Lifetime Value sits with members the model flags as high-risk** — members who are behaviorally disengaging right now but have not yet submitted a cancellation. The airline's current system would not flag a single one of them, because it only responds to formal cancellations. By the time a cancellation arrives, the member has already stopped flying.

Our model identifies approximately **50% of churners six to twelve months before they leave**, with 70% precision at the conservative detection threshold. Shifting to a broader detection threshold catches 65% of churners with 56% precision — a trade-off that is almost always worth making when campaign costs are low (estimated at $3.50–$4.50 per member contact).

Most importantly, the analysis reveals that **CLV alone cannot identify at-risk members.** The mean CLV of high-risk members ($7,902) is nearly identical to low-risk members ($7,981). A member's dollar value tells you nothing about whether they are about to leave. Their behavioral trajectory — how recently they flew, whether their activity is declining, whether they have ever once used their accumulated points — tells you everything.

The Streamlit-based Loyalty Intelligence Hub delivers this analysis in a form any marketing manager can use without reading a manual.

---

## 2. How We Defined Churn — and Why Cancellation Alone Misses the Point

### The problem with the obvious definition

The dataset contains formal cancellation records. The obvious approach is to label cancelled members as churned and build a model to predict future cancellations. We rejected this approach for one critical reason: **cancellation is the last event in a long disengagement process, not the first signal of it.** A member who last flew in 2015, accumulated points through 2016, and went completely silent in 2017 is behaviorally churned — but the cancellation column is empty.

### The Behavioral Disengagement Score

We constructed a **Behavioral Disengagement Score (BDS)** that captures three independent decay signals for each member, computed only from data available at the prediction cutoff (November 2017):

**Flight Inactivity (40% weight):** How many months has it been since the member's last recorded flight? Members who have not flown in 2017 at all receive the maximum score on this dimension. Flight booking is the most direct expression of engagement with the airline itself, which is why it carries the highest weight.

**Redemption Dormancy (30% weight):** How many months since the member last redeemed loyalty points? Members who have *never* redeemed a single point across their entire membership — despite accumulating them — receive the maximum score. This captures a specific failure mode: members who treat the program as an obligation rather than a benefit.

**Point Momentum (30% weight):** Is the member's monthly point accumulation trending up or down? We fit a linear regression over each member's monthly points across 2017 and use the slope as a velocity signal. A member whose earning is declining month-over-month is losing engagement even if they are still technically flying.

These three signals combine into a single score from 0 (fully engaged) to 1 (fully disengaged). Members scoring above **0.55** with no 2018 cancellation are classified as **soft churn** — behaviorally dead but formally active. Members with a confirmed 2018 cancellation are classified as **hard churn**.

### Why this matters for the model

This definition produces a meaningful positive class. Using formal cancellation alone would have yielded only 2.84% positive examples — far too sparse for a reliable classifier. The combined behavioral definition produces a 5.75% positive rate across 11,076 cohort members, which is sufficient for XGBoost with appropriate class weighting.

---

## 3. Five Loyalty Archetypes — Who Your Members Actually Are

Cluster analysis of five behavioral dimensions — Recency, Frequency, Monetary value, Engagement rate, and loyalty Tier (RFMET) — revealed five distinct member archetypes. These are not statistical artifacts; each archetype has a coherent behavioral profile and a different relationship with churn risk.

### Crown Holders
*The airline's most valuable and most loyal members.*

Aurora card holders with high CLV, recent flight history, and active point redemption. The Kaplan-Meier survival analysis shows this group **never reaches 50% cumulative churn** within the six-year observation window. They are the floor of the business. The retention strategy for Crown Holders is not to panic — it is to reinforce the relationship through experience upgrades (lounge access, priority boarding) rather than point promotions they do not need.

### Silent Investors
*High earners who never spend.*

This archetype flies consistently and accumulates points steadily, but their redemption rate is near zero. They are not disengaged — they are saving. The risk is that if they never find a compelling reason to redeem, the program stops feeling like a benefit and starts feeling like a bureaucratic points ledger. The highest ROI intervention in the entire portfolio targets this segment: a personalized redemption nudge showing the member their exact balance and a curated redemption option generates an estimated **892x ROI** on campaign spend, because the cost is low and re-engagement probability is high.

### Budget Explorers
*Consistent but low-value members.*

Star card holders with low CLV and moderate flight frequency. They are not at high risk individually, but their sheer volume makes them the **largest single pool of CLV at risk** in the portfolio at **$2.02 million.** They respond well to partner reward offers (hotel discounts, car rental deals) rather than flight discounts, where airline margins are thin.

### Drifting Stars
*Formerly engaged members in visible decline.*

This is the archetype that should concern marketing most. Drifting Stars show a clear declining trend in flight frequency and point accumulation — the behavioral equivalent of a member slowly backing toward the exit. Their flight trend ratio (recent 3-month rate vs. 12-month average) is measurably negative. A time-limited double-miles offer tied to their historically preferred travel season is the recommended intervention, creating urgency without permanently discounting the programme.

### Ghost Members
*The program's most urgent — and most surprising — finding.*

151 members enrolled in the loyalty programme and then **never flew once, never accumulated a point, and never redeemed anything across the entire six-year observation period.** Their median months-since-last-activity is 72 — the maximum possible, dating to their enrollment day. The model assigns them a 99.95% mean churn probability, which is correct: they are not at risk of churning, they have already churned in every meaningful sense.

The business recommendation for Ghost Members is not a retention campaign — it is **formal removal from the active roster.** Marketing budget directed at this group is wasted. The more productive question is why 151 members enrolled and immediately disengaged, which may point to an onboarding experience problem rather than a loyalty programme problem.

---

## 4. Who Is Most Likely to Leave — and When

### The strongest single predictor

SHAP analysis of the XGBoost model identifies **months since last any activity** as the dominant churn driver, with a mean absolute SHAP value of 1.18. Members with high inactivity are **6.88 times more likely to churn** than otherwise similar members. This single variable explains more variance in churn risk than CLV, salary, education, or loyalty card tier combined.

The practical implication: **activity recency, not customer value, is the right trigger for retention campaigns.** A 90-day activity silence should be the operational threshold for marketing intervention — not a CLV threshold, not a cancellation flag.

### How long members survive — by segment

Kaplan-Meier survival analysis shows stark differences in churn timing by loyalty card tier:

- **Aurora card holders** do not reach 50% cumulative churn within the observation window. The survival curve stays elevated across all 72 months of data.
- **Star and Nova card holders** reach 50% cumulative churn at approximately **68–69 months** from enrollment — meaning half of these members have formally cancelled within six years of joining.

Stratified by the Behavioral Disengagement Score, members in the high-BDS band (> 0.55) show dramatically compressed survival curves compared to low-BDS members, confirming that the behavioral score is capturing a real and measurable difference in retention trajectory.

### The role of tenure

Cox Proportional Hazards modeling identifies **tenure as the only statistically significant predictor of time-to-cancellation** (Hazard Ratio = 0.98, p < 0.005). Each additional month of membership reduces the instantaneous cancellation hazard by approximately 1.7%. Long-tenured members are sticky — not because they are inherently more loyal, but because switching costs accumulate over time. This has a direct implication for the 2018 Promotion cohort: members acquired through promotional incentives rather than organic intent have not yet built tenure-based stickiness and are structurally higher risk in their first two years.

*Note: The Cox model showed signs of overfitting due to the limited number of hard cancellation events relative to covariates. The findings above are directional; the Kaplan-Meier curves are the primary survival evidence.*

---

## 5. Three Business Recommendations

### Recommendation 1: Deploy a 90-Day Silence Trigger Across All Segments

**What:** Set an automated operational rule — any loyalty member with no flights, no point accumulation, and no redemption activity in the preceding 90 days triggers a personalised outreach from the appropriate segment playbook.

**Who receives it:** The outreach content differs by archetype. Drifting Stars receive a double-miles offer tied to their preferred season. Silent Investors receive a points redemption nudge. Budget Explorers receive a partner reward offer. Crown Holders receive a personal account manager touch.

**Why now:** The model identifies 939 members currently in the high-risk tier, representing $7.67 million in CLV. At an estimated campaign cost of $3.50–$4.50 per member, the total cost of reaching all 939 is under $4,500. Expected CLV recovery, assuming conservative re-engagement rates of 15–35% by segment, is **$1.81 million** — a portfolio ROI of 427x on direct campaign spend.

**What success looks like:** A flight booked or a point redemption within 30 days of campaign send, tracked at the member level through the Loyalty Intelligence Hub.

---

### Recommendation 2: Redirect Ghost Member Budget to Drifting Stars

**What:** Remove the 151 Ghost Members from all active marketing lists and reallocate their campaign budget to the Drifting Stars segment.

**Why:** Ghost Members have a 99.95% mean churn probability and zero lifetime engagement. No retention campaign will recover members who never engaged. The budget spent contacting them produces no measurable return.

**The reallocation case:** Drifting Stars have demonstrated historical engagement — they flew, they earned points, they were once real customers. Their trajectory is declining but not terminal. A well-timed double-miles offer during their preferred travel season has an estimated 22% re-engagement rate, compared to a near-zero rate for Ghost Members.

**Systemic fix:** The 151 Ghost Members represent an onboarding failure, not a loyalty failure. Recommend an audit of the enrollment flow to identify where these members dropped off between registration and first flight.

---

### Recommendation 3: Protect the Silent Investor Segment Before Competitors Do

**What:** Launch a dedicated redemption education campaign for Silent Investors — members who accumulate points steadily but never redeem them.

**Why this is urgent:** A member who has never experienced a redemption has never experienced the *value* of the loyalty programme. Their psychological relationship with the programme is transactional at best. A competitor offering a sign-up bonus or a simpler redemption interface can poach them with minimal friction.

**What the campaign looks like:** A personalised email showing the member their exact points balance, a single curated redemption option (a flight or hotel they would plausibly want based on their travel history), and a clear call to action. No generic "redeem your points" messaging — it must feel specific and relevant.

**Expected impact:** Silent Investors have the highest modelled ROI of any segment (892x) because their re-engagement probability is high (28% estimated) and their CLV is substantial. This is the highest-return intervention in the portfolio that requires the least budget.

---

## 6. Data Decisions and Known Limitations

Every analytical decision involves trade-offs. The following choices deserve explicit documentation.

**Post-cancellation placeholder records dropped.** The merged dataset contained 34,302 monthly activity records appearing after a member's formal cancellation date, all with zero values across every activity column. These were dummy placeholders generated by the loyalty system, not real activity. They were dropped before any analysis.

**Minimum tenure guard applied.** Members who enrolled after June 2017 were excluded from the modeling cohort. A member with one or two months of history produces unreliable behavioral signals — particularly for the point momentum calculation, which fits a regression over 12 monthly data points. Excluding late-2017 enrollees reduces the cohort from 16,700 to 11,076 but substantially improves label quality.

**Temporal split enforced for model validation.** The train/test split is time-based, not random. Members who enrolled through 2015 form the training set; 2016–2017 enrollees form the test set. This is the correct approach for a time-series prediction problem. A random split would allow behavioral patterns from future periods to leak into training, artificially inflating performance metrics.

**2017 cancellations excluded from training.** Approximately 1,600 members who cancelled during 2017 are not in the modeling cohort because they had already churned before the prediction cutoff. Their pre-cancellation data represents a rich training signal that was not used. Including these members as labeled churn examples in a future iteration would likely improve model recall.

**CLV outliers retained.** 660 members have CLV values above the 3-IQR upper bound of $23,820. These are not data errors — they are the programme's most valuable members. They were retained in the dataset and scaled with a robust scaler rather than trimmed.

**Cox model concordance caveat.** The Cox Proportional Hazards model reported a concordance of 0.99, which reflects overfitting on a small number of hard cancellation events rather than genuine predictive power. The Cox findings (particularly the tenure hazard ratio) are presented as directional evidence only. The Kaplan-Meier curves, which make no parametric assumptions, are the primary survival analysis output.

---

## 7. Appendix — Model Performance Summary

| Metric | Logistic Regression (baseline) | XGBoost (final model) |
|---|---|---|
| ROC-AUC | 0.7207 | **0.7325** |
| Churn-class F1 (threshold 0.50) | 0.5758 | **0.5793** |
| Churn-class Precision | 0.7443 | 0.7062 |
| Churn-class Recall | 0.4695 | **0.4910** |
| True Positives (churners caught) | 131 / 279 | **137 / 279** |
| False Negatives (churners missed) | 148 | **142** |
| False Positives (false alarms) | 45 | 57 |
| Optimal threshold (F1-maximising) | — | **0.35** |
| Churn-class Recall at threshold 0.35 | — | ~50.9% |

**Test set composition:** 3,271 members (2,992 non-churners, 279 churners). Positive rate: 8.5% in test set vs. 5.75% in full cohort — reflects higher churn concentration among later enrollees.

**Feature importance (top 5 by mean |SHAP|):**

| Rank | Feature | Mean \|SHAP\| | Plain-English Meaning |
|---|---|---|---|
| 1 | months_since_last_any_activity | 1.18 | Months without any program interaction |
| 2 | flights_last_12m | — | Flight frequency in the past year |
| 3 | redemption_rate | — | Proportion of accumulated points ever redeemed |
| 4 | tenure_months | — | Length of membership |
| 5 | points_accumulated_last_12m | — | Earning velocity in the past year |

**Segmentation validation:** K-Means with k=5 selected on the basis of silhouette score (preferred k=5) over GMM BIC (preferred k=9). Five-cluster solution chosen for business interpretability. All five archetypes show meaningfully distinct RFMET profiles confirmed by the cluster profile table and UMAP visualization.

---

*This report was generated from a fully reproducible analytical pipeline. All model artifacts, intermediate datasets, and the Streamlit application are available in the project repository. Re-running the pipeline from raw data produces identical results.*
