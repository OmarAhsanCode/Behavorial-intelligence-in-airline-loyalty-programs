# Chart helpers and feature name mappings for translation

ARCHETYPE_COLORS = {
    "Crown Holders": "#1D9E75",
    "Ghost Members": "#D85A30",
    "Drifting Stars": "#BA7517",
    "Silent Investors": "#534AB7",
    "Budget Explorers": "#888780"
}

FEATURE_TRANSLATIONS = {
    "months_since_last_any_activity": "Months without any activity",
    "flights_last_12m": "Flights in past 12 months",
    "tenure_months": "Tenure in months",
    "flight_trend_ratio": "Flight trend ratio (recent vs historical)",
    "months_since_last_redemption": "Months since last point redemption",
    "flights_last_6m": "Flights in past 6 months",
    "points_trend_ratio": "Points accumulation trend ratio",
    "distance_per_flight": "Average distance per flight",
    "CLV": "Customer Lifetime Value",
    "months_since_last_flight": "Months since last flight",
    "distance_last_12m": "Total distance flown (past 12m)",
    "redemption_rate": "Redemption rate",
    "loyalty_card_ordinal": "Loyalty Card tier level",
    "points_accumulated_last_12m": "Points accumulated (past 12m)",
    "salary_imputed": "Estimated Salary",
    "education_ordinal": "Education level",
    "active_months_last_12m": "Active months (past 12m)",
    "zero_flight_months_last_12m": "Months with 0 flights (past 12m)",
    "points_redeemed_last_12m": "Points redeemed (past 12m)",
    "dollar_redeemed_last_12m": "Dollar value redeemed (past 12m)",
    "Enrollment Year": "Enrollment Year",
    "Gender_Female": "Gender (Female)",
    "Marital Status_Single": "Marital Status (Single)",
    "Marital Status_Married": "Marital Status (Married)"
}

def translate_feature(feature_name):
    """
    Translates database feature names to plain-English human-readable labels.
    """
    return FEATURE_TRANSLATIONS.get(feature_name, feature_name.replace("_", " ").title())

def apply_chart_style(fig):
    """
    Applies standard Plotly white template and default margins.
    """
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=40, b=40, l=40, r=40)
    )
    return fig
