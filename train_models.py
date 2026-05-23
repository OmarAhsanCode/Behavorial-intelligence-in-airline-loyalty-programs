"""
train_models.py

Task 3.1 — Time-based train/test split
Task 3.2 — Logistic Regression baseline
Task 3.3 — XGBoost churn classifier

Justification for Temporal Split (Time-based):
1. Avoids target leakage: Random splitting would mix customers enrolled at different times. 
   Since churn behaviors and loyalty engagement change over time, using future customers 
   to predict past customers' behavior violates chronological progression.
2. Simulates real-world deployment: In production, the model will be trained on historical 
   members (e.g. those who enrolled in 2015 or earlier) and used to predict churn for 
   subsequent enrollment cohorts (e.g. those who enrolled in 2016-2017). A time-based 
   split mirrors this deployment pattern, providing a realistic estimate of out-of-time generalizability.
3. Protects the "no future information" constraint.
"""

import os
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

def plot_custom_confusion_matrix(cm, model_name, filepath):
    """
    Plots a highly stylized, interactive Plotly heatmap representing the confusion matrix.
    """
    z = cm
    x = ['Predicted: Not Churned (0)', 'Predicted: Churned (1)']
    y = ['Actual: Not Churned (0)', 'Actual: Churned (1)']
    
    # Calculate percentages for annotations
    total = z.sum()
    annot = [
        [f"{z[0][0]}<br>({z[0][0]/total*100:.1f}%)", f"{z[0][1]}<br>({z[0][1]/total*100:.1f}%)"],
        [f"{z[1][0]}<br>({z[1][0]/total*100:.1f}%)", f"{z[1][1]}<br>({z[1][1]/total*100:.1f}%)"]
    ]
    
    # Premium color theme (Indigo/Teal blend)
    colorscale = [
        [0.0, '#f8f9fa'],
        [0.1, '#e3f2fd'],
        [0.5, '#42a5f5'],
        [1.0, '#1565c0']
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        colorscale=colorscale,
        showscale=True,
        text=annot,
        texttemplate="%{text}",
        textfont={"size": 15, "family": "Inter, sans-serif", "color": "black"},
        hoverinfo="z"
    ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>{model_name} Confusion Matrix</b>',
            font=dict(family="Outfit, Inter, sans-serif", size=20, color="#1e293b")
        ),
        xaxis=dict(tickfont=dict(family="Inter, sans-serif", size=12)),
        yaxis=dict(tickfont=dict(family="Inter, sans-serif", size=12), autorange="reversed"),
        width=550,
        height=500,
        margin=dict(t=80, b=40, l=40, r=40),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.write_html(filepath)
    print(f"Confusion matrix for {model_name} saved to {filepath}")

def main():
    print("Starting Phase 4: Model Training and Evaluation...")
    
    # Paths
    data_dir = "."
    outputs_dir = os.path.join(data_dir, "outputs")
    models_dir = os.path.join(outputs_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Load files
    print("Loading modeling data...")
    cohort_path = os.path.join(outputs_dir, "churn_modeling_cohort.csv")
    model_ready_path = os.path.join(outputs_dir, "model_ready_data.csv")
    feature_names_path = os.path.join(outputs_dir, "feature_names.txt")
    
    cohort_df = pd.read_csv(cohort_path)
    model_df = pd.read_csv(model_ready_path)
    
    with open(feature_names_path) as f:
        feature_names = [line.strip() for line in f if line.strip()]
        
    # Map Enrollment Year to model_df for temporal split
    print("Mapping Enrollment Year for time-based split...")
    enrollment_mapping = cohort_df.set_index('Loyalty Number')['Enrollment Year']
    model_df['Enrollment Year'] = model_df['Loyalty Number'].map(enrollment_mapping)
    
    # One-hot encode pref_season
    print("One-hot encoding categorical features...")
    model_df_encoded = pd.get_dummies(model_df, columns=['pref_season'], prefix='pref_season', dtype=int)
    
    # Retrieve new dummy columns for pref_season
    pref_season_dummies = [col for col in model_df_encoded.columns if col.startswith('pref_season_')]
    
    # Construct final feature column list
    X_features = []
    for col in feature_names:
        if col == 'pref_season':
            X_features.extend(pref_season_dummies)
        else:
            X_features.append(col)
            
    print(f"Constructed model features list. Total features: {len(X_features)}")
    
    # --- Task 3.1: Time-based Train/Test Split ---
    print("\n--- Split Evaluation (Task 3.1) ---")
    train_mask = model_df_encoded['Enrollment Year'] <= 2015
    test_mask = model_df_encoded['Enrollment Year'].isin([2016, 2017])
    
    X_train = model_df_encoded.loc[train_mask, X_features].copy()
    y_train = model_df_encoded.loc[train_mask, 'Churned'].copy()
    
    X_test = model_df_encoded.loc[test_mask, X_features].copy()
    y_test = model_df_encoded.loc[test_mask, 'Churned'].copy()
    
    # Calculate churn rates
    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()
    
    print(f"Training set size: {X_train.shape[0]} members (Enrolled <= 2015)")
    print(f"Testing set size:  {X_test.shape[0]} members (Enrolled 2016-2017)")
    print(f"Training set churn rate: {train_churn_rate * 100:.2f}% ({y_train.sum()} churned)")
    print(f"Testing set churn rate:  {test_churn_rate * 100:.2f}% ({y_test.sum()} churned)")
    
    # Confirm churn rates are within 3 percentage points
    rate_diff = abs(train_churn_rate - test_churn_rate)
    print(f"Churn rate difference: {rate_diff * 100:.2f} percentage points")
    assert rate_diff <= 0.03, \
        f"Assertion Failed: Churn rate difference {rate_diff*100:.2f}% exceeds 3 percentage points!"
    print("Assertion Passed: Churn rates in train/test splits are similar.")
    
    # --- Task 3.2: Logistic Regression Baseline ---
    print("\n--- Logistic Regression Baseline (Task 3.2) ---")
    # Preprocess with StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    lr_prec = precision_score(y_test, y_pred_lr)
    lr_rec = recall_score(y_test, y_pred_lr)
    lr_f1 = f1_score(y_test, y_pred_lr)
    lr_auc = roc_auc_score(y_test, y_prob_lr)
    
    print("Logistic Regression Baseline Results:")
    print(f"  Precision (Class 1): {lr_prec:.4f}")
    print(f"  Recall (Class 1):    {lr_rec:.4f}")
    print(f"  F1-Score (Class 1):  {lr_f1:.4f}")
    print(f"  ROC-AUC:             {lr_auc:.4f}")
    
    # Save Confusion Matrix
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    plot_custom_confusion_matrix(cm_lr, "Logistic Regression Baseline", os.path.join(outputs_dir, "logistic_confusion_matrix.html"))
    
    # Save Model
    lr_model_path = os.path.join(models_dir, "logistic_model.pkl")
    with open(lr_model_path, 'wb') as f:
        pickle.dump(lr, f)
    print(f"Logistic model saved to {lr_model_path}")
    
    # --- Task 3.3: XGBoost Churn Classifier ---
    print("\n--- XGBoost Classifier (Task 3.3) ---")
    # Compute scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.4f}")
    
    # Define hyperparameter grid for GridSearchCV
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    print("Setting up GridSearchCV with 5-fold cross-validation...")
    xgb_base = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    # Train XGBoost
    print("Tuning hyperparameters on training data (this may take a few seconds)...")
    grid_search.fit(X_train, y_train)
    
    best_xgb = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best hyperparameters found: {best_params}")
    
    # Evaluate best model
    y_pred_xgb = best_xgb.predict(X_test)
    y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]
    
    xgb_prec = precision_score(y_test, y_pred_xgb)
    xgb_rec = recall_score(y_test, y_pred_xgb)
    xgb_f1 = f1_score(y_test, y_pred_xgb)
    xgb_auc = roc_auc_score(y_test, y_prob_xgb)
    
    print("\nXGBoost Classifier Results (Best Estimator):")
    print(f"  Precision (Class 1): {xgb_prec:.4f}")
    print(f"  Recall (Class 1):    {xgb_rec:.4f}")
    print(f"  F1-Score (Class 1):  {xgb_f1:.4f}")
    print(f"  ROC-AUC:             {xgb_auc:.4f}")
    
    # Save XGBoost Model
    xgb_model_path = os.path.join(models_dir, "xgboost_model.pkl")
    with open(xgb_model_path, 'wb') as f:
        pickle.dump(best_xgb, f)
    print(f"XGBoost model saved to {xgb_model_path}")
    
    # Save XGBoost Confusion Matrix
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    plot_custom_confusion_matrix(cm_xgb, "XGBoost Classifier", os.path.join(outputs_dir, "xgboost_confusion_matrix.html"))
    
    # --- Overlay ROC Curves ---
    print("\nGenerating overlaid ROC curves comparison...")
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    auc_lr = auc(fpr_lr, tpr_lr)
    
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    auc_xgb = auc(fpr_xgb, tpr_xgb)
    
    fig_roc = go.Figure()
    
    # Reference Line
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='grey', width=1.5, dash='dash'),
        name='Random Guess (AUC = 0.50)',
        showlegend=True
    ))
    
    # Logistic Regression Curve
    fig_roc.add_trace(go.Scatter(
        x=fpr_lr, y=tpr_lr,
        mode='lines',
        line=dict(color='#f59e0b', width=2.5),
        name=f'Logistic Regression (AUC = {auc_lr:.4f})'
    ))
    
    # XGBoost Curve
    fig_roc.add_trace(go.Scatter(
        x=fpr_xgb, y=tpr_xgb,
        mode='lines',
        line=dict(color='#0f172a', width=3),
        name=f'XGBoost Classifier (AUC = {auc_xgb:.4f})'
    ))
    
    fig_roc.update_layout(
        title=dict(
            text='<b>ROC Curves Comparison</b>',
            font=dict(family="Outfit, Inter, sans-serif", size=22, color="#1e293b")
        ),
        xaxis=dict(
            title='False Positive Rate (1 - Specificity)',
            gridcolor='#f1f5f9',
            zeroline=False,
            tickfont=dict(family="Inter, sans-serif", size=12)
        ),
        yaxis=dict(
            title='True Positive Rate (Sensitivity)',
            gridcolor='#f1f5f9',
            zeroline=False,
            tickfont=dict(family="Inter, sans-serif", size=12)
        ),
        legend=dict(
            x=0.55, y=0.15,
            bordercolor='#cbd5e1',
            borderwidth=1,
            font=dict(family="Inter, sans-serif", size=12)
        ),
        width=700,
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=100, b=60, l=60, r=40)
    )
    
    roc_plot_path = os.path.join(outputs_dir, "roc_curve_comparison.html")
    fig_roc.write_html(roc_plot_path)
    print(f"Overlaid ROC curve comparison saved to {roc_plot_path}")
    
    print("\nPhase 4 (Tasks 3.1 - 3.3) completed successfully.")

if __name__ == "__main__":
    main()
