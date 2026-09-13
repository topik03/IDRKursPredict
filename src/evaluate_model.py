#!/usr/bin/env python3
"""
evaluate_model.py - Evaluate XGBoost model performance
Purpose: Calculate and display detailed model evaluation metrics
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_directional_accuracy(y_true, y_pred):
    """Calculate directional accuracy (trend prediction)"""
    actual_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    correct = np.sum((actual_diff * pred_diff) > 0)
    total = len(actual_diff)
    return (correct / total) * 100 if total > 0 else 0

def calculate_mdae(y_true, y_pred):
    """Calculate Median Absolute Error"""
    return np.median(np.abs(y_true - y_pred))

def calculate_smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return np.mean(np.abs(y_true - y_pred) / denominator[mask]) * 100

def main():
    # Define file paths
    predictions_file = Path("data/processed/xgboost_predictions.csv")
    features_file = Path("data/processed/usd_idr_features.csv")
    
    print("="*60)
    print("USD/IDR Model Evaluation")
    print("="*60)
    
    # Read predictions
    print(f"\n[1] Reading predictions from: {predictions_file}")
    
    if not predictions_file.exists():
        print(f"ERROR: Predictions file not found: {predictions_file}")
        print("Please run train_xgboost.py first!")
        sys.exit(1)
    
    try:
        predictions_df = pd.read_csv(predictions_file)
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        print(f"   ✓ Loaded {len(predictions_df)} predictions")
    except Exception as e:
        print(f"ERROR reading file: {e}")
        sys.exit(1)
    
    # Read features to get full data
    print(f"\n[2] Reading features data...")
    
    if not features_file.exists():
        print(f"WARNING: Features file not found, skipping some analysis")
    else:
        features_df = pd.read_csv(features_file)
        features_df['Date'] = pd.to_datetime(features_df['Date'])
    
    # Extract actual and predicted values
    y_actual = predictions_df['Actual'].values
    y_pred = predictions_df['Predicted'].values
    
    # ============================================
    # CALCULATE METRICS
    # ============================================
    print(f"\n[3] Calculating evaluation metrics...")
    
    # Basic metrics
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mape = calculate_mape(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    dir_acc = calculate_directional_accuracy(y_actual, y_pred)
    
    # Additional metrics
    mdae = calculate_mdae(y_actual, y_pred)
    smape = calculate_smape(y_actual, y_pred)
    
    # Error statistics
    errors = y_actual - y_pred
    abs_errors = np.abs(errors)
    pct_errors = (errors / y_actual) * 100
    
    # ============================================
    # DISPLAY RESULTS
    # ============================================
    print(f"\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"\nEvaluation Period:")
    print(f"  Start Date: {predictions_df['Date'].min().strftime('%Y-%m-%d')}")
    print(f"  End Date: {predictions_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Total Predictions: {len(predictions_df)}")
    
    print(f"\n┌{'─'*50}┐")
    print(f"│{'METRIC':<30} {'VALUE':>18}│")
    print(f"├{'─'*50}┤")
    print(f"│{'MAE (Mean Absolute Error)':<30} {mae:>15,.2f} IDR│")
    print(f"│{'RMSE (Root Mean Squared Error)':<30} {rmse:>15,.2f} IDR│")
    print(f"│{'MAPE (Mean Absolute % Error)':<30} {mape:>15,.2f}%│")
    print(f"│{'SMAPE (Symmetric MAPE)':<30} {smape:>15,.2f}%│")
    print(f"│{'MDAE (Median Absolute Error)':<30} {mdae:>15,.2f} IDR│")
    print(f"│{'R² Score':<30} {r2:>15,.4f}   │")
    print(f"│{'Directional Accuracy':<30} {dir_acc:>15,.2f}%│")
    print(f"└{'─'*50}┘")
    
    # Error distribution
    print(f"\n[4] Error Distribution:")
    print(f"  Mean Error: {np.mean(errors):,.2f} IDR")
    print(f"  Std Error: {np.std(errors):,.2f} IDR")
    print(f"  Min Error: {np.min(errors):,.2f} IDR")
    print(f"  Max Error: {np.max(errors):,.2f} IDR")
    print(f"  Median Error: {np.median(errors):,.2f} IDR")
    
    # Percentile distribution
    print(f"\n[5] Error Percentiles:")
    print(f"  25th percentile: {np.percentile(abs_errors, 25):,.2f} IDR")
    print(f"  50th percentile (median): {np.percentile(abs_errors, 50):,.2f} IDR")
    print(f"  75th percentile: {np.percentile(abs_errors, 75):,.2f} IDR")
    print(f"  90th percentile: {np.percentile(abs_errors, 90):,.2f} IDR")
    print(f"  95th percentile: {np.percentile(abs_errors, 95):,.2f} IDR")
    print(f"  99th percentile: {np.percentile(abs_errors, 99):,.2f} IDR")
    
    # Accuracy buckets
    print(f"\n[6] Prediction Accuracy Buckets:")
    
    pct_within_1pct = (predictions_df['Pct_Error'] < 1).sum() / len(predictions_df) * 100
    pct_within_2pct = (predictions_df['Pct_Error'] < 2).sum() / len(predictions_df) * 100
    pct_within_5pct = (predictions_df['Pct_Error'] < 5).sum() / len(predictions_df) * 100
    pct_within_10pct = (predictions_df['Pct_Error'] < 10).sum() / len(predictions_df) * 100
    
    print(f"  Within 1% error: {pct_within_1pct:.2f}%")
    print(f"  Within 2% error: {pct_within_2pct:.2f}%")
    print(f"  Within 5% error: {pct_within_5pct:.2f}%")
    print(f"  Within 10% error: {pct_within_10pct:.2f}%")
    
    # Period comparison
    print(f"\n[7] Monthly Performance:")
    
    predictions_df['Month'] = predictions_df['Date'].dt.to_period('M')
    monthly_stats = predictions_df.groupby('Month').agg({
        'Pct_Error': 'mean',
        'Abs_Error': 'mean'
    }).round(2)
    
    print(f"  {'Month':<10} {'Avg MAPE':>12} {'Avg MAE':>15}")
    print(f"  {'-'*10} {'-'*12} {'-'*15}")
    
    for month, row in monthly_stats.iterrows():
        print(f"  {str(month):<10} {row['Pct_Error']:>12,.2f}% {row['Abs_Error']:>15,.0f}")
    
    # Baseline comparison
    print(f"\n[8] Comparison with Baseline (Naive Forecast):")
    print(f"  Baseline MAPE: ~0.00% (by definition)")
    print(f"  XGBoost MAPE: {mape:.2f}%")
    print(f"  Improvement: N/A (same-day prediction)")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print("""
Next steps:
- Run python src/visualize_result.py for visualizations
""")

if __name__ == "__main__":
    main()
