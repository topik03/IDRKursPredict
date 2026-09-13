#!/usr/bin/env python3
"""
train_baseline.py - Train and evaluate baseline Naive Forecast model
Purpose: Establish baseline performance using Naive Forecast (tomorrow = today)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_directional_accuracy(y_true, y_pred):
    """Calculate directional accuracy (trend prediction)"""
    # Compare actual direction with predicted direction
    actual_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    
    # Count correct directions
    correct = np.sum((actual_diff * pred_diff) > 0)
    total = len(actual_diff)
    
    return (correct / total) * 100 if total > 0 else 0

def main():
    # Define file paths
    input_file = Path("data/processed/usd_idr_features.csv")
    
    print("="*60)
    print("USD/IDR Baseline Model Training")
    print("="*60)
    
    # Read features data
    print(f"\n[1] Reading feature data from: {input_file}")
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Please run feature_engineering.py first!")
        sys.exit(1)
    
    try:
        df = pd.read_csv(input_file)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"   ✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"ERROR reading file: {e}")
        sys.exit(1)
    
    # ============================================
    # TIME SERIES SPLIT
    # ============================================
    print(f"\n[2] Splitting data (time series split)...")
    
    train_end = pd.Timestamp('2022-12-31')
    val_end = pd.Timestamp('2024-12-31')
    
    # Training set: before 2023
    train_df = df[df['Date'] < '2023-01-01'].copy()
    
    # Validation set: 2023-01-01 to 2024-12-31
    val_df = df[(df['Date'] >= '2023-01-01') & (df['Date'] < '2025-01-01')].copy()
    
    # Test set: 2025 onwards
    test_df = df[df['Date'] >= '2025-01-01'].copy()
    
    print(f"   - Training set: {len(train_df)} rows ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"   - Validation set: {len(val_df)} rows ({val_df['Date'].min().date()} to {val_df['Date'].max().date()})")
    print(f"   - Test set: {len(test_df)} rows ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
    
    # ============================================
    # NAIVE FORECAST BASELINE
    # ============================================
    print(f"\n[3] Training Naive Forecast baseline model...")
    print("   Logic: Tomorrow's rate = Today's rate")
    
    # Prepare test data
    X_test = test_df.drop(columns=['Date', 'target_next_day'])
    y_test = test_df['target_next_day'].values
    
    # Naive forecast: predict today's Close as tomorrow's value
    # The model predicts: next day = current day (shift of 1)
    y_pred = test_df['Close'].values
    
    # Calculate metrics
    print(f"\n[4] Evaluating baseline on test set...")
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = calculate_mape(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    dir_acc = calculate_directional_accuracy(y_test, y_pred)
    
    print(f"\n" + "="*60)
    print("BASELINE MODEL RESULTS (Naive Forecast)")
    print("="*60)
    print(f"Test Period: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
    print(f"Test Samples: {len(y_test)}")
    print(f"\nModel Performance Metrics:")
    print(f"  {'Metric':<25} {'Value':>15}")
    print(f"  {'-'*25} {'-'*15}")
    print(f"  {'MAE (Mean Abs Error)':<25} {mae:>15,.2f} IDR")
    print(f"  {'RMSE (Root MSE)':<25} {rmse:>15,.2f} IDR")
    print(f"  {'MAPE (Mean Abs % Error)':<25} {mape:>15,.2f}%")
    print(f"  {'R² Score':<25} {r2:>15,.4f}")
    print(f"  {'Directional Accuracy':<25} {dir_acc:>15,.2f}%")
    
    # Comparison with actual values
    print(f"\n[5] Sample Predictions (first 10):")
    print(f"   {'Date':<12} {'Actual':>12} {'Predicted':>12} {'Error':>12}")
    print(f"   {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    sample_df = test_df.head(10).copy()
    for _, row in sample_df.iterrows():
        actual = row['target_next_day']
        pred = row['Close']
        error = actual - pred
        print(f"   {row['Date'].strftime('%Y-%m-%d'):<12} {actual:>12,.0f} {pred:>12,.0f} {error:>12,.0f}")
    
    print("\n" + "="*60)
    print("Baseline Training Complete!")
    print("="*60)
    print("""
Next steps:
- Run python src/train_xgboost.py to train XGBoost model
- Compare XGBoost results with baseline
""")

if __name__ == "__main__":
    main()
