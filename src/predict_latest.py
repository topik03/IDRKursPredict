#!/usr/bin/env python3
"""
predict_latest.py - Make prediction for latest day
Purpose: Use trained XGBoost model to predict next day's USD/IDR rate
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib

def main():
    # Define file paths
    model_file = Path("models/xgboost_usd_idr_external.pkl")
    data_file = Path("data/processed/usd_idr_features_external.csv")
    output_file = Path("data/processed/latest_prediction.csv")
    
    print("="*60)
    print("Latest USD/IDR Prediction")
    print("="*60)
    
    # Check if model exists
    if not model_file.exists():
        print(f"ERROR: Model not found: {model_file}")
        print("Please run train_xgboost_external.py first!")
        sys.exit(1)
    
    # Check if data exists
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        print("Please run feature_engineering_external.py first!")
        sys.exit(1)
    
    # Load model
    print(f"\n[1] Loading model from: {model_file}")
    try:
        model = joblib.load(model_file)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)
    
    # Load data
    print(f"\n[2] Loading data from: {data_file}")
    try:
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"   ✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)
    
    # Get last row for prediction
    print(f"\n[3] Getting latest data...")
    last_row = df.iloc[-1]
    
    last_date = last_row['Date']
    last_close = last_row['Close']
    
    print(f"   - Last date: {last_date}")
    print(f"   - Last close: {last_close}")
    
    # Prepare features (exclude Date and target)
    exclude_cols = ['Date', 'target_next_day']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Get features for last row
    X_last = df[feature_cols].iloc[[-1]]
    
    # Make prediction
    print(f"\n[4] Making prediction...")
    try:
        predicted_next_day = model.predict(X_last)[0]
        print(f"   ✓ Prediction made")
    except Exception as e:
        print(f"ERROR making prediction: {e}")
        sys.exit(1)
    
    # Calculate difference
    difference = predicted_next_day - last_close
    
    # Determine direction
    direction = "Naik" if difference > 0 else "Turun"
    
    # Display results
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Last Date:        {last_date}")
    print(f"Last Close:       {last_close:,.2f} IDR")
    print(f"Predicted Next:   {predicted_next_day:,.2f} IDR")
    print(f"Difference:      {difference:+,.2f} IDR")
    print(f"Direction:       {direction}")
    
    # Save to CSV
    print(f"\n[5] Saving to: {output_file}")
    
    result_df = pd.DataFrame({
        'Date': [last_date],
        'last_close': [last_close],
        'predicted_next_day': [predicted_next_day],
        'difference': [difference],
        'direction': [direction]
    })
    
    result_df.to_csv(output_file, index=False)
    print("   ✓ Saved successfully")
    
    print("\n" + "="*60)
    print("Prediction Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
