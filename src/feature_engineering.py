#!/usr/bin/env python3
"""
feature_engineering.py - Create features for USD/IDR prediction
Purpose: Generate lag features, rolling statistics, calendar features, and target variable
"""

import pandas as pd
import sys
from pathlib import Path

def create_features(df):
    """Create all feature types"""
    
    # ============================================
    # LAG FEATURES
    # ============================================
    print("\n[2] Creating lag features...")
    
    lag_periods = [1, 2, 3, 7, 14, 30]
    for lag in lag_periods:
        df[f'lag_{lag}'] = df['Close'].shift(lag)
        print(f"   ✓ Created lag_{lag}")
    
    # ============================================
    # MOVING AVERAGE FEATURES
    # ============================================
    print("\n[3] Creating moving average features...")
    
    ma_windows = [7, 14, 30, 60]
    for window in ma_windows:
        df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
        print(f"   ✓ Created ma_{window}")
    
    # ============================================
    # ROLLING STANDARD DEVIATION FEATURES
    # ============================================
    print("\n[4] Creating rolling standard deviation features...")
    
    std_windows = [7, 14, 30]
    for window in std_windows:
        df[f'std_{window}'] = df['Close'].rolling(window=window).std()
        print(f"   ✓ Created std_{window}")
    
    # ============================================
    # RETURN FEATURES
    # ============================================
    print("\n[5] Creating return features...")
    
    # 1-day return
    df['return_1'] = df['Close'].pct_change(1)
    print(f"   ✓ Created return_1")
    
    # ============================================
    # CALENDAR FEATURES
    # ============================================
    print("\n[6] Creating calendar features...")
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['Date'].dt.dayofweek
    print(f"   ✓ Created day_of_week")
    
    # Month (1-12)
    df['month'] = df['Date'].dt.month
    print(f"   ✓ Created month")
    
    # Quarter (1-4)
    df['quarter'] = df['Date'].dt.quarter
    print(f"   ✓ Created quarter")
    
    # Is month end (boolean)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    print(f"   ✓ Created is_month_end")
    
    # Is month start
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    print(f"   ✓ Created is_month_start")
    
    # Day of month
    df['day_of_month'] = df['Date'].dt.day
    print(f"   ✓ Created day_of_month")
    
    # Week of year
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    print(f"   ✓ Created week_of_year")
    
    # ============================================
    # TARGET VARIABLE
    # ============================================
    print("\n[7] Creating target variable...")
    
    # Target: Next day's Close price
    df['target_next_day'] = df['Close'].shift(-1)
    print(f"   ✓ Created target_next_day")
    
    return df

def main():
    # Define file paths
    input_file = Path("data/processed/usd_idr_clean.csv")
    output_file = Path("data/processed/usd_idr_features.csv")
    
    # Create processed directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("USD/IDR Feature Engineering Pipeline")
    print("="*60)
    
    # Read cleaned data
    print(f"\n[1] Reading cleaned data from: {input_file}")
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Please run clean_data.py first!")
        sys.exit(1)
    
    try:
        df = pd.read_csv(input_file)
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"   ✓ Loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        print(f"ERROR reading file: {e}")
        sys.exit(1)
    
    print(f"   - Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Create features
    df = create_features(df)
    
    # ============================================
    # HANDLE MISSING VALUES
    # ============================================
    print("\n[8] Handling missing values...")
    
    # Count missing before
    missing_before = df.isnull().sum().sum()
    print(f"   - Total missing values before: {missing_before}")
    
    # Remove rows with missing values (from lag/rolling/target features)
    # Keep only rows where target_next_day is available
    df_clean = df.dropna(subset=['target_next_day'])
    
    # Also drop rows with too many NaN in features
    rows_removed = len(df) - len(df_clean)
    print(f"   - Rows removed due to missing values: {rows_removed}")
    
    df = df_clean.reset_index(drop=True)
    
    missing_after = df.isnull().sum().sum()
    print(f"   - Total missing values after: {missing_after}")
    
    # ============================================
    # SAVE FEATURES
    # ============================================
    print(f"\n[9] Final feature data:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Save to CSV
    print(f"\n[10] Saving features to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"   ✓ Saved successfully")
    
    # Feature summary
    print(f"\n[11] Feature Summary:")
    print(f"   Total features: {len(df.columns)}")
    feature_cols = [c for c in df.columns if c not in ['Date', 'target_next_day']]
    print(f"   Feature columns: {len(feature_cols)}")
    
    # List features by category
    print(f"\n[12] Features by category:")
    print(f"   - Original: Close, Open, High, Low, Change %")
    print(f"   - Lag features: lag_1, lag_2, lag_3, lag_7, lag_14, lag_30")
    print(f"   - Moving averages: ma_7, ma_14, ma_30, ma_60")
    print(f"   - Rolling std: std_7, std_14, std_30")
    print(f"   - Returns: return_1")
    print(f"   - Calendar: day_of_week, month, quarter, is_month_end, is_month_start, day_of_month, week_of_year")
    print(f"   - Target: target_next_day")
    
    # Sample data
    print(f"\n[13] Sample of feature data (first 5 rows):")
    display_cols = ['Date', 'Close', 'lag_1', 'ma_7', 'return_1', 'day_of_week', 'target_next_day']
    print(df[display_cols].head().to_string())
    
    print("\n" + "="*60)
    print("Feature Engineering Complete!")
    print("="*60)
    print("""
Next steps:
- Run python src/train_baseline.py to train baseline model
- Run python src/train_xgboost.py to train XGBoost model
""")

if __name__ == "__main__":
    main()
