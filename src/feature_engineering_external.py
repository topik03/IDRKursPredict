#!/usr/bin/env python3
"""
feature_engineering_external.py - Create features with external data
Purpose: Generate features using USD/IDR and external market indicators
"""

import pandas as pd
import sys
from pathlib import Path

def create_usd_idr_features(df):
    """Create features from USD/IDR data"""
    
    print("\n[2] Creating USD/IDR features...")
    
    # ============================================
    # LAG FEATURES (USD/IDR)
    # ============================================
    lag_periods = [1, 2, 3, 7, 14, 30]
    for lag in lag_periods:
        df[f'lag_{lag}'] = df['Close'].shift(lag)
        print(f"   ✓ Created lag_{lag}")
    
    # ============================================
    # MOVING AVERAGE FEATURES
    # ============================================
    ma_windows = [7, 14, 30, 60]
    for window in ma_windows:
        df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
        print(f"   ✓ Created ma_{window}")
    
    # ============================================
    # ROLLING STANDARD DEVIATION FEATURES
    # ============================================
    std_windows = [7, 14, 30]
    for window in std_windows:
        df[f'std_{window}'] = df['Close'].rolling(window=window).std()
        print(f"   ✓ Created std_{window}")
    
    # ============================================
    # RETURN FEATURES (USD/IDR)
    # ============================================
    df['return_1'] = df['Close'].pct_change(1)
    print(f"   ✓ Created return_1")
    
    # ============================================
    # MOMENTUM FEATURES (RSI & MACD)
    # ============================================
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    print(f"   ✓ Created rsi_14")
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    print(f"   ✓ Created macd & macd_signal")
    
    return df

def create_calendar_features(df):
    """Create calendar features"""
    
    print("\n[3] Creating calendar features...")
    
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
    
    return df

def create_external_features(df):
    """Create features from external market data"""
    
    print("\n[4] Creating external market features...")
    
    # List of external columns
    external_cols = ['dxy_close', 'gold_close', 'oil_close', 'vix_close', 
                 'ihsg_close', 'sp500_close', 'us10y_close']
    
    # Use external data as-is (same day 't' - no leakage)
    print(f"   ✓ Using external Close prices from day t (no leakage)")
    
    # Create return features for external data
    # Return = (price_t - price_{t-1}) / price_{t-1}
    return_mappings = {
        'dxy_return': 'dxy_close',
        'gold_return': 'gold_close',
        'oil_return': 'oil_close',
        'vix_return': 'vix_close',
        'ihsg_return': 'ihsg_close',
        'sp500_return': 'sp500_close',
        'us10y_return': 'us10y_close'
    }
    
    for return_col, close_col in return_mappings.items():
        if close_col in df.columns:
            df[return_col] = df[close_col].pct_change(1)
            print(f"   ✓ Created {return_col}")
            
    # ============================================
    # CROSS-ASSET CORRELATION
    # ============================================
    # Requires return_1 to be created first in create_usd_idr_features
    if 'dxy_return' in df.columns and 'return_1' in df.columns:
        df['corr_dxy_30'] = df['return_1'].rolling(window=30).corr(df['dxy_return'])
        print(f"   ✓ Created corr_dxy_30")
        
    if 'gold_return' in df.columns and 'return_1' in df.columns:
        df['corr_gold_30'] = df['return_1'].rolling(window=30).corr(df['gold_return'])
        print(f"   ✓ Created corr_gold_30")
    
    return df

def create_target(df):
    """Create target variable"""
    
    print("\n[5] Creating target variable...")
    
    # Target: Next day's Close price
    df['target_next_day'] = df['Close'].shift(-1)
    print(f"   ✓ Created target_next_day")
    
    return df

def main():
    # Define file paths
    input_file = Path("data/processed/usd_idr_with_external.csv")
    output_file = Path("data/processed/usd_idr_features_external.csv")
    
    # Create processed directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Feature Engineering with External Data Pipeline")
    print("="*60)
    
    # Read data with external features
    print(f"\n[1] Reading data from: {input_file}")
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Please run merge_external_features.py first!")
        sys.exit(1)
    
    try:
        df = pd.read_csv(input_file)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"   ✓ Loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        print(f"ERROR reading file: {e}")
        sys.exit(1)
    
    print(f"   - Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Create features
    df = create_usd_idr_features(df)
    df = create_calendar_features(df)
    df = create_external_features(df)
    df = create_target(df)
    
    # ============================================
    # HANDLE MISSING VALUES
    # ============================================
    print("\n[6] Handling missing values...")
    
    # Count missing before
    missing_before = df.isnull().sum().sum()
    print(f"   - Total missing values before: {missing_before}")
    
    # Remove rows with missing target (last row(s))
    df_clean = df.dropna(subset=['target_next_day'])
    
    # Also drop rows where essential features are NaN
    # (from lag/rolling features in first ~60 rows)
    df_clean = df_clean.dropna(subset=['lag_1', 'ma_7'])
    
    rows_removed = len(df) - len(df_clean)
    print(f"   - Rows removed due to missing values: {rows_removed}")
    
    df = df_clean.reset_index(drop=True)
    
    missing_after = df.isnull().sum().sum()
    print(f"   - Total missing values after: {missing_after}")
    
    # ============================================
    # SAVE FEATURES
    # ============================================
    print(f"\n[7] Final feature data:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Save to CSV
    print(f"\n[8] Saving features to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"   ✓ Saved successfully")
    
    # Feature summary
    print(f"\n[9] Feature Summary:")
    print(f"   Total columns: {len(df.columns)}")
    
    # List features by category
    print(f"\n[10] Features by category:")
    print(f"   - Original USD/IDR: Date, Close, Open, High, Low, Change %")
    print(f"   - External Close: dxy_close, gold_close, oil_close, vix_close, ihsg_close, sp500_close, us10y_close")
    print(f"   - Lag features: lag_1, lag_2, lag_3, lag_7, lag_14, lag_30")
    print(f"   - Moving averages: ma_7, ma_14, ma_30, ma_60")
    print(f"   - Rolling std: std_7, std_14, std_30")
    print(f"   - Returns: return_1, dxy_return, gold_return, oil_return, vix_return, ihsg_return, sp500_return, us10y_return")
    print(f"   - Calendar: day_of_week, month, quarter, is_month_end, is_month_start, day_of_month, week_of_year")
    print(f"   - Target: target_next_day")
    
    # Sample data
    print(f"\n[11] Sample of feature data (first 5 rows):")
    display_cols = ['Date', 'Close', 'dxy_close', 'lag_1', 'ma_7', 'return_1', 'target_next_day']
    print(df[display_cols].head().to_string())
    
    print("\n" + "="*60)
    print("Feature Engineering External Complete!")
    print("="*60)
    print("""
Next steps:
- Run python src/train_xgboost_external.py to train model with external data
""")

if __name__ == "__main__":
    main()
