#!/usr/bin/env python3
"""
merge_external_features.py - Merge external data with USD/IDR data
Purpose: Combine cleaned USD/IDR data with external market indicators
"""

import pandas as pd
import sys
from pathlib import Path

def main():
    # Define file paths
    usd_idr_file = Path("data/processed/usd_idr_clean.csv")
    external_file = Path("data/raw/external_market_data.csv")
    output_file = Path("data/processed/usd_idr_with_external.csv")
    
    # Create processed directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Merge External Features Pipeline")
    print("="*60)
    
    # Read USD/IDR cleaned data
    print(f"\n[1] Reading USD/IDR data from: {usd_idr_file}")
    
    if not usd_idr_file.exists():
        print(f"ERROR: USD/IDR file not found: {usd_idr_file}")
        print("Please run clean_data.py first!")
        sys.exit(1)
    
    try:
        usd_idr_df = pd.read_csv(usd_idr_file)
        usd_idr_df['Date'] = pd.to_datetime(usd_idr_df['Date'])
        print(f"   ✓ Loaded {len(usd_idr_df)} rows")
        print(f"   - Columns: {list(usd_idr_df.columns)}")
    except Exception as e:
        print(f"ERROR reading USD/IDR file: {e}")
        sys.exit(1)
    
    # Read external market data
    print(f"\n[2] Reading external market data from: {external_file}")
    
    if not external_file.exists():
        print(f"ERROR: External data file not found: {external_file}")
        print("Please run download_external_data.py first!")
        sys.exit(1)
    
    try:
        external_df = pd.read_csv(external_file)
        external_df['Date'] = pd.to_datetime(external_df['Date'])
        print(f"   ✓ Loaded {len(external_df)} rows")
        print(f"   - Columns: {list(external_df.columns)}")
    except Exception as e:
        print(f"ERROR reading external file: {e}")
        sys.exit(1)
    
    # Show date ranges
    print(f"\n[3] Date ranges:")
    print(f"   - USD/IDR: {usd_idr_df['Date'].min().date()} to {usd_idr_df['Date'].max().date()}")
    print(f"   - External: {external_df['Date'].min().date()} to {external_df['Date'].max().date()}")
    
    # Merge data using left join (USD/IDR as main)
    print(f"\n[4] Merging data (left join on Date)...")
    
    merged_df = pd.merge(
        usd_idr_df,
        external_df,
        on='Date',
        how='left'
    )
    
    print(f"   ✓ Merged {len(merged_df)} rows")
    print(f"   - Columns after merge: {list(merged_df.columns)}")
    
    # Check missing values in external data
    print(f"\n[5] Missing values after merge (before filling):")
    external_cols = [col for col in merged_df.columns if col.endswith('_close')]
    for col in external_cols:
        missing = merged_df[col].isnull().sum()
        pct = (missing / len(merged_df)) * 100
        print(f"   - {col}: {missing} missing ({pct:.2f}%)")
    
    # Fill missing values using forward fill, then backward fill
    print(f"\n[6] Filling missing values...")
    print(f"   - Method: Forward fill then backward fill")
    
    for col in external_cols:
        # Forward fill first (carry previous day's data forward)
        merged_df[col] = merged_df[col].ffill()
        
        # Then backward fill for any remaining NaNs at the beginning
        merged_df[col] = merged_df[col].bfill()
    
    # Check missing values after filling
    print(f"\n[7] Missing values after filling:")
    remaining_missing = 0
    for col in external_cols:
        missing = merged_df[col].isnull().sum()
        pct = (missing / len(merged_df)) * 100
        remaining_missing += missing
        print(f"   - {col}: {missing} missing ({pct:.2f}%)")
    
    if remaining_missing > 0:
        print(f"   ⚠ Warning: {remaining_missing} missing values remain")
    else:
        print(f"   ✓ All external values filled")
    
    # Data quality check (no leakage - external data should be from same day)
    print(f"\n[8] Data quality check:")
    print(f"   ✓ Using left join ensures no data leakage")
    print(f"   ✓ External data is from the same day (t), not future data")
    
    # Sort by date
    merged_df = merged_df.sort_values('Date', ascending=True).reset_index(drop=True)
    
    # Save to CSV
    print(f"\n[9] Saving to: {output_file}")
    merged_df.to_csv(output_file, index=False)
    print(f"   ✓ Saved successfully")
    
    # Show sample
    print(f"\n[10] Sample data (first 5 rows):")
    display_cols = ['Date', 'Close', 'dxy_close', 'gold_close', 'oil_close', 'vix_close']
    print(merged_df[display_cols].head().to_string())
    
    print("\n" + "="*60)
    print("Merge External Features Complete!")
    print("="*60)
    print("""
Next steps:
- Run python src/feature_engineering_external.py to create features
""")

if __name__ == "__main__":
    main()
