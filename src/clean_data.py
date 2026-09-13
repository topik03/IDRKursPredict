#!/usr/bin/env python3
"""
clean_data.py - Clean and prepare the USD/IDR dataset
Purpose: Clean raw data, handle missing values, format columns, and save processed data
"""

import pandas as pd
import sys
from pathlib import Path

def clean_numeric_column(series):
    """Remove commas from numeric strings and convert to float"""
    if series.dtype == object:
        # Remove commas and convert to float
        return series.astype(str).str.replace(',', '', regex=False).str.strip()
    return series

def clean_change_pct(series):
    """Clean Change % column - remove % sign and convert to float"""
    if series.dtype == object:
        return series.astype(str).str.replace('%', '', regex=False).str.strip()
    return series

def main():
    # Define file paths
    input_file = Path("data/raw/USD_IDR Historical Data.csv")
    output_file = Path("data/processed/usd_idr_clean.csv")
    
    # Create processed directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("USD/IDR Data Cleaning Pipeline")
    print("="*60)
    
    # Read the dataset
    print(f"\n[1] Reading dataset from: {input_file}")
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(input_file)
        print(f"   ✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"ERROR reading file: {e}")
        sys.exit(1)
    
    print(f"\n[2] Initial data info:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {list(df.columns)}")
    
    # Step 1: Convert Date to datetime
    print(f"\n[3] Converting Date column...")
    try:
        # Try to parse dates - format is MM/DD/YYYY
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        print(f"   ✓ Date converted to datetime")
    except Exception as e:
        print(f"   ⚠ Could not parse date with format %m/%d/%Y: {e}")
        try:
            df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
            print(f"   ✓ Date converted (inferred format)")
        except Exception as e2:
            print(f"   ERROR: Failed to convert Date: {e2}")
            sys.exit(1)
    
    print(f"   - Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Step 2: Sort by date (oldest to newest)
    print(f"\n[4] Sorting data by date (oldest to newest)...")
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)
    print(f"   ✓ Data sorted")
    
    # Step 3: Rename Price to Close
    print(f"\n[5] Renaming columns...")
    if 'Price' in df.columns:
        df = df.rename(columns={'Price': 'Close'})
        print(f"   ✓ 'Price' renamed to 'Close'")
    
    # Step 4: Clean numeric columns (remove commas)
    print(f"\n[6] Cleaning numeric columns...")
    numeric_cols = ['Close', 'Open', 'High', 'Low']
    
    for col in numeric_cols:
        if col in df.columns:
            # Convert to string, remove commas, convert to float
            df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"   ✓ Cleaned {col}")
    
    # Step 5: Clean Change %
    print(f"\n[7] Cleaning Change % column...")
    if 'Change %' in df.columns:
        df['Change %'] = df['Change %'].astype(str).str.replace('%', '', regex=False).str.strip()
        df['Change %'] = pd.to_numeric(df['Change %'], errors='coerce')
        print(f"   ✓ Cleaned and converted Change %")
    
    # Step 6: Handle Vol. column
    print(f"\n[8] Checking Vol. column...")
    if 'Vol.' in df.columns:
        vol_missing = df['Vol.'].isnull().sum()
        vol_empty = (df['Vol.'] == '').sum() if df['Vol.'].dtype == object else 0
        total_missing = vol_missing + vol_empty
        pct_missing = (total_missing / len(df)) * 100
        
        print(f"   - Vol. missing/empty: {total_missing} ({pct_missing:.2f}%)")
        
        if pct_missing > 50:
            df = df.drop(columns=['Vol.'])
            print(f"   ✓ Dropped Vol. column (>50% missing)")
        else:
            # Try to clean Vol. if it has values
            try:
                # Handle 'K' and 'M' suffixes (e.g., 0.15K = 150)
                vol_series = df['Vol.'].astype(str).str.strip()
                # Multiply by 1000 for 'K', by 1000000 for 'M'
                vol_series = vol_series.str.replace('K', 'e3', regex=False)
                vol_series = vol_series.str.replace('M', 'e6', regex=False)
                df['Vol.'] = pd.to_numeric(vol_series, errors='coerce')
                print(f"   ✓ Cleaned Vol. column")
            except Exception as e:
                print(f"   ⚠ Could not clean Vol.: {e}")
                df = df.drop(columns=['Vol.'])
                print(f"   ✓ Dropped Vol. column")
    
    # Step 7: Remove duplicate dates
    print(f"\n[9] Checking for duplicate dates...")
    dup_count = df['Date'].duplicated().sum()
    if dup_count > 0:
        df = df[~df['Date'].duplicated()]
        print(f"   ✓ Removed {dup_count} duplicate date(s)")
    else:
        print(f"   ✓ No duplicate dates found")
    
    print(f"\n[10] Final cleaned data:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Save to CSV
    print(f"\n[11] Saving cleaned data to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"   ✓ Saved successfully")
    
    # Show sample of cleaned data
    print(f"\n[12] Sample of cleaned data (first 5 rows):")
    print(df.head().to_string())
    
    print(f"\n[13] Sample of cleaned data (last 5 rows):")
    print(df.tail().to_string())
    
    # Data types
    print(f"\n[14] Final data types:")
    print(df.dtypes.to_string())
    
    print("\n" + "="*60)
    print("Data Cleaning Complete!")
    print("="*60)
    print("""
Next steps:
- Run python src/feature_engineering.py to create features
""")

if __name__ == "__main__":
    main()
