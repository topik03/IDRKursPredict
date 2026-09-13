#!/usr/bin/env python3
"""
check_data.py - Check and explore the raw USD/IDR dataset
Purpose: Examine the raw dataset structure, content, and quality
"""

import pandas as pd
import sys
from pathlib import Path

def main():
    # Define file paths
    data_file = Path("data/raw/USD_IDR Historical Data.csv")
    
    print("="*60)
    print("USD/IDR Historical Data - Initial Exploration")
    print("="*60)
    
    # Check if file exists
    if not data_file.exists():
        print(f"ERROR: File not found: {data_file}")
        print("Please ensure the dataset is in the correct location.")
        sys.exit(1)
    
    print(f"\n[1] Reading dataset from: {data_file}")
    
    # Read the dataset
    try:
        df = pd.read_csv(data_file)
        print(f"   ✓ Dataset loaded successfully")
    except Exception as e:
        print(f"ERROR reading file: {e}")
        sys.exit(1)
    
    # Shape
    print(f"\n[2] Dataset Shape:")
    print(f"   - Rows: {df.shape[0]}")
    print(f"   - Columns: {df.shape[1]}")
    
    # Column names
    print(f"\n[3] Column Names:")
    for col in df.columns:
        print(f"   - {col}")
    
    # Data types
    print(f"\n[4] Data Types:")
    print(df.dtypes.to_string())
    
    # First 5 rows
    print(f"\n[5] First 5 Rows:")
    print(df.head().to_string())
    
    # Last 5 rows
    print(f"\n[6] Last 5 Rows:")
    print(df.tail().to_string())
    
    # Missing values
    print(f"\n[7] Missing Values:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        pct = (count / len(df)) * 100
        print(f"   - {col}: {count} ({pct:.2f}%)")
    
    # Duplicate rows
    print(f"\n[8] Duplicate Rows:")
    duplicates = df.duplicated().sum()
    print(f"   - Total duplicate rows: {duplicates}")
    
    # Date range analysis
    print(f"\n[9] Date Analysis:")
    print("   Note: Dates appear to be in MM/DD/YYYY format")
    print("   Data appears to be sorted from newest to oldest (descending)")
    
    # Sample date values
    print(f"\n[10] Sample Date Values:")
    print(f"   First date (newest): {df['Date'].iloc[0]}")
    print(f"   Last date (oldest): {df['Date'].iloc[-1]}")
    
    # Numerical columns sample
    print(f"\n[11] Sample Numeric Values:")
    for col in ['Price', 'Open', 'High', 'Low']:
        print(f"   - {col}: {df[col].iloc[0]}")
    
    # Change % sample
    print(f"\n[12] Change % Sample:")
    print(f"   - Change %: {df['Change %'].iloc[0]}")
    
    # Vol. sample
    print(f"\n[13] Volume Sample:")
    print(f"   - Vol.: '{df['Vol.'].iloc[0]}'")
    
    print("\n" + "="*60)
    print("Exploration Complete!")
    print("="*60)
    print("""
Next steps:
- Run python src/clean_data.py to clean and prepare the data
""")

if __name__ == "__main__":
    main()
