#!/usr/bin/env python3
"""
update_raw_data_yf.py - Auto update raw USD/IDR data from Yahoo Finance
Purpose: Fetches latest USD/IDR data since the last date in the raw CSV and appends it.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
from pathlib import Path

def main():
    raw_file = Path("data/raw/USD_IDR Historical Data.csv")
    
    print("="*60)
    print("Auto-Update USD/IDR Data from Yahoo Finance")
    print("="*60)
    
    if not raw_file.exists():
        print(f"ERROR: Raw file not found at {raw_file}")
        sys.exit(1)
        
    # 1. Read existing raw data
    print(f"\n[1] Reading existing data...")
    df = pd.read_csv(raw_file)
    
    # Try parsing date to find the last date
    try:
        parsed_dates = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    except Exception:
        parsed_dates = pd.to_datetime(df['Date'], infer_datetime_format=True)
        
    last_date = parsed_dates.max()
    print(f"   ✓ Last recorded date: {last_date.strftime('%Y-%m-%d')}")
    
    today = datetime.now()
    if last_date.date() >= today.date():
        print(f"   ✓ Data is already up to date. No new data to download.")
        return
        
    # Fetch data starting from the last date to get correct percentage change
    start_fetch = last_date.strftime('%Y-%m-%d')
    end_fetch = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"\n[2] Downloading new data from {start_fetch} to {end_fetch}...")
    try:
        new_data = yf.download('IDR=X', start=start_fetch, end=end_fetch, progress=False)
    except Exception as e:
        print(f"ERROR fetching from Yahoo Finance: {e}")
        sys.exit(1)
        
    if new_data.empty:
        print(f"   ⚠ No new data available from Yahoo Finance.")
        return
        
    print(f"   ✓ Downloaded {len(new_data)} rows.")
    
    # Calculate Change %
    # Change % is (Close - Previous Close) / Previous Close * 100
    if 'Close' in new_data.columns:
        close_col = new_data['Close']
    else:
        close_col = new_data.iloc[:, 0] # fallback if multi-index
        
    change_pct = close_col.pct_change() * 100
    
    # Drop the first row which is the 'last_date' (already exists in our CSV)
    new_data = new_data.iloc[1:].copy()
    change_pct = change_pct.iloc[1:]
    
    if new_data.empty:
        print(f"   ✓ Data is already up to date (no new trading days).")
        return
        
    # 3. Format new data to match the raw CSV
    print(f"\n[3] Formatting new data...")
    formatted_df = pd.DataFrame()
    formatted_df['Date'] = new_data.index.strftime('%m/%d/%Y')
    
    # Handling yfinance returning multi-level columns in recent versions
    def get_col(col_name):
        if col_name in new_data.columns:
            # If it's a dataframe (e.g. from multi-index), get the first series
            if isinstance(new_data[col_name], pd.DataFrame):
                return new_data[col_name].iloc[:, 0].values
            return new_data[col_name].values
        return [0] * len(new_data)
        
    formatted_df['Price'] = get_col('Close')
    formatted_df['Open'] = get_col('Open')
    formatted_df['High'] = get_col('High')
    formatted_df['Low'] = get_col('Low')
    formatted_df['Vol.'] = '' # Empty volume
    
    # Format Change % nicely
    formatted_df['Change %'] = change_pct.values
    formatted_df['Change %'] = formatted_df['Change %'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "0.00%")
    
    print(f"   - Prepared {len(formatted_df)} new rows to append.")
    
    # 4. Append and save
    print(f"\n[4] Appending to {raw_file}...")
    updated_df = pd.concat([df, formatted_df], ignore_index=True)
    
    updated_df.to_csv(raw_file, index=False)
    print(f"   ✓ Successfully updated {raw_file} with latest data up to {new_data.index[-1].strftime('%Y-%m-%d')}.")

if __name__ == "__main__":
    main()
