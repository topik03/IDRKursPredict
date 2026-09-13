#!/usr/bin/env python3
"""
download_external_data.py - Download external market data from Yahoo Finance
Purpose: Fetch market data for features to improve USD/IDR prediction
"""

import pandas as pd
import sys
from pathlib import Path
import yfinance as yf

from datetime import datetime, timedelta

def download_ticker_data(ticker, start_date, end_date, feature_name):
    """Download data for a single ticker"""
    print(f"   Downloading {ticker} ({feature_name})...")
    
    try:
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"   ⚠ No data for {ticker}")
            return None
        
        # Extract Close price and reset index to get Date as column
        close_col = 'Close'
        if close_col in data.columns:
            close_prices = data[close_col].reset_index()
            close_prices.columns = ['Date', feature_name]
        else:
            # Handle single column case - get first available price column
            first_col = data.columns[0]
            close_prices = pd.DataFrame({
                'Date': data.index,
                feature_name: data[first_col]
            })
            close_prices = close_prices.reset_index(drop=True)
        
        print(f"   ✓ Downloaded {len(close_prices)} rows for {ticker}")
        return close_prices
    
    except Exception as e:
        print(f"   ⚠ Error downloading {ticker}: {str(e)}")
        return None

def main():
    # Define file paths
    output_file = Path("data/raw/external_market_data.csv")
    
    # Create data/raw directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("External Market Data Download Pipeline")
    print("="*60)
    
    # Define date range (matching USD/IDR dataset)
    start_date = "2015-01-01"
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"\nDate range: {start_date} to {end_date}")
    
    # Define tickers and their feature names
    tickers = {
        'DX-Y.NYB': 'dxy_close',       # US Dollar Index
        'GC=F': 'gold_close',        # Gold Futures
        'CL=F': 'oil_close',         # Crude Oil Futures
        '^VIX': 'vix_close',        # VIX Index
        '^JKSE': 'ihsg_close',       # IHSG (Indonesia)
        '^GSPC': 'sp500_close',      # S&P 500
        '^TNX': 'us10y_close',      # US 10-Year Treasury Yield
    }
    
    # Download all tickers
    print(f"\n[1] Downloading {len(tickers)} external data sources...")
    
    downloaded_data = []
    
    for ticker, feature_name in tickers.items():
        data = download_ticker_data(ticker, start_date, end_date, feature_name)
        
        if data is not None:
            downloaded_data.append(data)
    
    if not downloaded_data:
        print("ERROR: No data could be downloaded!")
        sys.exit(1)
    
    print(f"\n[2] Merging external data...")
    
    # Merge all data on Date
    # Start with the first dataset
    merged_df = downloaded_data[0]
    
    # Merge with remaining datasets
    for df in downloaded_data[1:]:
        merged_df = pd.merge(merged_df, df, on='Date', how='outer')
    
    # Sort by date
    merged_df = merged_df.sort_values('Date', ascending=True).reset_index(drop=True)
    
    print(f"   ✓ Merged {len(merged_df)} rows")
    print(f"   - Columns: {list(merged_df.columns)}")
    
    # Check for missing values
    print(f"\n[3] Data quality check:")
    for col in merged_df.columns:
        if col != 'Date':
            missing = merged_df[col].isnull().sum()
            pct = (missing / len(merged_df)) * 100
            print(f"   - {col}: {missing} missing ({pct:.2f}%)")
    
    # Convert Date to datetime
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    
    # Save to CSV
    print(f"\n[4] Saving to: {output_file}")
    merged_df.to_csv(output_file, index=False)
    print(f"   ✓ Saved successfully")
    
    # Show sample
    print(f"\n[5] Sample data (first 5 rows):")
    print(merged_df.head().to_string())
    
    print(f"\n[6] Sample data (last 5 rows):")
    print(merged_df.tail().to_string())
    
    print("\n" + "="*60)
    print("External Data Download Complete!")
    print("="*60)
    print("""
Next steps:
- Run python src/merge_external_features.py to merge with USD/IDR data
""")

if __name__ == "__main__":
    main()
