#!/usr/bin/env python3
"""
visualize_external_result.py - Visualize results with external features
Purpose: Create visualization plots for model with external market data
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def main():
    # Define file paths
    predictions_file = Path("data/processed/xgboost_external_predictions.csv")
    output_dir = Path("reports/figures")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("External Model Visualization Pipeline")
    print("="*60)
    
    # Read predictions
    print(f"\n[1] Reading predictions from: {predictions_file}")
    
    if not predictions_file.exists():
        print(f"ERROR: Predictions file not found: {predictions_file}")
        print("Please run train_xgboost_external.py first!")
        sys.exit(1)
    
    try:
        predictions_df = pd.read_csv(predictions_file)
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        print(f"   ✓ Loaded {len(predictions_df)} predictions")
    except Exception as e:
        print(f"ERROR reading file: {e}")
        sys.exit(1)
    
    # Check if previous predictions exist for comparison
    prev_predictions_file = Path("data/processed/xgboost_predictions.csv")
    has_comparison = prev_predictions_file.exists()
    
    if has_comparison:
        try:
            prev_df = pd.read_csv(prev_predictions_file)
            prev_df['Date'] = pd.to_datetime(prev_df['Date'])
            has_comparison = True
            print(f"   ✓ Loaded previous predictions for comparison")
        except Exception as e:
            has_comparison = False
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # ============================================
    # PLOT 1: ACTUAL VS PREDICTED
    # ============================================
    print(f"\n[2] Creating Actual vs Predicted plot...")
    
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    
    ax1.plot(predictions_df['Date'], predictions_df['Actual'], 
             label='Actual', color='#2E86AB', linewidth=1.5, alpha=0.9)
    ax1.plot(predictions_df['Date'], predictions_df['Predicted'], 
             label='Predicted (External)', color='#E94F37', linewidth=1.5, alpha=0.7, linestyle='--')
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('USD/IDR Exchange Rate', fontsize=12)
    ax1.set_title('USD/IDR Exchange Rate: Actual vs Predicted\n(XGBoost with External Features)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add metrics annotation
    mae = predictions_df['Abs_Error'].mean()
    mape = predictions_df['Pct_Error'].mean()
    textstr = f'MAE: {mae:,.0f}\nMAPE: {mape:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    plot1_path = output_dir / "actual_vs_predicted_external.png"
    fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {plot1_path}")
    plt.close(fig1)
    
    # ============================================
    # PLOT 2: ERROR/RESIDUAL PLOT
    # ============================================
    print(f"\n[3] Creating Error/Residual plot...")
    
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 2a: Prediction Error over time
    ax2a = axes[0, 0]
    ax2a.plot(predictions_df['Date'], predictions_df['Error'], 
              color='#2E86AB', linewidth=1, alpha=0.8)
    ax2a.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax2a.fill_between(predictions_df['Date'], predictions_df['Error'], 0, 
                     where=(predictions_df['Error'] > 0), alpha=0.3, color='green', label='Underprediction')
    ax2a.fill_between(predictions_df['Date'], predictions_df['Error'], 0, 
                     where=(predictions_df['Error'] < 0), alpha=0.3, color='red', label='Overprediction')
    ax2a.set_xlabel('Date', fontsize=10)
    ax2a.set_ylabel('Error (Actual - Predicted)', fontsize=10)
    ax2a.set_title('Prediction Error Over Time', fontsize=12, fontweight='bold')
    ax2a.legend(loc='best', fontsize=8)
    ax2a.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2a.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2b: Absolute Error over time
    ax2b = axes[0, 1]
    ax2b.plot(predictions_df['Date'], predictions_df['Abs_Error'], 
             color='#E94F37', linewidth=1, alpha=0.8)
    ax2b.set_xlabel('Date', fontsize=10)
    ax2b.set_ylabel('Absolute Error', fontsize=10)
    ax2b.set_title('Absolute Error Over Time', fontsize=12, fontweight='bold')
    ax2b.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2b.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2c: Error distribution histogram
    ax2c = axes[1, 0]
    ax2c.hist(predictions_df['Error'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax2c.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2c.axvline(x=predictions_df['Error'].mean(), color='green', linestyle='--', linewidth=2, label=f"Mean: {predictions_df['Error'].mean():.0f}")
    ax2c.set_xlabel('Error (Actual - Predicted)', fontsize=10)
    ax2c.set_ylabel('Frequency', fontsize=10)
    ax2c.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax2c.legend(loc='best', fontsize=8)
    
    # Plot 2d: Scatter plot Actual vs Predicted
    ax2d = axes[1, 1]
    ax2d.scatter(predictions_df['Actual'], predictions_df['Predicted'], 
                 alpha=0.5, color='#2E86AB', s=30)
    
    # Add perfect prediction line
    min_val = min(predictions_df['Actual'].min(), predictions_df['Predicted'].min())
    max_val = max(predictions_df['Actual'].max(), predictions_df['Predicted'].max())
    ax2d.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2d.set_xlabel('Actual USD/IDR', fontsize=10)
    ax2d.set_ylabel('Predicted USD/IDR', fontsize=10)
    ax2d.set_title('Actual vs Predicted Scatter', fontsize=12, fontweight='bold')
    ax2d.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    
    plot2_path = output_dir / "error_analysis_external.png"
    fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {plot2_path}")
    plt.close(fig2)
    
    # ============================================
    # PLOT 3: MODEL COMPARISON (if available)
    # ============================================
    print(f"\n[4] Creating Model Comparison plot...")
    
    if has_comparison:
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        
        # Merge predictions on Date for comparison
        merged = pd.merge(
            predictions_df[['Date', 'Actual', 'Predicted']], 
            prev_df[['Date', 'Predicted']], 
            on='Date', 
            suffixes=('_External', '_Base'),
            how='inner'
        )
        
        if len(merged) > 0:
            ax3.plot(merged['Date'], merged['Actual'], 
                     label='Actual', color='#2E86AB', linewidth=1.5, alpha=0.9)
            ax3.plot(merged['Date'], merged['Predicted_Base'], 
                     label='XGBoost (USD/IDR only)', color='#A23B72', linewidth=1.2, alpha=0.7, linestyle='-.')
            ax3.plot(merged['Date'], merged['Predicted_External'], 
                     label='XGBoost (External)', color='#F18F01', linewidth=1.2, alpha=0.7, linestyle='--')
            
            ax3.set_xlabel('Date', fontsize=12)
            ax3.set_ylabel('USD/IDR Exchange Rate', fontsize=12)
            ax3.set_title('Model Comparison: XGBoost vs XGBoost with External Features', fontsize=14, fontweight='bold')
            ax3.legend(loc='best', fontsize=10)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            plot3_path = output_dir / "model_comparison.png"
            fig3.savefig(plot3_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ Saved: {plot3_path}")
            plt.close(fig3)
    else:
        print(f"   ⚠ Skipped - no previous predictions for comparison")
    
    # ============================================
    # SUMMARY
    # ============================================
    print(f"\n[5] Visualization Summary:")
    print(f"   Total plots created: {3 if has_comparison else 2}")
    print(f"   Output directory: {output_dir}")
    
    print("\nPlots generated:")
    print(f"  1. actual_vs_predicted_external.png - Actual vs Predicted")
    print(f"  2. error_analysis_external.png - Error/residual analysis")
    if has_comparison:
        print(f"  3. model_comparison.png - Model comparison")
    
    # List files in output directory
    print(f"\n[6] Files in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            print(f"   - {f.name}")
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print("""
All external pipeline scripts completed!

Run order for external data pipeline:

1. python src/download_external_data.py    - Download external market data
2. python src/merge_external_features.py   - Merge with USD/IDR data
3. python src/feature_engineering_external.py - Create features
4. python src/train_xgboost_external.py  - Train XGBoost with external
5. python src/compare_models.py          - Compare all models
6. python src/visualize_external_result.py - Visualize results
""")

if __name__ == "__main__":
    main()
