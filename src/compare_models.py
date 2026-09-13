#!/usr/bin/env python3
"""
compare_models.py - Compare all model performances with visualization
Purpose: Evaluate and compare Baseline, XGBoost, and XGBoost with external features
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_directional_accuracy(y_true, y_pred):
    """Calculate directional accuracy (trend prediction)"""
    actual_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    correct = np.sum((actual_diff * pred_diff) > 0)
    total = len(actual_diff)
    return (correct / total) * 100 if total > 0 else 0

def get_baseline_metrics():
    """Get baseline metrics from Naive Forecast"""
    features_file = Path("data/processed/usd_idr_features.csv")
    if not features_file.exists():
        return None
    
    df = pd.read_csv(features_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Test set
    test_df = df[df['Date'] >= '2025-01-01'].copy()
    
    # Baseline: tomorrow = today
    y_actual = test_df['target_next_day'].values
    y_pred = test_df['Close'].values
    
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mape = calculate_mape(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    dir_acc = calculate_directional_accuracy(y_actual, y_pred)
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2, 'Directional Accuracy': dir_acc}

def get_xgboost_metrics():
    """Get XGBoost metrics (without external)"""
    predictions_file = Path("data/processed/xgboost_predictions.csv")
    if not predictions_file.exists():
        return None
    
    df = pd.read_csv(predictions_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    y_actual = df['Actual'].values
    y_pred = df['Predicted'].values
    
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mape = calculate_mape(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    dir_acc = calculate_directional_accuracy(y_actual, y_pred)
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2, 'Directional Accuracy': dir_acc}

def get_xgboost_external_metrics():
    """Get XGBoost metrics (with external)"""
    predictions_file = Path("data/processed/xgboost_external_predictions.csv")
    if not predictions_file.exists():
        return None
    
    df = pd.read_csv(predictions_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    y_actual = df['Actual'].values
    y_pred = df['Predicted'].values
    
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mape = calculate_mape(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    dir_acc = calculate_directional_accuracy(y_actual, y_pred)
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2, 'Directional Accuracy': dir_acc}

def create_comparison_chart(comparison_df, output_path):
    """Create bar chart comparison"""
    print(f"\n[3] Creating comparison chart...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Comparison - All Metrics', fontsize=14, fontweight='bold')
    
    models = comparison_df['Model']
    metrics = ['MAE', 'RMSE', 'MAPE', 'R2', 'Directional Accuracy']
    colors = ['#2E86AB', '#E94F37', '#F18F01']
    
    # MAE
    ax = axes[0, 0]
    ax.bar(models, comparison_df['MAE'], color=colors)
    ax.set_title('MAE (Lower is Better)')
    ax.set_ylabel('IDR')
    ax.tick_params(axis='x', rotation=45)
    
    # RMSE
    ax = axes[0, 1]
    ax.bar(models, comparison_df['RMSE'], color=colors)
    ax.set_title('RMSE (Lower is Better)')
    ax.set_ylabel('IDR')
    ax.tick_params(axis='x', rotation=45)
    
    # MAPE
    ax = axes[0, 2]
    ax.bar(models, comparison_df['MAPE'], color=colors)
    ax.set_title('MAPE % (Lower is Better)')
    ax.set_ylabel('%')
    ax.tick_params(axis='x', rotation=45)
    
    # R2
    ax = axes[1, 0]
    ax.bar(models, comparison_df['R2'], color=colors)
    ax.set_title('R² Score (Higher is Better)')
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=45)
    
    # Directional Accuracy
    ax = axes[1, 1]
    ax.bar(models, comparison_df['Directional Accuracy'], color=colors)
    ax.set_title('Directional Accuracy (Higher is Better)')
    ax.set_ylabel('%')
    ax.tick_params(axis='x', rotation=45)
    
    # Hide last empty subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")
    plt.close()

def main():
    output_file = Path("reports/model_comparison.csv")
    chart_file = Path("reports/figures/model_comparison.png")
    
    print("="*60)
    print("Model Comparison Pipeline")
    print("="*60)
    
    # Get metrics
    baseline_metrics = get_baseline_metrics()
    xgboost_metrics = get_xgboost_metrics()
    xgboost_external_metrics = get_xgboost_external_metrics()
    
    # Build comparison table
    models_data = []
    
    if baseline_metrics:
        models_data.append({
            'Model': 'Baseline (Naive)',
            'MAE': baseline_metrics['MAE'],
            'RMSE': baseline_metrics['RMSE'],
            'MAPE': baseline_metrics['MAPE'],
            'R2': baseline_metrics['R2'],
            'Directional Accuracy': baseline_metrics['Directional Accuracy']
        })
    
    if xgboost_metrics:
        models_data.append({
            'Model': 'XGBoost',
            'MAE': xgboost_metrics['MAE'],
            'RMSE': xgboost_metrics['RMSE'],
            'MAPE': xgboost_metrics['MAPE'],
            'R2': xgboost_metrics['R2'],
            'Directional Accuracy': xgboost_metrics['Directional Accuracy']
        })
    
    if xgboost_external_metrics:
        models_data.append({
            'Model': 'XGBoost External',
            'MAE': xgboost_external_metrics['MAE'],
            'RMSE': xgboost_external_metrics['RMSE'],
            'MAPE': xgboost_external_metrics['MAPE'],
            'R2': xgboost_external_metrics['R2'],
            'Directional Accuracy': xgboost_external_metrics['Directional Accuracy']
        })
    
    if not models_data:
        print("ERROR: No model results found!")
        print("Please run training scripts first:")
        print("  - train_baseline.py")
        print("  - train_xgboost.py")
        print("  - train_xgboost_external.py")
        sys.exit(1)
    
    comparison_df = pd.DataFrame(models_data)
    
    # Display comparison
    print(f"\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Model':<25} {'MAE':>12} {'RMSE':>12} {'MAPE':>10} {'R²':>10} {'Dir Acc':>10}")
    print("-" * 84)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['Model']:<25} {row['MAE']:>12,.2f} {row['RMSE']:>12,.2f} {row['MAPE']:>9.2f}% {row['R2']:>10.4f} {row['Directional Accuracy']:>9.2f}%")
    
    # Best model
    print(f"\n[1] Best model analysis:")
    best_mae = comparison_df.loc[comparison_df['MAE'].idxmin(), 'Model']
    best_mape = comparison_df.loc[comparison_df['MAPE'].idxmin(), 'Model']
    best_dir = comparison_df.loc[comparison_df['Directional Accuracy'].idxmax(), 'Model']
    print(f"   - Best by MAE: {best_mae}")
    print(f"   - Best by MAPE: {best_mape}")
    print(f"   - Best by Dir Acc: {best_dir}")
    
    # XGBoost comparison
    if xgboost_metrics and xgboost_external_metrics:
        print(f"\n[2] XGBoost improvement:")
        mae_diff = xgboost_external_metrics['MAE'] - xgboost_metrics['MAE']
        mape_diff = xgboost_external_metrics['MAPE'] - xgboost_metrics['MAPE']
        
        if mae_diff < 0:
            print(f"   - External improved MAE by {abs(mae_diff):,.2f} IDR")
        else:
            print(f"   - External increased MAE by {abs(mae_diff):,.2f} IDR")
    
    # Save to CSV
    print(f"\n[4] Saving comparison to: {output_file}")
    comparison_df.to_csv(output_file, index=False)
    print("   ✓ Saved successfully")
    
    # Create chart
    create_comparison_chart(comparison_df, chart_file)
    
    print("\n" + "="*60)
    print("Model Comparison Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
