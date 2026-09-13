#!/usr/bin/env python3
"""
model_explainability.py - Model explainability and feature importance
Purpose: Analyze and visualize feature importance from XGBoost model
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

def create_return_features(df):
    """Create return-based features for prediction"""
    df = df.copy()
    
    for lag in [1, 2, 3, 7, 14]:
        df[f'return_lag_{lag}'] = df['return_1'].shift(lag)
    
    for window in [7, 14, 30]:
        df[f'return_ma_{window}'] = df['return_1'].rolling(window=window).mean()
    
    for window in [7, 14, 30]:
        df[f'return_std_{window}'] = df['return_1'].rolling(window=window).std()
    
    return df

def main():
    # Define file paths
    model_file = Path("models/xgboost_usd_idr_improved.pkl")
    metadata_file = Path("models/xgboost_usd_idr_improved_metadata.pkl")
    data_file = Path("data/processed/usd_idr_features_external.csv")
    output_csv = Path("reports/feature_importance.csv")
    output_plot = Path("reports/figures/feature_importance_top20.png")
    shap_output = Path("reports/figures/shap_summary.png")
    
    print("="*60)
    print("Model Explainability")
    print("="*60)
    
    # Check if model exists
    if not model_file.exists():
        print(f"ERROR: Model not found: {model_file}")
        print("Please run train_xgboost_improved.py first!")
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
    
    # Get feature columns from metadata
    print(f"\n[2.5] Loading model metadata from: {metadata_file}")
    try:
        metadata = joblib.load(metadata_file)
        feature_cols = metadata['feature_cols']
        print(f"   ✓ Loaded {len(feature_cols)} feature columns from metadata")
    except Exception as e:
        print(f"ERROR loading metadata: {e}")
        sys.exit(1)
        
    # Create return features
    print(f"\n[2.7] Creating return-based features...")
    df = create_return_features(df)
    
    # Use test data (2025 onwards) for explainability
    test_df = df[df['Date'] >= '2025-01-01'].copy()
    X_test = test_df[feature_cols]
    
    print(f"\n[3] Extracting feature importance (using test data: {len(X_test)} rows)...")
    
    # Get feature importance from XGBoost
    importance = model.feature_importances_
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Save to CSV
    print(f"\n[4] Saving feature importance to: {output_csv}")
    importance_df.to_csv(output_csv, index=False)
    print("   ✓ Saved successfully")
    
    # Create feature importance plot
    print(f"\n[5] Creating feature importance plot...")
    
    top_n = 20
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['importance'], align='center', color='#2E86AB')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance Score')
    ax.set_title(f'Top {top_n} Feature Importance (Improved XGBoost)')
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_plot}")
    plt.close()
    
    # Try SHAP if available
    print(f"\n[6] Attempting SHAP analysis...")
    try:
        import shap
        print("   SHAP library found, creating SHAP summary plot...")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # SHAP summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_test, show=False, plot_type="dot")
        plt.tight_layout()
        plt.savefig(shap_output, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {shap_output}")
        plt.close()
        
    except ImportError:
        print("   ⚠ SHAP not installed, skipping SHAP analysis")
    except Exception as e:
        print(f"   ⚠ Could not create SHAP plot: {e}")
    
    # Display top features
    print(f"\n[7] Top 20 Feature Importance:")
    print("-" * 40)
    for i, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:<25} {row['importance']:.4f}")
    
    print(f"\n[8] Feature importance CSV saved to: {output_csv}")
    print(f"   Feature importance plot saved to: {output_plot}")
    
    print("\n" + "="*60)
    print("Explainability Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
