#!/usr/bin/env python3
"""
train_xgboost.py - Train and evaluate XGBoost model for USD/IDR prediction
Purpose: Train XGBoost regressor, evaluate on test set, and save model
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
import xgboost as xgb
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

def main():
    # Define file paths
    input_file = Path("data/processed/usd_idr_features.csv")
    model_output = Path("models/xgboost_usd_idr.pkl")
    predictions_output = Path("data/processed/xgboost_predictions.csv")
    
    # Create directories
    model_output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("USD/IDR XGBoost Model Training")
    print("="*60)
    
    # Read features data
    print(f"\n[1] Reading feature data from: {input_file}")
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Please run feature_engineering.py first!")
        sys.exit(1)
    
    try:
        df = pd.read_csv(input_file)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"   ✓ Loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        print(f"ERROR reading file: {e}")
        sys.exit(1)
    
    # ============================================
    # TIME SERIES SPLIT
    # ============================================
    print(f"\n[2] Splitting data (time series split)...")
    
    # Training set: before 2023
    train_df = df[df['Date'] < '2023-01-01'].copy()
    
    # Validation set: 2023-01-01 to 2024-12-31
    val_df = df[(df['Date'] >= '2023-01-01') & (df['Date'] < '2025-01-01')].copy()
    
    # Test set: 2025 onwards
    test_df = df[df['Date'] >= '2025-01-01'].copy()
    
    print(f"   - Training set: {len(train_df)} rows ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"   - Validation set: {len(val_df)} rows ({val_df['Date'].min().date()} to {val_df['Date'].max().date()})")
    print(f"   - Test set: {len(test_df)} rows ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
    
    # ============================================
    # PREPARE FEATURES
    # ============================================
    print(f"\n[3] Preparing features...")
    
    # Exclude non-feature columns
    exclude_cols = ['Date', 'target_next_day']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"   - Using {len(feature_cols)} features:")
    print(f"   - {feature_cols}")
    
    # Split data
    X_train = train_df[feature_cols]
    y_train = train_df['target_next_day']
    
    X_val = val_df[feature_cols]
    y_val = val_df['target_next_day']
    
    X_test = test_df[feature_cols]
    y_test = test_df['target_next_day']
    
    print(f"   - X_train shape: {X_train.shape}")
    print(f"   - X_val shape: {X_val.shape}")
    print(f"   - X_test shape: {X_test.shape}")
    
    # ============================================
    # TRAIN XGBOOST MODEL
    # ============================================
    print(f"\n[4] Training XGBoost model...")
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50
    }
    
    print(f"   - Parameters:")
    for key, value in params.items():
        if key != 'early_stopping_rounds':
            print(f"     {key}: {value}")
    
    # Create and train model
    model = xgb.XGBRegressor(**params)
    
    # Fit with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"   ✓ Model trained successfully")
    print(f"   - Best iteration: {model.best_iteration}")
    
    # ============================================
    # EVALUATE ON TEST SET
    # ============================================
    print(f"\n[5] Evaluating on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = calculate_mape(y_test.values, y_pred)
    r2 = r2_score(y_test, y_pred)
    dir_acc = calculate_directional_accuracy(y_test.values, y_pred)
    
    print(f"\n" + "="*60)
    print("XGBOOST MODEL RESULTS")
    print("="*60)
    print(f"Test Period: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
    print(f"Test Samples: {len(y_test)}")
    print(f"\nModel Performance Metrics:")
    print(f"  {'Metric':<25} {'Value':>15}")
    print(f"  {'-'*25} {'-'*15}")
    print(f"  {'MAE (Mean Abs Error)':<25} {mae:>15,.2f} IDR")
    print(f"  {'RMSE (Root MSE)':<25} {rmse:>15,.2f} IDR")
    print(f"  {'MAPE (Mean Abs % Error)':<25} {mape:>15,.2f}%")
    print(f"  {'R² Score':<25} {r2:>15,.4f}")
    print(f"  {'Directional Accuracy':<25} {dir_acc:>15,.2f}%")
    
    # ============================================
    # FEATURE IMPORTANCE
    # ============================================
    print(f"\n[6] Top 10 Feature Importance:")
    
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance_df.head(10).iterrows():
        print(f"   {row['feature']:<20} {row['importance']:.4f}")
    
    # ============================================
    # SAVE MODEL
    # ============================================
    print(f"\n[7] Saving model to: {model_output}")
    joblib.dump(model, model_output)
    print(f"   ✓ Model saved successfully")
    
    # ============================================
    # SAVE PREDICTIONS
    # ============================================
    print(f"\n[8] Saving predictions to: {predictions_output}")
    
    predictions_df = pd.DataFrame({
        'Date': test_df['Date'].values,
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Error': y_test.values - y_pred,
        'Abs_Error': np.abs(y_test.values - y_pred),
        'Pct_Error': np.abs((y_test.values - y_pred) / y_test.values) * 100
    })
    
    predictions_df.to_csv(predictions_output, index=False)
    print(f"   ✓ Predictions saved successfully")
    
    # Sample predictions
    print(f"\n[9] Sample Predictions (first 10):")
    print(f"   {'Date':<12} {'Actual':>12} {'Predicted':>12} {'Error':>12}")
    print(f"   {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for _, row in predictions_df.head(10).iterrows():
        print(f"   {pd.to_datetime(row['Date']).strftime('%Y-%m-%d'):<12} {row['Actual']:>12,.0f} {row['Predicted']:>12,.0f} {row['Error']:>12,.0f}")
    
    print("\n" + "="*60)
    print("XGBoost Training Complete!")
    print("="*60)
    print("""
Next steps:
- Run python src/evaluate_model.py for detailed evaluation
- Run python src/visualize_result.py for visualization
""")

if __name__ == "__main__":
    main()
