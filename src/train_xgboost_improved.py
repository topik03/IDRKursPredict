#!/usr/bin/env python3
"""
train_xgboost_improved.py - Improved XGBoost model for USD/IDR prediction
Purpose: Train XGBoost using returns (percentage changes) for better extrapolation
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
    input_file = Path("data/processed/usd_idr_features_external.csv")
    model_output = Path("models/xgboost_usd_idr_improved.pkl")
    predictions_output = Path("data/processed/xgboost_improved_predictions.csv")
    
    # Create directories
    model_output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("USD/IDR XGBoost IMPROVED Model Training")
    print("="*60)
    
    # Read features data
    print(f"\n[1] Reading feature data from: {input_file}")
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(input_file)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"   ✓ Loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        print(f"ERROR reading file: {e}")
        sys.exit(1)
    
    # ============================================
    # CREATE RETURN-BASED TARGET
    # ============================================
    print(f"\n[2] Creating return-based target...")
    
    # Instead of predicting absolute next day price,
    # we predict the RETURN (percentage change) from today to tomorrow
    # This normalizes the price level issue
    
    # Return = (tomorrow_price - today_price) / today_price
    df['return_target'] = (df['target_next_day'] - df['Close']) / df['Close']
    
    print(f"   ✓ Return target created")
    print(f"   - Return range: {df['return_target'].min():.4f} to {df['return_target'].max():.4f}")
    print(f"   - Return mean: {df['return_target'].mean():.6f}")
    
    # ============================================
    # CREATE RETURN-BASED FEATURES
    # ============================================
    print(f"\n[3] Creating return-based features...")
    
    # Convert price features to returns
    return_features = []
    
    # Lag returns (past 1,2,3,7 days returns)
    for lag in [1, 2, 3, 7, 14]:
        df[f'return_lag_{lag}'] = df['return_1'].shift(lag)
        return_features.append(f'return_lag_{lag}')
        print(f"   ✓ Created return_lag_{lag}")
    
    # Moving average of returns (volatility indicator)
    for window in [7, 14, 30]:
        df[f'return_ma_{window}'] = df['return_1'].rolling(window=window).mean()
        return_features.append(f'return_ma_{window}')
        print(f"   ✓ Created return_ma_{window}")
    
    # Rolling std of returns (volatility)
    for window in [7, 14, 30]:
        df[f'return_std_{window}'] = df['return_1'].rolling(window=window).std()
        return_features.append(f'return_std_{window}')
        print(f"   ✓ Created return_std_{window}")
    
    print(f"\n[3.5] Handling missing values using Forward Fill...")
    # Fill missing values from lags chronologically before splitting
    df = df.ffill().bfill()
    print(f"   ✓ Missing values imputed chronologically")

    # ============================================
    # TIME SERIES SPLIT
    # ============================================
    print(f"\n[4] Splitting data (time series split)...")
    
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
    print(f"\n[5] Preparing features...")
    
# Use return-based features + external returns + calendar
    # NOTE: Must explicitly exclude 'return_target' (it's the target variable, not a feature!)
    exclude_from_features = ['Date', 'target_next_day', 'return_target', 'return_1']
    
    # Include return-based lag features (return_lag_*)
    return_feature_cols = [col for col in df.columns if 'return_lag_' in col]
    # Include return moving averages (return_ma_*)
    return_feature_cols += [col for col in df.columns if 'return_ma_' in col]
    # Include return std features (return_std_*)
    return_feature_cols += [col for col in df.columns if 'return_std_' in col]
    
    # Include external returns (dxy_return, gold_return, etc.)
    external_return_cols = [col for col in df.columns if '_return' in col and col != 'return_1' and 'return_lag' not in col and 'return_ma' not in col and 'return_std' not in col]
    
    # Include calendar features
    calendar_cols = ['day_of_week', 'month', 'quarter', 'is_month_end', 'is_month_start', 'day_of_month', 'week_of_year']
    
    # Include volatility features
    vol_cols = ['std_7', 'std_14', 'std_30']
    
    # Include momentum and correlation features
    momentum_corr_cols = ['rsi_14', 'macd', 'macd_signal', 'corr_dxy_30', 'corr_gold_30']
    
    feature_cols = return_feature_cols + external_return_cols + calendar_cols + vol_cols + momentum_corr_cols
    feature_cols = [col for col in feature_cols if col in df.columns and col not in exclude_from_features]
    feature_cols = list(set(feature_cols))  # Remove duplicates
    
    print(f"   - Using {len(feature_cols)} return-based features")
    
    # Split data
    X_train = train_df[feature_cols]
    y_train = train_df['return_target']
    
    X_val = val_df[feature_cols]
    y_val = val_df['return_target']
    
    X_test = test_df[feature_cols]
    y_test_return = test_df['return_target']
    
    # Store actual prices for later conversion
    y_test_actual = test_df['target_next_day']
    test_close_prices = test_df['Close']
    
    print(f"   - X_train shape: {X_train.shape}")
    print(f"   - X_val shape: {X_val.shape}")
    print(f"   - X_test shape: {X_test.shape}")
    
    # ============================================
    # TRAIN XGBOOST MODEL
    # ============================================
    print(f"\n[6] Training XGBoost model (return-based)...")
    
    # XGBoost parameters - tuned for return prediction
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 500,
        'max_depth': 5,  # Slightly reduced to prevent overfitting
        'learning_rate': 0.05,  # Increased to allow more active fitting
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,  # Decreased to allow more aggressive predictions
        'random_state': 42,
        'n_jobs': 2,
        'early_stopping_rounds': 20
    }
    
    print(f"   - Parameters:")
    for key, value in params.items():
        if key != 'early_stopping_rounds':
            print(f"     {key}: {value}")
    
    # Data is already imputed chronologically via ffill before splitting
    X_train_imputed = X_train.copy()
    X_val_imputed = X_val.copy()
    X_test_imputed = X_test.copy()


    # Create and train model
    model = xgb.XGBRegressor(**params)

    # Fit with early stopping
    model.fit(
        X_train_imputed, y_train,
        eval_set=[(X_val_imputed, y_val)],
        verbose=False
    )

    
    print(f"   ✓ Model trained successfully")
    print(f"   - Best iteration: {model.best_iteration}")
    
    # ============================================
    # PREDICT RETURNS
    # ============================================
    print(f"\n[7] Predicting returns...")
    
    # Predict returns
    predicted_returns = model.predict(X_test)
    
    # Convert predicted returns to absolute prices
    # predicted_price = current_price * (1 + predicted_return)
    predicted_prices = test_close_prices.values * (1 + predicted_returns)
    
    # ============================================
    # EVALUATE ON TEST SET
    # ============================================
    print(f"\n[8] Evaluating on test set...")
    
    # Calculate metrics on ABSOLUTE prices
    mae = mean_absolute_error(y_test_actual.values, predicted_prices)
    rmse = np.sqrt(mean_squared_error(y_test_actual.values, predicted_prices))
    mape = calculate_mape(y_test_actual.values, predicted_prices)
    
    # R² on returns
    r2_returns = r2_score(y_test_return.values, predicted_returns)
    
    # Directional accuracy on returns
    dir_acc = calculate_directional_accuracy(y_test_return.values, predicted_returns)
    
    print(f"\n" + "="*60)
    print("IMPROVED XGBOOST MODEL RESULTS")
    print("="*60)
    print(f"Test Period: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
    print(f"Test Samples: {len(y_test_actual)}")
    print(f"\nModel Performance Metrics:")
    print(f"  {'Metric':<25} {'Value':>15}")
    print(f"  {'-'*25} {'-'*15}")
    print(f"  {'MAE (Mean Abs Error)':<25} {mae:>15,.2f} IDR")
    print(f"  {'RMSE (Root MSE)':<25} {rmse:>15,.2f} IDR")
    print(f"  {'MAPE (Mean Abs % Error)':<25} {mape:>15,.2f}%")
    print(f"  {'R² Score (returns)':<25} {r2_returns:>15,.4f}")
    print(f"  {'Directional Accuracy':<25} {dir_acc:>15,.2f}%")
    
    # Compare with original model (if available)
    print(f"\n[9] Comparison with original model...")
    
    original_predictions_file = Path("data/processed/xgboost_external_predictions.csv")
    if original_predictions_file.exists():
        try:
            orig_df = pd.read_csv(original_predictions_file)
            orig_mae = orig_df['Abs_Error'].mean()
            orig_mape = orig_df['Pct_Error'].mean()
            
            print(f"\n   Original model:")
            print(f"   - MAE: {orig_mae:,.2f} IDR")
            print(f"   - MAPE: {orig_mape:.2f}%")
            
            print(f"\n   Improved model:")
            print(f"   - MAE: {mae:,.2f} IDR")
            print(f"   - MAPE: {mape:.2f}%")
            
            mae_improvement = orig_mae - mae
            mape_improvement = orig_mape - mape
            
            print(f"\n   Improvement:")
            print(f"   - MAE improvement: {mae_improvement:+,.2f} IDR ({mae_improvement/orig_mae*100:.1f}% better)")
            print(f"   - MAPE improvement: {mape_improvement:+.2f}% ({mape_improvement/orig_mape*100:.1f}% better)")
        except Exception as e:
            print(f"   ⚠ Could not load original predictions: {e}")
    
    # ============================================
    # SAVE MODEL
    # ============================================
    print(f"\n[10] Saving model to: {model_output}")
    joblib.dump(model, model_output)
    print(f"   ✓ Model saved successfully")
    
    # Also save model metadata for prediction
    model_metadata = {
        'feature_cols': feature_cols,
        'last_date': test_df['Date'].max(),
        'last_close': test_df['Close'].iloc[-1],
        'model_type': 'return_based'
    }

    metadata_file = Path("models/xgboost_usd_idr_improved_metadata.pkl")
    joblib.dump(model_metadata, metadata_file)
    print(f"   ✓ Metadata saved")
    
    # ============================================
    # SAVE PREDICTIONS
    # ============================================
    print(f"\n[11] Saving predictions to: {predictions_output}")
    
    predictions_df = pd.DataFrame({
        'Date': test_df['Date'].values,
        'Actual': y_test_actual.values,
        'Predicted': predicted_prices,
        'Predicted_Return': predicted_returns,
        'Actual_Return': y_test_return.values,
        'Error': y_test_actual.values - predicted_prices,
        'Abs_Error': np.abs(y_test_actual.values - predicted_prices),
        'Pct_Error': np.abs((y_test_actual.values - predicted_prices) / y_test_actual.values) * 100
    })
    
    predictions_df.to_csv(predictions_output, index=False)
    print(f"   ✓ Predictions saved successfully")
    
    # Sample predictions
    print(f"\n[12] Sample Predictions (first 10):")
    print(f"   {'Date':<12} {'Actual':>12} {'Predicted':>12} {'Return':>10} {'Error':>12}")
    print(f"   {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")
    
    for _, row in predictions_df.head(10).iterrows():
        print(f"   {pd.to_datetime(row['Date']).strftime('%Y-%m-%d'):<12} {row['Actual']:>12,.0f} {row['Predicted']:>12,.0f} {row['Predicted_Return']:>10.4f} {row['Error']:>12,.0f}")
    
    print("\n" + "="*60)
    print("Improved XGBoost Training Complete!")
    print("="*60)
    print("""
Next steps:
- Run python src/predict_latest_improved.py to make predictions
- Run streamlit to view dashboard
""")

if __name__ == "__main__":
    main()
