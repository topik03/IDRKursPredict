#!/usr/bin/env python3
"""
predict_latest_improved.py - Make prediction for latest day using improved model
Purpose: Use trained improved XGBoost model to predict next day's USD/IDR rate with prediction constraints and confidence information
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
from scipy import stats

def create_return_features(df):
    """Create return-based features for prediction"""
    df = df.copy()
    
    # Return lags (past 1,2,3,7,14 days returns)
    for lag in [1, 2, 3, 7, 14]:
        df[f'return_lag_{lag}'] = df['return_1'].shift(lag)
    
    # Moving average of returns (volatility indicator)
    for window in [7, 14, 30]:
        df[f'return_ma_{window}'] = df['return_1'].rolling(window=window).mean()
    
    # Rolling std of returns (volatility)
    for window in [7, 14, 30]:
        df[f'return_std_{window}'] = df['return_1'].rolling(window=window).std()
    
    return df

def get_prediction_constraints(df, last_close):
    """Calculate prediction constraints based on historical volatility"""
    returns = df['return_1'].dropna()
    recent_returns = returns.tail(30)
    std_return = recent_returns.std()
    mean_return = recent_returns.mean()
    max_change_pct = 3 * std_return
    max_change_pct = min(max_change_pct, 0.03)
    
    return {
        'std_return': std_return,
        'mean_return': mean_return,
        'max_change_pct': max_change_pct,
        'max_up': last_close * (1 + max_change_pct),
        'max_down': last_close * (1 - max_change_pct)
    }

def get_prediction_intervals(predicted_return, std_return, last_close, confidence_levels=[0.90, 0.95]):
    """Calculate prediction intervals using historical volatility
    Args:
        predicted_return: Predicted return from model
        std_return: Standard deviation of returns
        last_close: Last closing price
        confidence_levels: List of confidence levels (e.g., [0.90, 0.95])
    Returns:
        Dictionary with intervals for each confidence level
    """
    intervals = {}
    
    for conf_level in confidence_levels:
        # Calculate z-score for two-tailed confidence level
        alpha = 1 - conf_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        # Calculate interval in return space
        return_margin = z_score * std_return
        
        # Convert to price space
        lower_return = predicted_return - return_margin
        upper_return = predicted_return + return_margin
        
        lower_price = last_close * (1 + lower_return)
        upper_price = last_close * (1 + upper_return)
        
        intervals[f'{int(conf_level*100)}%'] = {
            'lower': lower_price,
            'upper': upper_price,
            'lower_return': lower_return * 100,
            'upper_return': upper_return * 100
        }
    
    return intervals

def get_directional_probability(predicted_return, std_return):
    """Calculate probability of up/down movement based on normal distribution
    Args:
        predicted_return: Predicted return from model
        std_return: Standard deviation of returns
    Returns:
        Dictionary with up/down probabilities
    """
    # Use normal distribution to calculate probability
    # P(return > 0) = 1 - CDF(0) = 1 - CDF(-predicted_return / std_return)
    z_score = predicted_return / std_return if std_return > 0 else 0
    
    # Probability of positive return (harga naik)
    prob_up = 1 - stats.norm.cdf(0, loc=predicted_return, scale=std_return)
    prob_down = 1 - prob_up
    
    # If prediction is negative, swap
    if predicted_return < 0:
        prob_up, prob_down = prob_down, prob_up
    
    return {
        'prob_up': prob_up * 100,
        'prob_down': prob_down * 100,
        'z_score': z_score
    }

def get_historical_accuracy():
    """Get historical model accuracy from predictions file
    Returns:
        Dictionary with accuracy metrics
    """
    predictions_file = Path("data/processed/xgboost_improved_predictions.csv")
    
    if not predictions_file.exists():
        return None
    
    try:
        df = pd.read_csv(predictions_file)
        
        # Calculate metrics
        mae = df['Abs_Error'].mean()
        mape = df['Pct_Error'].mean()
        rmse = np.sqrt((df['Error'] ** 2).mean())
        
        # Directional accuracy
        correct_direction = (df['Error'] > 0).sum()
        total = len(df)
        dir_accuracy = (correct_direction / total) * 100 if total > 0 else 0
        
        return {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'directional_accuracy': dir_accuracy,
            'total_predictions': total
        }
    except Exception:
        return None

def get_risk_level(predicted_return, std_return, mape):
    """Determine risk level based on prediction confidence and historical error
    Args:
        predicted_return: Predicted return from model
        std_return: Standard deviation of returns
        mape: Historical Mean Absolute Percentage Error
    Returns:
        Risk level (Rendah/Sedang/Tinggi) with description
    """
    # Combine factors: prediction confidence + historical error
    confidence_factor = abs(predicted_return) / std_return if std_return > 0 else 0
    
    # Low risk: high confidence in strong direction + low historical MAPE
    # High risk: low confidence or high historical MAPE
    
    # Calculate risk score (lower is better)
    if mape < 1.0:
        error_factor = 1  # Low risk from error
    elif mape < 2.0:
        error_factor = 2  # Medium risk
    else:
        error_factor = 3  # High risk
    
    # Confidence score
    if confidence_factor > 2:
        conf_factor = 1  # High confidence
    elif confidence_factor > 1:
        conf_factor = 2  # Medium confidence
    else:
        conf_factor = 3  # Low confidence
    
    risk_score = error_factor + conf_factor
    
    if risk_score <= 2:
        return {
            'level': 'Rendah',
            'score': risk_score,
            'description': 'Model memiliki kepercayaan tinggi dengan error historis rendah'
        }
    elif risk_score <= 4:
        return {
            'level': 'Sedang',
            'score': risk_score,
            'description': 'Model memiliki kepercayaan sedang dengan error moderat'
        }
    else:
        return {
            'level': 'Tinggi',
            'score': risk_score,
            'description': 'Model memiliki kepercayaan rendah atau error historis tinggi'
        }

def main():
    model_file = Path("models/xgboost_usd_idr_improved.pkl")
    metadata_file = Path("models/xgboost_usd_idr_improved_metadata.pkl")
    data_file = Path("data/processed/usd_idr_features_external.csv")
    output_file = Path("data/processed/latest_prediction_improved.csv")
    
    print("="*60)
    print("Latest USD/IDR Prediction (Improved Model)")
    print("="*60)
    
    if not model_file.exists():
        print(f"ERROR: Model not found: {model_file}")
        print("Please run train_xgboost_improved.py first!")
        sys.exit(1)
    
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
    
    # Load metadata (feature columns that model was trained with)
    print(f"\n[2] Loading model metadata...")
    try:
        metadata = joblib.load(metadata_file)
        feature_cols = metadata['feature_cols']
        print(f"   ✓ Model expects {len(feature_cols)} features")
    except Exception as e:
        print(f"ERROR loading metadata: {e}")
        sys.exit(1)
    
    # Load data
    print(f"\n[3] Loading data from: {data_file}")
    try:
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"   ✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)
    
    # Get last row for prediction
    print(f"\n[4] Getting latest data...")
    last_row = df.iloc[-1]
    last_date = last_row['Date']
    last_close = last_row['Close']
    print(f"   - Last date: {last_date}")
    print(f"   - Last close: {last_close:,.2f} IDR")
    
    # Create return-based features (same as training)
    print(f"\n[5] Creating return-based features...")
    df_features = create_return_features(df)
    
    # Forward fill any missing values in the historical sequence before extracting the last row
    df_features = df_features.ffill().bfill()
    
    # Prepare features - USE EXACT SAME FEATURES as model was trained with
    print(f"\n[6] Preparing features for prediction...")
    # Validate feature columns
    missing_cols = [c for c in feature_cols if c not in df_features.columns]
    if missing_cols:
        print(f"ERROR: Missing required feature columns: {missing_cols}")
        sys.exit(1)

    X_last = df_features[feature_cols].iloc[[-1]].copy()

    # Features are already imputed sequentially via ffill on df_features
    # Fallback to 0 only for any remaining NaNs (e.g. return variables if data is completely missing)
    if X_last.isna().any().any():
        X_last = X_last.fillna(0)

    # Final safety: ensure all required feature columns are present & no NaN left
    if X_last.isna().any().any():
        print("ERROR: NaN values still present after imputation")
        sys.exit(1)

    print(f"   - Features shape: {X_last.shape}")


    
    # Make prediction
    print(f"\n[7] Making prediction...")
    try:
        predicted_return = model.predict(X_last)[0]
        print(f"   ✓ Raw predicted return: {predicted_return:.6f} ({predicted_return*100:.4f}%)")
    except Exception as e:
        print(f"ERROR making prediction: {e}")
        sys.exit(1)
    
    # Apply prediction constraints
    print(f"\n[8] Applying prediction constraints...")
    constraints = get_prediction_constraints(df, last_close)
    
    print(f"   - Historical std: {constraints['std_return']:.6f}")
    print(f"   - Mean return: {constraints['mean_return']:.6f}")
    print(f"   - Max allowed change: {constraints['max_change_pct']*100:.2f}%")
    
    # Calculate predicted price
    raw_predicted_next_day = last_close * (1 + predicted_return)
    
    # The improved model predicts returns which are naturally bounded,
    # so we no longer forcefully cap the prediction based on past volatility bounds.
    # This allows the model to be more 'active' during sudden market shifts.
    predicted_next_day = raw_predicted_next_day
    
    # Calculate difference
    difference = predicted_next_day - last_close
    pct_difference = (predicted_next_day / last_close - 1) * 100
    
# Determine direction
    direction = "Naik ↑" if difference > 0 else "Turun ↓"
    
    # Calculate confidence information
    intervals = get_prediction_intervals(
        predicted_return, 
        constraints['std_return'], 
        last_close,
        confidence_levels=[0.90, 0.95]
    )
    
    dir_prob = get_directional_probability(
        predicted_return, 
        constraints['std_return']
    )
    
    hist_accuracy = get_historical_accuracy()
    mape = hist_accuracy['mape'] if hist_accuracy else 2.0
    
    risk_level = get_risk_level(
        predicted_return, 
        constraints['std_return'],
        mape
    )
    
    # Calculate confidence level
    if abs(pct_difference) < 0.5:
        confidence = "Tinggi"
        confidence_pct = 85
    elif abs(pct_difference) < 1.0:
        confidence = "Sedang"
        confidence_pct = 70
    else:
        confidence = "Rendah"
        confidence_pct = 50
    
    # Display results
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS (Improved Model)")
    print(f"{'='*60}")
    print(f"Last Date:         {last_date}")
    print(f"Last Close:       {last_close:,.2f} IDR")
    print(f"Predicted Next:   {predicted_next_day:,.2f} IDR")
    print(f"Change:           {difference:+,.2f} IDR ({pct_difference:+.4f}%)")
    print(f"Direction:        {direction}")
    print(f"Confidence:       {confidence} ({confidence_pct}%)")
    print(f"Max Allowed:      {constraints['max_change_pct']*100:.2f}%")

    
    # Display confidence information
    print(f"\n{'='*60}")
    print("CONFIDENCE INFORMATION")
    print(f"{'='*60}")
    
    # 1. Prediction Intervals
    print(f"\n[1] Prediction Intervals (Rentang Prediksi):")
    print(f"    90% Yakin:  {intervals['90%']['lower']:,.0f} - {intervals['90%']['upper']:,.0f} IDR")
    print(f"    95% Yakin:  {intervals['95%']['lower']:,.0f} - {intervals['95%']['upper']:,.0f} IDR")
    
    # 2. Directional Probability
    print(f"\n[2] Directional Probability (Probabilitas Arah):")
    print(f"    Probabilitas Naik:  {dir_prob['prob_up']:.1f}%")
    print(f"    Probabilitas Turun:  {dir_prob['prob_down']:.1f}%")
    
    # 3. Historical Accuracy
    print(f"\n[3] Historical Accuracy (Akurasi Historis):")
    if hist_accuracy:
        print(f"    MAPE: {hist_accuracy['mape']:.2f}%")
        print(f"    MAE: {hist_accuracy['mae']:,.0f} IDR")
        print(f"    Directional Accuracy: {hist_accuracy['directional_accuracy']:.1f}%")
    else:
        print(f"    Data tidak tersedia")
    
    # 4. Risk Level
    print(f"\n[4] Risk Level (Tingkat Risiko):")
    print(f"    Level: {risk_level['level']}")
    print(f"    Description: {risk_level['description']}")
    
    # Save to CSV
    print(f"\n[9] Saving to: {output_file}")
    
    result_df = pd.DataFrame({
        'Date': [last_date],
        'Last_Close': [last_close],
        'Predicted_Next_Day': [predicted_next_day],
        'Predicted_Return': [predicted_return],
        'Raw_Predicted': [raw_predicted_next_day],
        'Difference': [difference],
        'Pct_Difference': [pct_difference],
        'Direction': [direction],
        'Confidence': [confidence],
        'Confidence_Pct': [confidence_pct],
        'Max_Allowed_Pct': [constraints['max_change_pct'] * 100],
        'Constraints_Applied': [raw_predicted_next_day != predicted_next_day]
    })
    
    result_df.to_csv(output_file, index=False)
    print("   ✓ Saved successfully")
    
    print("\n" + "="*60)
    print("Prediction Complete!")
    print("="*60)
    print(f"""
Note: 
- Prediction is now fully unconstrained to dynamically respond to external features.
- Model predicts daily return percentage based on momentum and correlation.
    """)

if __name__ == "__main__":
    main()
