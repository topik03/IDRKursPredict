#!/usr/bin/env python3
"""
streamlit_app.py - USD/IDR Forecasting Streamlit Dashboard
 Purpose: Interactive dashboard for USD/IDR exchange rate forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib
from datetime import datetime, timedelta

# ============================================
# CONFIGURATION
# ============================================
st.set_page_config(
    page_title="USD/IDR Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

# Use improved model for correct predictions (external model gives ~15,700 which is too low!)
FEATURES_FILE = DATA_DIR / "usd_idr_features_external.csv"
PREDICTIONS_FILE = DATA_DIR / "xgboost_improved_predictions.csv"
MODEL_FILE = MODEL_DIR / "xgboost_usd_idr_improved.pkl"
METADATA_FILE = MODEL_DIR / "xgboost_usd_idr_improved_metadata.pkl"

# ============================================
# DATA LOADING FUNCTIONS
# ============================================
@st.cache_data(ttl=3600)
def load_features_data():
    """Load features data"""
    try:
        df = pd.read_csv(FEATURES_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading features: {e}")
        return None

@st.cache_data(ttl=3600)
def load_predictions_data():
    """Load predictions data"""
    try:
        df = pd.read_csv(PREDICTIONS_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return None

@st.cache_data(ttl=3600)
def load_model():
    """Load trained model"""
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data(ttl=3600)
def load_metadata():
    """Load model metadata (feature columns)"""
    try:
        metadata = joblib.load(METADATA_FILE)
        return metadata
    except Exception as e:
        st.warning(f"Metadata not found, using default features: {e}")
        return None

def create_return_features(df):
    """Create return-based features for improved model"""
    df = df.copy()
    
    # Return lags
    for lag in [1, 2, 3, 7, 14]:
        df[f'return_lag_{lag}'] = df['return_1'].shift(lag)
    
    # Moving average of returns
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
    max_change_pct = min(max_change_pct, 0.03)  # Cap at 3%
    
    return {
        'max_up': last_close * (1 + max_change_pct),
        'max_down': last_close * (1 - max_change_pct),
        'std_return': std_return,
        'mean_return': mean_return,
        'max_change_pct': max_change_pct
    }

def get_historical_accuracy_app(predictions_file_path):
    """Get historical accuracy metrics from improved predictions csv"""
    try:
        p = Path(predictions_file_path)
        if not p.exists():
            return None

        df = pd.read_csv(p)
        for col in ['Error', 'Abs_Error', 'Pct_Error', 'Actual', 'Predicted']:
            if col not in df.columns:
                return None

        mae = df['Abs_Error'].mean()
        mape = df['Pct_Error'].mean()
        rmse = np.sqrt((df['Error'] ** 2).mean())

        correct_direction = (df['Error'] > 0).sum()
        total = len(df)
        dir_accuracy = (correct_direction / total) * 100 if total > 0 else 0

        return {
            'mae': float(mae),
            'mape': float(mape),
            'rmse': float(rmse),
            'directional_accuracy': float(dir_accuracy),
            'total_predictions': int(total)
        }
    except Exception:
        return None

def get_risk_level_app(predicted_return, std_return, mape):
    """Determine risk level based on prediction confidence and historical error (replica of CLI logic)."""
    confidence_factor = abs(predicted_return) / std_return if std_return > 0 else 0

    if mape < 1.0:
        error_factor = 1
    elif mape < 2.0:
        error_factor = 2
    else:
        error_factor = 3

    if confidence_factor > 2:
        conf_factor = 1
    elif confidence_factor > 1:
        conf_factor = 2
    else:
        conf_factor = 3

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

def get_prediction_intervals_app(predicted_return, std_return, last_close, confidence_levels=[0.90, 0.95]):
    """Calculate prediction intervals using historical volatility"""
    from scipy import stats
    intervals = {}
    
    for conf_level in confidence_levels:
        alpha = 1 - conf_level
        z_score = stats.norm.ppf(1 - alpha/2)
        return_margin = z_score * std_return
        
        lower_return = predicted_return - return_margin
        upper_return = predicted_return + return_margin
        
        lower_price = last_close * (1 + lower_return)
        upper_price = last_close * (1 + upper_return)
        
        intervals[f'{int(conf_level*100)}%'] = {
            'lower': lower_price,
            'upper': upper_price
        }
    
    return intervals

def get_directional_probability_app(predicted_return, std_return):
    """Calculate probability of up/down movement based on normal distribution"""
    from scipy import stats
    
    z_score = predicted_return / std_return if std_return > 0 else 0
    
    prob_up = 1 - stats.norm.cdf(0, loc=predicted_return, scale=std_return)
    prob_down = 1 - prob_up
    
    if predicted_return < 0:
        prob_up, prob_down = prob_down, prob_up
    
    return {
        'prob_up': prob_up * 100,
        'prob_down': prob_down * 100,
        'z_score': z_score
    }

def get_latest_prediction(features_df, model):
    """Get latest prediction using improved model"""
    try:
        # Get last row
        last_row = features_df.iloc[-1]
        last_close = last_row['Close']
        last_date = last_row['Date']

        metadata = load_metadata()

        # Improved model path
        if metadata is not None:
            feature_cols = metadata.get('feature_cols', [])
            impute_values = metadata.get('impute_values', None)

            if not feature_cols:
                raise ValueError("Metadata tidak berisi feature_cols")

            # Create return features as used during training/prediction
            df_features = create_return_features(features_df.copy())
            
            # Forward fill the historical sequence to impute missing values before extracting last row
            df_features = df_features.ffill().bfill()

            # Prepare features for prediction
            missing_cols = [c for c in feature_cols if c not in df_features.columns]
            if missing_cols:
                raise ValueError(f"Missing required feature columns: {missing_cols}")

            X_last = df_features[feature_cols].iloc[[-1]].copy()

            nan_count_before = int(X_last.isna().sum().sum())

            impute_strategy = "ffill_historical_sequence"
            
            # Features are already imputed sequentially via ffill on df_features
            # Fallback to 0 only for any remaining NaNs
            if X_last.isna().any().any():
                X_last = X_last.fillna(0)

            nan_count_after = int(X_last.isna().sum().sum())

            if X_last.isna().any().any():
                # Safety
                X_last = X_last.fillna(0)

            # Ensure 2D for model input (scikit-learn accepts DataFrame directly)
            # X_last is already a single-row DataFrame here.
            # (Do not call .to_frame() because X_last is not a Series.)
            # X_last = X_last

            # Predict raw return
            predicted_return = float(model.predict(X_last)[0])
            raw_predicted_next_day = last_close * (1 + predicted_return)

            # Apply constraints
            constraints = get_prediction_constraints(features_df, last_close)
            std_return = float(constraints['std_return'])
            predicted = raw_predicted_next_day
            constraints_applied = False

            # Prediction is naturally unconstrained
            predicted = raw_predicted_next_day
            constraints_applied = False

            difference = predicted - last_close
            pct_difference = (predicted / last_close - 1) * 100
            direction = 'Naik ↑' if difference > 0 else 'Turun ↓'

            # Confidence intervals & probabilities
            intervals = get_prediction_intervals_app(predicted_return, std_return, last_close)
            dir_prob = get_directional_probability_app(predicted_return, std_return)

            # Confidence tier as in CLI improved
            if abs(pct_difference) < 0.5:
                confidence = "Tinggi"
                confidence_pct = 85
            elif abs(pct_difference) < 1.0:
                confidence = "Sedang"
                confidence_pct = 70
            else:
                confidence = "Rendah"
                confidence_pct = 50

            # Risk level & historical accuracy
            hist_accuracy = get_historical_accuracy_app(DATA_DIR / "xgboost_improved_predictions.csv")
            mape = hist_accuracy['mape'] if hist_accuracy else 2.0
            risk_level = get_risk_level_app(predicted_return, std_return, mape)

            # Diagnostics for UI
            X_last_series = X_last.iloc[0].copy()
            X_last_values = {str(k): (None if pd.isna(v) else float(v)) for k, v in X_last_series.items()}

            diagnostics = {
                'metadata_feature_cols_count': int(len(feature_cols)),
                'metadata_impute_strategy': impute_strategy,
                'missing_cols': missing_cols,
                'nan_count_before': nan_count_before,
                'nan_count_after': nan_count_after,
                'features_used': feature_cols,
                'x_last_values': X_last_values,
                'model_file_used': str(MODEL_FILE),
                'intervals': intervals,
                'dir_prob': dir_prob,
            }

            return {
                'last_date': last_date,
                'last_close': last_close,
                'predicted': predicted,
                'difference': difference,
                'pct_difference': pct_difference,
                'predicted_return': predicted_return,
                'raw_predicted_next_day': raw_predicted_next_day,
                'direction': direction,
                'intervals': intervals,
                'dir_prob': dir_prob,
                'std_return': std_return,
                'constraints': constraints,
                'constraints_applied': constraints_applied,
                'confidence': confidence,
                'confidence_pct': confidence_pct,
                'risk_level': risk_level,
                'hist_accuracy': hist_accuracy,
                'diagnostics': diagnostics
            }

        # Fallback: external model path
        exclude_cols = ['Date', 'target_next_day']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        X_last = features_df[feature_cols].iloc[[-1]]

        predicted = float(model.predict(X_last)[0])

        return {
            'last_date': last_date,
            'last_close': last_close,
            'predicted': predicted,
            'difference': predicted - last_close,
            'pct_difference': (predicted / last_close - 1) * 100,
            'direction': 'Naik ↑' if predicted > last_close else 'Turun ↓'
        }
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# ============================================
# PAGE FUNCTIONS
# ============================================
def render_overview(prediction, predictions_df, features_df=None):
    """Render Overview page with detailed visualizations"""
    st.markdown("## 📊 Overview")
    
    # Latest prediction card
    if prediction:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Last Date",
                prediction['last_date'].strftime("%Y-%m-%d"),
                delta=None
            )
        
        with col2:
            st.metric(
                "Last Close (IDR)",
                f"{prediction['last_close']:,.0f}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Predicted Next Day (IDR)",
                f"{prediction['predicted']:,.0f}",
                delta=f"{prediction['difference']:+,.0f}"
            )
        
        with col4:
            direction = prediction['direction']
            direction_color = "normal" if "Naik" in direction else "inverse"
            st.metric(
                "Direction",
                direction,
                delta=None
            )
    
    # Display confidence information if available
    if prediction and 'intervals' in prediction and prediction['intervals']:
        st.markdown("### 📊 Confidence Information")
        
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        
        with col_c1:
            # Prediction Intervals
            intervals = prediction['intervals']
            st.markdown("**Prediction Intervals:**")
            st.caption(f"90% Yakin: {intervals['90%']['lower']:,.0f} - {intervals['90%']['upper']:,.0f}")
            st.caption(f"95% Yakin: {intervals['95%']['lower']:,.0f} - {intervals['95%']['upper']:,.0f}")
        
        with col_c2:
            # Directional Probability
            dir_prob = prediction.get('dir_prob', {})
            st.markdown("**Directional Probability:**")
            st.caption(f"Naik: {dir_prob.get('prob_up', 0):.1f}%")
            st.caption(f"Turun: {dir_prob.get('prob_down', 0):.1f}%")
        
        with col_c3:
            # Volatility
            std_return = prediction.get('std_return', 0)
            st.markdown("**Volatility (Std Dev):**")
            st.caption(f"Return Std: {std_return*100:.4f}%")
        
        with col_c4:
            # Risk Level (simplified)
            if predictions_df is not None and len(predictions_df) > 0:
                mape = predictions_df['Pct_Error'].mean()
                if mape < 1.0:
                    risk = "Rendah"
                    risk_color = "green"
                elif mape < 2.0:
                    risk = "Sedang"
                    risk_color = "orange"
                else:
                    risk = "Tinggi"
                    risk_color = "red"
                st.markdown("**Risk Level:**")
                st.caption(f"Level: {risk}")
                st.caption(f"MAPE: {mape:.2f}%")
    
    # ============================================
    # DEBUG: Detailed prediction info
    # ============================================
    if prediction and ('predicted_return' in prediction or 'constraints' in prediction):
        with st.expander("🔍 Prediction Debug Detail (Improved Model)", expanded=False):
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Prediction Core**")
                st.caption(f"Last Date: {prediction['last_date'].strftime('%Y-%m-%d')}")
                st.caption(f"Last Close: {prediction['last_close']:,.2f}")
                st.caption(f"Raw Predicted Next Day: {prediction.get('raw_predicted_next_day', np.nan):,.2f}")
                st.caption(f"Constrained Predicted Next Day: {prediction['predicted']:,.2f}")
                st.caption(f"Difference: {prediction['difference']:+,.2f} IDR ({prediction.get('pct_difference', 0):+.4f}%)")

                st.caption(f"Predicted Return (raw): {prediction.get('predicted_return', np.nan):+.6f} ({prediction.get('predicted_return', 0)*100:+.4f}%)")
                st.caption(f"Constraints applied: {'✅' if prediction.get('constraints_applied') else '❌'}")

                intervals = prediction.get('intervals', {})
                if intervals:
                    st.markdown("**Prediction Intervals (Quick View)**")
                    st.caption(f"90%: {intervals['90%']['lower']:,.0f} - {intervals['90%']['upper']:,.0f}")
                    st.caption(f"95%: {intervals['95%']['lower']:,.0f} - {intervals['95%']['upper']:,.0f}")

            with c2:
                constraints = prediction.get('constraints', {})
                st.markdown("**Constraints (volatility-based)**")
                st.caption(f"Std Return (30d): {constraints.get('std_return', 0)*100:.4f}%")
                st.caption(f"Mean Return (30d): {constraints.get('mean_return', 0)*100:.6f}%")
                st.caption(f"Max allowed change: {constraints.get('max_change_pct', 0)*100:.2f}%")
                st.caption(f"Max Up: {constraints.get('max_up', np.nan):,.2f}")
                st.caption(f"Max Down: {constraints.get('max_down', np.nan):,.2f}")

                dir_prob = prediction.get('dir_prob', {})
                st.markdown("**Directional Probability**")
                st.caption(f"Naik: {dir_prob.get('prob_up', 0):.1f}%")
                st.caption(f"Turun: {dir_prob.get('prob_down', 0):.1f}%")
                if 'z_score' in dir_prob:
                    st.caption(f"Z-score: {dir_prob.get('z_score', 0):.4f}")

                st.markdown("**Confidence & Risk**")
                st.caption(f"Confidence: {prediction.get('confidence')} ({prediction.get('confidence_pct')}%)")
                risk = prediction.get('risk_level', {})
                if risk:
                    st.caption(f"Risk Level: {risk.get('level')}")
                    st.caption(f"Risk Description: {risk.get('description')}")

                hist = prediction.get('hist_accuracy')
                if hist:
                    st.markdown("**Historical Accuracy (from test set)**")
                    st.caption(f"MAE: {hist.get('mae', 0):,.0f} IDR")
                    st.caption(f"MAPE: {hist.get('mape', 0):.2f}%")
                    st.caption(f"RMSE: {hist.get('rmse', 0):,.0f} IDR")
                    st.caption(f"Directional Accuracy: {hist.get('directional_accuracy', 0):.1f}%")
                    st.caption(f"Total predictions: {hist.get('total_predictions', 0)}")

            # ===============================
            # Additional detailed diagnostics
            # ===============================
            diagnostics = prediction.get('diagnostics', None)

            if diagnostics:
                st.divider()
                with st.expander("🧪 Data & Model Diagnostics", expanded=False):
                    st.caption(f"Model file used: {diagnostics.get('model_file_used', '-')}")
                    st.caption(f"Metadata feature columns count: {diagnostics.get('metadata_feature_cols_count', '-')}")
                    st.caption(f"Impute strategy: {diagnostics.get('metadata_impute_strategy', '-')}")
                    st.caption(f"Missing cols (should be empty): {diagnostics.get('missing_cols', [])}")
                    st.caption(f"NaN count before impute: {diagnostics.get('nan_count_before', '-')}")
                    st.caption(f"NaN count after impute: {diagnostics.get('nan_count_after', '-')}")

                with st.expander("📦 Features Used (Last Row) - FULL", expanded=False):
                    x_last_values = diagnostics.get('x_last_values', {})
                    if x_last_values:
                        feat_df = (
                            pd.DataFrame(
                                [{"feature": k, "value": v} for k, v in x_last_values.items()]
                            )
                            .sort_values("feature")
                        )
                        feat_df["value"] = feat_df["value"].map(lambda x: float(x) if x is not None else np.nan)
                        st.dataframe(
                            feat_df,
                            use_container_width=True,
                            hide_index=True,
                            height=650
                        )
                    else:
                        st.warning("No feature diagnostics available.")

                with st.expander("📊 Interval & Probability Detail", expanded=False):
                    intervals = diagnostics.get('intervals', {})
                    dir_prob = diagnostics.get('dir_prob', {})

                    if intervals:
                        interval_rows = []
                        for k, v in intervals.items():
                            interval_rows.append({
                                "confidence": k,
                                "lower": v["lower"],
                                "upper": v["upper"]
                            })
                        interval_df = pd.DataFrame(interval_rows)
                        interval_df["lower"] = interval_df["lower"].apply(lambda x: f"{x:,.0f}")
                        interval_df["upper"] = interval_df["upper"].apply(lambda x: f"{x:,.0f}")
                        st.table(interval_df)

                    st.caption(f"Probability Naik (%): {dir_prob.get('prob_up', 0):.1f}")
                    st.caption(f"Probability Turun (%): {dir_prob.get('prob_down', 0):.1f}")
                    if 'z_score' in dir_prob:
                        st.caption(f"Z-score: {dir_prob.get('z_score', 0):.4f}")

        st.divider()
    
    # ============================================
    # CHART 1: Recent Price Trend with Moving Averages
    # ============================================
    st.markdown("### 📈 Recent Price Trend (Last 90 Days)")
    
    if features_df is not None and len(features_df) > 0:
        # Get last 90 days of data
        recent_df = features_df.tail(90).copy()
        
        fig_price = go.Figure()
        
        # Close price line
        fig_price.add_trace(go.Scatter(
            x=recent_df['Date'],
            y=recent_df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#2E86AB', width=2),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.1)'
        ))
        
        # Add Moving Averages if available
        if 'ma_7' in recent_df.columns:
            fig_price.add_trace(go.Scatter(
                x=recent_df['Date'],
                y=recent_df['ma_7'],
                mode='lines',
                name='MA 7',
                line=dict(color='#F18F01', width=1.5, dash='dot')
            ))
        
        if 'ma_14' in recent_df.columns:
            fig_price.add_trace(go.Scatter(
                x=recent_df['Date'],
                y=recent_df['ma_14'],
                mode='lines',
                name='MA 14',
                line=dict(color='#C73E1D', width=1.5, dash='dot')
            ))
        
        if 'ma_30' in recent_df.columns:
            fig_price.add_trace(go.Scatter(
                x=recent_df['Date'],
                y=recent_df['ma_30'],
                mode='lines',
                name='MA 30',
                line=dict(color='#6A994E', width=1.5, dash='dot')
            ))
        
        # Add latest prediction point
        if prediction:
            fig_price.add_trace(go.Scatter(
                x=[prediction['last_date'] + pd.Timedelta(days=1)],
                y=[prediction['predicted']],
                mode='markers',
                name='Predicted Next Day',
                marker=dict(
                    symbol='diamond',
                    size=12,
                    color='#E94F37',
                    line=dict(width=2, color='white')
                )
            ))
        
        fig_price.update_layout(
            title='USD/IDR Price Trend - Last 90 Days',
            xaxis_title='Date',
            yaxis_title='USD/IDR Exchange Rate',
            hovermode='x unified',
            template='plotly_white',
            height=400,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.warning("No historical data available for visualization")
    
    st.divider()
    
    # ============================================
    # CHART 2 & 3: Model Performance & Error Analysis
    # ============================================
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 🎯 Model Prediction Accuracy")
        
        if predictions_df is not None and len(predictions_df) > 0:
            # Get last 30 predictions for visualization
            recent_pred = predictions_df.tail(30).copy()
            
            fig_accuracy = go.Figure()
            
            # Actual line
            fig_accuracy.add_trace(go.Scatter(
                x=recent_pred['Date'],
                y=recent_pred['Actual'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='#2E86AB', width=2),
                marker=dict(size=6)
            ))
            
            # Predicted line
            fig_accuracy.add_trace(go.Scatter(
                x=recent_pred['Date'],
                y=recent_pred['Predicted'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='#E94F37', width=2, dash='dash'),
                marker=dict(size=6, symbol='diamond')
            ))
            
            fig_accuracy.update_layout(
                title='Actual vs Predicted (Last 30 Days)',
                xaxis_title='Date',
                yaxis_title='USD/IDR',
                hovermode='x unified',
                template='plotly_white',
                height=350,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig_accuracy, use_container_width=True)
        else:
            st.warning("No prediction data available")
    
    with col_right:
        st.markdown("### 📉 Error Distribution")
        
        if predictions_df is not None and len(predictions_df) > 0:
            fig_error = go.Figure()
            
            # Error histogram
            fig_error.add_trace(go.Histogram(
                x=predictions_df['Error'],
                name='Prediction Error',
                marker_color='#2E86AB',
                opacity=0.75,
                nbinsx=30
            ))
            
            # Add vertical line at zero
            fig_error.add_vline(x=0, line_dash="dash", line_color="red", 
                             annotation_text="Zero Error", annotation_position="top right")
            
            # Add mean line
            mean_error = predictions_df['Error'].mean()
            fig_error.add_vline(x=mean_error, line_dash="dash", line_color="green",
                            annotation_text=f"Mean: {mean_error:,.0f}", annotation_position="top left")
            
            fig_error.update_layout(
                title='Error Distribution (Actual - Predicted)',
                xaxis_title='Error (IDR)',
                yaxis_title='Frequency',
                template='plotly_white',
                height=350,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig_error, use_container_width=True)
        else:
            st.warning("No error data available")
    
    st.divider()
    
    # ============================================
    # CHART 4: Error Over Time
    # ============================================
    st.markdown("### 📊 Prediction Error Over Time")
    
    if predictions_df is not None and len(predictions_df) > 0:
        # Get last 60 days of predictions
        error_df = predictions_df.tail(60).copy()
        
        fig_error_time = go.Figure()
        
        # Error bars with color coding
        colors = ['#2E86AB' if e >= 0 else '#E94F37' for e in error_df['Error']]
        
        fig_error_time.add_trace(go.Bar(
            x=error_df['Date'],
            y=error_df['Error'],
            name='Error',
            marker=dict(color=colors)
        ))
        
        # Add zero line
        fig_error_time.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig_error_time.update_layout(
            title='Prediction Error Over Time (Last 60 Days)',
            xaxis_title='Date',
            yaxis_title='Error (Actual - Predicted)',
            template='plotly_white',
            height=350,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig_error_time, use_container_width=True)
    
    st.divider()
    
    # ============================================
    # Summary Statistics Table
    # ============================================
    st.markdown("### 📈 Model Summary Statistics")
    
    if predictions_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        mae = predictions_df['Abs_Error'].mean()
        mape = predictions_df['Pct_Error'].mean()
        rmse = np.sqrt((predictions_df['Error'] ** 2).mean())
        r2 = 1 - (predictions_df['Error'].var() / predictions_df['Actual'].var())
        
        with col1:
            st.metric("Mean Absolute Error (MAE)", f"{mae:,.0f} IDR")
        
        with col2:
            st.metric("Mean Absolute % Error (MAPE)", f"{mape:.2f}%")
        
        with col3:
            st.metric("Root Mean Square Error (RMSE)", f"{rmse:,.0f} IDR")
        
        with col4:
            st.metric("R² Score", f"{r2:.4f}")
        
        # Additional statistics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            min_error = predictions_df['Error'].min()
            st.metric("Min Error", f"{min_error:+,.0f} IDR")
        
        with col6:
            max_error = predictions_df['Error'].max()
            st.metric("Max Error", f"{max_error:+,.0f} IDR")
        
        with col7:
            std_error = predictions_df['Error'].std()
            st.metric("Error Std Dev", f"{std_error:,.0f} IDR")
        
        with col8:
            st.metric("Total Predictions", len(predictions_df))

def render_historical_trends(features_df):
    """Render Historical Trends page"""
    st.markdown("## 📈 Historical Trends")
    
    if features_df is None:
        st.warning("No data available")
        return
    
    # Date range selector
    min_date = features_df['Date'].min().date()
    max_date = features_df['Date'].max().date()
    
    date_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    
    # Filter data
    mask = (features_df['Date'].dt.date >= date_range[0]) & (features_df['Date'].dt.date <= date_range[1])
    filtered_df = features_df[mask]
    
    # Create chart
    fig = go.Figure()
    
    # Add close price line
    fig.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#2E86AB', width=2),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.1)'
    ))
    
    # Add moving averages if available
    if 'ma_7' in filtered_df.columns:
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['ma_7'],
            mode='lines',
            name='MA 7',
            line=dict(color='#F18F01', width=1, dash='dot')
        ))
    
    if 'ma_30' in filtered_df.columns:
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['ma_30'],
            mode='lines',
            name='MA 30',
            line=dict(color='#C73E1D', width=1, dash='dot')
        ))
    
    # Update layout
    fig.update_layout(
        title='USD/IDR Exchange Rate Over Time',
        xaxis_title='Date',
        yaxis_title='USD/IDR',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics for selected range
    st.markdown("### 📊 Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Min", f"{filtered_df['Close'].min():,.0f}")
    with col2:
        st.metric("Max", f"{filtered_df['Close'].max():,.0f}")
    with col3:
        st.metric("Mean", f"{filtered_df['Close'].mean():,.0f}")
    with col4:
        st.metric("Std", f"{filtered_df['Close'].std():,.0f}")

def render_model_performance(predictions_df):
    """Render Model Performance page"""
    st.markdown("## 🎯 Model Performance")
    
    if predictions_df is None:
        st.warning("No prediction data available")
        return
    
    # Date range selector
    min_date = predictions_df['Date'].min().date()
    max_date = predictions_df['Date'].max().date()
    
    date_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    
    # Filter data
    mask = (predictions_df['Date'].dt.date >= date_range[0]) & (predictions_df['Date'].dt.date <= date_range[1])
    filtered_df = predictions_df[mask]
    
    # Chart 1: Actual vs Predicted
    st.markdown("### Actual vs Predicted")
    
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Actual'],
        mode='lines',
        name='Actual',
        line=dict(color='#2E86AB', width=2)
    ))
    
    fig1.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Predicted'],
        mode='lines',
        name='Predicted',
        line=dict(color='#E94F37', width=2, dash='dash')
    ))
    
    fig1.update_layout(
        title='Actual vs Predicted Exchange Rate',
        xaxis_title='Date',
        yaxis_title='USD/IDR',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
# Chart 2: Error Distribution
    st.markdown("### Error Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Error over time
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            x=filtered_df['Date'],
            y=filtered_df['Error'],
            name='Error',
            marker=dict(
                color=filtered_df['Error'],
                colorscale='RdYlGn',
                cmid=0
            )
        ))
        
        fig2.update_layout(
            title='Prediction Error Over Time',
            xaxis_title='Date',
            yaxis_title='Error (Actual - Predicted)',
            template='plotly_white',
            height=300
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Error distribution
        fig3 = go.Figure()
        
        fig3.add_trace(go.Histogram(
            x=filtered_df['Error'],
            name='Error Distribution',
            marker_color='#2E86AB',
            opacity=0.75
        ))
        
        fig3.update_layout(
            title='Error Distribution',
            xaxis_title='Error',
            yaxis_title='Frequency',
            template='plotly_white',
            height=300
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    # Metrics
    st.markdown("### 📊 Performance Metrics")
    
    mae = filtered_df['Abs_Error'].mean()
    mape = filtered_df['Pct_Error'].mean()
    rmse = np.sqrt((filtered_df['Error'] ** 2).mean())
    r2 = 1 - (filtered_df['Error'].var() / filtered_df['Actual'].var())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAE", f"{mae:,.0f} IDR")
    with col2:
        st.metric("MAPE", f"{mape:.2f}%")
    with col3:
        st.metric("RMSE", f"{rmse:,.0f} IDR")
    with col4:
        st.metric("R² Score", f"{r2:.4f}")

def render_predictions_table(predictions_df):
    """Render Predictions Table page"""
    st.markdown("## 📋 Predictions Table")
    
    if predictions_df is None:
        st.warning("No prediction data available")
        return
    
    # Date range selector
    min_date = predictions_df['Date'].min().date()
    max_date = predictions_df['Date'].max().date()
    
    date_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    
    # Filter data
    mask = (predictions_df['Date'].dt.date >= date_range[0]) & (predictions_df['Date'].dt.date <= date_range[1])
    filtered_df = predictions_df[mask].copy()
    
    # Format columns
    display_df = filtered_df.copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df['Actual'] = display_df['Actual'].apply(lambda x: f"{x:,.0f}")
    display_df['Predicted'] = display_df['Predicted'].apply(lambda x: f"{x:,.2f}")
    display_df['Error'] = display_df['Error'].apply(lambda x: f"{x:+,.2f}")
    display_df['Abs_Error'] = display_df['Abs_Error'].apply(lambda x: f"{x:,.2f}")
    display_df['Pct_Error'] = display_df['Pct_Error'].apply(lambda x: f"{x:.2f}%")
    
    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=500
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="usd_idr_predictions.csv",
        mime="text/csv"
    )

def render_model_insights():
    """Render Model Insights / Explainability page"""
    st.markdown("## 🧠 Model Insights (Explainability)")
    st.markdown("Pahami alasan di balik setiap prediksi AI. Fitur apa yang paling berpengaruh terhadap tebakan harga Rupiah?")
    
    import os
    feature_importance_path = "reports/figures/feature_importance_top20.png"
    shap_path = "reports/figures/shap_summary.png"
    csv_path = "reports/feature_importance.csv"
    
    if os.path.exists(shap_path):
        st.markdown("### 🎯 Dampak Arah (SHAP Values)")
        st.info("Grafik ini menunjukkan seberapa kuat dorongan suatu fitur terhadap prediksi, dan ke arah mana dorongannya. Semakin merah warnanya, semakin tinggi nilai asli fitur tersebut.")
        st.image(shap_path, use_column_width=True)
        st.markdown("---")
        
    if os.path.exists(feature_importance_path):
        st.markdown("### 📊 Top 20 Fitur Paling Berpengaruh")
        st.info("Grafik ini menunjukkan secara umum (secara global), fitur apa yang paling sering dipakai oleh AI untuk membuat tebakan yang akurat.")
        st.image(feature_importance_path, use_column_width=True)
        st.markdown("---")
        
    if os.path.exists(csv_path):
        st.markdown("### 📋 Detail Skor Fitur")
        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Failed to load data: {e}")

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main application"""
    # Load data
    features_df = load_features_data()
    predictions_df = load_predictions_data()
    model = load_model()
    
    # Get latest prediction
    prediction = get_latest_prediction(features_df, model) if (features_df is not None and model is not None) else None
    
    # Sidebar
    st.sidebar.title("📈 USD/IDR Forecast")
    st.sidebar.markdown("---")
    
    # Navigation
    pages = {
        "Overview": "overview",
        "Historical Trends": "trends",
        "Model Performance": "performance",
        "Predictions Table": "table",
        "Model Insights": "insights"
    }
    
    selected_page = st.sidebar.radio("Navigation", list(pages.keys()))
    
    st.sidebar.markdown("---")
    
    # Auto-Update Button
    st.sidebar.markdown("### 🔄 Auto-Update")
    if st.sidebar.button("Update Data & Retrain Model"):
        with st.spinner("Downloading latest data and retraining model... (Takes ~1-2 minutes)"):
            import runpy
            import io
            from contextlib import redirect_stdout, redirect_stderr
            import traceback
            
            f = io.StringIO()
            try:
                with redirect_stdout(f), redirect_stderr(f):
                    runpy.run_path("src/run_pipeline.py", run_name="__main__")
                
                st.sidebar.success("✅ Update successful!")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.sidebar.error("❌ Update failed!")
                with st.sidebar.expander("Show Error Logs"):
                    error_trace = traceback.format_exc()
                    st.text(f.getvalue() + "\n" + error_trace)
            except SystemExit as e:
                if e.code == 0 or e.code is None:
                    st.sidebar.success("✅ Update successful!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.sidebar.error("❌ Update failed (SystemExit)!")
                    with st.sidebar.expander("Show Error Logs"):
                        st.text(f.getvalue() + f"\nScript exited with code {e.code}")
                
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "USD/IDR Exchange Rate Forecasting Dashboard\n\n"
        "Powered by XGBoost ML Model"
    )
    
# Render selected page
    if selected_page == "Overview":
        render_overview(prediction, predictions_df, features_df)
    elif selected_page == "Historical Trends":
        render_historical_trends(features_df)
    elif selected_page == "Model Performance":
        render_model_performance(predictions_df)
    elif selected_page == "Predictions Table":
        render_predictions_table(predictions_df)
    elif selected_page == "Model Insights":
        render_model_insights()

if __name__ == "__main__":
    main()
