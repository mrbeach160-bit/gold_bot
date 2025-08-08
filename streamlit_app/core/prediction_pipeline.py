"""
Prediction Pipeline Module

Contains model prediction and ensemble logic.
"""

import numpy as np
import pandas as pd

# Import streamlit for temporary compatibility (should be refactored to use proper logging)
try:
    import streamlit as st
except ImportError:
    # Fallback for when streamlit is not available (e.g., in tests)
    class MockStreamlit:
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
    st = MockStreamlit()


def predict_with_models(models, data):
    """
    Make predictions with ensemble of models and better error handling
    
    Args:
        models: Dictionary of trained models
        data: DataFrame with market data and indicators
        
    Returns:
        tuple: (signal, confidence, predicted_price)
    """
    try:
        min_data_needed = 60
        if len(data) < min_data_needed + 1:
            return "HOLD", 0.5, data['close'].iloc[-1] if not data.empty else 0

        if models.get('meta') is None:
            st.warning("⚠️ Meta learner model tidak tersedia")
            return "HOLD", 0.5, data['close'].iloc[-1]

        # LSTM Prediction with validation
        lstm_prediction_price, lstm_pred_diff = _predict_with_lstm(models, data)
        
        # XGBoost Prediction
        xgb_pred, xgb_confidence = _predict_with_xgboost(models, data)
        
        # Other models predictions with simplified error handling
        cnn_pred, svc_pred, nb_pred, svc_confidence, nb_confidence = _predict_with_other_models(models, data)

        # Meta learner prediction
        final_signal, final_confidence = _predict_with_meta_learner(
            models, data, lstm_pred_diff, xgb_pred, xgb_confidence,
            cnn_pred, svc_pred, svc_confidence, nb_pred, nb_confidence
        )
        
        return final_signal, final_confidence, lstm_prediction_price

    except Exception as e:
        st.error(f"❌ Critical error in model prediction: {str(e)}")
        return "HOLD", 0.5, data['close'].iloc[-1] if not data.empty else 0


def _predict_with_lstm(models, data):
    """Predict with LSTM model"""
    try:
        sequence_length_lstm = 60
        if len(data) < sequence_length_lstm:
            return data['close'].iloc[-1], 0
            
        last_lstm_data = data['close'].tail(sequence_length_lstm).values
        
        if models.get('scaler') is None:
            st.warning("⚠️ LSTM scaler tidak tersedia")
            return data['close'].iloc[-1], 0
            
        scaled_lstm_data = models['scaler'].transform(last_lstm_data.reshape(-1, 1))
        X_lstm = np.reshape(scaled_lstm_data, (1, sequence_length_lstm, 1))
        
        lstm_prediction_scaled = models['lstm'].predict(X_lstm, verbose=0)[0][0]
        lstm_prediction_price = models['scaler'].inverse_transform([[lstm_prediction_scaled]])[0][0]
        
        # Validate LSTM prediction
        current_price = data['close'].iloc[-1]
        if abs(lstm_prediction_price - current_price) / current_price > 0.1:  # >10% change unrealistic
            st.warning("⚠️ LSTM prediction tidak realistis, menggunakan harga saat ini")
            lstm_prediction_price = current_price
            
        lstm_pred_diff = (lstm_prediction_price - current_price) / current_price
        
        return lstm_prediction_price, lstm_pred_diff
        
    except Exception as e:
        st.warning(f"⚠️ LSTM prediction error: {e}")
        return data['close'].iloc[-1], 0


def _predict_with_xgboost(models, data):
    """Predict with XGBoost model"""
    try:
        xgb_feature_names = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 'ATR_14', 'STOCHk_14_3_3', 'price_change', 'high_low_ratio', 'open_close_ratio']
        temp_df_xgb = data.copy()
        
        # Add derived features if missing
        if 'price_change' not in temp_df_xgb.columns: 
            temp_df_xgb['price_change'] = temp_df_xgb['close'].pct_change()
        if 'high_low_ratio' not in temp_df_xgb.columns: 
            temp_df_xgb['high_low_ratio'] = temp_df_xgb['high'] / temp_df_xgb['low']
        if 'open_close_ratio' not in temp_df_xgb.columns: 
            temp_df_xgb['open_close_ratio'] = temp_df_xgb['open'] / temp_df_xgb['close']
        
        available_xgb_features = [f for f in xgb_feature_names if f in temp_df_xgb.columns]
        X_xgb_latest = temp_df_xgb[available_xgb_features].tail(1).fillna(0)
        
        if not X_xgb_latest.empty and models.get('xgb'):
            xgb_pred = models['xgb'].predict(X_xgb_latest)[0]
            xgb_confidence = models['xgb'].predict_proba(X_xgb_latest)[0][xgb_pred]
            return xgb_pred, xgb_confidence
        else:
            return 0, 0.5
            
    except Exception as e:
        st.warning(f"⚠️ XGBoost prediction error: {e}")
        return 0, 0.5


def _predict_with_other_models(models, data):
    """Predict with CNN, SVC, and Naive Bayes models"""
    try:
        cnn_pred, svc_pred, nb_pred = 0, 0, 0
        svc_confidence, nb_confidence = 0.5, 0.5

        # CNN
        cnn_feature_names = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20']
        available_cnn_features = [f for f in cnn_feature_names if f in data.columns]
        sequence_length_cnn = 20
        if len(data) >= sequence_length_cnn and models.get('cnn') and models.get('cnn_scaler'):
            last_cnn_data_features = data[available_cnn_features].tail(sequence_length_cnn).fillna(0)
            scaled_cnn_data = models['cnn_scaler'].transform(last_cnn_data_features)
            X_cnn = np.reshape(scaled_cnn_data, (1, sequence_length_cnn, len(available_cnn_features)))
            cnn_pred_proba = models['cnn'].predict(X_cnn, verbose=0)[0][0]
            cnn_pred = 1 if cnn_pred_proba > 0.5 else 0

        # SVC
        svc_feature_names = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 'ATR_14', 'STOCHk_14_3_3']
        available_svc_features = [f for f in svc_feature_names if f in data.columns]
        if models.get('svc') and models.get('svc_scaler'):
            X_svc_latest = data[available_svc_features].tail(1).fillna(0)
            if not X_svc_latest.empty:
                X_svc_latest_scaled = models['svc_scaler'].transform(X_svc_latest)
                svc_pred = models['svc'].predict(X_svc_latest_scaled)[0]
                svc_confidence = models['svc'].predict_proba(X_svc_latest_scaled)[0][svc_pred]

        # Naive Bayes
        nb_feature_names = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 'ATR_14', 'STOCHk_14_3_3']
        available_nb_features = [f for f in nb_feature_names if f in data.columns]
        if models.get('nb'):
            X_nb_latest = data[available_nb_features].tail(1).fillna(0)
            if not X_nb_latest.empty:
                nb_pred = models['nb'].predict(X_nb_latest)[0]
                nb_confidence = models['nb'].predict_proba(X_nb_latest)[0][nb_pred]

        return cnn_pred, svc_pred, nb_pred, svc_confidence, nb_confidence
        
    except Exception as e:
        st.warning(f"⚠️ Other models prediction error: {e}")
        return 0, 0, 0, 0.5, 0.5


def _predict_with_meta_learner(models, data, lstm_pred_diff, xgb_pred, xgb_confidence,
                              cnn_pred, svc_pred, svc_confidence, nb_pred, nb_confidence):
    """Predict with meta learner ensemble"""
    try:
        current_data_point = data.tail(1).copy()
        
        # Create prediction series
        lstm_preds_single = pd.Series(lstm_pred_diff, index=current_data_point.index)
        xgb_preds_single = pd.Series(xgb_pred, index=current_data_point.index)
        xgb_conf_single = pd.Series(xgb_confidence, index=current_data_point.index)
        cnn_preds_single = pd.Series(cnn_pred, index=current_data_point.index)
        svc_preds_single = pd.Series(svc_pred, index=current_data_point.index)
        svc_conf_single = pd.Series(svc_confidence, index=current_data_point.index)
        nb_preds_single = pd.Series(nb_pred, index=current_data_point.index)
        nb_conf_single = pd.Series(nb_confidence, index=current_data_point.index)
        
        # Import meta learner utilities (these should be moved to this module eventually)
        try:
            from utils.meta_learner import prepare_data_for_meta_learner, get_meta_signal
        except ImportError:
            st.warning("⚠️ Meta learner utilities not available")
            return "HOLD", 0.5
        
        X_meta_pred, _ = prepare_data_for_meta_learner(
            current_data_point, lstm_preds_single, xgb_preds_single, xgb_conf_single,
            cnn_preds_single, svc_preds_single, svc_conf_single, nb_preds_single, nb_conf_single
        )

        if not X_meta_pred.empty:
            meta_signal_series = get_meta_signal(
                current_data_point, lstm_preds_single, xgb_preds_single, xgb_conf_single,
                cnn_preds_single, svc_preds_single, svc_conf_single, nb_preds_single, nb_conf_single,
                models['meta']
            )
            meta_signal_numeric = meta_signal_series.iloc[0]

            final_signal = "BUY" if meta_signal_numeric == 1 else "SELL" if meta_signal_numeric == -1 else "HOLD"
            
            # Calculate confidence from meta model
            final_confidence = 0.5
            if final_signal != "HOLD":
                meta_proba = models['meta'].predict_proba(X_meta_pred)[0]
                class_index = np.where(models['meta'].classes_ == meta_signal_numeric)[0]
                if len(class_index) > 0:
                    final_confidence = meta_proba[class_index[0]]
            
            return final_signal, final_confidence
        else:
            return "HOLD", 0.5
            
    except Exception as e:
        st.warning(f"⚠️ Meta learner error: {e}")
        return "HOLD", 0.5


def validate_prediction_result(signal, confidence, predicted_price, current_price):
    """
    Validate prediction results for reasonableness
    
    Args:
        signal: Predicted signal
        confidence: Prediction confidence
        predicted_price: Predicted price
        current_price: Current market price
        
    Returns:
        tuple: (is_valid, validated_signal, validated_confidence, validated_price)
    """
    validated_signal = signal
    validated_confidence = confidence
    validated_price = predicted_price
    
    # Validate signal type
    if signal not in ['BUY', 'SELL', 'HOLD']:
        validated_signal = 'HOLD'
    
    # Validate confidence range
    if confidence < 0 or confidence > 1:
        validated_confidence = max(0, min(1, confidence))
    
    # Validate predicted price reasonableness
    if predicted_price <= 0 or (current_price > 0 and abs(predicted_price - current_price) / current_price > 0.5):
        validated_price = current_price
    
    is_valid = (validated_signal == signal and 
                abs(validated_confidence - confidence) < 0.01 and 
                abs(validated_price - predicted_price) < 0.01)
    
    return is_valid, validated_signal, validated_confidence, validated_price