"""
Models module for machine learning model operations.

This module handles loading, training, and prediction with various ML models
including LSTM, XGBoost, SVM, Naive Bayes, CNN, and Meta learners.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from .config import (
    is_feature_enabled, get_model_directory, TF_AVAILABLE,
    LABEL_MAP, configure_tensorflow
)

# Import ML libraries if available
if TF_AVAILABLE:
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model, Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        st.error("TensorFlow not properly installed")
        TF_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Import meta learner if available
if is_feature_enabled('UTILS_AVAILABLE'):
    from utils.meta_learner import train_meta_learner, prepare_data_for_meta_learner, get_meta_signal


def predict_with_models(models: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate predictions using ensemble of models.
    
    Args:
        models: Dictionary of loaded models
        data: Input data for prediction
        
    Returns:
        Dictionary with predictions and metadata
    """
    try:
        if data.empty or len(data) < 10:
            return {
                "ensemble_signal": 0,
                "confidence": 0.0,
                "predicted_price": data['close'].iloc[-1] if not data.empty else 0,
                "individual_predictions": {},
                "error": "Insufficient data for prediction"
            }
        
        # Prepare data for prediction
        recent_data = data.tail(60).copy()
        current_price = recent_data['close'].iloc[-1]
        
        individual_predictions = {}
        valid_predictions = []
        
        # LSTM Prediction
        if 'lstm' in models and models['lstm'] is not None:
            try:
                lstm_pred, lstm_conf = _predict_lstm(models['lstm'], models.get('scaler'), recent_data)
                individual_predictions['lstm'] = {'signal': lstm_pred, 'confidence': lstm_conf}
                valid_predictions.append(lstm_pred)
            except Exception as e:
                individual_predictions['lstm'] = {'error': str(e)}
        
        # XGBoost Prediction
        if 'xgb' in models and models['xgb'] is not None:
            try:
                xgb_pred, xgb_conf = _predict_xgboost(models['xgb'], recent_data)
                individual_predictions['xgb'] = {'signal': xgb_pred, 'confidence': xgb_conf}
                valid_predictions.append(xgb_pred)
            except Exception as e:
                individual_predictions['xgb'] = {'error': str(e)}
        
        # SVM Prediction
        if 'svc' in models and models['svc'] is not None:
            try:
                svc_pred, svc_conf = _predict_svm(models['svc'], models.get('svc_scaler'), recent_data)
                individual_predictions['svc'] = {'signal': svc_pred, 'confidence': svc_conf}
                valid_predictions.append(svc_pred)
            except Exception as e:
                individual_predictions['svc'] = {'error': str(e)}
        
        # Naive Bayes Prediction
        if 'nb' in models and models['nb'] is not None:
            try:
                nb_pred, nb_conf = _predict_naive_bayes(models['nb'], recent_data)
                individual_predictions['nb'] = {'signal': nb_pred, 'confidence': nb_conf}
                valid_predictions.append(nb_pred)
            except Exception as e:
                individual_predictions['nb'] = {'error': str(e)}
        
        # CNN Prediction
        if 'cnn' in models and models['cnn'] is not None:
            try:
                cnn_pred, cnn_conf = _predict_cnn(models['cnn'], models.get('cnn_scaler'), recent_data)
                individual_predictions['cnn'] = {'signal': cnn_pred, 'confidence': cnn_conf}
                valid_predictions.append(cnn_pred)
            except Exception as e:
                individual_predictions['cnn'] = {'error': str(e)}
        
        # Meta Learner Prediction
        if 'meta' in models and models['meta'] is not None:
            try:
                meta_pred, meta_conf = _predict_meta_learner(models['meta'], recent_data, individual_predictions)
                individual_predictions['meta'] = {'signal': meta_pred, 'confidence': meta_conf}
                valid_predictions.append(meta_pred)
            except Exception as e:
                individual_predictions['meta'] = {'error': str(e)}
        
        # Ensemble prediction
        if valid_predictions:
            # Weighted voting (can be enhanced with model performance weights)
            ensemble_signal = int(np.sign(np.mean(valid_predictions)))
            confidence = min(0.95, len(valid_predictions) / len(models) * 0.8 + 0.2)
        else:
            ensemble_signal = 0
            confidence = 0.0
        
        # Price prediction (simple approach based on signals)
        if ensemble_signal == 1:
            predicted_price = current_price * 1.001  # Small upward movement
        elif ensemble_signal == -1:
            predicted_price = current_price * 0.999  # Small downward movement
        else:
            predicted_price = current_price
        
        return {
            "ensemble_signal": ensemble_signal,
            "confidence": confidence,
            "predicted_price": predicted_price,
            "individual_predictions": individual_predictions,
            "current_price": current_price,
            "valid_models": len(valid_predictions),
            "total_models": len(models)
        }
        
    except Exception as e:
        st.error(f"Error in model prediction: {e}")
        return {
            "ensemble_signal": 0,
            "confidence": 0.0,
            "predicted_price": data['close'].iloc[-1] if not data.empty else 0,
            "individual_predictions": {},
            "error": str(e)
        }


def load_all_models(symbol: str, timeframe_key: str) -> Dict[str, Any]:
    """
    Load all trained models for the given symbol and timeframe.
    
    Args:
        symbol: Trading symbol
        timeframe_key: Timeframe identifier
        
    Returns:
        Dictionary of loaded models
    """
    models = {}
    model_dir = get_model_directory()
    symbol_fn = sanitize_filename(symbol)
    
    try:
        # Load LSTM model and scaler
        lstm_path = os.path.join(model_dir, f'lstm_model_{symbol_fn}_{timeframe_key}.h5')
        scaler_path = os.path.join(model_dir, f'lstm_scaler_{symbol_fn}_{timeframe_key}.pkl')
        
        if os.path.exists(lstm_path) and os.path.exists(scaler_path) and TF_AVAILABLE:
            try:
                models['lstm'] = load_model(lstm_path)
                models['scaler'] = joblib.load(scaler_path)
                st.success("✅ LSTM model loaded")
            except Exception as e:
                st.warning(f"⚠️ Failed to load LSTM: {e}")
        
        # Load XGBoost model
        xgb_path = os.path.join(model_dir, f'xgb_model_{symbol_fn}_{timeframe_key}.pkl')
        if os.path.exists(xgb_path) and XGB_AVAILABLE:
            try:
                models['xgb'] = joblib.load(xgb_path)
                st.success("✅ XGBoost model loaded")
            except Exception as e:
                st.warning(f"⚠️ Failed to load XGBoost: {e}")
        
        # Load SVM model and scaler
        svc_path = os.path.join(model_dir, f'svc_model_{symbol_fn}_{timeframe_key}.pkl')
        svc_scaler_path = os.path.join(model_dir, f'svc_scaler_{symbol_fn}_{timeframe_key}.pkl')
        
        if os.path.exists(svc_path) and os.path.exists(svc_scaler_path):
            try:
                models['svc'] = joblib.load(svc_path)
                models['svc_scaler'] = joblib.load(svc_scaler_path)
                st.success("✅ SVM model loaded")
            except Exception as e:
                st.warning(f"⚠️ Failed to load SVM: {e}")
        
        # Load Naive Bayes model
        nb_path = os.path.join(model_dir, f'nb_model_{symbol_fn}_{timeframe_key}.pkl')
        if os.path.exists(nb_path):
            try:
                models['nb'] = joblib.load(nb_path)
                st.success("✅ Naive Bayes model loaded")
            except Exception as e:
                st.warning(f"⚠️ Failed to load Naive Bayes: {e}")
        
        # Load CNN model and scaler
        cnn_path = os.path.join(model_dir, f'cnn_model_{symbol_fn}_{timeframe_key}.h5')
        cnn_scaler_path = os.path.join(model_dir, f'cnn_scaler_{symbol_fn}_{timeframe_key}.pkl')
        
        if os.path.exists(cnn_path) and os.path.exists(cnn_scaler_path) and TF_AVAILABLE:
            try:
                models['cnn'] = load_model(cnn_path)
                models['cnn_scaler'] = joblib.load(cnn_scaler_path)
                st.success("✅ CNN model loaded")
            except Exception as e:
                st.warning(f"⚠️ Failed to load CNN: {e}")
        
        # Load Meta Learner
        meta_path = os.path.join(model_dir, f'meta_model_{symbol_fn}_{timeframe_key}.pkl')
        if os.path.exists(meta_path):
            try:
                models['meta'] = joblib.load(meta_path)
                st.success("✅ Meta Learner loaded")
            except Exception as e:
                st.warning(f"⚠️ Failed to load Meta Learner: {e}")
        
        if not models:
            st.warning(f"⚠️ No models found for {symbol} ({timeframe_key})")
        else:
            st.info(f"📊 Loaded {len(models)} models for {symbol}")
        
        return models
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}


def train_and_save_all_models(df: pd.DataFrame, symbol: str, timeframe_key: str) -> Dict[str, Any]:
    """
    Train and save all models with proper error handling.
    
    Args:
        df: Training data
        symbol: Trading symbol
        timeframe_key: Timeframe identifier
        
    Returns:
        Dictionary of training results
    """
    model_dir = get_model_directory()
    os.makedirs(model_dir, exist_ok=True)
    symbol_fn = sanitize_filename(symbol)
    
    results = {}
    
    with st.status(f"🏗️ Training models for {symbol} ({timeframe_key})...", expanded=True) as status:
        try:
            # Train LSTM
            if TF_AVAILABLE:
                status.update(label="Training LSTM model...")
                try:
                    lstm_model, scaler = train_simple_lstm(df, symbol, timeframe_key)
                    results['lstm'] = {'success': True, 'model': lstm_model, 'scaler': scaler}
                    st.write("✅ LSTM training completed")
                except Exception as e:
                    results['lstm'] = {'success': False, 'error': str(e)}
                    st.write(f"❌ LSTM training failed: {e}")
            
            # Train XGBoost
            if XGB_AVAILABLE:
                status.update(label="Training XGBoost model...")
                try:
                    xgb_model = train_simple_xgboost(df.copy(), symbol, timeframe_key)
                    results['xgb'] = {'success': True, 'model': xgb_model}
                    st.write("✅ XGBoost training completed")
                except Exception as e:
                    results['xgb'] = {'success': False, 'error': str(e)}
                    st.write(f"❌ XGBoost training failed: {e}")
            
            # Train CNN
            if TF_AVAILABLE:
                status.update(label="Training CNN model...")
                try:
                    cnn_model = train_simple_cnn(df.copy(), symbol, timeframe_key)
                    results['cnn'] = {'success': True, 'model': cnn_model}
                    st.write("✅ CNN training completed")
                except Exception as e:
                    results['cnn'] = {'success': False, 'error': str(e)}
                    st.write(f"❌ CNN training failed: {e}")
            
            # Train SVM
            status.update(label="Training SVM model...")
            try:
                svc_model, svc_scaler = train_simple_svc(df.copy(), symbol, timeframe_key)
                results['svc'] = {'success': True, 'model': svc_model, 'scaler': svc_scaler}
                st.write("✅ SVM training completed")
            except Exception as e:
                results['svc'] = {'success': False, 'error': str(e)}
                st.write(f"❌ SVM training failed: {e}")
            
            # Train Naive Bayes
            status.update(label="Training Naive Bayes model...")
            try:
                nb_model = train_simple_naive_bayes(df.copy(), symbol, timeframe_key)
                results['nb'] = {'success': True, 'model': nb_model}
                st.write("✅ Naive Bayes training completed")
            except Exception as e:
                results['nb'] = {'success': False, 'error': str(e)}
                st.write(f"❌ Naive Bayes training failed: {e}")
            
            # Train Meta Learner
            if is_feature_enabled('UTILS_AVAILABLE'):
                status.update(label="Training Meta Learner...")
                try:
                    meta_model = _train_meta_learner_wrapper(df.copy(), symbol, timeframe_key)
                    results['meta'] = {'success': True, 'model': meta_model}
                    st.write("✅ Meta Learner training completed")
                except Exception as e:
                    results['meta'] = {'success': False, 'error': str(e)}
                    st.write(f"❌ Meta Learner training failed: {e}")
            
            status.update(label="Training completed!", state="complete")
            
            # Summary
            successful_models = [k for k, v in results.items() if v.get('success', False)]
            st.success(f"🎉 Training completed! Successfully trained {len(successful_models)} models")
            
        except Exception as e:
            st.error(f"❌ Training process failed: {e}")
            results['overall_error'] = str(e)
    
    return results


# Individual model training functions
def train_simple_lstm(data: pd.DataFrame, symbol: str, timeframe: str) -> Tuple[Any, Any]:
    """Train a simple LSTM model."""
    if not TF_AVAILABLE:
        raise ValueError("TensorFlow not available")
    
    df = data.copy()
    
    # Prepare features
    features = ['open', 'high', 'low', 'close', 'volume']
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 4:
        raise ValueError(f"Insufficient features for LSTM training: {available_features}")
    
    # Create target
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    if len(df) < 100:
        raise ValueError(f"Insufficient data for LSTM training: {len(df)} rows")
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[available_features])
    
    # Create sequences
    sequence_length = 10
    X, y = [], []
    
    for i in range(sequence_length, len(scaled_data) - 1):
        X.append(scaled_data[i-sequence_length:i])
        y.append(df['target'].iloc[i])
    
    X, y = np.array(X), np.array(y)
    
    if len(X) == 0:
        raise ValueError("No sequences created for training")
    
    # Create and train model
    model = create_simple_lstm_model((sequence_length, len(available_features)))
    
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    
    model.fit(
        X, y,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Save model
    model_dir = get_model_directory()
    symbol_fn = sanitize_filename(symbol)
    model.save(os.path.join(model_dir, f'lstm_model_{symbol_fn}_{timeframe}.h5'))
    joblib.dump(scaler, os.path.join(model_dir, f'lstm_scaler_{symbol_fn}_{timeframe}.pkl'))
    
    return model, scaler


def create_simple_lstm_model(input_shape: Tuple[int, int]) -> Any:
    """Create a simple LSTM model architecture."""
    if not TF_AVAILABLE:
        raise ValueError("TensorFlow not available")
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_simple_xgboost(data: pd.DataFrame, symbol: str, timeframe: str) -> Any:
    """Train a simple XGBoost model."""
    if not XGB_AVAILABLE:
        raise ValueError("XGBoost not available")
    
    df = data.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    if len(df) < 50:
        raise ValueError(f"Insufficient data for XGBoost training: {len(df)} rows")
    
    # Select features
    features = ['open', 'high', 'low', 'close']
    if 'rsi' in df.columns: features.append('rsi')
    if 'MACD_12_26_9' in df.columns: features.append('MACD_12_26_9')
    if 'EMA_10' in df.columns: features.append('EMA_10')
    if 'EMA_20' in df.columns: features.append('EMA_20')
    if 'ATR_14' in df.columns: features.append('ATR_14')
    if 'STOCHk_14_3_3' in df.columns: features.append('STOCHk_14_3_3')
    
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features].fillna(0)
    y = df['target']
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X, y)
    
    # Save model
    model_dir = get_model_directory()
    symbol_fn = sanitize_filename(symbol)
    joblib.dump(model, os.path.join(model_dir, f'xgb_model_{symbol_fn}_{timeframe}.pkl'))
    
    return model


def train_simple_svc(data: pd.DataFrame, symbol: str, timeframe: str) -> Tuple[Any, Any]:
    """Train a simple SVM model."""
    df = data.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    if len(df) < 50:
        raise ValueError(f"Insufficient data for SVM training: {len(df)} rows")
    
    # Select features
    features = ['open', 'high', 'low', 'close']
    if 'rsi' in df.columns: features.append('rsi')
    if 'MACD_12_26_9' in df.columns: features.append('MACD_12_26_9')
    if 'EMA_10' in df.columns: features.append('EMA_10')
    if 'EMA_20' in df.columns: features.append('EMA_20')
    if 'ATR_14' in df.columns: features.append('ATR_14')
    if 'STOCHk_14_3_3' in df.columns: features.append('STOCHk_14_3_3')
    
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features].fillna(0)
    y = df['target']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_scaled, y)
    
    # Save model and scaler
    model_dir = get_model_directory()
    symbol_fn = sanitize_filename(symbol)
    joblib.dump(model, os.path.join(model_dir, f'svc_model_{symbol_fn}_{timeframe}.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, f'svc_scaler_{symbol_fn}_{timeframe}.pkl'))
    
    return model, scaler


def train_simple_naive_bayes(data: pd.DataFrame, symbol: str, timeframe: str) -> Any:
    """Train a simple Gaussian Naive Bayes model."""
    df = data.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    if len(df) < 50:
        raise ValueError(f"Insufficient data for Naive Bayes training: {len(df)} rows")
    
    features = ['open', 'high', 'low', 'close']
    if 'rsi' in df.columns: features.append('rsi')
    if 'MACD_12_26_9' in df.columns: features.append('MACD_12_26_9')
    if 'EMA_10' in df.columns: features.append('EMA_10')
    if 'EMA_20' in df.columns: features.append('EMA_20')
    if 'ATR_14' in df.columns: features.append('ATR_14')
    if 'STOCHk_14_3_3' in df.columns: features.append('STOCHk_14_3_3')
    
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features].fillna(0)
    y = df['target']
    
    model = GaussianNB()
    model.fit(X, y)
    
    # Save model
    model_dir = get_model_directory()
    symbol_fn = sanitize_filename(symbol)
    joblib.dump(model, os.path.join(model_dir, f'nb_model_{symbol_fn}_{timeframe}.pkl'))
    
    return model


def train_simple_cnn(data: pd.DataFrame, symbol: str, timeframe: str) -> Any:
    """Train a simple CNN model."""
    if not TF_AVAILABLE:
        raise ValueError("TensorFlow not available")
    
    df = data.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    if len(df) < 100:
        raise ValueError(f"Insufficient data for CNN training: {len(df)} rows")
    
    # Prepare features
    features = ['open', 'high', 'low', 'close', 'volume']
    available_features = [f for f in features if f in df.columns]
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[available_features])
    
    # Create sequences
    sequence_length = 10
    X, y = [], []
    
    for i in range(sequence_length, len(scaled_data) - 1):
        X.append(scaled_data[i-sequence_length:i])
        y.append(df['target'].iloc[i])
    
    X, y = np.array(X), np.array(y)
    
    if len(X) == 0:
        raise ValueError("No sequences created for training")
    
    # Create and train model
    model = create_simple_cnn_model((sequence_length, len(available_features)))
    
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    
    model.fit(
        X, y,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Save model and scaler
    model_dir = get_model_directory()
    symbol_fn = sanitize_filename(symbol)
    model.save(os.path.join(model_dir, f'cnn_model_{symbol_fn}_{timeframe}.h5'))
    joblib.dump(scaler, os.path.join(model_dir, f'cnn_scaler_{symbol_fn}_{timeframe}.pkl'))
    
    return model


def create_simple_cnn_model(input_shape: Tuple[int, int]) -> Any:
    """Create a simple CNN model architecture."""
    if not TF_AVAILABLE:
        raise ValueError("TensorFlow not available")
    
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# Helper functions for predictions
def _predict_lstm(model: Any, scaler: Any, data: pd.DataFrame) -> Tuple[int, float]:
    """Make prediction with LSTM model."""
    features = ['open', 'high', 'low', 'close', 'volume']
    available_features = [f for f in features if f in data.columns]
    
    scaled_data = scaler.transform(data[available_features].fillna(0).tail(10))
    X = scaled_data.reshape(1, scaled_data.shape[0], scaled_data.shape[1])
    
    prediction = model.predict(X, verbose=0)[0][0]
    signal = 1 if prediction > 0.5 else -1
    confidence = abs(prediction - 0.5) * 2
    
    return signal, confidence


def _predict_xgboost(model: Any, data: pd.DataFrame) -> Tuple[int, float]:
    """Make prediction with XGBoost model."""
    features = ['open', 'high', 'low', 'close']
    if 'rsi' in data.columns: features.append('rsi')
    if 'MACD_12_26_9' in data.columns: features.append('MACD_12_26_9')
    if 'EMA_10' in data.columns: features.append('EMA_10')
    if 'EMA_20' in data.columns: features.append('EMA_20')
    if 'ATR_14' in data.columns: features.append('ATR_14')
    if 'STOCHk_14_3_3' in data.columns: features.append('STOCHk_14_3_3')
    
    available_features = [f for f in features if f in data.columns]
    X = data[available_features].fillna(0).tail(1)
    
    prediction_proba = model.predict_proba(X)[0]
    signal = 1 if prediction_proba[1] > 0.5 else -1
    confidence = max(prediction_proba)
    
    return signal, confidence


def _predict_svm(model: Any, scaler: Any, data: pd.DataFrame) -> Tuple[int, float]:
    """Make prediction with SVM model."""
    features = ['open', 'high', 'low', 'close']
    if 'rsi' in data.columns: features.append('rsi')
    if 'MACD_12_26_9' in data.columns: features.append('MACD_12_26_9')
    if 'EMA_10' in data.columns: features.append('EMA_10')
    if 'EMA_20' in data.columns: features.append('EMA_20')
    if 'ATR_14' in data.columns: features.append('ATR_14')
    if 'STOCHk_14_3_3' in data.columns: features.append('STOCHk_14_3_3')
    
    available_features = [f for f in features if f in data.columns]
    X = data[available_features].fillna(0).tail(1)
    X_scaled = scaler.transform(X)
    
    prediction_proba = model.predict_proba(X_scaled)[0]
    signal = 1 if prediction_proba[1] > 0.5 else -1
    confidence = max(prediction_proba)
    
    return signal, confidence


def _predict_naive_bayes(model: Any, data: pd.DataFrame) -> Tuple[int, float]:
    """Make prediction with Naive Bayes model."""
    features = ['open', 'high', 'low', 'close']
    if 'rsi' in data.columns: features.append('rsi')
    if 'MACD_12_26_9' in data.columns: features.append('MACD_12_26_9')
    if 'EMA_10' in data.columns: features.append('EMA_10')
    if 'EMA_20' in data.columns: features.append('EMA_20')
    if 'ATR_14' in data.columns: features.append('ATR_14')
    if 'STOCHk_14_3_3' in data.columns: features.append('STOCHk_14_3_3')
    
    available_features = [f for f in features if f in data.columns]
    X = data[available_features].fillna(0).tail(1)
    
    prediction_proba = model.predict_proba(X)[0]
    signal = 1 if prediction_proba[1] > 0.5 else -1
    confidence = max(prediction_proba)
    
    return signal, confidence


def _predict_cnn(model: Any, scaler: Any, data: pd.DataFrame) -> Tuple[int, float]:
    """Make prediction with CNN model."""
    features = ['open', 'high', 'low', 'close', 'volume']
    available_features = [f for f in features if f in data.columns]
    
    scaled_data = scaler.transform(data[available_features].fillna(0).tail(10))
    X = scaled_data.reshape(1, scaled_data.shape[0], scaled_data.shape[1])
    
    prediction = model.predict(X, verbose=0)[0][0]
    signal = 1 if prediction > 0.5 else -1
    confidence = abs(prediction - 0.5) * 2
    
    return signal, confidence


def _predict_meta_learner(model: Any, data: pd.DataFrame, individual_predictions: Dict) -> Tuple[int, float]:
    """Make prediction with Meta Learner."""
    if not is_feature_enabled('UTILS_AVAILABLE'):
        raise ValueError("Meta learner utils not available")
    
    # Use utils function
    return get_meta_signal(model, data, individual_predictions)


def _train_meta_learner_wrapper(data: pd.DataFrame, symbol: str, timeframe: str) -> Any:
    """Wrapper for training meta learner."""
    if not is_feature_enabled('UTILS_AVAILABLE'):
        raise ValueError("Meta learner utils not available")
    
    return train_meta_learner(data, symbol, timeframe)


def sanitize_filename(name: str) -> str:
    """Clean string to be used as safe filename."""
    return re.sub(r'[^a-zA-Z0-9_-]', '', name.replace('/', '_').replace('\\', '_'))