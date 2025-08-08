# model_trainer.py - Model training and management functions
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Import utility functions
from ..utils.formatters import sanitize_filename

# Configure TensorFlow
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Force CPU usage
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("TensorFlow not available for model training")


def create_simple_lstm_model(input_shape):
    """Create a simple LSTM model"""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available")
        
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_simple_lstm(data, symbol, timeframe):
    """Train a simple LSTM model"""
    if not TF_AVAILABLE:
        st.error("TensorFlow not available for LSTM training")
        return None, None
        
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

    sequence_length = 60
    X, y = [], []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = create_simple_lstm_model((X.shape[1], 1))
    model.fit(X, y, batch_size=32, epochs=5, verbose=0)

    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    symbol_fn = sanitize_filename(symbol)

    model.save(os.path.join(model_dir, f'lstm_model_{symbol_fn}_{timeframe}.keras'))
    joblib.dump(scaler, os.path.join(model_dir, f'scaler_{symbol_fn}_{timeframe}.pkl'))

    return model, scaler


def train_simple_xgboost(data, symbol, timeframe):
    """Train a simple XGBoost model"""
    df = data.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()

    if len(df) < 50:
        st.warning("Data tidak cukup untuk training XGBoost")
        return None

    feature_names = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 'ATR_14', 'STOCHk_14_3_3', 'price_change', 'high_low_ratio', 'open_close_ratio']
    
    # Add derived features if missing
    if 'price_change' not in df.columns:
        df['price_change'] = df['close'].pct_change()
    if 'high_low_ratio' not in df.columns:
        df['high_low_ratio'] = df['high'] / df['low']
    if 'open_close_ratio' not in df.columns:
        df['open_close_ratio'] = df['open'] / df['close']

    available_features = [f for f in feature_names if f in df.columns]
    
    if len(available_features) < 5:
        st.warning(f"Feature tidak cukup untuk XGBoost: {available_features}")
        return None

    X = df[available_features].fillna(0)
    y = df['target']

    model = XGBClassifier(n_estimators=50, max_depth=6, random_state=42, verbosity=0)
    model.fit(X, y)

    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    symbol_fn = sanitize_filename(symbol)

    joblib.dump(model, os.path.join(model_dir, f'xgb_model_{symbol_fn}_{timeframe}.pkl'))
    return model


def create_simple_cnn_model(input_shape):
    """Create a simple CNN model"""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available")
        
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        Dropout(0.5),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_simple_cnn(data, symbol, timeframe):
    """Train a simple CNN model"""
    if not TF_AVAILABLE:
        st.error("TensorFlow not available for CNN training")
        return None
        
    df = data.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()

    if len(df) < 100:
        st.warning("Data tidak cukup untuk training CNN")
        return None

    feature_names = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20']
    available_features = [f for f in feature_names if f in df.columns]

    if len(available_features) < 4:
        st.warning(f"Feature tidak cukup untuk CNN: {available_features}")
        return None

    sequence_length = 20
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[available_features])

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(df['target'].iloc[i])

    X, y = np.array(X), np.array(y)

    model = create_simple_cnn_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    symbol_fn = sanitize_filename(symbol)

    model.save(os.path.join(model_dir, f'cnn_model_{symbol_fn}_{timeframe}.keras'))
    joblib.dump(scaler, os.path.join(model_dir, f'cnn_scaler_{symbol_fn}_{timeframe}.pkl'))

    return model


def train_simple_svc(data, symbol, timeframe):
    """Train a simple SVC model"""
    df = data.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()

    if len(df) < 50:
        st.warning("Data tidak cukup untuk training SVC")
        return None, None

    feature_names = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 'ATR_14', 'STOCHk_14_3_3']
    available_features = [f for f in feature_names if f in df.columns]

    if len(available_features) < 5:
        st.warning(f"Feature tidak cukup untuk SVC: {available_features}")
        return None, None

    X = df[available_features].fillna(0)
    y = df['target']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(probability=True, random_state=42)
    model.fit(X_scaled, y)

    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    symbol_fn = sanitize_filename(symbol)

    joblib.dump(model, os.path.join(model_dir, f'svc_model_{symbol_fn}_{timeframe}.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, f'svc_scaler_{symbol_fn}_{timeframe}.pkl'))

    return model, scaler


def train_simple_naive_bayes(data, symbol, timeframe):
    """Train a simple Naive Bayes model"""
    df = data.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()

    if len(df) < 50:
        st.warning("Data tidak cukup untuk training Naive Bayes")
        return None

    feature_names = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 'ATR_14', 'STOCHk_14_3_3']
    available_features = [f for f in feature_names if f in df.columns]

    if len(available_features) < 5:
        st.warning(f"Feature tidak cukup untuk Naive Bayes: {available_features}")
        return None

    X = df[available_features].fillna(0)
    y = df['target']

    model = GaussianNB()
    model.fit(X, y)

    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    symbol_fn = sanitize_filename(symbol)

    joblib.dump(model, os.path.join(model_dir, f'nb_model_{symbol_fn}_{timeframe}.pkl'))
    return model


def train_and_save_all_models(df, symbol, timeframe_key):
    """Train and save all models with proper error handling and batch prediction."""
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    symbol_fn = sanitize_filename(symbol)

    with st.status(f"🏗️ Memulai proses training untuk {symbol} ({timeframe_key})...", expanded=True) as status:
        try:
            status.update(label="Langkah 1/6: Melatih model LSTM...")
            lstm_model, scaler = train_simple_lstm(df, symbol, timeframe_key)
            st.write(f"Training LSTM untuk {symbol} selesai.")

            status.update(label="Langkah 2/6: Melatih model XGBoost...")
            xgb_model = train_simple_xgboost(df.copy(), symbol, timeframe_key)
            st.write(f"Training XGBoost untuk {symbol} selesai.")

            status.update(label="Langkah 3/6: Melatih model CNN...")
            cnn_model = train_simple_cnn(df.copy(), symbol, timeframe_key)
            st.write(f"Training CNN untuk {symbol} selesai.")

            status.update(label="Langkah 4/6: Melatih model SVC...")
            svc_model, svc_scaler = train_simple_svc(df.copy(), symbol, timeframe_key)
            st.write(f"Training SVC untuk {symbol} selesai.")

            status.update(label="Langkah 5/6: Melatih model Naive Bayes...")
            nb_model = train_simple_naive_bayes(df.copy(), symbol, timeframe_key)
            st.write(f"Training Naive Bayes untuk {symbol} selesai.")

            status.update(label="Semua model berhasil dilatih dan disimpan!")

            return {
                'lstm': lstm_model,
                'scaler': scaler,
                'xgb': xgb_model,
                'cnn': cnn_model,
                'svc': svc_model,
                'svc_scaler': svc_scaler,
                'nb': nb_model
            }

        except Exception as e:
            st.error(f"❌ Error dalam training models: {str(e)}")
            return None