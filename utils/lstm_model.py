# utils/lstm_model.py
# Versi yang dimodifikasi untuk menyimpan model dan scaler dengan nama file dinamis.

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import os # <-- Tambahan
import re # <-- Tambahan
import joblib # <-- Tambahan

def sanitize_filename(name): # <-- Tambahan
    """Membersihkan string untuk digunakan sebagai nama file yang aman."""
    return re.sub(r'[^a-zA-Z0-9_-]', '', name)

# <-- DIUBAH: Menambahkan parameter 'symbol' dan 'timeframe_key'
def train_lstm_model(df, symbol, timeframe_key):
    """
    Melatih model LSTM dan menyimpannya dengan nama file dinamis.
    """
    # <-- DIUBAH: Pesan log menyertakan simbol
    print(f"Memulai training model LSTM untuk {symbol} timeframe {timeframe_key}...")
    data = df[['close']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    tf.keras.backend.clear_session()
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(units=50), Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    print("Training model LSTM selesai.")

    # <-- DIUBAH: Menambahkan logika penyimpanan model & scaler
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    symbol_fn = sanitize_filename(symbol)
    
    # Simpan model
    model_path = os.path.join(model_dir, f'lstm_model_{symbol_fn}_{timeframe_key}.keras')
    print(f"Menyimpan model LSTM ke {model_path}...")
    model.save(model_path)
    
    # Simpan scaler
    scaler_path = os.path.join(model_dir, f'scaler_{symbol_fn}_{timeframe_key}.pkl')
    print(f"Menyimpan scaler ke {scaler_path}...")
    joblib.dump(scaler, scaler_path)
    
    return model, scaler

def predict_next_n_close(model, scaler, df, n=1):
    # ... (Fungsi ini tidak diubah)
    last_sequence = df[["close"]].values[-60:]
    predictions = []
    for _ in range(n):
        scaled_seq = scaler.transform(last_sequence)
        X = np.reshape(scaled_seq, (1, 60, 1))
        pred = model.predict(X, verbose=0)
        pred_unscaled = scaler.inverse_transform(pred)
        predictions.append(pred_unscaled[0][0])
        last_sequence = np.append(last_sequence[1:], [[pred_unscaled[0][0]]], axis=0)
    return predictions


def convert_to_signal(direction, confidence, threshold=0.6):
    if direction == "TURUN" and confidence > threshold:
        return -1
    elif direction == "NAIK" and confidence > threshold:
        return 1
    else:
        return 0
