# utils/cnn_model.py
# Versi yang dimodifikasi untuk menyimpan model dengan nama file dinamis berdasarkan simbol & timeframe.

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import os # <-- Tambahan
import re # <-- Tambahan

def sanitize_filename(name): # <-- Tambahan
    """Membersihkan string untuk digunakan sebagai nama file yang aman."""
    return re.sub(r'[^a-zA-Z0-9_-]', '', name)

def prepare_data_for_cnn(df, look_forward_periods=5, sequence_length=30):
    # ... (Fungsi ini tidak diubah)
    df['future_close'] = df['close'].shift(-look_forward_periods)
    df.dropna(inplace=True)
    df['target'] = (df['future_close'] > df['close']).astype(int)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    X, y = [], []
    for i in range(len(df) - sequence_length):
        seq = df.iloc[i:i+sequence_length][['open', 'high', 'low', 'close']].values
        label = df.iloc[i+sequence_length]['target']
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y)

# <-- DIUBAH: Menambahkan parameter 'symbol' dan 'timeframe_key'
def train_cnn_model(df, symbol, timeframe_key):
    """
    Melatih model CNN dan menyimpannya dengan nama file dinamis.
    """
    X, y = prepare_data_for_cnn(df)
    
    if len(X) == 0:
        print("Tidak cukup data untuk melatih model CNN.")
        return None

    tf.keras.backend.clear_session()
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=X.shape[1:]), MaxPooling1D(pool_size=2), Dropout(0.3),
        Conv1D(filters=64, kernel_size=3, activation='relu'), MaxPooling1D(pool_size=2), Dropout(0.3),
        Flatten(), Dense(50, activation='relu'), Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # <-- DIUBAH: Pesan log menyertakan simbol
    print(f"Memulai training model CNN untuk {symbol} timeframe {timeframe_key}...")
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    print("Training model CNN selesai.")
    
    # <-- DIUBAH: Menambahkan logika penyimpanan model
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    symbol_fn = sanitize_filename(symbol)
    model_path = os.path.join(model_dir, f'cnn_model_{symbol_fn}_{timeframe_key}.keras')
    print(f"Menyimpan model CNN ke {model_path}...")
    model.save(model_path)
    
    return model

def predict_cnn_direction(model, latest_data_df, sequence_length=30):
    # ... (Fungsi ini tidak diubah)
    if model is None or len(latest_data_df) < sequence_length: return "N/A", 0.0
    data_seq = latest_data_df.tail(sequence_length).copy()
    for col in ['open', 'high', 'low', 'close']:
        data_seq[col] = (data_seq[col] - data_seq[col].mean()) / data_seq[col].std()
    X_pred = data_seq[['open', 'high', 'low', 'close']].values
    X_pred = np.reshape(X_pred, (1, X_pred.shape[0], X_pred.shape[1]))
    prediction_proba = model.predict(X_pred, verbose=0)[0][0]
    direction = "NAIK" if prediction_proba > 0.5 else "TURUN"
    confidence = prediction_proba if direction == "NAIK" else 1 - prediction_proba
    return direction, confidence


def convert_to_signal(direction, confidence, threshold=0.6):
    if direction == "TURUN" and confidence > threshold:
        return -1
    elif direction == "NAIK" and confidence > threshold:
        return 1
    else:
        return 0
