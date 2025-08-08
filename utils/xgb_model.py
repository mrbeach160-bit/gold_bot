# utils/xgboost_model.py
# Versi yang dimodifikasi untuk menyimpan model dengan nama file dinamis berdasarkan simbol & timeframe.

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import re # <-- Tambahan

def sanitize_filename(name): # <-- Tambahan
    """Membersihkan string untuk digunakan sebagai nama file yang aman."""
    return re.sub(r'[^a-zA-Z0-9_-]', '', name)

def prepare_data_for_classification(df, look_forward_periods=5):
    # ... (Fungsi ini tidak diubah)
    df['future_close'] = df['close'].shift(-look_forward_periods)
    df.dropna(inplace=True)
    df['target'] = (df['future_close'] > df['close']).astype(int)
    features = df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'future_close', 'target', 'ema_signal'], errors='ignore')
    return features, df['target']

# <-- DIUBAH: Menambahkan parameter 'symbol'
def train_xgboost_model(df, symbol, timeframe_key):
    """
    Melatih model XGBoost dan menyimpannya dengan nama file dinamis.
    """
    X, y = prepare_data_for_classification(df.copy())
    
    if X.empty or y.empty or len(X.columns) == 0:
        print("Tidak cukup data atau fitur untuk melatih model XGBoost.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
    # <-- DIUBAH: Pesan log menyertakan simbol
    print(f"Memulai training model XGBoost untuk {symbol} timeframe {timeframe_key}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi Model XGBoost di data tes: {accuracy:.2f}")

    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    
    symbol_fn = sanitize_filename(symbol) # <-- Tambahan
    # <-- DIUBAH: Nama file menyertakan simbol
    model_path = os.path.join(model_dir, f'xgboost_model_{symbol_fn}_{timeframe_key}.json')
    print(f"Menyimpan model ke {model_path}...")
    model.save_model(model_path)

    return model

def predict_direction(model, latest_data_df):
    # ... (Fungsi ini tidak diubah)
    if model is None: return "N/A", 0.0
    training_features = model.get_booster().feature_names
    prediction_features = latest_data_df[training_features]
    try:
        prediction = model.predict(prediction_features)
        prediction_proba = model.predict_proba(prediction_features)
        final_prediction = int(prediction[-1])
        final_proba = prediction_proba[-1]
        direction = "NAIK" if final_prediction == 1 else "TURUN"
        confidence = float(max(final_proba))
        return direction, confidence
    except Exception as e:
        print(f"Error saat prediksi XGBoost: {e}")
        return "N/A", 0.0


def convert_to_signal(direction, confidence, threshold=0.6):
    if direction == "TURUN" and confidence > threshold:
        return -1
    elif direction == "NAIK" and confidence > threshold:
        return 1
    else:
        return 0
