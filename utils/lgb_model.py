import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

def train_lgb_model(df):
    """
    Melatih model LightGBM untuk memprediksi arah harga (naik/turun).
    """
    df = df.copy()
    df['rsi'] = df.ta.rsi()
    df['MACDh_12_26_9'] = df.ta.macd()['MACDh_12_26_9']
    df['ADX_14'] = df.ta.adx()['ADX_14']

    # Label: apakah close berikutnya naik?
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)

    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'MACDh_12_26_9', 'ADX_14']
    X = df[feature_cols]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"? LightGBM trained  Accuracy: {acc:.4f}")

    # Simpan model
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(model, os.path.join(model_dir, 'lgb_model.pkl'))
    print("?? Model disimpan: model/lgb_model.pkl")

    return model

def predict_lgb_direction(model, latest_df):
    """
    Prediksi arah harga (naik/turun) dan confidence dengan model LightGBM.
    """
    latest_df = latest_df.copy()
    latest_df['rsi'] = latest_df.ta.rsi()
    latest_df['MACDh_12_26_9'] = latest_df.ta.macd()['MACDh_12_26_9']
    latest_df['ADX_14'] = latest_df.ta.adx()['ADX_14']

    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'MACDh_12_26_9', 'ADX_14']
    latest_df.dropna(inplace=True)
    X = latest_df[feature_cols].tail(1)

    if X.empty:
        return "HOLD", 0.0

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()
    direction = "NAIK" if pred == 1 else "TURUN"
    return direction, prob



def convert_to_signal(direction, confidence, threshold=0.6):
    if direction == "TURUN" and confidence > threshold:
        return -1
    elif direction == "NAIK" and confidence > threshold:
        return 1
    else:
        return 0
