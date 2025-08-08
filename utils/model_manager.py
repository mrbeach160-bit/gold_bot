import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def prepare_data_for_meta_learner(df, lstm_predictions, xgb_predictions, xgb_confidences, cnn_predictions, svc_predictions, svc_confidences, nb_predictions, nb_confidences):
    """
    Menyiapkan fitur dan target untuk Meta-Learner.
    """
    lstm_predictions = lstm_predictions.reindex(df.index).fillna(0)
    xgb_predictions = xgb_predictions.reindex(df.index).fillna(0)
    xgb_confidences = xgb_confidences.reindex(df.index).fillna(0)
    cnn_predictions = cnn_predictions.reindex(df.index).fillna(0)
    svc_predictions = svc_predictions.reindex(df.index).fillna(0)
    svc_confidences = svc_confidences.reindex(df.index).fillna(0)
    nb_predictions = nb_predictions.reindex(df.index).fillna(0)
    nb_confidences = nb_confidences.reindex(df.index).fillna(0)

    df['rsi'] = df['rsi'].fillna(0)
    df['bb_percent'] = df['bb_percent'].fillna(0.5)
    df['MACDh_12_26_9'] = df['MACDh_12_26_9'].fillna(0)
    df['ADX_14'] = df['ADX_14'].fillna(0)
    df['dist_to_support'] = df['dist_to_support'].fillna(0)
    df['dist_to_resistance'] = df['dist_to_resistance'].fillna(0)
    df['ema_signal_numeric'] = df['ema_signal_numeric'].fillna(0)

    X = pd.DataFrame({
        'lstm_pred_diff': lstm_predictions,
        'xgb_pred': xgb_predictions,
        'xgb_confidence': xgb_confidences,
        'cnn_pred': cnn_predictions,
        'svc_pred': svc_predictions,
        'svc_confidence': svc_confidences,
        'nb_pred': nb_predictions,
        'nb_confidence': nb_confidences,
        'ema_signal_numeric': df['ema_signal_numeric'],
        'rsi': df['rsi'],
        'bb_percent': df['bb_percent'],
        'MACDh_12_26_9': df['MACDh_12_26_9'],
        'ADX_14': df['ADX_14'],
        'dist_to_support': df['dist_to_support'],
        'dist_to_resistance': df['dist_to_resistance']
    })

    y = df['target_meta']

    # --- PERUBAHAN DI SINI ---
    # Mengubah 'inner' join menjadi 'left' join agar tidak kehilangan data fitur
    combined = X.join(y, how='left') 
    X = combined.drop(columns=['target_meta'])
    y = combined['target_meta']

    return X, y

def train_meta_learner(df, lstm_predictions, xgb_predictions, xgb_confidences, cnn_predictions, svc_predictions, svc_confidences, nb_predictions, nb_confidences, timeframe_key):
    """
    Melatih Master AI (RandomForestClassifier) dengan parameter tetap.
    """
    X, y = prepare_data_for_meta_learner(df, lstm_predictions, xgb_predictions, xgb_confidences, cnn_predictions, svc_predictions, svc_confidences, nb_predictions, nb_confidences)

    # Menghapus baris di mana target (y) adalah NaN. Ini akan secara otomatis menyelaraskan X.
    valid_indices = y.dropna().index
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    if X.empty or y.empty:
        print("Tidak cukup data yang valid untuk melatih Meta-Learner.")
        return None

    if len(y.unique()) < 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Kembali ke metode training langsung yang lebih stabil
    model = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=4, random_state=42, verbose=0)
    
    print(f"Memulai training Master AI (RandomForestClassifier) untuk timeframe {timeframe_key}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi Master AI di data tes: {accuracy:.2f}")

    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, f'meta_learner_randomforest_{timeframe_key}.pkl')
    print(f"Menyimpan Master AI ke {model_path}...")
    joblib.dump(model, model_path)

    return model

def get_meta_signal(df, lstm_predictions, xgb_predictions, xgb_confidences, cnn_predictions, svc_predictions, svc_confidences, nb_predictions, nb_confidences, meta_model):
    """
    Mendapatkan sinyal dari Master AI.
    """
    if meta_model is None:
        return pd.Series(0, index=df.index)

    last_row_df = df.iloc[[-1]].copy()
    last_row_df['target_meta'] = 0

    lstm_preds_single = lstm_predictions.reindex(last_row_df.index).fillna(0)
    xgb_preds_single = xgb_predictions.reindex(last_row_df.index).fillna(0)
    xgb_conf_single = xgb_confidences.reindex(last_row_df.index).fillna(0)
    cnn_preds_single = cnn_predictions.reindex(last_row_df.index).fillna(0)
    svc_preds_single = svc_predictions.reindex(last_row_df.index).fillna(0)
    svc_conf_single = svc_confidences.reindex(last_row_df.index).fillna(0)
    nb_preds_single = nb_predictions.reindex(last_row_df.index).fillna(0)
    nb_conf_single = nb_confidences.reindex(last_row_df.index).fillna(0)

    X_pred, _ = prepare_data_for_meta_learner(
        last_row_df, lstm_preds_single, xgb_preds_single,
        xgb_conf_single, cnn_preds_single, svc_preds_single, svc_conf_single,
        nb_preds_single, nb_conf_single
    )

    if X_pred.empty:
        return pd.Series(0, index=df.index)

    expected_cols = [
        'lstm_pred_diff', 'xgb_pred', 'xgb_confidence', 'cnn_pred', 'svc_pred', 'svc_confidence',
        'nb_pred', 'nb_confidence',
        'ema_signal_numeric', 'rsi', 'bb_percent', 'MACDh_12_26_9',
        'ADX_14', 'dist_to_support', 'dist_to_resistance'
    ]
    
    for col in expected_cols:
        if col not in X_pred.columns:
            X_pred[col] = 0

    X_pred = X_pred[expected_cols]

    meta_pred = meta_model.predict(X_pred)[0]
    
    final_signal_value = meta_pred
    if meta_pred == 0:
        final_signal_value = -1 # SELL

    return pd.Series(final_signal_value, index=df.index)