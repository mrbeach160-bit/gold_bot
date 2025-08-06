import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier


def prepare_data_for_meta_learner(df, lstm_predictions, xgb_predictions, xgb_confidences,
                                  cnn_predictions, svc_predictions, svc_confidences,
                                  nb_predictions, nb_confidences):
    """
    Menyiapkan fitur dan target untuk Meta-Learner.
    Target sekarang: 1 = BUY, 0 = HOLD, -1 = SELL
    """
    # Reindex and fill
    preds = [lstm_predictions, xgb_predictions, xgb_confidences,
             cnn_predictions, svc_predictions, svc_confidences,
             nb_predictions, nb_confidences]
    preds = [p.reindex(df.index).fillna(0) for p in preds]
    lstm_p, xgb_p, xgb_c, cnn_p, svc_p, svc_c, nb_p, nb_c = preds

    # Fill missing indicator values
    df = df.fillna({
        'rsi': 0,
        'bb_percent': 0.5,
        'MACDh_12_26_9': 0,
        'ADX_14': 0,
        'dist_to_support': 0,
        'dist_to_resistance': 0,
        'ema_signal_numeric': 0
    })

    # Engineered features
    vote_agreement = ((xgb_p == cnn_p) & (cnn_p == svc_p)).astype(int)
    avg_confidence = (xgb_c + svc_c + nb_c) / 3

    X = pd.DataFrame({
        'lstm_pred_diff': lstm_p,
        'xgb_pred': xgb_p,
        'xgb_confidence': xgb_c,
        'cnn_pred': cnn_p,
        'svc_pred': svc_p,
        'svc_confidence': svc_c,
        'nb_pred': nb_p,
        'nb_confidence': nb_c,
        'ema_signal_numeric': df['ema_signal_numeric'],
        'rsi': df['rsi'],
        'bb_percent': df['bb_percent'],
        'MACDh_12_26_9': df['MACDh_12_26_9'],
        'ADX_14': df['ADX_14'],
        'dist_to_support': df['dist_to_support'],
        'dist_to_resistance': df['dist_to_resistance'],
        'vote_agreement': vote_agreement,
        'avg_confidence': avg_confidence
    })
    # Logika ini sudah benar, masalahnya ada pada pembuatan 'target_meta' di app.py
    y = df['target_meta'].apply(lambda v: -1 if v == -1 else (1 if v == 1 else 0))
    combined = X.join(y.rename('target_meta'), how='inner')
    combined.dropna(subset=['target_meta'], inplace=True)
    return combined.drop(columns=['target_meta']), combined['target_meta']


def train_meta_learner(df, lstm_predictions, xgb_predictions, xgb_confidences,
                       cnn_predictions, svc_predictions, svc_confidences,
                       nb_predictions, nb_confidences, timeframe_key,
                       tune_hyperparams=False, n_iter=50):
    """
    Melatih Master AI menggunakan LightGBM.
    Jika tune_hyperparams=True, lakukan RandomizedSearchCV.
    Model disimpan dengan nama meta_learner_randomforest_<timeframe>.pkl agar kompatibel.
    """
    X, y = prepare_data_for_meta_learner(
        df, lstm_predictions, xgb_predictions, xgb_confidences,
        cnn_predictions, svc_predictions, svc_confidences,
        nb_predictions, nb_confidences
    )
    if len(y.unique()) < 2:
        print("Target tidak cukup beragam untuk klasifikasi.")
        # --- PERBAIKAN: Memberikan detail lebih jika kelas target kurang dari 3
        if len(y.unique()) > 0:
            print(f"Hanya ditemukan kelas: {y.unique()}. Seharusnya ada [-1, 0, 1].")
            print("Pastikan data training memiliki pergerakan naik, turun, dan netral yang cukup.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    base_model = LGBMClassifier(random_state=42)
    if tune_hyperparams:
        param_dist = {
            'n_estimators': [100, 200, 300],
            'num_leaves': [31, 63, 127],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        search = RandomizedSearchCV(
            base_model, param_distributions=param_dist,
            n_iter=n_iter, scoring='f1_macro', cv=3,
            random_state=42, n_jobs=-1
        )
        print("Memulai hyperparameter tuning LightGBM...")
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print(f"Best params: {search.best_params_}")
    else:
        model = base_model
        print(f"Melatih LGBMClassifier standar untuk timeframe {timeframe_key}...")
        model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    # --- PERBAIKAN: Menambahkan zero_division=0 untuk menghindari error jika suatu kelas tidak ada di hasil prediksi
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    model_path = f"model/meta_learner_randomforest_{timeframe_key}.pkl"
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model disimpan di {model_path}")
    return model


def get_meta_signal(df, lstm_predictions, xgb_predictions, xgb_confidences,
                    cnn_predictions, svc_predictions, svc_confidences,
                    nb_predictions, nb_confidences, meta_model, threshold=0.52): # <-- PERBAIKAN: Threshold default diturunkan
    """
    Menghasilkan sinyal akhir: 1 (BUY), 0 (HOLD), -1 (SELL).
    """
    if meta_model is None:
        return pd.Series(0, index=df.index)

    last_idx = df.index[-1:]
    preds = [p.reindex(last_idx).fillna(0) for p in [
        lstm_predictions, xgb_predictions, xgb_confidences,
        cnn_predictions, svc_predictions, svc_confidences,
        nb_predictions, nb_confidences
    ]]
    # Pastikan data yang dikirim ke prepare_data_for_meta_learner adalah untuk satu baris terakhir
    X_pred, _ = prepare_data_for_meta_learner(df.loc[last_idx], *preds)

    if X_pred.empty:
        return pd.Series(0, index=df.index)

    proba = meta_model.predict_proba(X_pred)[0]
    max_conf = np.max(proba)
    pred_class = meta_model.classes_[np.argmax(proba)]

    # Mengembalikan sinyal prediksi jika confidence melewati threshold, jika tidak maka HOLD
    final_signal = pred_class if max_conf >= threshold else 0
    # Pastikan series yang dikembalikan memiliki index yang sama dengan dataframe input terakhir
    return pd.Series(final_signal, index=last_idx)