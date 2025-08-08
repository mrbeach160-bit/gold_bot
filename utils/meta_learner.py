import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Import dependency manager for robust handling
try:
    from utils.dependency_manager import is_available, require_dependency
    DEPENDENCY_MANAGER_AVAILABLE = True
except ImportError:
    DEPENDENCY_MANAGER_AVAILABLE = False

# Import LightGBM with fallback
if DEPENDENCY_MANAGER_AVAILABLE and is_available('lightgbm'):
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
else:
    try:
        from lightgbm import LGBMClassifier
        LIGHTGBM_AVAILABLE = True
    except ImportError:
        LIGHTGBM_AVAILABLE = False
        print("Warning: LightGBM not available, meta learner training will be disabled")
        
        # Create a dummy classifier for fallback
        class LGBMClassifier:
            def __init__(self, *args, **kwargs):
                self.classes_ = np.array([-1, 0, 1])
                
            def fit(self, X, y):
                return self
                
            def predict(self, X):
                return np.zeros(len(X))
                
            def predict_proba(self, X):
                return np.array([[0.33, 0.34, 0.33]] * len(X))


def prepare_data_for_meta_learner(df, lstm_predictions, xgb_predictions, xgb_confidences,
                                  cnn_predictions, svc_predictions, svc_confidences,
                                  nb_predictions, nb_confidences):
    """
    Prepare features and targets for Meta-Learner with proper validation.
    
    Args:
        df: DataFrame containing market data and target_meta column
        lstm_predictions: LSTM model predictions 
        xgb_predictions: XGBoost model predictions
        xgb_confidences: XGBoost confidence scores
        cnn_predictions: CNN model predictions
        svc_predictions: SVC model predictions
        svc_confidences: SVC confidence scores
        nb_predictions: Naive Bayes predictions
        nb_confidences: Naive Bayes confidence scores
    
    Returns:
        tuple: (features DataFrame, target Series)
        
    Target mapping:
        - -1 = SELL (bearish signal)
        - 0 = HOLD (neutral/uncertain)  
        - 1 = BUY (bullish signal)
    
    Raises:
        ValueError: If target_meta column is missing or has invalid values
        ValueError: If prediction arrays have mismatched lengths
    """
    # Validate required target column exists
    if 'target_meta' not in df.columns:
        raise ValueError(
            "target_meta column missing from dataframe. "
            "Ensure target values are properly computed before calling meta learner."
        )
    
    # Validate prediction inputs
    prediction_arrays = [
        ('lstm_predictions', lstm_predictions),
        ('xgb_predictions', xgb_predictions), 
        ('xgb_confidences', xgb_confidences),
        ('cnn_predictions', cnn_predictions),
        ('svc_predictions', svc_predictions),
        ('svc_confidences', svc_confidences),
        ('nb_predictions', nb_predictions),
        ('nb_confidences', nb_confidences)
    ]
    
    for name, pred_array in prediction_arrays:
        if pred_array is None:
            raise ValueError(f"{name} cannot be None")
        if len(pred_array) == 0:
            raise ValueError(f"{name} cannot be empty")

    # Reindex predictions to match dataframe and fill missing values
    preds = [lstm_predictions, xgb_predictions, xgb_confidences,
             cnn_predictions, svc_predictions, svc_confidences,
             nb_predictions, nb_confidences]
    preds = [p.reindex(df.index).fillna(0) for p in preds]
    lstm_p, xgb_p, xgb_c, cnn_p, svc_p, svc_c, nb_p, nb_c = preds

    # Fill missing indicator values with sensible defaults
    df = df.fillna({
        'rsi': 50,  # Neutral RSI
        'bb_percent': 0.5,  # Middle of Bollinger Bands
        'MACDh_12_26_9': 0,  # Neutral MACD
        'ADX_14': 25,  # Moderate trend strength
        'dist_to_support': 0,  # No distance data
        'dist_to_resistance': 0,  # No distance data  
        'ema_signal_numeric': 0  # Neutral EMA signal
    })

    # Engineered consensus features
    vote_agreement = ((xgb_p == cnn_p) & (cnn_p == svc_p)).astype(int)
    avg_confidence = (xgb_c + svc_c + nb_c) / 3
    
    # Confidence-weighted ensemble signals
    confidence_sum = xgb_c + svc_c + nb_c
    weighted_signal = (xgb_p * xgb_c + svc_p * svc_c + nb_p * nb_c) / np.maximum(confidence_sum, 0.001)

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
        'avg_confidence': avg_confidence,
        'weighted_signal': weighted_signal
    })
    
    # Create clear target mapping with validation
    def create_target(value):
        """
        Convert target_meta value to standardized class labels.
        
        Args:
            value: Raw target value
            
        Returns:
            int: -1 (SELL), 0 (HOLD), or 1 (BUY)
        """
        if pd.isna(value):
            return 0  # HOLD for missing values
        elif value > 0.5:  # Strong bullish signal
            return 1  # BUY
        elif value < -0.5:  # Strong bearish signal  
            return -1  # SELL
        else:  # Weak or neutral signal
            return 0  # HOLD
    
    # Apply target creation with clear logic
    y = df['target_meta'].apply(create_target)
    
    # Validate target distribution
    target_counts = y.value_counts()
    if len(target_counts) < 2:
        print(f"Warning: Only {len(target_counts)} target classes found: {target_counts.index.tolist()}")
        print("Consider adjusting target creation thresholds for better class balance")
    
    # Combine features and targets, handling missing data
    combined = X.join(y.rename('target_meta'), how='inner')
    
    # Remove rows with NaN targets (data leakage prevention)
    initial_rows = len(combined)
    combined.dropna(subset=['target_meta'], inplace=True)
    
    if len(combined) < initial_rows:
        print(f"Removed {initial_rows - len(combined)} rows with missing targets")
    
    if len(combined) == 0:
        raise ValueError("No valid samples remaining after data preparation")
    
    return combined.drop(columns=['target_meta']), combined['target_meta']


def train_meta_learner(df, lstm_predictions, xgb_predictions, xgb_confidences,
                       cnn_predictions, svc_predictions, svc_confidences,
                       nb_predictions, nb_confidences, timeframe_key,
                       tune_hyperparams=False, n_iter=50):
    """
    Train Master AI using LightGBM or fallback to simpler model.
    Model is saved with compatible naming for backward compatibility.
    
    Args:
        df: DataFrame with market data and targets
        *_predictions: Various model predictions
        *_confidences: Model confidence scores  
        timeframe_key: Timeframe identifier for model naming
        tune_hyperparams: Whether to perform hyperparameter tuning
        n_iter: Number of iterations for hyperparameter search
        
    Returns:
        Trained model or None if training fails
    """
    if not LIGHTGBM_AVAILABLE:
        print("Warning: LightGBM not available, using fallback classifier")
    
    try:
        X, y = prepare_data_for_meta_learner(
            df, lstm_predictions, xgb_predictions, xgb_confidences,
            cnn_predictions, svc_predictions, svc_confidences,
            nb_predictions, nb_confidences
        )
        
        if len(y.unique()) < 2:
            print("Target classes insufficient for classification.")
            if len(y.unique()) > 0:
                print(f"Found classes: {y.unique()}. Expected [-1, 0, 1].")
                print("Ensure training data has sufficient up, down, and neutral movements.")
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        if LIGHTGBM_AVAILABLE:
            base_model = LGBMClassifier(random_state=42)
        else:
            # Fallback to RandomForest if available
            try:
                from sklearn.ensemble import RandomForestClassifier
                base_model = RandomForestClassifier(n_estimators=100, random_state=42)
                print("Using RandomForest as fallback classifier")
            except ImportError:
                print("Error: No suitable classifier available")
                return None
        
        if tune_hyperparams and LIGHTGBM_AVAILABLE:
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
            print("Starting hyperparameter tuning...")
            search.fit(X_train, y_train)
            model = search.best_estimator_
            print(f"Best params: {search.best_params_}")
        else:
            model = base_model
            print(f"Training {type(model).__name__} for timeframe {timeframe_key}...")
            model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=3, zero_division=0))

        # Save model with path management
        try:
            from utils.path_manager import get_model_path
            model_path = get_model_path(f'meta_learner_randomforest_{timeframe_key}.pkl')
        except ImportError:
            model_path = f"model/meta_learner_randomforest_{timeframe_key}.pkl"
            os.makedirs('model', exist_ok=True)
        
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        return model
        
    except Exception as e:
        print(f"Error training meta learner: {e}")
        return None


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