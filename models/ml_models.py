# models/ml_models.py
"""
Consolidated ML models with standardized interfaces.
Migrated from utils/ with minimal changes for backward compatibility.
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime
from typing import Dict, Any, List

# Import technical indicators
try:
    from utils.indicators import add_indicators
    INDICATORS_AVAILABLE = True
except ImportError:
    INDICATORS_AVAILABLE = False
    print("Warning: Technical indicators utility not available")

# Try to import TensorFlow, handle gracefully if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available, LSTM model will be disabled")

# Try to import LightGBM, handle gracefully if not available  
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available, LightGBM model will be disabled")

# Try to import XGBoost, handle gracefully if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available, XGBoost model will be disabled")

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .base import BaseModel


def calculate_trading_metrics(y_true, y_pred, y_prob=None):
    """Calculate trading-specific evaluation metrics.
    
    Args:
        y_true: True binary labels (1 for up, 0 for down)
        y_pred: Predicted binary labels  
        y_prob: Prediction probabilities (optional)
    
    Returns:
        Dict with trading metrics
    """
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Calculate win rate and trading-specific metrics
        total_trades = len(y_pred)
        if total_trades > 0:
            correct_predictions = sum(y_true == y_pred)
            metrics['win_rate'] = correct_predictions / total_trades
            
            # Calculate directional accuracy for buy/sell separately
            buy_signals = y_pred == 1
            sell_signals = y_pred == 0
            
            if sum(buy_signals) > 0:
                buy_accuracy = sum((y_true == 1) & (y_pred == 1)) / sum(buy_signals)
                metrics['buy_accuracy'] = buy_accuracy
            else:
                metrics['buy_accuracy'] = 0.0
            
            if sum(sell_signals) > 0:
                sell_accuracy = sum((y_true == 0) & (y_pred == 0)) / sum(sell_signals)
                metrics['sell_accuracy'] = sell_accuracy
            else:
                metrics['sell_accuracy'] = 0.0
            
            # Confidence-based metrics if probabilities provided
            if y_prob is not None:
                high_confidence_mask = y_prob > 0.7
                if sum(high_confidence_mask) > 0:
                    high_conf_accuracy = sum((y_true == y_pred) & high_confidence_mask) / sum(high_confidence_mask)
                    metrics['high_confidence_accuracy'] = high_conf_accuracy
                else:
                    metrics['high_confidence_accuracy'] = 0.0
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating trading metrics: {e}")
        return {'accuracy': 0.0, 'error': str(e)}


class LSTMModel(BaseModel):
    """LSTM neural network model migrated from utils/lstm_model.py"""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.scaler = None
        self.sequence_length = 60
        if not TENSORFLOW_AVAILABLE:
            print(f"Warning: TensorFlow not available, LSTM model for {symbol} {timeframe} will not function")
        
    def train(self, data: pd.DataFrame) -> bool:
        """Train LSTM model with provided data."""
        if not TENSORFLOW_AVAILABLE:
            print("Cannot train LSTM model: TensorFlow not available")
            return False
            
        try:
            print(f"Training LSTM model for {self.symbol} timeframe {self.timeframe}...")
            
            # Prepare data with TimeSeriesSplit validation
            close_data = data[['close']].values
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(close_data)
            
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            if len(X) < 100:
                print(f"Insufficient data for LSTM training: {len(X)} samples")
                return False
            
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for smaller datasets
            val_losses = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Build enhanced LSTM model
                fold_model = Sequential([
                    LSTM(units=100, return_sequences=True, dropout=0.2, input_shape=(X.shape[1], 1)),
                    BatchNormalization(),
                    LSTM(units=50, return_sequences=True, dropout=0.2),
                    LSTM(units=25, dropout=0.2),
                    Dense(50, activation='relu'),
                    Dropout(0.3),
                    Dense(1, activation='sigmoid')
                ])
                
                fold_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
                
                # Train with early stopping
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                history = fold_model.fit(
                    X_train_fold, y_train_fold,
                    validation_data=(X_val_fold, y_val_fold),
                    epochs=50, batch_size=32, verbose=0,
                    callbacks=[early_stopping]
                )
                
                val_loss = min(history.history['val_loss'])
                val_losses.append(val_loss)
            
            # Train final model on most recent data (preserving temporal order)
            train_size = int(0.8 * len(X))
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Clear session and build enhanced model
            tf.keras.backend.clear_session()
            self.model = Sequential([
                LSTM(units=100, return_sequences=True, dropout=0.2, input_shape=(X.shape[1], 1)),
                BatchNormalization(),
                LSTM(units=50, return_sequences=True, dropout=0.2),
                LSTM(units=25, dropout=0.2),
                Dense(50, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
            
            # Train with early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50, batch_size=32, verbose=0,
                callbacks=[early_stopping]
            )
            
            # Print training results
            final_loss = min(history.history['val_loss'])
            cv_loss = np.mean(val_losses)
            print(f"LSTM training completed.")
            print(f"Cross-validation loss: {cv_loss:.6f} (+/- {np.std(val_losses)*2:.6f})")
            print(f"Final validation loss: {final_loss:.6f}")
            print(f"Training stopped at epoch: {len(history.history['loss'])}")
            
            self._trained = True
            return True
            
        except Exception as e:
            print(f"Error training LSTM model: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with LSTM model."""
        if not TENSORFLOW_AVAILABLE:
            return self._unavailable_prediction("TensorFlow not available")
            
        if not self._trained or self.model is None or self.scaler is None:
            return self._default_prediction()
        
        try:
            # Predict next close price (minimal change from original)
            last_sequence = data[["close"]].values[-self.sequence_length:]
            scaled_seq = self.scaler.transform(last_sequence)
            X = np.reshape(scaled_seq, (1, self.sequence_length, 1))
            pred = self.model.predict(X, verbose=0)
            pred_unscaled = self.scaler.inverse_transform(pred)
            
            current_price = data['close'].iloc[-1]
            predicted_price = pred_unscaled[0][0]
            
            # Convert to direction and confidence
            price_diff = predicted_price - current_price
            price_change_pct = abs(price_diff) / current_price
            
            if price_diff > 0:
                direction = "BUY"
            elif price_diff < 0:
                direction = "SELL"
            else:
                direction = "HOLD"
                
            # Confidence based on percentage change
            confidence = min(price_change_pct * 10, 1.0)  # Scale to 0-1
            
            return {
                'direction': direction,
                'confidence': confidence,
                'probability': confidence,
                'model_name': 'LSTM',
                'timestamp': datetime.now(),
                'features_used': ['close'],
                'predicted_price': predicted_price,
                'current_price': current_price
            }
            
        except Exception as e:
            print(f"Error making LSTM prediction: {e}")
            return self._default_prediction()
    
    def save(self) -> bool:
        """Save LSTM model and scaler."""
        if not TENSORFLOW_AVAILABLE:
            print("Cannot save LSTM model: TensorFlow not available")
            return False
            
        try:
            os.makedirs('model', exist_ok=True)
            
            # Save model
            model_path = self.get_model_path('.keras')
            self.model.save(model_path)
            
            # Save scaler
            scaler_path = self.get_model_path('_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            
            print(f"LSTM model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving LSTM model: {e}")
            return False
    
    def load(self) -> bool:
        """Load LSTM model and scaler."""
        if not TENSORFLOW_AVAILABLE:
            print("Cannot load LSTM model: TensorFlow not available")
            return False
            
        try:
            model_path = self.get_model_path('.keras')
            scaler_path = self.get_model_path('_scaler.pkl')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                self._trained = True
                print(f"LSTM model loaded from {model_path}")
                return True
            else:
                print(f"LSTM model files not found: {model_path}, {scaler_path}")
                return False
                
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return LSTM model metadata."""
        return {
            'name': 'LSTM',
            'type': 'neural_network',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'sequence_length': self.sequence_length,
            'trained': self._trained,
            'features': ['close'],
            'model_path': self.get_model_path('.keras'),
            'available': TENSORFLOW_AVAILABLE
        }
    
    def _unavailable_prediction(self, reason: str) -> Dict[str, Any]:
        """Return prediction when dependencies unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'LSTM',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': reason
        }
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when model unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'LSTM',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': 'Model not trained or unavailable'
        }


class LightGBMModel(BaseModel):
    """LightGBM gradient boosting model migrated from utils/lgb_model.py"""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.feature_columns = None
        if not LIGHTGBM_AVAILABLE:
            print(f"Warning: LightGBM not available, LightGBM model for {symbol} {timeframe} will not function")
        
    def train(self, data: pd.DataFrame) -> bool:
        """Train LightGBM model with provided data."""
        if not LIGHTGBM_AVAILABLE:
            print("Cannot train LightGBM model: LightGBM not available")
            return False
            
        try:
            print(f"Training LightGBM model for {self.symbol} timeframe {self.timeframe}...")
            
            # Prepare features with proper technical indicators
            df = data.copy()
            
            # Calculate proper technical indicators if missing
            if not all(col in df.columns for col in ['rsi', 'MACDh_12_26_9', 'ADX_14']):
                if INDICATORS_AVAILABLE:
                    print("Calculating technical indicators...")
                    try:
                        df = add_indicators(df)
                        print("Technical indicators calculated successfully")
                    except Exception as e:
                        print(f"Failed to calculate indicators: {e}")
                        print("Cannot train model without proper technical indicators")
                        return False
                else:
                    print("Technical indicators missing and calculator not available")
                    return False
            
            # Validate that we have sufficient data after indicator calculation
            if len(df) < 100:
                print(f"Insufficient data after indicator calculation: {len(df)} rows")
                return False
            
            # Create target: next close higher than current
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df.dropna(inplace=True)
            
            if len(df) < 50:
                print("Insufficient data after preprocessing")
                return False
            
            # Use validated technical indicators
            self.feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'MACDh_12_26_9', 'ADX_14']
            X = df[self.feature_columns]
            y = df['target']
            
            # Use TimeSeriesSplit to prevent data leakage
            tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for smaller datasets
            accuracies = []
            
            # Train on the full dataset for final model, but validate with time series split
            for train_idx, test_idx in tscv.split(X):
                X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
                y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train fold model for validation
                fold_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
                fold_model.fit(X_train_fold, y_train_fold)
                fold_accuracy = fold_model.score(X_test_fold, y_test_fold)
                accuracies.append(fold_accuracy)
            
            # Train final model on all data (preserving temporal order)
            train_size = int(0.8 * len(X))
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            self.model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Calculate final accuracy and additional metrics
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]  # Probability of positive class
            
            # Calculate comprehensive metrics
            final_metrics = calculate_trading_metrics(y_test, y_pred, y_prob)
            cv_accuracy = np.mean(accuracies)
            
            print(f"LightGBM training completed.")
            print(f"Cross-validation accuracy: {cv_accuracy:.4f} (+/- {np.std(accuracies)*2:.4f})")
            print(f"Final test metrics:")
            for metric, value in final_metrics.items():
                if metric != 'error':
                    print(f"  {metric}: {value:.4f}")
            
            self._trained = True
            return True
            
        except Exception as e:
            print(f"Error training LightGBM model: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            return self._unavailable_prediction("LightGBM not available")
            
        if not self._trained or self.model is None:
            return self._default_prediction()
        
        try:
            # Prepare latest data for prediction
            latest_df = data.copy()
            
            # Calculate indicators if missing
            if not all(col in latest_df.columns for col in ['rsi', 'MACDh_12_26_9', 'ADX_14']):
                if INDICATORS_AVAILABLE:
                    try:
                        latest_df = add_indicators(latest_df)
                    except Exception as e:
                        print(f"Failed to calculate indicators for prediction: {e}")
                        return self._default_prediction()
                else:
                    print("Cannot make prediction: technical indicators missing")
                    return self._default_prediction()
            
            latest_df.dropna(inplace=True)
            
            if len(latest_df) == 0:
                print("No valid data for prediction after preprocessing")
                return self._default_prediction()
            
            X = latest_df[self.feature_columns].tail(1)
            
            if X.empty:
                return self._default_prediction()
            
            pred = self.model.predict(X)[0]
            prob = self.model.predict_proba(X)[0].max()
            
            direction = "BUY" if pred == 1 else "SELL"
            
            return {
                'direction': direction,
                'confidence': prob,
                'probability': prob,
                'model_name': 'LightGBM',
                'timestamp': datetime.now(),
                'features_used': self.feature_columns,
                'prediction': int(pred)
            }
            
        except Exception as e:
            print(f"Error making LightGBM prediction: {e}")
            return self._default_prediction()
    
    def save(self) -> bool:
        """Save LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            print("Cannot save LightGBM model: LightGBM not available")
            return False
            
        try:
            os.makedirs('model', exist_ok=True)
            model_path = self.get_model_path('.pkl')
            joblib.dump(self.model, model_path)
            
            # Save feature columns
            metadata = {
                'feature_columns': self.feature_columns,
                'symbol': self.symbol,
                'timeframe': self.timeframe
            }
            metadata_path = self.get_model_path('_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            print(f"LightGBM model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving LightGBM model: {e}")
            return False
    
    def load(self) -> bool:
        """Load LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            print("Cannot load LightGBM model: LightGBM not available")
            return False
            
        try:
            model_path = self.get_model_path('.pkl')
            metadata_path = self.get_model_path('_metadata.json')
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                
                # Load metadata if available
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.feature_columns = metadata.get('feature_columns')
                else:
                    # Default feature columns for backward compatibility
                    self.feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'MACDh_12_26_9', 'ADX_14']
                
                self._trained = True
                print(f"LightGBM model loaded from {model_path}")
                return True
            else:
                print(f"LightGBM model file not found: {model_path}")
                return False
                
        except Exception as e:
            print(f"Error loading LightGBM model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return LightGBM model metadata."""
        return {
            'name': 'LightGBM',
            'type': 'gradient_boosting',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'trained': self._trained,
            'features': self.feature_columns,
            'model_path': self.get_model_path('.pkl'),
            'available': LIGHTGBM_AVAILABLE
        }
    
    def _unavailable_prediction(self, reason: str) -> Dict[str, Any]:
        """Return prediction when dependencies unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'LightGBM',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': reason
        }
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when model unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'LightGBM',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': 'Model not trained or unavailable'
        }


class XGBoostModel(BaseModel):
    """XGBoost gradient boosting model migrated from utils/xgb_model.py"""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.feature_columns = None
        if not XGBOOST_AVAILABLE:
            print(f"Warning: XGBoost not available, XGBoost model for {symbol} {timeframe} will not function")
        
    def train(self, data: pd.DataFrame) -> bool:
        """Train XGBoost model with provided data."""
        if not XGBOOST_AVAILABLE:
            print("Cannot train XGBoost model: XGBoost not available")
            return False
            
        try:
            print(f"Training XGBoost model for {self.symbol} timeframe {self.timeframe}...")
            
            # Prepare data safely
            df = data.copy()
            
            # Add indicators if missing
            if INDICATORS_AVAILABLE and not all(col in df.columns for col in ['rsi', 'MACDh_12_26_9', 'ADX_14']):
                try:
                    df = add_indicators(df)
                except Exception as e:
                    print(f"Failed to calculate indicators: {e}")
                    return False
            
            # Create enhanced targets with multiple horizons
            df['target_1'] = (df['close'].shift(-1) > df['close']).astype(int)  # 1 period ahead
            df['target_5'] = (df['close'].shift(-5) > df['close']).astype(int)  # 5 periods ahead
            
            # Use the 5-period target as main target
            df.dropna(inplace=True)
            
            if len(df) < 100:
                print(f"Insufficient data for XGBoost training: {len(df)} rows")
                return False
            
            # Select features more safely
            price_features = ['open', 'high', 'low', 'close', 'volume']
            indicator_features = []
            
            # Add available indicator features
            possible_indicators = ['rsi', 'MACDh_12_26_9', 'ADX_14', 'EMA_10', 'EMA_20', 'ATR_14', 'STOCH_k']
            for ind in possible_indicators:
                if ind in df.columns:
                    indicator_features.append(ind)
            
            self.feature_columns = price_features + indicator_features
            
            # Ensure we have features to work with
            if len(self.feature_columns) < 5:
                print("Insufficient features for XGBoost training")
                return False
            
            X = df[self.feature_columns]
            y = df['target_5']  # Use 5-period ahead target
            
            if X.empty or y.empty:
                print("Empty features or targets after preprocessing")
                return False
            
            # Use TimeSeriesSplit for validation
            tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for smaller datasets
            accuracies = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
                y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
                
                # Check if we have both classes in training fold
                if len(y_train_fold.unique()) < 2:
                    print(f"Skipping fold with only one class: {y_train_fold.unique()}")
                    continue
                
                fold_model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss', 
                    use_label_encoder=False,
                    n_estimators=50,  # Reduced for smaller datasets
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
                
                fold_model.fit(X_train_fold, y_train_fold)
                fold_accuracy = fold_model.score(X_test_fold, y_test_fold)
                accuracies.append(fold_accuracy)
            
            # If no valid folds, skip cross-validation
            if not accuracies:
                print("No valid cross-validation folds - training on single split")
                accuracies = [0.5]  # Default accuracy
            
            # Train final model using temporal split
            train_size = int(0.8 * len(X))
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            # Check if we have both classes in training data
            if len(y_train.unique()) < 2:
                print("Training data has only one class - adjusting split")
                # Use more data for training to ensure both classes
                train_size = int(0.9 * len(X))
                X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
                y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
                
                if len(y_train.unique()) < 2:
                    print("Still only one class - using stratified approach")
                    # Ensure we have at least one sample of each class
                    class_0_indices = y[y == 0].index
                    class_1_indices = y[y == 1].index
                    
                    if len(class_0_indices) == 0 or len(class_1_indices) == 0:
                        print("Dataset has only one class overall - cannot train classifier")
                        return False
                    
                    # Take most samples for training, but ensure both classes present
                    train_0 = class_0_indices[:max(1, int(0.8 * len(class_0_indices)))]
                    train_1 = class_1_indices[:max(1, int(0.8 * len(class_1_indices)))]
                    test_0 = class_0_indices[len(train_0):]
                    test_1 = class_1_indices[len(train_1):]
                    
                    train_indices = list(train_0) + list(train_1)
                    test_indices = list(test_0) + list(test_1)
                    
                    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
                    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss', 
                use_label_encoder=False,
                n_estimators=50,  # Reduced for smaller datasets
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Calculate accuracy
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]  # Probability of positive class
            
            # Calculate comprehensive metrics
            final_metrics = calculate_trading_metrics(y_test, y_pred, y_prob)
            cv_accuracy = np.mean(accuracies)
            
            print(f"XGBoost training completed.")
            print(f"Cross-validation accuracy: {cv_accuracy:.4f} (+/- {np.std(accuracies)*2:.4f})")
            print(f"Final test metrics:")
            for metric, value in final_metrics.items():
                if metric != 'error':
                    print(f"  {metric}: {value:.4f}")
            
            # Print feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                print("Feature importance:")
                for i, feature in enumerate(self.feature_columns):
                    print(f"  {feature}: {importance[i]:.4f}")
            
            self._trained = True
            return True
            
        except Exception as e:
            print(f"Error training XGBoost model: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with XGBoost model."""
        if not XGBOOST_AVAILABLE:
            return self._unavailable_prediction("XGBoost not available")
            
        if not self._trained or self.model is None:
            return self._default_prediction()
        
        try:
            # Prepare data with indicators if needed
            df = data.copy()
            
            if INDICATORS_AVAILABLE and not all(col in df.columns for col in self.feature_columns[-3:]):  # Check for indicators
                try:
                    df = add_indicators(df)
                except Exception as e:
                    print(f"Failed to calculate indicators for prediction: {e}")
                    return self._default_prediction()
            
            # Check if we have all required features
            missing_features = [col for col in self.feature_columns if col not in df.columns]
            if missing_features:
                print(f"Missing features for prediction: {missing_features}")
                return self._default_prediction()
            
            prediction_features = df[self.feature_columns].tail(1)
            
            if prediction_features.empty:
                return self._default_prediction()
            
            prediction = self.model.predict(prediction_features)
            prediction_proba = self.model.predict_proba(prediction_features)
            
            final_prediction = int(prediction[0])
            final_proba = prediction_proba[0]
            
            direction = "BUY" if final_prediction == 1 else "SELL"
            confidence = float(max(final_proba))
            
            return {
                'direction': direction,
                'confidence': confidence,
                'probability': confidence,
                'model_name': 'XGBoost',
                'timestamp': datetime.now(),
                'features_used': self.feature_columns,
                'prediction': final_prediction
            }
            
        except Exception as e:
            print(f"Error making XGBoost prediction: {e}")
            return self._default_prediction()
    
    def save(self) -> bool:
        """Save XGBoost model."""
        if not XGBOOST_AVAILABLE:
            print("Cannot save XGBoost model: XGBoost not available")
            return False
            
        try:
            os.makedirs('model', exist_ok=True)
            model_path = self.get_model_path('.json')
            self.model.save_model(model_path)
            
            # Save feature columns
            metadata = {
                'feature_columns': self.feature_columns,
                'symbol': self.symbol,
                'timeframe': self.timeframe
            }
            metadata_path = self.get_model_path('_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            print(f"XGBoost model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving XGBoost model: {e}")
            return False
    
    def load(self) -> bool:
        """Load XGBoost model."""
        if not XGBOOST_AVAILABLE:
            print("Cannot load XGBoost model: XGBoost not available")
            return False
            
        try:
            model_path = self.get_model_path('.json')
            metadata_path = self.get_model_path('_metadata.json')
            
            if os.path.exists(model_path):
                self.model = xgb.XGBClassifier()
                self.model.load_model(model_path)
                
                # Load metadata if available
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.feature_columns = metadata.get('feature_columns')
                
                self._trained = True
                print(f"XGBoost model loaded from {model_path}")
                return True
            else:
                print(f"XGBoost model file not found: {model_path}")
                return False
                
        except Exception as e:
            print(f"Error loading XGBoost model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return XGBoost model metadata."""
        return {
            'name': 'XGBoost',
            'type': 'gradient_boosting',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'trained': self._trained,
            'features': self.feature_columns,
            'model_path': self.get_model_path('.json'),
            'available': XGBOOST_AVAILABLE
        }
    
    def _unavailable_prediction(self, reason: str) -> Dict[str, Any]:
        """Return prediction when dependencies unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'XGBoost',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': reason
        }
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when model unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'XGBoost',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': 'Model not trained or unavailable'
        }