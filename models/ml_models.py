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

# Try to import TensorFlow, handle gracefully if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available, LSTM and CNN models will be disabled")

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

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from .base import BaseModel


class LSTMModel(BaseModel):
    """Enhanced LSTM neural network model with regularization and optimization."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.scaler = None
        
        # Enhanced configurable parameters
        self.sequence_length = 60
        self.dropout_rate = 0.3
        self.use_batch_norm = True
        self.epochs = 50
        self.batch_size = 32
        self.lstm_units = [50, 50]  # Support multiple LSTM layers
        self.use_early_stopping = True
        self.patience = 10
        
        if not TENSORFLOW_AVAILABLE:
            print(f"Warning: TensorFlow not available, LSTM model for {symbol} {timeframe} will not function")
        
    def train(self, data: pd.DataFrame) -> bool:
        """Train enhanced LSTM model with dropout, batch normalization and early stopping."""
        if not TENSORFLOW_AVAILABLE:
            print("Cannot train LSTM model: TensorFlow not available")
            return False
            
        try:
            print(f"Training enhanced LSTM model for {self.symbol} timeframe {self.timeframe}...")
            print(f"Configuration: epochs={self.epochs}, dropout={self.dropout_rate}, batch_norm={self.use_batch_norm}")
            
            # Prepare data with multiple features for better performance
            feature_columns = ['close']
            
            # Add volume if available for better predictions
            if 'volume' in data.columns:
                feature_columns.append('volume')
            
            # Add basic technical indicators if available
            if 'rsi' in data.columns:
                feature_columns.append('rsi')
            if 'MACD_12_26_9' in data.columns:
                feature_columns.append('MACD_12_26_9')
            
            feature_data = data[feature_columns].values
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(feature_data)
            
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i])
                y.append(scaled_data[i, 0])  # Predict close price
            
            X, y = np.array(X), np.array(y)
            
            if len(X) < 50:
                print(f"Insufficient data for LSTM training: {len(X)} sequences")
                return False
            
            # Split data for validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Clear session and build enhanced model
            tf.keras.backend.clear_session()
            
            model = Sequential()
            
            # First LSTM layer with dropout
            model.add(LSTM(
                units=self.lstm_units[0], 
                return_sequences=len(self.lstm_units) > 1, 
                input_shape=(X.shape[1], X.shape[2])
            ))
            model.add(Dropout(self.dropout_rate))
            
            if self.use_batch_norm:
                model.add(BatchNormalization())
            
            # Additional LSTM layers if configured
            for i, units in enumerate(self.lstm_units[1:], 1):
                return_sequences = i < len(self.lstm_units) - 1
                model.add(LSTM(units=units, return_sequences=return_sequences))
                model.add(Dropout(self.dropout_rate))
                
                if self.use_batch_norm:
                    model.add(BatchNormalization())
            
            # Dense output layer
            model.add(Dense(1))
            
            # Compile with improved optimizer settings
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            self.model = model
            
            # Setup callbacks for better training
            callbacks = []
            
            if self.use_early_stopping:
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=1
                )
                callbacks.append(early_stopping)
            
            # Reduce learning rate on plateau
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.patience // 2,
                min_lr=0.0001,
                verbose=1
            )
            callbacks.append(reduce_lr)
            
            print(f"Starting training with {len(X_train)} training samples, {len(X_val)} validation samples...")
            
            # Train with validation data and callbacks
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Print training results
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            print(f"LSTM training completed. Final loss: {final_loss:.6f}, Val loss: {final_val_loss:.6f}")
            
            self._trained = True
            return True
            
        except Exception as e:
            print(f"Error training LSTM model: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with enhanced LSTM model."""
        if not TENSORFLOW_AVAILABLE:
            return self._unavailable_prediction("TensorFlow not available")
            
        if not self._trained or self.model is None or self.scaler is None:
            return self._default_prediction()
        
        try:
            # Prepare features same as training
            feature_columns = ['close']
            
            # Add same features used in training
            if 'volume' in data.columns:
                feature_columns.append('volume')
            if 'rsi' in data.columns:
                feature_columns.append('rsi')
            if 'MACD_12_26_9' in data.columns:
                feature_columns.append('MACD_12_26_9')
            
            # Get last sequence for prediction
            if len(data) < self.sequence_length:
                return self._default_prediction()
            
            last_sequence = data[feature_columns].values[-self.sequence_length:]
            scaled_seq = self.scaler.transform(last_sequence)
            X = np.reshape(scaled_seq, (1, self.sequence_length, len(feature_columns)))
            
            pred = self.model.predict(X, verbose=0)
            
            # Inverse transform only the close price (first feature)
            pred_with_features = np.zeros((1, len(feature_columns)))
            pred_with_features[0, 0] = pred[0, 0]  # Set predicted close price
            
            # Fill other features with current values for inverse transform
            current_features = scaled_seq[-1:, :]
            pred_with_features[0, 1:] = current_features[0, 1:]
            
            pred_unscaled = self.scaler.inverse_transform(pred_with_features)
            
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
                
            # Enhanced confidence calculation
            confidence = min(price_change_pct * 15, 1.0)  # Slightly more sensitive
            
            return {
                'direction': direction,
                'confidence': confidence,
                'probability': confidence,
                'model_name': 'LSTM',
                'timestamp': datetime.now(),
                'features_used': feature_columns,
                'predicted_price': predicted_price,
                'current_price': current_price,
                'price_change_pct': price_change_pct
            }
            
        except Exception as e:
            print(f"Error making LSTM prediction: {e}")
            return self._default_prediction()
    
    def save(self) -> bool:
        """Save enhanced LSTM model, scaler, and configuration."""
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
            
            # Save enhanced configuration
            config_path = self.get_model_path('_config.json')
            config = {
                'sequence_length': self.sequence_length,
                'dropout_rate': self.dropout_rate,
                'use_batch_norm': self.use_batch_norm,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'lstm_units': self.lstm_units,
                'use_early_stopping': self.use_early_stopping,
                'patience': self.patience,
                'symbol': self.symbol,
                'timeframe': self.timeframe
            }
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            print(f"Enhanced LSTM model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving LSTM model: {e}")
            return False
    
    def load(self) -> bool:
        """Load enhanced LSTM model, scaler, and configuration."""
        if not TENSORFLOW_AVAILABLE:
            print("Cannot load LSTM model: TensorFlow not available")
            return False
            
        try:
            model_path = self.get_model_path('.keras')
            scaler_path = self.get_model_path('_scaler.pkl')
            config_path = self.get_model_path('_config.json')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                
                # Load enhanced configuration if available
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        self.sequence_length = config.get('sequence_length', 60)
                        self.dropout_rate = config.get('dropout_rate', 0.3)
                        self.use_batch_norm = config.get('use_batch_norm', True)
                        self.epochs = config.get('epochs', 50)
                        self.batch_size = config.get('batch_size', 32)
                        self.lstm_units = config.get('lstm_units', [50, 50])
                        self.use_early_stopping = config.get('use_early_stopping', True)
                        self.patience = config.get('patience', 10)
                
                self._trained = True
                print(f"Enhanced LSTM model loaded from {model_path}")
                return True
            else:
                print(f"LSTM model files not found: {model_path}, {scaler_path}")
                return False
                
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return enhanced LSTM model metadata."""
        return {
            'name': 'LSTM',
            'type': 'neural_network',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'sequence_length': self.sequence_length,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lstm_units': self.lstm_units,
            'use_early_stopping': self.use_early_stopping,
            'patience': self.patience,
            'trained': self._trained,
            'features': ['close', 'volume', 'rsi', 'MACD_12_26_9'],
            'model_path': self.get_model_path('.keras'),
            'available': TENSORFLOW_AVAILABLE,
            'enhanced': True
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
            
            # Prepare features (minimal change from original)
            df = data.copy()
            
            # Add basic technical indicators if they don't exist
            if 'rsi' not in df.columns:
                # Use simple price-based features if ta library not available
                df['rsi'] = 50.0  # Neutral RSI
            if 'MACDh_12_26_9' not in df.columns:
                df['MACDh_12_26_9'] = 0.0  # Neutral MACD
            if 'ADX_14' not in df.columns:
                df['ADX_14'] = 25.0  # Neutral ADX
            
            # Create target: next close higher than current
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df.dropna(inplace=True)
            
            self.feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'MACDh_12_26_9', 'ADX_14']
            X = df[self.feature_columns]
            y = df['target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            self.model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Calculate accuracy
            accuracy = self.model.score(X_test, y_test)
            print(f"LightGBM training completed. Accuracy: {accuracy:.4f}")
            
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
            # Prepare latest data for prediction (minimal change from original)
            latest_df = data.copy()
            
            # Add basic indicators if missing
            if 'rsi' not in latest_df.columns:
                latest_df['rsi'] = 50.0
            if 'MACDh_12_26_9' not in latest_df.columns:
                latest_df['MACDh_12_26_9'] = 0.0
            if 'ADX_14' not in latest_df.columns:
                latest_df['ADX_14'] = 25.0
            
            latest_df.dropna(inplace=True)
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
    """Enhanced XGBoost gradient boosting model with optimized parameters for financial data."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.feature_columns = None
        self.hyperparameter_tuning = True  # Enable tuning by default
        
        # Enhanced parameters optimized for financial data
        self.n_estimators = 300  # Increased from 100
        self.learning_rate = 0.05  # Reduced for better generalization
        self.max_depth = 6  # Increased from 3 for better complexity
        self.subsample = 0.8  # Add subsampling for regularization
        self.colsample_bytree = 0.8  # Feature subsampling
        self.reg_alpha = 0.1  # L1 regularization
        self.reg_lambda = 1.0  # L2 regularization
        
        if not XGBOOST_AVAILABLE:
            print(f"Warning: XGBoost not available, XGBoost model for {symbol} {timeframe} will not function")
        
    def train(self, data: pd.DataFrame) -> bool:
        """Train enhanced XGBoost model with optimized parameters."""
        if not XGBOOST_AVAILABLE:
            print("Cannot train XGBoost model: XGBoost not available")
            return False
            
        try:
            print(f"Training enhanced XGBoost model for {self.symbol} timeframe {self.timeframe}...")
            print(f"Configuration: n_estimators={self.n_estimators}, lr={self.learning_rate}, max_depth={self.max_depth}")
            
            # Enhanced data preparation
            df = data.copy()
            df['future_close'] = df['close'].shift(-5)  # 5 periods ahead
            df.dropna(inplace=True)
            df['target'] = (df['future_close'] > df['close']).astype(int)
            
            # Create more robust features
            base_features = ['open', 'high', 'low', 'close', 'volume']
            
            # Add price-based features
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['volume_price_ratio'] = df['volume'] / df['close']
            
            # Add technical indicators if available, otherwise create basic ones
            if 'rsi' not in df.columns:
                # Simple RSI approximation
                df['price_momentum'] = df['close'].rolling(14).mean() / df['close']
            else:
                df['price_momentum'] = df['rsi'] / 100
            
            if 'MACD_12_26_9' not in df.columns:
                # Simple momentum indicator
                df['momentum_signal'] = (df['close'].rolling(12).mean() - df['close'].rolling(26).mean()) / df['close']
            else:
                df['momentum_signal'] = df['MACD_12_26_9']
            
            # Add moving averages
            df['ma_5'] = df['close'].rolling(5).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            df['price_to_ma5'] = df['close'] / df['ma_5']
            df['price_to_ma20'] = df['close'] / df['ma_20']
            
            # Add volatility features
            df['volatility'] = df['close'].rolling(10).std()
            df['volatility_ratio'] = df['volatility'] / df['close']
            
            # Select features for training
            feature_columns = [
                'price_change', 'high_low_ratio', 'volume_price_ratio',
                'price_momentum', 'momentum_signal', 
                'price_to_ma5', 'price_to_ma20', 'volatility_ratio'
            ]
            
            # Add original OHLCV if not using derived features only
            feature_columns.extend([col for col in base_features if col in df.columns])
            
            # Remove target and future columns from features
            exclude_columns = ['future_close', 'target', 'ma_5', 'ma_20', 'volatility']
            available_features = [col for col in feature_columns if col in df.columns and col not in exclude_columns]
            
            if len(available_features) == 0:
                print("No suitable features available for XGBoost training.")
                return False
            
            X = df[available_features].fillna(0)
            y = df['target']
            
            # Remove any infinite values
            X = X.replace([np.inf, -np.inf], 0)
            
            if X.empty or y.empty:
                print("Not enough data for training XGBoost model.")
                return False
            
            self.feature_columns = available_features
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            if self.hyperparameter_tuning:
                # Use RandomizedSearchCV for hyperparameter optimization
                from sklearn.model_selection import RandomizedSearchCV
                
                param_dist = {
                    'n_estimators': [200, 300, 400, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'max_depth': [4, 5, 6, 7, 8],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0.5, 1.0, 2.0]
                }
                
                base_model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    use_label_encoder=False,
                    random_state=42,
                    n_jobs=-1
                )
                
                print("Starting hyperparameter tuning for XGBoost...")
                search = RandomizedSearchCV(
                    base_model, param_distributions=param_dist,
                    n_iter=20, scoring='roc_auc', cv=3,
                    random_state=42, n_jobs=-1
                )
                search.fit(X_train, y_train)
                
                self.model = search.best_estimator_
                print(f"Best XGBoost parameters: {search.best_params_}")
                print(f"Best CV score: {search.best_score_:.4f}")
            else:
                # Use optimized default parameters
                self.model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    use_label_encoder=False,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    reg_alpha=self.reg_alpha,
                    reg_lambda=self.reg_lambda,
                    random_state=42,
                    n_jobs=-1
                )
                
                self.model.fit(X_train, y_train)
            
            # Calculate accuracy and other metrics
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate AUC if possible
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_test, y_pred_proba)
                print(f"Enhanced XGBoost training completed. Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            except:
                print(f"Enhanced XGBoost training completed. Accuracy: {accuracy:.4f}")
            
            self._trained = True
            return True
            
        except Exception as e:
            print(f"Error training XGBoost model: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with enhanced XGBoost model."""
        if not XGBOOST_AVAILABLE:
            return self._unavailable_prediction("XGBoost not available")
            
        if not self._trained or self.model is None:
            return self._default_prediction()
        
        try:
            # Prepare prediction data with same feature engineering as training
            df = data.copy()
            
            # Create same engineered features as in training
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['volume_price_ratio'] = df['volume'] / df['close']
            
            # Add technical indicators if available, otherwise create basic ones
            if 'rsi' not in df.columns:
                df['price_momentum'] = df['close'].rolling(14).mean() / df['close']
            else:
                df['price_momentum'] = df['rsi'] / 100
            
            if 'MACD_12_26_9' not in df.columns:
                df['momentum_signal'] = (df['close'].rolling(12).mean() - df['close'].rolling(26).mean()) / df['close']
            else:
                df['momentum_signal'] = df['MACD_12_26_9']
            
            # Add moving averages
            df['ma_5'] = df['close'].rolling(5).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            df['price_to_ma5'] = df['close'] / df['ma_5']
            df['price_to_ma20'] = df['close'] / df['ma_20']
            
            # Add volatility features
            df['volatility'] = df['close'].rolling(10).std()
            df['volatility_ratio'] = df['volatility'] / df['close']
            
            # Select features used in training
            prediction_data = df[self.feature_columns].tail(1).fillna(0)
            
            # Remove any infinite values
            prediction_data = prediction_data.replace([np.inf, -np.inf], 0)
            
            if prediction_data.empty:
                return self._default_prediction()
            
            prediction = self.model.predict(prediction_data)
            prediction_proba = self.model.predict_proba(prediction_data)
            
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
                'prediction': final_prediction,
                'enhanced': True
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


class CNNModel(BaseModel):
    """Enhanced Convolutional Neural Network model for financial time series prediction."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.scaler = None
        
        # Enhanced CNN configuration
        self.sequence_length = 30  # Optimized for CNN
        self.dropout_rate = 0.4  # Higher dropout for CNN
        self.use_batch_norm = True
        self.epochs = 30
        self.batch_size = 32
        
        # CNN architecture parameters
        self.filter_sizes = [32, 64, 128]  # Multiple filter sizes
        self.kernel_sizes = [3, 5, 7]  # Different kernel sizes for multi-scale features
        self.pool_size = 2
        self.use_global_pooling = True
        self.use_early_stopping = True
        self.patience = 10
        
        if not TENSORFLOW_AVAILABLE:
            print(f"Warning: TensorFlow not available, CNN model for {symbol} {timeframe} will not function")
    
    def train(self, data: pd.DataFrame) -> bool:
        """Train enhanced CNN model with multi-scale feature extraction."""
        if not TENSORFLOW_AVAILABLE:
            print("Cannot train CNN model: TensorFlow not available")
            return False
            
        try:
            print(f"Training enhanced CNN model for {self.symbol} timeframe {self.timeframe}...")
            print(f"Configuration: sequence_length={self.sequence_length}, filters={self.filter_sizes}")
            
            # Prepare data with multiple features
            feature_columns = ['open', 'high', 'low', 'close']
            
            # Add volume if available
            if 'volume' in data.columns:
                feature_columns.append('volume')
            
            # Prepare target (classification: price will go up or down)
            df = data.copy()
            df['future_close'] = df['close'].shift(-5)  # 5 periods ahead
            df.dropna(inplace=True)
            df['target'] = (df['future_close'] > df['close']).astype(int)
            
            # Normalize features
            feature_data = df[feature_columns].values
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(feature_data)
            
            # Create sequences for CNN
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i])
                y.append(df['target'].iloc[i])
            
            X, y = np.array(X), np.array(y)
            
            if len(X) < 50:
                print(f"Insufficient data for CNN training: {len(X)} sequences")
                return False
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build enhanced CNN model
            tf.keras.backend.clear_session()
            
            model = Sequential()
            
            # First convolutional block with multiple filter sizes
            model.add(Conv1D(
                filters=self.filter_sizes[0], 
                kernel_size=self.kernel_sizes[0], 
                activation='relu',
                input_shape=(X.shape[1], X.shape[2])
            ))
            if self.use_batch_norm:
                model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=self.pool_size))
            model.add(Dropout(self.dropout_rate))
            
            # Second convolutional block
            model.add(Conv1D(
                filters=self.filter_sizes[1], 
                kernel_size=self.kernel_sizes[1], 
                activation='relu'
            ))
            if self.use_batch_norm:
                model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=self.pool_size))
            model.add(Dropout(self.dropout_rate))
            
            # Third convolutional block
            model.add(Conv1D(
                filters=self.filter_sizes[2], 
                kernel_size=self.kernel_sizes[2], 
                activation='relu'
            ))
            if self.use_batch_norm:
                model.add(BatchNormalization())
            
            # Global pooling for better feature extraction
            if self.use_global_pooling:
                model.add(GlobalMaxPooling1D())
            else:
                model.add(Flatten())
            
            model.add(Dropout(self.dropout_rate))
            
            # Dense layers
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(self.dropout_rate))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(self.dropout_rate))
            model.add(Dense(1, activation='sigmoid'))
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            
            # Setup callbacks
            callbacks = []
            
            if self.use_early_stopping:
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=1
                )
                callbacks.append(early_stopping)
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.patience // 2,
                min_lr=0.0001,
                verbose=1
            )
            callbacks.append(reduce_lr)
            
            print(f"Starting CNN training with {len(X_train)} training samples, {len(X_val)} validation samples...")
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Print training results
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            final_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            
            print(f"Enhanced CNN training completed.")
            print(f"Final - Loss: {final_loss:.4f}, Acc: {final_acc:.4f}")
            print(f"Final - Val Loss: {final_val_loss:.4f}, Val Acc: {final_val_acc:.4f}")
            
            self._trained = True
            return True
            
        except Exception as e:
            print(f"Error training CNN model: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with enhanced CNN model."""
        if not TENSORFLOW_AVAILABLE:
            return self._unavailable_prediction("TensorFlow not available")
            
        if not self._trained or self.model is None or self.scaler is None:
            return self._default_prediction()
        
        try:
            # Prepare features same as training
            feature_columns = ['open', 'high', 'low', 'close']
            
            if 'volume' in data.columns:
                feature_columns.append('volume')
            
            # Get last sequence for prediction
            if len(data) < self.sequence_length:
                return self._default_prediction()
            
            last_sequence = data[feature_columns].values[-self.sequence_length:]
            scaled_seq = self.scaler.transform(last_sequence)
            X = np.reshape(scaled_seq, (1, self.sequence_length, len(feature_columns)))
            
            pred_proba = self.model.predict(X, verbose=0)[0][0]
            
            # Convert to direction and confidence
            if pred_proba > 0.5:
                direction = "BUY"
                confidence = pred_proba
            else:
                direction = "SELL"
                confidence = 1 - pred_proba
            
            return {
                'direction': direction,
                'confidence': confidence,
                'probability': pred_proba,
                'model_name': 'CNN',
                'timestamp': datetime.now(),
                'features_used': feature_columns,
                'sequence_length': self.sequence_length,
                'enhanced': True
            }
            
        except Exception as e:
            print(f"Error making CNN prediction: {e}")
            return self._default_prediction()
    
    def save(self) -> bool:
        """Save enhanced CNN model, scaler, and configuration."""
        if not TENSORFLOW_AVAILABLE:
            print("Cannot save CNN model: TensorFlow not available")
            return False
            
        try:
            os.makedirs('model', exist_ok=True)
            
            # Save model
            model_path = self.get_model_path('.keras')
            self.model.save(model_path)
            
            # Save scaler
            scaler_path = self.get_model_path('_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            
            # Save configuration
            config_path = self.get_model_path('_config.json')
            config = {
                'sequence_length': self.sequence_length,
                'dropout_rate': self.dropout_rate,
                'use_batch_norm': self.use_batch_norm,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'filter_sizes': self.filter_sizes,
                'kernel_sizes': self.kernel_sizes,
                'pool_size': self.pool_size,
                'use_global_pooling': self.use_global_pooling,
                'use_early_stopping': self.use_early_stopping,
                'patience': self.patience,
                'symbol': self.symbol,
                'timeframe': self.timeframe
            }
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            print(f"Enhanced CNN model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving CNN model: {e}")
            return False
    
    def load(self) -> bool:
        """Load enhanced CNN model, scaler, and configuration."""
        if not TENSORFLOW_AVAILABLE:
            print("Cannot load CNN model: TensorFlow not available")
            return False
            
        try:
            model_path = self.get_model_path('.keras')
            scaler_path = self.get_model_path('_scaler.pkl')
            config_path = self.get_model_path('_config.json')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                
                # Load configuration if available
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        self.sequence_length = config.get('sequence_length', 30)
                        self.dropout_rate = config.get('dropout_rate', 0.4)
                        self.use_batch_norm = config.get('use_batch_norm', True)
                        self.epochs = config.get('epochs', 30)
                        self.batch_size = config.get('batch_size', 32)
                        self.filter_sizes = config.get('filter_sizes', [32, 64, 128])
                        self.kernel_sizes = config.get('kernel_sizes', [3, 5, 7])
                        self.pool_size = config.get('pool_size', 2)
                        self.use_global_pooling = config.get('use_global_pooling', True)
                        self.use_early_stopping = config.get('use_early_stopping', True)
                        self.patience = config.get('patience', 10)
                
                self._trained = True
                print(f"Enhanced CNN model loaded from {model_path}")
                return True
            else:
                print(f"CNN model files not found: {model_path}, {scaler_path}")
                return False
                
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return enhanced CNN model metadata."""
        return {
            'name': 'CNN',
            'type': 'convolutional_neural_network',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'sequence_length': self.sequence_length,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'filter_sizes': self.filter_sizes,
            'kernel_sizes': self.kernel_sizes,
            'use_global_pooling': self.use_global_pooling,
            'trained': self._trained,
            'features': ['open', 'high', 'low', 'close', 'volume'],
            'model_path': self.get_model_path('.keras'),
            'available': TENSORFLOW_AVAILABLE,
            'enhanced': True
        }
    
    def _unavailable_prediction(self, reason: str) -> Dict[str, Any]:
        """Return prediction when dependencies unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'CNN',
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
            'model_name': 'CNN',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': 'Model not trained or unavailable'
        }


class SVCModel(BaseModel):
    """Support Vector Classifier model with hyperparameter optimization."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.scaler = None
        self.feature_columns = None
        self.hyperparameter_tuning = True  # Enable tuning by default
        
    def train(self, data: pd.DataFrame) -> bool:
        """Train SVC model with provided data."""
        try:
            print(f"Training SVC model for {self.symbol} timeframe {self.timeframe}...")
            
            # Prepare data for classification
            df = data.copy()
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df.dropna(inplace=True)
            
            if len(df) < 50:
                print(f"Insufficient data for SVC training: {len(df)} rows")
                return False
            
            # Define feature columns
            base_features = ['open', 'high', 'low', 'close', 'volume']
            technical_features = ['rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 'ATR_14', 'STOCHk_14_3_3']
            
            # Add basic technical indicators if missing
            if 'rsi' not in df.columns:
                df['rsi'] = 50.0  # Neutral RSI
            if 'MACD_12_26_9' not in df.columns:
                df['MACD_12_26_9'] = 0.0  # Neutral MACD
            if 'EMA_10' not in df.columns:
                df['EMA_10'] = df['close']  # Use close price as fallback
            if 'EMA_20' not in df.columns:
                df['EMA_20'] = df['close']  # Use close price as fallback
            if 'ATR_14' not in df.columns:
                df['ATR_14'] = abs(df['high'] - df['low'])  # Simple ATR approximation
            if 'STOCHk_14_3_3' not in df.columns:
                df['STOCHk_14_3_3'] = 50.0  # Neutral Stochastic
            
            all_features = base_features + technical_features
            self.feature_columns = [f for f in all_features if f in df.columns]
            
            X = df[self.feature_columns].fillna(0)
            y = df['target']
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            if self.hyperparameter_tuning:
                # Hyperparameter optimization
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'poly', 'linear']
                }
                
                print("Starting hyperparameter tuning for SVC...")
                self.model = GridSearchCV(
                    SVC(probability=True, random_state=42),
                    param_grid, cv=3, scoring='accuracy', n_jobs=-1
                )
                self.model.fit(X_train, y_train)
                print(f"Best SVC parameters: {self.model.best_params_}")
                
                # Get the best model
                best_model = self.model.best_estimator_
            else:
                # Use default parameters optimized for financial data
                self.model = SVC(
                    kernel='rbf', 
                    C=10, 
                    gamma='scale', 
                    probability=True, 
                    random_state=42,
                    class_weight='balanced'  # Handle imbalanced data
                )
                self.model.fit(X_train, y_train)
                best_model = self.model
            
            # Calculate accuracy
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"SVC training completed. Accuracy: {accuracy:.4f}")
            
            self._trained = True
            return True
            
        except Exception as e:
            print(f"Error training SVC model: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with SVC model."""
        if not self._trained or self.model is None or self.scaler is None:
            return self._default_prediction()
        
        try:
            # Prepare latest data for prediction
            latest_df = data.copy()
            
            # Add missing indicators with defaults
            if 'rsi' not in latest_df.columns:
                latest_df['rsi'] = 50.0
            if 'MACD_12_26_9' not in latest_df.columns:
                latest_df['MACD_12_26_9'] = 0.0
            if 'EMA_10' not in latest_df.columns:
                latest_df['EMA_10'] = latest_df['close']
            if 'EMA_20' not in latest_df.columns:
                latest_df['EMA_20'] = latest_df['close']
            if 'ATR_14' not in latest_df.columns:
                latest_df['ATR_14'] = abs(latest_df['high'] - latest_df['low'])
            if 'STOCHk_14_3_3' not in latest_df.columns:
                latest_df['STOCHk_14_3_3'] = 50.0
            
            X = latest_df[self.feature_columns].tail(1).fillna(0)
            
            if X.empty:
                return self._default_prediction()
            
            X_scaled = self.scaler.transform(X)
            
            # Get the actual model (handle GridSearchCV vs direct model)
            actual_model = self.model.best_estimator_ if hasattr(self.model, 'best_estimator_') else self.model
            
            pred = actual_model.predict(X_scaled)[0]
            prob = actual_model.predict_proba(X_scaled)[0].max()
            
            direction = "BUY" if pred == 1 else "SELL"
            
            return {
                'direction': direction,
                'confidence': prob,
                'probability': prob,
                'model_name': 'SVC',
                'timestamp': datetime.now(),
                'features_used': self.feature_columns,
                'prediction': int(pred)
            }
            
        except Exception as e:
            print(f"Error making SVC prediction: {e}")
            return self._default_prediction()
    
    def save(self) -> bool:
        """Save SVC model and scaler."""
        try:
            os.makedirs('model', exist_ok=True)
            model_path = self.get_model_path('.pkl')
            scaler_path = self.get_model_path('_scaler.pkl')
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            # Save metadata
            metadata = {
                'feature_columns': self.feature_columns,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'hyperparameter_tuning': self.hyperparameter_tuning
            }
            metadata_path = self.get_model_path('_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            print(f"SVC model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving SVC model: {e}")
            return False
    
    def load(self) -> bool:
        """Load SVC model and scaler."""
        try:
            model_path = self.get_model_path('.pkl')
            scaler_path = self.get_model_path('_scaler.pkl')
            metadata_path = self.get_model_path('_metadata.json')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                # Load metadata if available
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.feature_columns = metadata.get('feature_columns')
                        self.hyperparameter_tuning = metadata.get('hyperparameter_tuning', True)
                else:
                    # Default feature columns for backward compatibility
                    self.feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'MACD_12_26_9']
                
                self._trained = True
                print(f"SVC model loaded from {model_path}")
                return True
            else:
                print(f"SVC model files not found: {model_path}, {scaler_path}")
                return False
                
        except Exception as e:
            print(f"Error loading SVC model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return SVC model metadata."""
        return {
            'name': 'SVC',
            'type': 'support_vector_machine',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'trained': self._trained,
            'features': self.feature_columns,
            'model_path': self.get_model_path('.pkl'),
            'hyperparameter_tuning': self.hyperparameter_tuning
        }
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when model unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'SVC',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': 'Model not trained or unavailable'
        }


class NaiveBayesModel(BaseModel):
    """Gaussian Naive Bayes model for classification."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.scaler = None
        self.feature_columns = None
        
    def train(self, data: pd.DataFrame) -> bool:
        """Train Naive Bayes model with provided data."""
        try:
            print(f"Training Naive Bayes model for {self.symbol} timeframe {self.timeframe}...")
            
            # Prepare data for classification
            df = data.copy()
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df.dropna(inplace=True)
            
            if len(df) < 30:
                print(f"Insufficient data for Naive Bayes training: {len(df)} rows")
                return False
            
            # Define feature columns (Naive Bayes works well with fewer features)
            base_features = ['open', 'high', 'low', 'close', 'volume']
            
            # Add basic technical indicators if missing
            if 'rsi' not in df.columns:
                df['rsi'] = 50.0
            if 'MACD_12_26_9' not in df.columns:
                df['MACD_12_26_9'] = 0.0
            
            available_features = [f for f in base_features + ['rsi', 'MACD_12_26_9'] if f in df.columns]
            self.feature_columns = available_features
            
            X = df[self.feature_columns].fillna(0)
            y = df['target']
            
            # Scale features for better performance
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.model = GaussianNB()
            self.model.fit(X_train, y_train)
            
            # Calculate accuracy
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Naive Bayes training completed. Accuracy: {accuracy:.4f}")
            
            self._trained = True
            return True
            
        except Exception as e:
            print(f"Error training Naive Bayes model: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with Naive Bayes model."""
        if not self._trained or self.model is None or self.scaler is None:
            return self._default_prediction()
        
        try:
            # Prepare latest data for prediction
            latest_df = data.copy()
            
            # Add missing indicators with defaults
            if 'rsi' not in latest_df.columns:
                latest_df['rsi'] = 50.0
            if 'MACD_12_26_9' not in latest_df.columns:
                latest_df['MACD_12_26_9'] = 0.0
            
            X = latest_df[self.feature_columns].tail(1).fillna(0)
            
            if X.empty:
                return self._default_prediction()
            
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)[0]
            prob = self.model.predict_proba(X_scaled)[0].max()
            
            direction = "BUY" if pred == 1 else "SELL"
            
            return {
                'direction': direction,
                'confidence': prob,
                'probability': prob,
                'model_name': 'NaiveBayes',
                'timestamp': datetime.now(),
                'features_used': self.feature_columns,
                'prediction': int(pred)
            }
            
        except Exception as e:
            print(f"Error making Naive Bayes prediction: {e}")
            return self._default_prediction()
    
    def save(self) -> bool:
        """Save Naive Bayes model and scaler."""
        try:
            os.makedirs('model', exist_ok=True)
            model_path = self.get_model_path('.pkl')
            scaler_path = self.get_model_path('_scaler.pkl')
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            # Save metadata
            metadata = {
                'feature_columns': self.feature_columns,
                'symbol': self.symbol,
                'timeframe': self.timeframe
            }
            metadata_path = self.get_model_path('_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            print(f"Naive Bayes model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving Naive Bayes model: {e}")
            return False
    
    def load(self) -> bool:
        """Load Naive Bayes model and scaler."""
        try:
            model_path = self.get_model_path('.pkl')
            scaler_path = self.get_model_path('_scaler.pkl')
            metadata_path = self.get_model_path('_metadata.json')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                # Load metadata if available
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.feature_columns = metadata.get('feature_columns')
                else:
                    # Default feature columns for backward compatibility
                    self.feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'MACD_12_26_9']
                
                self._trained = True
                print(f"Naive Bayes model loaded from {model_path}")
                return True
            else:
                print(f"Naive Bayes model files not found: {model_path}, {scaler_path}")
                return False
                
        except Exception as e:
            print(f"Error loading Naive Bayes model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return Naive Bayes model metadata."""
        return {
            'name': 'NaiveBayes',
            'type': 'naive_bayes',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'trained': self._trained,
            'features': self.feature_columns,
            'model_path': self.get_model_path('.pkl')
        }
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when model unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'NaiveBayes',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': 'Model not trained or unavailable'
        }