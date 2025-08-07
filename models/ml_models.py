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
    from tensorflow.keras.layers import Dense, LSTM
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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from .base import BaseModel


class LSTMModel(BaseModel):
    """LSTM neural network model migrated from utils/lstm_model.py"""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.scaler = None
        self.sequence_length = 60
        if not TENSORFLOW_AVAILABLE:
            print(f"Warning: TensorFlow not available, LSTM model for {symbol} {timeframe} will not function")
        
    def train(self, data: pd.DataFrame) -> bool:
        """Train LSTM model with enhanced architecture and configurable parameters."""
        if not TENSORFLOW_AVAILABLE:
            print("Cannot train LSTM model: TensorFlow not available")
            return False
            
        try:
            print(f"Training enhanced LSTM model for {self.symbol} timeframe {self.timeframe}...")
            
            # Prepare data (minimal change from original)
            close_data = data[['close']].values
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(close_data)
            
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Clear session and build enhanced LSTM model
            tf.keras.backend.clear_session()
            
            # Enhanced LSTM architecture with dropout and batch normalization
            from tensorflow.keras.layers import Dropout, BatchNormalization
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            self.model = Sequential([
                # First LSTM layer with return_sequences=True for stacking
                LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.3),
                BatchNormalization(),
                
                # Second LSTM layer
                LSTM(units=100, return_sequences=True),
                Dropout(0.4),
                BatchNormalization(),
                
                # Third LSTM layer
                LSTM(units=50, return_sequences=False),
                Dropout(0.5),
                
                # Dense layers with regularization
                Dense(50, activation='relu'),
                Dropout(0.3),
                Dense(25, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            # Compile with optimized settings
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            # Enhanced training with callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001,
                verbose=1
            )
            
            # Train with enhanced parameters (50-200 epochs with early stopping)
            print(f"Training enhanced LSTM model with {len(X)} samples...")
            history = self.model.fit(
                X, y, 
                epochs=100,  # Increased from 5 to 100 with early stopping
                batch_size=32, 
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Get training metrics
            final_loss = min(history.history['val_loss']) if 'val_loss' in history.history else history.history['loss'][-1]
            epochs_trained = len(history.history['loss'])
            
            print(f"Enhanced LSTM training completed after {epochs_trained} epochs.")
            print(f"Final validation loss: {final_loss:.6f}")
            
            self._trained = True
            return True
            
        except Exception as e:
            print(f"Error training enhanced LSTM model: {e}")
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
    """XGBoost gradient boosting model migrated from utils/xgb_model.py"""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.feature_columns = None
        if not XGBOOST_AVAILABLE:
            print(f"Warning: XGBoost not available, XGBoost model for {symbol} {timeframe} will not function")
        
    def train(self, data: pd.DataFrame) -> bool:
        """Train XGBoost model with optimized hyperparameters."""
        if not XGBOOST_AVAILABLE:
            print("Cannot train XGBoost model: XGBoost not available")
            return False
            
        try:
            print(f"Training optimized XGBoost model for {self.symbol} timeframe {self.timeframe}...")
            
            # Prepare data for classification (minimal change from original)
            df = data.copy()
            df['future_close'] = df['close'].shift(-5)  # 5 periods ahead
            df.dropna(inplace=True)
            df['target'] = (df['future_close'] > df['close']).astype(int)
            
            # Remove unwanted columns and prepare features
            features = df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'future_close', 'target', 'ema_signal'], errors='ignore')
            X = features
            y = df['target']
            
            if X.empty or y.empty or len(X.columns) == 0:
                print("Not enough data or features for training XGBoost model.")
                return False
            
            self.feature_columns = list(X.columns)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Enhanced XGBoost parameters optimized for financial data
            optimized_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss', 
                'use_label_encoder': False,
                'n_estimators': 300,  # Increased from 100 to 300
                'learning_rate': 0.05,  # Reduced from 0.1 to 0.05 for better convergence
                'max_depth': 6,  # Increased from 3 to 6 for better feature extraction
                'subsample': 0.8,  # Added for regularization
                'colsample_bytree': 0.8,  # Added for regularization
                'reg_alpha': 0.1,  # L1 regularization
                'reg_lambda': 1.0,  # L2 regularization
                'min_child_weight': 3,  # Increased for regularization
                'gamma': 0.1,  # Added for regularization
                'random_state': 42,
                'n_jobs': -1  # Use all cores
            }
            
            self.model = xgb.XGBClassifier(**optimized_params)
            
            print(f"Training XGBoost with optimized parameters...")
            print(f"Parameters: n_estimators={optimized_params['n_estimators']}, "
                  f"learning_rate={optimized_params['learning_rate']}, "
                  f"max_depth={optimized_params['max_depth']}")
            
            # Train with early stopping (using different approach for newer XGBoost)
            eval_set = [(X_test, y_test)]
            try:
                # Try newer XGBoost API first
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
            except TypeError:
                # Fallback for older XGBoost versions
                self.model.fit(X_train, y_train, verbose=False)
            
            # Calculate accuracy
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Optimized XGBoost training completed. Accuracy: {accuracy:.4f}")
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                top_features = sorted(zip(self.feature_columns, importance), key=lambda x: x[1], reverse=True)[:5]
                print(f"Top 5 features: {[f'{name}: {imp:.4f}' for name, imp in top_features]}")
            
            self._trained = True
            return True
            
        except Exception as e:
            print(f"Error training optimized XGBoost model: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with XGBoost model."""
        if not XGBOOST_AVAILABLE:
            return self._unavailable_prediction("XGBoost not available")
            
        if not self._trained or self.model is None:
            return self._default_prediction()
        
        try:
            # Prepare prediction features (minimal change from original)
            training_features = self.model.get_booster().feature_names
            prediction_features = data[training_features].tail(1)
            
            prediction = self.model.predict(prediction_features)
            prediction_proba = self.model.predict_proba(prediction_features)
            
            final_prediction = int(prediction[-1])
            final_proba = prediction_proba[-1]
            
            direction = "BUY" if final_prediction == 1 else "SELL"
            confidence = float(max(final_proba))
            
            return {
                'direction': direction,
                'confidence': confidence,
                'probability': confidence,
                'model_name': 'XGBoost',
                'timestamp': datetime.now(),
                'features_used': training_features,
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


class SVCModel(BaseModel):
    """Support Vector Classifier model with hyperparameter optimization."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def train(self, data: pd.DataFrame) -> bool:
        """Train SVC model with provided data."""
        try:
            print(f"Training SVC model for {self.symbol} timeframe {self.timeframe}...")
            
            # Prepare features
            df = data.copy()
            
            # Add basic technical indicators if they don't exist
            if 'rsi' not in df.columns:
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
            
            if X.empty or y.empty:
                print("Not enough data for SVC training.")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Optimized SVC parameters for financial data
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly']
            }
            
            # Use GridSearchCV for hyperparameter optimization
            svc_base = SVC(probability=True, random_state=42)
            self.model = GridSearchCV(svc_base, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            self.model.fit(X_train, y_train)
            
            # Calculate accuracy
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"SVC training completed. Best params: {self.model.best_params_}")
            print(f"SVC accuracy: {accuracy:.4f}")
            
            self._trained = True
            return True
            
        except Exception as e:
            print(f"Error training SVC model: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with SVC model."""
        if not self._trained or self.model is None:
            return self._default_prediction()
        
        try:
            # Prepare latest data for prediction
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
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            pred = self.model.predict(X_scaled)[0]
            prob = self.model.predict_proba(X_scaled)[0].max()
            
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
        """Save SVC model."""
        try:
            os.makedirs('model', exist_ok=True)
            model_path = self.get_model_path('.pkl')
            joblib.dump(self.model, model_path)
            
            # Save scaler and feature columns
            scaler_path = self.get_model_path('_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            
            metadata = {
                'feature_columns': self.feature_columns,
                'symbol': self.symbol,
                'timeframe': self.timeframe
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
        """Load SVC model."""
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
                    self.feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'MACDh_12_26_9', 'ADX_14']
                
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
            'available': True
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
    """Gaussian Naive Bayes model for financial prediction."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.feature_columns = None
        
    def train(self, data: pd.DataFrame) -> bool:
        """Train Naive Bayes model with provided data."""
        try:
            print(f"Training Naive Bayes model for {self.symbol} timeframe {self.timeframe}...")
            
            # Prepare features
            df = data.copy()
            
            # Add basic technical indicators if they don't exist
            if 'rsi' not in df.columns:
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
            
            if X.empty or y.empty:
                print("Not enough data for Naive Bayes training.")
                return False
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
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
        if not self._trained or self.model is None:
            return self._default_prediction()
        
        try:
            # Prepare latest data for prediction
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
                'model_name': 'NaiveBayes',
                'timestamp': datetime.now(),
                'features_used': self.feature_columns,
                'prediction': int(pred)
            }
            
        except Exception as e:
            print(f"Error making Naive Bayes prediction: {e}")
            return self._default_prediction()
    
    def save(self) -> bool:
        """Save Naive Bayes model."""
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
            
            print(f"Naive Bayes model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving Naive Bayes model: {e}")
            return False
    
    def load(self) -> bool:
        """Load Naive Bayes model."""
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
                print(f"Naive Bayes model loaded from {model_path}")
                return True
            else:
                print(f"Naive Bayes model file not found: {model_path}")
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
            'model_path': self.get_model_path('.pkl'),
            'available': True
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


class CNNModel(BaseModel):
    """Convolutional Neural Network model with enhanced architecture."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.scaler = None
        self.sequence_length = 30
        if not TENSORFLOW_AVAILABLE:
            print(f"Warning: TensorFlow not available, CNN model for {symbol} {timeframe} will not function")
        
    def train(self, data: pd.DataFrame) -> bool:
        """Train CNN model with enhanced architecture."""
        if not TENSORFLOW_AVAILABLE:
            print("Cannot train CNN model: TensorFlow not available")
            return False
            
        try:
            print(f"Training CNN model for {self.symbol} timeframe {self.timeframe}...")
            
            # Prepare data for CNN
            df = data.copy()
            
            # Create future target
            df['future_close'] = df['close'].shift(-5)  # 5 periods ahead
            df.dropna(inplace=True)
            df['target'] = (df['future_close'] > df['close']).astype(int)
            
            # Normalize price data
            price_cols = ['open', 'high', 'low', 'close']
            self.scaler = MinMaxScaler()
            df[price_cols] = self.scaler.fit_transform(df[price_cols])
            
            # Create sequences
            X, y = [], []
            for i in range(len(df) - self.sequence_length):
                seq = df.iloc[i:i+self.sequence_length][price_cols].values
                label = df.iloc[i+self.sequence_length]['target']
                X.append(seq)
                y.append(label)
            
            X, y = np.array(X), np.array(y)
            
            if len(X) == 0:
                print("Not enough data for CNN training.")
                return False
            
            # Clear session and build enhanced CNN model
            tf.keras.backend.clear_session()
            
            # Enhanced CNN architecture with multiple Conv1D layers
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, BatchNormalization
            
            self.model = Sequential([
                # First convolutional block
                Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=X.shape[1:]),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Dropout(0.3),
                
                # Second convolutional block with different kernel size
                Conv1D(filters=64, kernel_size=5, activation='relu'),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Dropout(0.3),
                
                # Third convolutional block
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                
                # Global pooling and dense layers
                GlobalMaxPooling1D(),
                Dense(100, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(50, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            
            # Compile with optimized settings
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Add early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train with validation split and early stopping
            print(f"Training enhanced CNN model with {len(X)} samples...")
            history = self.model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            final_accuracy = max(history.history['val_accuracy']) if 'val_accuracy' in history.history else 0.0
            print(f"CNN training completed. Best validation accuracy: {final_accuracy:.4f}")
            
            self._trained = True
            return True
            
        except Exception as e:
            print(f"Error training CNN model: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with CNN model."""
        if not TENSORFLOW_AVAILABLE:
            return self._unavailable_prediction("TensorFlow not available")
            
        if not self._trained or self.model is None or self.scaler is None:
            return self._default_prediction()
        
        try:
            # Prepare latest sequence
            latest_data = data.copy()
            
            if len(latest_data) < self.sequence_length:
                return self._default_prediction()
            
            # Get last sequence and normalize
            price_cols = ['open', 'high', 'low', 'close']
            data_seq = latest_data[price_cols].tail(self.sequence_length).copy()
            data_seq_scaled = self.scaler.transform(data_seq)
            
            # Reshape for CNN
            X_pred = np.reshape(data_seq_scaled, (1, self.sequence_length, len(price_cols)))
            
            # Make prediction
            prediction_proba = self.model.predict(X_pred, verbose=0)[0][0]
            
            # Convert to direction and confidence
            direction = "BUY" if prediction_proba > 0.5 else "SELL"
            confidence = prediction_proba if direction == "BUY" else 1 - prediction_proba
            
            return {
                'direction': direction,
                'confidence': confidence,
                'probability': prediction_proba,
                'model_name': 'CNN',
                'timestamp': datetime.now(),
                'features_used': price_cols,
                'predicted_probability': float(prediction_proba)
            }
            
        except Exception as e:
            print(f"Error making CNN prediction: {e}")
            return self._default_prediction()
    
    def save(self) -> bool:
        """Save CNN model and scaler."""
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
            
            # Save metadata
            metadata = {
                'sequence_length': self.sequence_length,
                'symbol': self.symbol,
                'timeframe': self.timeframe
            }
            metadata_path = self.get_model_path('_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            print(f"CNN model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving CNN model: {e}")
            return False
    
    def load(self) -> bool:
        """Load CNN model and scaler."""
        if not TENSORFLOW_AVAILABLE:
            print("Cannot load CNN model: TensorFlow not available")
            return False
            
        try:
            model_path = self.get_model_path('.keras')
            scaler_path = self.get_model_path('_scaler.pkl')
            metadata_path = self.get_model_path('_metadata.json')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                # Import load_model here to avoid circular import
                from tensorflow.keras.models import load_model
                
                self.model = load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                
                # Load metadata if available
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.sequence_length = metadata.get('sequence_length', 30)
                
                self._trained = True
                print(f"CNN model loaded from {model_path}")
                return True
            else:
                print(f"CNN model files not found: {model_path}, {scaler_path}")
                return False
                
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return CNN model metadata."""
        return {
            'name': 'CNN',
            'type': 'convolutional_neural_network',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'sequence_length': self.sequence_length,
            'trained': self._trained,
            'features': ['open', 'high', 'low', 'close'],
            'model_path': self.get_model_path('.keras'),
            'available': TENSORFLOW_AVAILABLE
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