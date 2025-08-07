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
from typing import Dict, Any, List, Tuple

# Import dependency manager for robust dependency handling
try:
    from utils.dependency_manager import dependency_manager, is_available
    DEPENDENCY_MANAGER_AVAILABLE = True
except ImportError:
    # Fallback for systems without dependency manager
    DEPENDENCY_MANAGER_AVAILABLE = False
    print("Warning: Dependency manager not available, using legacy imports")

# Import technical indicators
try:
    from utils.indicators import add_indicators
    INDICATORS_AVAILABLE = True
except ImportError:
    INDICATORS_AVAILABLE = False
    print("Warning: Technical indicators utility not available")

# Import dependencies with centralized checking
if DEPENDENCY_MANAGER_AVAILABLE:
    TENSORFLOW_AVAILABLE = is_available('tensorflow')
    LIGHTGBM_AVAILABLE = is_available('lightgbm')
    XGBOOST_AVAILABLE = is_available('xgboost')
    
    if TENSORFLOW_AVAILABLE:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential, Model, load_model
        from tensorflow.keras.layers import (Dense, LSTM, Dropout, BatchNormalization, 
                                           Input, MultiHeadAttention, LayerNormalization,
                                           Add, GlobalAveragePooling1D, Concatenate)
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
    
    if LIGHTGBM_AVAILABLE:
        import lightgbm as lgb
    
    if XGBOOST_AVAILABLE:
        import xgboost as xgb
        
else:
    # Legacy import handling with try/except blocks
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential, Model, load_model
        from tensorflow.keras.layers import (Dense, LSTM, Dropout, BatchNormalization, 
                                           Input, MultiHeadAttention, LayerNormalization,
                                           Add, GlobalAveragePooling1D, Concatenate)
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False
        print("Warning: TensorFlow not available, LSTM model will be disabled")

    try:
        import lightgbm as lgb
        LIGHTGBM_AVAILABLE = True
    except ImportError:
        LIGHTGBM_AVAILABLE = False
        print("Warning: LightGBM not available, LightGBM model will be disabled")

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


class AdvancedLSTMModel(BaseModel):
    """Enhanced LSTM with attention mechanism for advanced predictions."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.scaler = None
        self.feature_scaler = None
        self.sequence_length = 60
        self.use_attention = True
        self.use_multivariate = True
        if not TENSORFLOW_AVAILABLE:
            print(f"Warning: TensorFlow not available, Advanced LSTM model for {symbol} {timeframe} will not function")
    
    def build_attention_lstm_model(self, input_shape, feature_dim=None):
        """Build advanced LSTM model with attention mechanism."""
        # Main sequence input (price data)
        sequence_input = Input(shape=input_shape, name='sequence_input')
        
        # LSTM layers with return_sequences=True for attention
        lstm1 = LSTM(128, return_sequences=True, dropout=0.2, name='lstm1')(sequence_input)
        lstm1_norm = BatchNormalization(name='lstm1_norm')(lstm1)
        
        # Self-attention layer
        if self.use_attention:
            attention = MultiHeadAttention(
                num_heads=8, 
                key_dim=16, 
                dropout=0.1, 
                name='self_attention'
            )(lstm1_norm, lstm1_norm)
            
            # Add & Norm layer (residual connection)
            attention_add = Add(name='attention_add')([lstm1_norm, attention])
            attention_norm = LayerNormalization(name='attention_norm')(attention_add)
        else:
            attention_norm = lstm1_norm
        
        # Second LSTM layer
        lstm2 = LSTM(64, return_sequences=True, dropout=0.2, name='lstm2')(attention_norm)
        lstm2_norm = BatchNormalization(name='lstm2_norm')(lstm2)
        
        # Third LSTM layer (final)
        lstm3 = LSTM(32, dropout=0.2, name='lstm3')(lstm2_norm)
        lstm3_norm = BatchNormalization(name='lstm3_norm')(lstm3)
        
        # Feature input (if using multivariate features)
        if feature_dim and self.use_multivariate:
            feature_input = Input(shape=(feature_dim,), name='feature_input')
            feature_dense = Dense(16, activation='relu', name='feature_dense')(feature_input)
            feature_norm = BatchNormalization(name='feature_norm')(feature_dense)
            
            # Combine LSTM output with features
            combined = Concatenate(name='combine_features')([lstm3_norm, feature_norm])
            inputs = [sequence_input, feature_input]
        else:
            combined = lstm3_norm
            inputs = sequence_input
        
        # Dense layers for final prediction
        dense1 = Dense(16, activation='relu', name='dense1')(combined)
        dropout1 = Dropout(0.3, name='dropout1')(dense1)
        
        # Multi-class output for BUY/HOLD/SELL
        output = Dense(3, activation='softmax', name='predictions')(dropout1)
        
        # Create model
        model = Model(inputs=inputs, outputs=output, name='AdvancedLSTM')
        
        return model
        
    def prepare_lstm_data(self, data: pd.DataFrame) -> Tuple:
        """Prepare data for LSTM training with enhanced features."""
        from .advanced_features import AdvancedDataPipeline
        from .advanced_indicators import EnhancedIndicators
        
        # Apply advanced feature engineering
        feature_pipeline = AdvancedDataPipeline(self.timeframe.replace('m', 'min'))
        enhanced_data = feature_pipeline.transform(data.copy())
        
        # Apply enhanced indicators
        indicator_calculator = EnhancedIndicators()
        enhanced_data = indicator_calculator.add_all_indicators(enhanced_data)
        
        # Prepare target variable (multi-class: 0=SELL, 1=HOLD, 2=BUY)
        future_returns = enhanced_data['close'].pct_change(periods=5).shift(-5)  # 5-period ahead returns
        
        # Define thresholds for classification
        buy_threshold = 0.002   # 0.2% gain
        sell_threshold = -0.002 # 0.2% loss
        
        targets = pd.Series(index=enhanced_data.index, data=1)  # Default to HOLD
        targets[future_returns > buy_threshold] = 2   # BUY
        targets[future_returns < sell_threshold] = 0  # SELL
        
        # Select features for LSTM sequence
        sequence_features = ['open', 'high', 'low', 'close', 'volume']
        if 'volume' not in enhanced_data.columns:
            sequence_features = ['open', 'high', 'low', 'close']
        
        # Additional features (non-sequence) for multivariate model
        feature_columns = [col for col in enhanced_data.columns 
                          if col not in sequence_features + ['datetime'] 
                          and enhanced_data[col].dtype in ['float64', 'int64']]
        
        # Scale sequence data
        sequence_data = enhanced_data[sequence_features].values
        self.scaler = MinMaxScaler()
        scaled_sequence = self.scaler.fit_transform(sequence_data)
        
        # Scale additional features
        if feature_columns and self.use_multivariate:
            feature_data = enhanced_data[feature_columns].fillna(0).values
            self.feature_scaler = MinMaxScaler()
            scaled_features = self.feature_scaler.fit_transform(feature_data)
        else:
            scaled_features = None
        
        # Create sequences for LSTM
        X_sequences, X_features, y = [], [], []
        
        for i in range(self.sequence_length, len(scaled_sequence) - 5):
            if not np.isnan(targets.iloc[i]):
                X_sequences.append(scaled_sequence[i-self.sequence_length:i])
                if scaled_features is not None:
                    X_features.append(scaled_features[i])
                y.append(targets.iloc[i])
        
        X_sequences = np.array(X_sequences)
        X_features = np.array(X_features) if scaled_features is not None else None
        y = np.array(y)
        
        return X_sequences, X_features, y
    
    def train(self, data: pd.DataFrame) -> bool:
        """Train advanced LSTM model with enhanced features."""
        if not TENSORFLOW_AVAILABLE:
            print("Cannot train Advanced LSTM model: TensorFlow not available")
            return False
            
        try:
            print(f"Training Advanced LSTM model for {self.symbol} timeframe {self.timeframe}...")
            
            # Prepare enhanced data
            X_sequences, X_features, y = self.prepare_lstm_data(data)
            
            if len(X_sequences) < 100:
                print(f"Insufficient data for LSTM training: {len(X_sequences)} samples")
                return False
            
            print(f"Prepared {len(X_sequences)} training samples")
            print(f"Sequence shape: {X_sequences.shape}")
            if X_features is not None:
                print(f"Feature shape: {X_features.shape}")
            
            # Train-validation split (temporal)
            train_size = int(0.8 * len(X_sequences))
            X_seq_train, X_seq_val = X_sequences[:train_size], X_sequences[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            if X_features is not None:
                X_feat_train, X_feat_val = X_features[:train_size], X_features[train_size:]
                train_data = [X_seq_train, X_feat_train]
                val_data = [X_seq_val, X_feat_val]
                feature_dim = X_features.shape[1]
            else:
                train_data = X_seq_train
                val_data = X_seq_val
                feature_dim = None
            
            # Build advanced model
            tf.keras.backend.clear_session()
            self.model = self.build_attention_lstm_model(
                input_shape=(X_sequences.shape[1], X_sequences.shape[2]),
                feature_dim=feature_dim
            )
            
            # Compile with appropriate loss for multi-class classification
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("Model architecture:")
            self.model.summary()
            
            # Train with early stopping and learning rate reduction
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001
                )
            ]
            
            history = self.model.fit(
                train_data, y_train,
                validation_data=(val_data, y_val),
                epochs=100, batch_size=32, verbose=1,
                callbacks=callbacks
            )
            
            # Evaluate model
            val_loss, val_acc = self.model.evaluate(val_data, y_val, verbose=0)
            print(f"Advanced LSTM training completed.")
            print(f"Final validation loss: {val_loss:.6f}")
            print(f"Final validation accuracy: {val_acc:.4f}")
            print(f"Training stopped at epoch: {len(history.history['loss'])}")
            
            self._trained = True
            return True
            
        except Exception as e:
            print(f"Error training Advanced LSTM model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate advanced LSTM prediction with confidence."""
        if not TENSORFLOW_AVAILABLE:
            return self._unavailable_prediction("TensorFlow not available")
            
        if not self._trained or self.model is None:
            return self._default_prediction()
        
        try:
            # Prepare data for prediction
            X_sequences, X_features, _ = self.prepare_lstm_data(data)
            
            if len(X_sequences) == 0:
                return self._default_prediction()
            
            # Use the last sequence for prediction
            last_sequence = X_sequences[-1:] 
            
            if X_features is not None and self.use_multivariate:
                last_features = X_features[-1:]
                prediction_input = [last_sequence, last_features]
            else:
                prediction_input = last_sequence
            
            # Get prediction probabilities
            probabilities = self.model.predict(prediction_input, verbose=0)[0]
            
            # Extract class predictions
            predicted_class = np.argmax(probabilities)
            confidence = np.max(probabilities)
            
            # Map class to direction
            class_to_direction = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            direction = class_to_direction[predicted_class]
            
            # Apply confidence threshold
            if confidence < 0.6:
                direction = 'HOLD'
                predicted_class = 1
                confidence = probabilities[1]  # Use HOLD confidence
            
            return {
                'direction': direction,
                'confidence': float(confidence),
                'probability': float(confidence),
                'model_name': 'AdvancedLSTM',
                'timestamp': datetime.now(),
                'features_used': ['enhanced_features', 'attention_mechanism'],
                'class_probabilities': {
                    'SELL': float(probabilities[0]),
                    'HOLD': float(probabilities[1]),
                    'BUY': float(probabilities[2])
                },
                'prediction_class': int(predicted_class)
            }
            
        except Exception as e:
            print(f"Error making Advanced LSTM prediction: {e}")
            return self._default_prediction()
    
    def save(self) -> bool:
        """Save Advanced LSTM model and scalers."""
        if not TENSORFLOW_AVAILABLE:
            print("Cannot save Advanced LSTM model: TensorFlow not available")
            return False
            
        try:
            os.makedirs('model', exist_ok=True)
            
            # Save model
            model_path = self.get_model_path('.keras')
            self.model.save(model_path)
            
            # Save scalers
            scaler_path = self.get_model_path('_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            
            if self.feature_scaler is not None:
                feature_scaler_path = self.get_model_path('_feature_scaler.pkl')
                joblib.dump(self.feature_scaler, feature_scaler_path)
            
            print(f"Advanced LSTM model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving Advanced LSTM model: {e}")
            return False
    
    def load(self) -> bool:
        """Load Advanced LSTM model and scalers."""
        if not TENSORFLOW_AVAILABLE:
            print("Cannot load Advanced LSTM model: TensorFlow not available")
            return False
            
        try:
            # Load model
            model_path = self.get_model_path('.keras')
            if not os.path.exists(model_path):
                print(f"Advanced LSTM model file not found: {model_path}")
                return False
                
            self.model = load_model(model_path, compile=False)
            
            # Re-compile the model for predictions
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Load scalers
            scaler_path = self.get_model_path('_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                print(f"Advanced LSTM scaler file not found: {scaler_path}")
                return False
            
            feature_scaler_path = self.get_model_path('_feature_scaler.pkl')
            if os.path.exists(feature_scaler_path):
                self.feature_scaler = joblib.load(feature_scaler_path)
            
            self._trained = True
            print(f"Advanced LSTM model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading Advanced LSTM model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return Advanced LSTM model information."""
        return {
            'name': 'AdvancedLSTM',
            'type': 'neural_network_with_attention',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'sequence_length': self.sequence_length,
            'use_attention': self.use_attention,
            'use_multivariate': self.use_multivariate,
            'trained': self._trained,
            'features': ['enhanced_features', 'attention_mechanism'],
            'model_path': self.get_model_path('.keras'),
            'available': TENSORFLOW_AVAILABLE
        }
    
    def _unavailable_prediction(self, reason: str) -> Dict[str, Any]:
        """Return prediction when dependencies unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'AdvancedLSTM',
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
            'model_name': 'AdvancedLSTM',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': 'Model not trained or unavailable'
        }


# For backward compatibility, create an alias
LSTMModel = AdvancedLSTMModel


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


# Additional ML Models for Advanced Ensemble
if DEPENDENCY_MANAGER_AVAILABLE:
    HAS_SKLEARN_EXTRA = is_available('sklearn')
    
    if HAS_SKLEARN_EXTRA:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
else:
    # Legacy sklearn import handling
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        HAS_SKLEARN_EXTRA = True
    except ImportError:
        HAS_SKLEARN_EXTRA = False


class RandomForestModel(BaseModel):
    """Random Forest model for ensemble diversity."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.feature_columns = None
        self.feature_scaler = None
        if not HAS_SKLEARN_EXTRA:
            print(f"Warning: sklearn not available, Random Forest model for {symbol} {timeframe} will not function")
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for Random Forest training."""
        from .advanced_features import AdvancedDataPipeline
        from .advanced_indicators import EnhancedIndicators
        
        # Apply advanced feature engineering
        feature_pipeline = AdvancedDataPipeline(self.timeframe.replace('m', 'min'))
        enhanced_data = feature_pipeline.transform(data.copy())
        
        # Apply enhanced indicators
        indicator_calculator = EnhancedIndicators()
        enhanced_data = indicator_calculator.add_all_indicators(enhanced_data)
        
        # Prepare target variable (multi-class)
        future_returns = enhanced_data['close'].pct_change(periods=5).shift(-5)
        
        buy_threshold = 0.002
        sell_threshold = -0.002
        
        targets = pd.Series(index=enhanced_data.index, data=1)  # Default to HOLD
        targets[future_returns > buy_threshold] = 2   # BUY
        targets[future_returns < sell_threshold] = 0  # SELL
        
        # Select features (exclude target-related columns)
        exclude_cols = ['datetime', 'close'] if 'datetime' in enhanced_data.columns else ['close']
        feature_columns = [col for col in enhanced_data.columns 
                          if col not in exclude_cols and enhanced_data[col].dtype in ['float64', 'int64']]
        
        features = enhanced_data[feature_columns].fillna(0)
        
        return features, targets
    
    def train(self, data: pd.DataFrame) -> bool:
        """Train Random Forest model."""
        if not HAS_SKLEARN_EXTRA:
            print("Cannot train Random Forest model: sklearn not available")
            return False
            
        try:
            print(f"Training Random Forest model for {self.symbol} timeframe {self.timeframe}...")
            
            # Prepare features
            features, targets = self.prepare_features(data)
            
            # Remove rows with NaN targets
            valid_mask = ~targets.isna()
            features = features[valid_mask]
            targets = targets[valid_mask]
            
            if len(features) < 50:
                print(f"Insufficient data for Random Forest training: {len(features)} samples")
                return False
            
            print(f"Training with {len(features)} samples and {features.shape[1]} features")
            
            # Feature scaling
            self.feature_scaler = StandardScaler()
            features_scaled = self.feature_scaler.fit_transform(features)
            
            # Train-test split (temporal)
            train_size = int(0.8 * len(features))
            X_train, X_test = features_scaled[:train_size], features_scaled[train_size:]
            y_train, y_test = targets[:train_size], targets[train_size:]
            
            # Train Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_accuracy = self.model.score(X_train, y_train)
            test_accuracy = self.model.score(X_test, y_test)
            
            print(f"Random Forest training completed.")
            print(f"Training accuracy: {train_accuracy:.4f}")
            print(f"Testing accuracy: {test_accuracy:.4f}")
            
            # Feature importance
            feature_importance = dict(zip(features.columns, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            print("Top 10 important features:")
            for feature, importance in top_features:
                print(f"  {feature}: {importance:.4f}")
            
            self.feature_columns = list(features.columns)
            self._trained = True
            return True
            
        except Exception as e:
            print(f"Error training Random Forest model: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate Random Forest prediction."""
        if not HAS_SKLEARN_EXTRA:
            return self._unavailable_prediction("sklearn not available")
            
        if not self._trained or self.model is None:
            return self._default_prediction()
        
        try:
            # Prepare features
            features, _ = self.prepare_features(data)
            
            if len(features) == 0:
                return self._default_prediction()
            
            # Use last row for prediction
            last_features = features.tail(1)[self.feature_columns].fillna(0)
            features_scaled = self.feature_scaler.transform(last_features)
            
            # Predict
            probabilities = self.model.predict_proba(features_scaled)[0]
            predicted_class = np.argmax(probabilities)
            confidence = np.max(probabilities)
            
            # Map class to direction
            class_to_direction = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            direction = class_to_direction[predicted_class]
            
            # Apply confidence threshold
            if confidence < 0.55:
                direction = 'HOLD'
                confidence = probabilities[1]
            
            return {
                'direction': direction,
                'confidence': float(confidence),
                'probability': float(confidence),
                'model_name': 'RandomForest',
                'timestamp': datetime.now(),
                'features_used': self.feature_columns or [],
                'class_probabilities': {
                    'SELL': float(probabilities[0]),
                    'HOLD': float(probabilities[1]),
                    'BUY': float(probabilities[2])
                }
            }
            
        except Exception as e:
            print(f"Error making Random Forest prediction: {e}")
            return self._default_prediction()
    
    def save(self) -> bool:
        """Save Random Forest model."""
        try:
            os.makedirs('model', exist_ok=True)
            model_path = self.get_model_path('.pkl')
            scaler_path = self.get_model_path('_scaler.pkl')
            
            joblib.dump(self.model, model_path)
            if self.feature_scaler:
                joblib.dump(self.feature_scaler, scaler_path)
            
            # Save feature columns
            feature_path = self.get_model_path('_features.pkl')
            joblib.dump(self.feature_columns, feature_path)
            
            print(f"Random Forest model saved to {model_path}")
            return True
        except Exception as e:
            print(f"Error saving Random Forest model: {e}")
            return False
    
    def load(self) -> bool:
        """Load Random Forest model."""
        try:
            model_path = self.get_model_path('.pkl')
            scaler_path = self.get_model_path('_scaler.pkl')
            feature_path = self.get_model_path('_features.pkl')
            
            if not os.path.exists(model_path):
                print(f"Random Forest model file not found: {model_path}")
                return False
            
            self.model = joblib.load(model_path)
            
            if os.path.exists(scaler_path):
                self.feature_scaler = joblib.load(scaler_path)
            
            if os.path.exists(feature_path):
                self.feature_columns = joblib.load(feature_path)
            
            self._trained = True
            print(f"Random Forest model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading Random Forest model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return Random Forest model information."""
        return {
            'name': 'RandomForest',
            'type': 'ensemble_tree',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'trained': self._trained,
            'features': self.feature_columns or [],
            'available': HAS_SKLEARN_EXTRA
        }
    
    def _unavailable_prediction(self, reason: str) -> Dict[str, Any]:
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'RandomForest',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': reason
        }
    
    def _default_prediction(self) -> Dict[str, Any]:
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'RandomForest',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': 'Model not trained or unavailable'
        }


class SVMModel(BaseModel):
    """Support Vector Machine model for ensemble diversity."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.feature_columns = None
        self.feature_scaler = None
        if not HAS_SKLEARN_EXTRA:
            print(f"Warning: sklearn not available, SVM model for {symbol} {timeframe} will not function")
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for SVM training."""
        from .advanced_features import AdvancedDataPipeline
        from .advanced_indicators import EnhancedIndicators
        
        # Apply advanced feature engineering
        feature_pipeline = AdvancedDataPipeline(self.timeframe.replace('m', 'min'))
        enhanced_data = feature_pipeline.transform(data.copy())
        
        # Apply enhanced indicators
        indicator_calculator = EnhancedIndicators()
        enhanced_data = indicator_calculator.add_all_indicators(enhanced_data)
        
        # Prepare target variable (multi-class)
        future_returns = enhanced_data['close'].pct_change(periods=5).shift(-5)
        
        buy_threshold = 0.002
        sell_threshold = -0.002
        
        targets = pd.Series(index=enhanced_data.index, data=1)  # Default to HOLD
        targets[future_returns > buy_threshold] = 2   # BUY
        targets[future_returns < sell_threshold] = 0  # SELL
        
        # Select most relevant features (SVM works better with fewer features)
        exclude_cols = ['datetime', 'close'] if 'datetime' in enhanced_data.columns else ['close']
        feature_columns = [col for col in enhanced_data.columns 
                          if col not in exclude_cols and enhanced_data[col].dtype in ['float64', 'int64']]
        
        # Select top features based on correlation with price changes
        price_changes = enhanced_data['close'].pct_change()
        feature_correlations = {}
        
        for col in feature_columns:
            corr = enhanced_data[col].corr(price_changes)
            if not np.isnan(corr):
                feature_correlations[col] = abs(corr)
        
        # Select top 20 features
        top_features = sorted(feature_correlations.items(), key=lambda x: x[1], reverse=True)[:20]
        selected_features = [f[0] for f in top_features]
        
        features = enhanced_data[selected_features].fillna(0)
        
        return features, targets
    
    def train(self, data: pd.DataFrame) -> bool:
        """Train SVM model."""
        if not HAS_SKLEARN_EXTRA:
            print("Cannot train SVM model: sklearn not available")
            return False
            
        try:
            print(f"Training SVM model for {self.symbol} timeframe {self.timeframe}...")
            
            # Prepare features
            features, targets = self.prepare_features(data)
            
            # Remove rows with NaN targets
            valid_mask = ~targets.isna()
            features = features[valid_mask]
            targets = targets[valid_mask]
            
            if len(features) < 50:
                print(f"Insufficient data for SVM training: {len(features)} samples")
                return False
            
            print(f"Training with {len(features)} samples and {features.shape[1]} features")
            
            # Feature scaling (crucial for SVM)
            self.feature_scaler = StandardScaler()
            features_scaled = self.feature_scaler.fit_transform(features)
            
            # Train-test split (temporal)
            train_size = int(0.8 * len(features))
            X_train, X_test = features_scaled[:train_size], features_scaled[train_size:]
            y_train, y_test = targets[:train_size], targets[train_size:]
            
            # Train SVM with RBF kernel
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,  # Enable probability estimates
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_accuracy = self.model.score(X_train, y_train)
            test_accuracy = self.model.score(X_test, y_test)
            
            print(f"SVM training completed.")
            print(f"Training accuracy: {train_accuracy:.4f}")
            print(f"Testing accuracy: {test_accuracy:.4f}")
            
            self.feature_columns = list(features.columns)
            self._trained = True
            return True
            
        except Exception as e:
            print(f"Error training SVM model: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate SVM prediction."""
        if not HAS_SKLEARN_EXTRA:
            return self._unavailable_prediction("sklearn not available")
            
        if not self._trained or self.model is None:
            return self._default_prediction()
        
        try:
            # Prepare features
            features, _ = self.prepare_features(data)
            
            if len(features) == 0:
                return self._default_prediction()
            
            # Use last row for prediction
            last_features = features.tail(1)[self.feature_columns].fillna(0)
            features_scaled = self.feature_scaler.transform(last_features)
            
            # Predict
            probabilities = self.model.predict_proba(features_scaled)[0]
            predicted_class = np.argmax(probabilities)
            confidence = np.max(probabilities)
            
            # Map class to direction
            class_to_direction = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            direction = class_to_direction[predicted_class]
            
            # Apply confidence threshold
            if confidence < 0.6:
                direction = 'HOLD'
                confidence = probabilities[1]
            
            return {
                'direction': direction,
                'confidence': float(confidence),
                'probability': float(confidence),
                'model_name': 'SVM',
                'timestamp': datetime.now(),
                'features_used': self.feature_columns or [],
                'class_probabilities': {
                    'SELL': float(probabilities[0]),
                    'HOLD': float(probabilities[1]),
                    'BUY': float(probabilities[2])
                }
            }
            
        except Exception as e:
            print(f"Error making SVM prediction: {e}")
            return self._default_prediction()
    
    def save(self) -> bool:
        """Save SVM model."""
        try:
            os.makedirs('model', exist_ok=True)
            model_path = self.get_model_path('.pkl')
            scaler_path = self.get_model_path('_scaler.pkl')
            
            joblib.dump(self.model, model_path)
            if self.feature_scaler:
                joblib.dump(self.feature_scaler, scaler_path)
            
            # Save feature columns
            feature_path = self.get_model_path('_features.pkl')
            joblib.dump(self.feature_columns, feature_path)
            
            print(f"SVM model saved to {model_path}")
            return True
        except Exception as e:
            print(f"Error saving SVM model: {e}")
            return False
    
    def load(self) -> bool:
        """Load SVM model."""
        try:
            model_path = self.get_model_path('.pkl')
            scaler_path = self.get_model_path('_scaler.pkl')
            feature_path = self.get_model_path('_features.pkl')
            
            if not os.path.exists(model_path):
                print(f"SVM model file not found: {model_path}")
                return False
            
            self.model = joblib.load(model_path)
            
            if os.path.exists(scaler_path):
                self.feature_scaler = joblib.load(scaler_path)
            
            if os.path.exists(feature_path):
                self.feature_columns = joblib.load(feature_path)
            
            self._trained = True
            print(f"SVM model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading SVM model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return SVM model information."""
        return {
            'name': 'SVM',
            'type': 'support_vector_machine',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'trained': self._trained,
            'features': self.feature_columns or [],
            'available': HAS_SKLEARN_EXTRA
        }
    
    def _unavailable_prediction(self, reason: str) -> Dict[str, Any]:
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'SVM',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': reason
        }
    
    def _default_prediction(self) -> Dict[str, Any]:
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'SVM',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': 'Model not trained or unavailable'
        }