"""
Prediction Service for the modular application.
Handles real model predictions and ensemble logic.
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler

from .model_registry import ModelRegistry
from .feature_service import FeatureService


class PredictionService:
    """Service for making predictions using trained models."""
    
    def __init__(self, model_registry: ModelRegistry, feature_service: FeatureService):
        self.model_registry = model_registry
        self.feature_service = feature_service
        self.ensemble_threshold = 0.55  # Minimum confidence for non-HOLD signal
    
    def _predict_lstm(self, prices60: np.ndarray, model: Any, scaler: Any) -> Dict[str, Any]:
        """
        Make LSTM prediction using the last 60 close prices.
        
        Args:
            prices60: Array of 60 close prices
            model: Trained LSTM model
            scaler: MinMaxScaler for LSTM
            
        Returns:
            Dictionary with direction, confidence, and raw prediction
        """
        try:
            # Scale the prices
            prices_scaled = scaler.transform(prices60.reshape(-1, 1))
            
            # Reshape for LSTM: (1, 60, 1)
            X = prices_scaled.reshape(1, 60, 1)
            
            # Make prediction
            pred_scaled = model.predict(X, verbose=0)
            
            # Inverse transform to get predicted price
            pred_price = scaler.inverse_transform(pred_scaled)[0, 0]
            
            # Calculate percentage change vs last close
            last_close = prices60[-1]
            delta_pct = (pred_price - last_close) / last_close
            
            # Determine direction and confidence
            if abs(delta_pct) < 0.001:  # Less than 0.1% change
                direction = "HOLD"
                confidence = 0.5
            elif delta_pct > 0:
                direction = "BUY"
                confidence = min(0.95, 0.5 + abs(delta_pct) * 100)  # Scale confidence
            else:
                direction = "SELL"
                confidence = min(0.95, 0.5 + abs(delta_pct) * 100)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'raw': {
                    'predicted_price': pred_price,
                    'last_close': last_close,
                    'delta_pct': delta_pct
                }
            }
            
        except Exception as e:
            st.warning(f"LSTM prediction error: {str(e)}")
            return {
                'direction': "HOLD",
                'confidence': 0.5,
                'raw': {'error': str(e)}
            }
    
    def _predict_cnn(self, feature_window: np.ndarray, model: Any, scaler: Any) -> Dict[str, Any]:
        """
        Make CNN prediction using windowed features.
        
        Args:
            feature_window: Array of shape (20, num_features)
            model: Trained CNN model
            scaler: Scaler for CNN features
            
        Returns:
            Dictionary with direction, confidence, and raw prediction
        """
        try:
            # Scale the feature window
            features_scaled = scaler.transform(feature_window)
            
            # Reshape for CNN: (1, 20, num_features)
            X = features_scaled.reshape(1, features_scaled.shape[0], features_scaled.shape[1])
            
            # Make prediction
            pred_proba = model.predict(X, verbose=0)[0, 0]
            
            # Convert probability to direction and confidence
            if pred_proba > 0.55:
                direction = "BUY"
                confidence = pred_proba
            elif pred_proba < 0.45:
                direction = "SELL"
                confidence = 1 - pred_proba
            else:
                direction = "HOLD"
                confidence = 0.5
            
            return {
                'direction': direction,
                'confidence': confidence,
                'raw': {
                    'probability': pred_proba,
                    'binary_prediction': int(pred_proba > 0.5)
                }
            }
            
        except Exception as e:
            st.warning(f"CNN prediction error: {str(e)}")
            return {
                'direction': "HOLD",
                'confidence': 0.5,
                'raw': {'error': str(e)}
            }
    
    def _predict_xgb(self, row_features: np.ndarray, model: Any) -> Dict[str, Any]:
        """
        Make XGBoost prediction using feature row.
        
        Args:
            row_features: Array of features for latest row
            model: Trained XGBoost model
            
        Returns:
            Dictionary with direction, confidence, and raw prediction
        """
        try:
            # Make prediction
            pred_class = model.predict(row_features)[0]
            
            # Get probabilities if available
            try:
                pred_proba = model.predict_proba(row_features)[0]
                confidence = pred_proba[pred_class]
            except:
                # Fallback to decision function if predict_proba not available
                try:
                    decision = model.decision_function(row_features)[0]
                    confidence = min(0.95, 0.5 + abs(decision) * 0.1)
                except:
                    confidence = 0.6  # Default confidence
            
            # Convert prediction to direction
            if pred_class == 1:
                direction = "BUY"
            elif pred_class == 0:
                direction = "SELL"
            else:
                direction = "HOLD"
            
            return {
                'direction': direction,
                'confidence': confidence,
                'raw': {
                    'prediction': int(pred_class),
                    'probabilities': pred_proba.tolist() if 'pred_proba' in locals() else None
                }
            }
            
        except Exception as e:
            st.warning(f"XGBoost prediction error: {str(e)}")
            return {
                'direction': "HOLD",
                'confidence': 0.5,
                'raw': {'error': str(e)}
            }
    
    def _predict_svc(self, row_features: np.ndarray, model: Any, scaler: Any) -> Dict[str, Any]:
        """
        Make SVC prediction using scaled features.
        
        Args:
            row_features: Array of features for latest row
            model: Trained SVC model
            scaler: Scaler for SVC features
            
        Returns:
            Dictionary with direction, confidence, and raw prediction
        """
        try:
            # Scale features
            features_scaled = scaler.transform(row_features)
            
            # Make prediction
            pred_class = model.predict(features_scaled)[0]
            
            # Get probabilities
            try:
                pred_proba = model.predict_proba(features_scaled)[0]
                confidence = pred_proba[pred_class]
            except:
                confidence = 0.6  # Default confidence
            
            # Convert prediction to direction
            if pred_class == 1:
                direction = "BUY"
            elif pred_class == 0:
                direction = "SELL"
            else:
                direction = "HOLD"
            
            return {
                'direction': direction,
                'confidence': confidence,
                'raw': {
                    'prediction': int(pred_class),
                    'probabilities': pred_proba.tolist() if 'pred_proba' in locals() else None
                }
            }
            
        except Exception as e:
            st.warning(f"SVC prediction error: {str(e)}")
            return {
                'direction': "HOLD",
                'confidence': 0.5,
                'raw': {'error': str(e)}
            }
    
    def _predict_nb(self, row_features: np.ndarray, model: Any) -> Dict[str, Any]:
        """
        Make Naive Bayes prediction using features.
        
        Args:
            row_features: Array of features for latest row
            model: Trained Naive Bayes model
            
        Returns:
            Dictionary with direction, confidence, and raw prediction
        """
        try:
            # Make prediction
            pred_class = model.predict(row_features)[0]
            
            # Get probabilities
            pred_proba = model.predict_proba(row_features)[0]
            confidence = pred_proba[pred_class]
            
            # Convert prediction to direction
            if pred_class == 1:
                direction = "BUY"
            elif pred_class == 0:
                direction = "SELL"
            else:
                direction = "HOLD"
            
            return {
                'direction': direction,
                'confidence': confidence,
                'raw': {
                    'prediction': int(pred_class),
                    'probabilities': pred_proba.tolist()
                }
            }
            
        except Exception as e:
            st.warning(f"Naive Bayes prediction error: {str(e)}")
            return {
                'direction': "HOLD",
                'confidence': 0.5,
                'raw': {'error': str(e)}
            }
    
    def make_predictions(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Make predictions using all available models and create ensemble.
        
        Args:
            data: DataFrame with historical data and features
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Dictionary with individual model predictions and ensemble result
        """
        try:
            # Load models
            models, load_errors = self.model_registry.load_for_prediction(symbol, timeframe)
            
            if not models:
                return {
                    'ensemble': {
                        'direction': 'HOLD',
                        'confidence': 0.0,
                        'message': f'No trained models found for {symbol} {timeframe}'
                    },
                    'individual': {},
                    'load_errors': load_errors
                }
            
            # Log load errors as warnings
            if load_errors:
                for model_name, error in load_errors.items():
                    if 'optional' not in error.lower():
                        st.warning(f"Failed to load {model_name}: {error}")
            
            individual_predictions = {}
            
            # LSTM prediction
            if 'lstm' in models and 'lstm_scaler' in models:
                try:
                    prices60 = self.feature_service.prepare_lstm_sequence(data)
                    individual_predictions['lstm'] = self._predict_lstm(
                        prices60, models['lstm'], models['lstm_scaler']
                    )
                except Exception as e:
                    st.warning(f"LSTM prediction failed: {str(e)}")
                    individual_predictions['lstm'] = {
                        'direction': 'HOLD', 'confidence': 0.5, 'raw': {'error': str(e)}
                    }
            
            # CNN prediction
            if 'cnn' in models and 'cnn_scaler' in models:
                try:
                    feature_window = self.feature_service.prepare_cnn_window(data)
                    individual_predictions['cnn'] = self._predict_cnn(
                        feature_window, models['cnn'], models['cnn_scaler']
                    )
                except Exception as e:
                    st.warning(f"CNN prediction failed: {str(e)}")
                    individual_predictions['cnn'] = {
                        'direction': 'HOLD', 'confidence': 0.5, 'raw': {'error': str(e)}
                    }
            
            # XGBoost prediction
            if 'xgb' in models:
                try:
                    row_features = self.feature_service.get_latest_row_features(data, 'xgb')
                    if row_features.size > 0:
                        individual_predictions['xgb'] = self._predict_xgb(row_features, models['xgb'])
                except Exception as e:
                    st.warning(f"XGBoost prediction failed: {str(e)}")
                    individual_predictions['xgb'] = {
                        'direction': 'HOLD', 'confidence': 0.5, 'raw': {'error': str(e)}
                    }
            
            # SVC prediction
            if 'svc' in models and 'svc_scaler' in models:
                try:
                    row_features = self.feature_service.get_latest_row_features(data, 'svc')
                    if row_features.size > 0:
                        individual_predictions['svc'] = self._predict_svc(
                            row_features, models['svc'], models['svc_scaler']
                        )
                except Exception as e:
                    st.warning(f"SVC prediction failed: {str(e)}")
                    individual_predictions['svc'] = {
                        'direction': 'HOLD', 'confidence': 0.5, 'raw': {'error': str(e)}
                    }
            
            # Naive Bayes prediction
            if 'nb' in models:
                try:
                    row_features = self.feature_service.get_latest_row_features(data, 'nb')
                    if row_features.size > 0:
                        individual_predictions['nb'] = self._predict_nb(row_features, models['nb'])
                except Exception as e:
                    st.warning(f"Naive Bayes prediction failed: {str(e)}")
                    individual_predictions['nb'] = {
                        'direction': 'HOLD', 'confidence': 0.5, 'raw': {'error': str(e)}
                    }
            
            # Create ensemble prediction
            ensemble = self._create_ensemble(individual_predictions)
            
            return {
                'ensemble': ensemble,
                'individual': individual_predictions,
                'load_errors': load_errors
            }
            
        except Exception as e:
            st.error(f"Prediction service error: {str(e)}")
            return {
                'ensemble': {
                    'direction': 'HOLD',
                    'confidence': 0.0,
                    'message': f'Prediction error: {str(e)}'
                },
                'individual': {},
                'load_errors': {}
            }
    
    def _create_ensemble(self, individual_predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create ensemble prediction from individual model predictions.
        
        Args:
            individual_predictions: Dictionary of individual model predictions
            
        Returns:
            Dictionary with ensemble prediction
        """
        if not individual_predictions:
            return {
                'direction': 'HOLD',
                'confidence': 0.0,
                'message': 'No individual predictions available'
            }
        
        # Count votes weighted by confidence
        vote_weights = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        total_confidence = 0.0
        
        for model_name, prediction in individual_predictions.items():
            direction = prediction.get('direction', 'HOLD')
            confidence = prediction.get('confidence', 0.5)
            
            vote_weights[direction] += confidence
            total_confidence += confidence
        
        if total_confidence == 0:
            return {
                'direction': 'HOLD',
                'confidence': 0.0,
                'message': 'No valid predictions'
            }
        
        # Normalize weights
        for direction in vote_weights:
            vote_weights[direction] /= total_confidence
        
        # Find winning direction
        winning_direction = max(vote_weights, key=vote_weights.get)
        winning_weight = vote_weights[winning_direction]
        
        # Apply threshold - if not confident enough, default to HOLD
        if winning_weight < self.ensemble_threshold and winning_direction != 'HOLD':
            final_direction = 'HOLD'
            final_confidence = 0.5
            message = f'Ensemble below threshold ({winning_weight:.3f} < {self.ensemble_threshold})'
        else:
            final_direction = winning_direction
            final_confidence = winning_weight
            message = f'Ensemble confidence: {final_confidence:.3f}'
        
        return {
            'direction': final_direction,
            'confidence': final_confidence,
            'message': message,
            'vote_weights': vote_weights,
            'model_count': len(individual_predictions)
        }