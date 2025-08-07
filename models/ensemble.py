# models/ensemble.py
"""
Ensemble methods for combining model predictions.
Migrated meta learner from utils/meta_learner.py with additional ensemble methods.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseModel


class MetaLearner(BaseModel):
    """Meta learner for ensemble predictions, migrated from utils/meta_learner.py"""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.expected_features = None
        
    def prepare_data_for_meta_learner(self, df, lstm_predictions, xgb_predictions, xgb_confidences,
                                      cnn_predictions, svc_predictions, svc_confidences,
                                      nb_predictions, nb_confidences):
        """Prepare features and target for Meta-Learner (minimal change from original)."""
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
        
        y = df['target_meta'].apply(lambda v: -1 if v == -1 else (1 if v == 1 else 0))
        combined = X.join(y.rename('target_meta'), how='inner')
        combined.dropna(subset=['target_meta'], inplace=True)
        return combined.drop(columns=['target_meta']), combined['target_meta']
    
    def train(self, data: pd.DataFrame, lstm_predictions=None, xgb_predictions=None, 
              xgb_confidences=None, cnn_predictions=None, svc_predictions=None, 
              svc_confidences=None, nb_predictions=None, nb_confidences=None,
              tune_hyperparams=True, n_iter=50) -> bool:  # Changed default to True
        """Train meta learner using LightGBM with tuning enabled by default."""
        try:
            print(f"Training Meta Learner for {self.symbol} timeframe {self.timeframe}...")
            print(f"Hyperparameter tuning: {'Enabled' if tune_hyperparams else 'Disabled'}")
            
            # Prepare data
            X, y = self.prepare_data_for_meta_learner(
                data, lstm_predictions, xgb_predictions, xgb_confidences,
                cnn_predictions, svc_predictions, svc_confidences,
                nb_predictions, nb_confidences
            )
            
            if len(y.unique()) < 2:
                print("Target not diverse enough for classification.")
                if len(y.unique()) > 0:
                    print(f"Only found classes: {y.unique()}. Should have [-1, 0, 1].")
                    print("Ensure training data has sufficient up, down, and neutral movements.")
                return False

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            base_model = LGBMClassifier(random_state=42)
            if tune_hyperparams:
                # Enhanced parameter distribution for better financial data performance
                param_dist = {
                    'n_estimators': [100, 200, 300, 500],
                    'num_leaves': [31, 63, 127, 255],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'subsample': [0.6, 0.7, 0.8, 0.9],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0, 0.1, 0.5, 1.0]
                }
                search = RandomizedSearchCV(
                    base_model, param_distributions=param_dist,
                    n_iter=n_iter, scoring='f1_macro', cv=3,
                    random_state=42, n_jobs=-1
                )
                print(f"Starting enhanced hyperparameter tuning for Meta Learner ({n_iter} iterations)...")
                search.fit(X_train, y_train)
                self.model = search.best_estimator_
                print(f"Best Meta Learner parameters: {search.best_params_}")
                print(f"Best CV score: {search.best_score_:.4f}")
            else:
                self.model = base_model
                print(f"Training standard LGBMClassifier for timeframe {self.timeframe}...")
                self.model.fit(X_train, y_train)

            # Enhanced evaluation
            y_pred = self.model.predict(X_test)
            print(confusion_matrix(y_test, y_pred))
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, digits=3, zero_division=0))

            self.expected_features = list(X.columns)
            self._trained = True
            print("Enhanced Meta Learner training completed.")
            return True
            
        except Exception as e:
            print(f"Error training Meta Learner: {e}")
            return False
    
    def predict(self, data: pd.DataFrame, lstm_predictions=None, xgb_predictions=None,
                xgb_confidences=None, cnn_predictions=None, svc_predictions=None,
                svc_confidences=None, nb_predictions=None, nb_confidences=None,
                threshold=0.52) -> Dict[str, Any]:
        """Generate meta learner prediction (minimal change from original)."""
        if not self._trained or self.model is None:
            return self._default_prediction()
        
        try:
            last_idx = data.index[-1:]
            preds = [p.reindex(last_idx).fillna(0) for p in [
                lstm_predictions, xgb_predictions, xgb_confidences,
                cnn_predictions, svc_predictions, svc_confidences,
                nb_predictions, nb_confidences
            ]]
            
            X_pred, _ = self.prepare_data_for_meta_learner(data.loc[last_idx], *preds)

            if X_pred.empty:
                return self._default_prediction()

            proba = self.model.predict_proba(X_pred)[0]
            max_conf = np.max(proba)
            pred_class = self.model.classes_[np.argmax(proba)]

            # Return prediction if confidence exceeds threshold, otherwise HOLD
            if max_conf >= threshold:
                if pred_class == 1:
                    direction = "BUY"
                elif pred_class == -1:
                    direction = "SELL"
                else:
                    direction = "HOLD"
            else:
                direction = "HOLD"
                pred_class = 0

            return {
                'direction': direction,
                'confidence': max_conf,
                'probability': max_conf,
                'model_name': 'MetaLearner',
                'timestamp': datetime.now(),
                'features_used': self.expected_features or [],
                'prediction_class': pred_class,
                'threshold': threshold
            }
            
        except Exception as e:
            print(f"Error making Meta Learner prediction: {e}")
            return self._default_prediction()
    
    def save(self) -> bool:
        """Save meta learner model."""
        try:
            os.makedirs('model', exist_ok=True)
            # Use same naming convention as original for compatibility
            model_path = os.path.join('model', f'meta_learner_randomforest_{self.timeframe}.pkl')
            joblib.dump(self.model, model_path)
            
            # Save metadata
            metadata = {
                'expected_features': self.expected_features,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'model_type': 'meta_learner'
            }
            metadata_path = self.get_model_path('_metadata.json')
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f)
            
            print(f"Meta Learner saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving Meta Learner: {e}")
            return False
    
    def load(self) -> bool:
        """Load meta learner model."""
        try:
            # Try new path first, fall back to original naming convention
            model_path = self.get_model_path('.pkl')
            original_path = os.path.join('model', f'meta_learner_randomforest_{self.timeframe}.pkl')
            
            path_to_use = model_path if os.path.exists(model_path) else original_path
            
            if os.path.exists(path_to_use):
                self.model = joblib.load(path_to_use)
                
                # Load metadata if available
                metadata_path = self.get_model_path('_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        import json
                        metadata = json.load(f)
                        self.expected_features = metadata.get('expected_features')
                
                self._trained = True
                print(f"Meta Learner loaded from {path_to_use}")
                return True
            else:
                print(f"Meta Learner model file not found: {path_to_use}")
                return False
                
        except Exception as e:
            print(f"Error loading Meta Learner: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return meta learner model metadata."""
        return {
            'name': 'MetaLearner',
            'type': 'ensemble_meta',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'trained': self._trained,
            'features': self.expected_features,
            'model_path': self.get_model_path('.pkl')
        }
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when model unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'MetaLearner',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': 'Model not trained or unavailable'
        }


class VotingEnsemble:
    """Simple voting ensemble for combining multiple model predictions."""
    
    def __init__(self, models: List[BaseModel]):
        """Initialize with list of trained models."""
        self.models = models
        self.name = "VotingEnsemble"
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make ensemble prediction using majority voting."""
        if not self.models:
            return self._default_prediction()
        
        try:
            predictions = []
            confidences = []
            
            for model in self.models:
                if model.is_trained():
                    pred = model.predict(data)
                    predictions.append(pred['direction'])
                    confidences.append(pred['confidence'])
            
            if not predictions:
                return self._default_prediction()
            
            # Count votes for each direction
            vote_counts = {}
            for pred in predictions:
                vote_counts[pred] = vote_counts.get(pred, 0) + 1
            
            # Get majority vote
            majority_direction = max(vote_counts, key=vote_counts.get)
            majority_votes = vote_counts[majority_direction]
            confidence = majority_votes / len(predictions)
            
            # Average confidence of models voting for majority
            avg_confidence = np.mean([
                conf for pred, conf in zip(predictions, confidences) 
                if pred == majority_direction
            ])
            
            return {
                'direction': majority_direction,
                'confidence': confidence,
                'probability': avg_confidence,
                'model_name': 'VotingEnsemble',
                'timestamp': datetime.now(),
                'features_used': ['ensemble_voting'],
                'vote_counts': vote_counts,
                'total_models': len(predictions)
            }
            
        except Exception as e:
            print(f"Error in voting ensemble prediction: {e}")
            return self._default_prediction()
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when ensemble unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'VotingEnsemble',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': 'No trained models available'
        }


class WeightedVotingEnsemble:
    """Weighted voting ensemble using model confidence as weights."""
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        """Initialize with list of trained models and optional weights."""
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.name = "WeightedVotingEnsemble"
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make ensemble prediction using weighted voting."""
        if not self.models:
            return self._default_prediction()
        
        try:
            weighted_votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            total_weight = 0
            predictions_made = 0
            
            for model, weight in zip(self.models, self.weights):
                if model.is_trained():
                    pred = model.predict(data)
                    direction = pred['direction']
                    confidence = pred['confidence']
                    
                    # Weight by both assigned weight and model confidence
                    effective_weight = weight * confidence
                    weighted_votes[direction] += effective_weight
                    total_weight += effective_weight
                    predictions_made += 1
            
            if predictions_made == 0 or total_weight == 0:
                return self._default_prediction()
            
            # Normalize weights and find winner
            for direction in weighted_votes:
                weighted_votes[direction] /= total_weight
            
            winning_direction = max(weighted_votes, key=weighted_votes.get)
            winning_confidence = weighted_votes[winning_direction]
            
            return {
                'direction': winning_direction,
                'confidence': winning_confidence,
                'probability': winning_confidence,
                'model_name': 'WeightedVotingEnsemble',
                'timestamp': datetime.now(),
                'features_used': ['weighted_ensemble'],
                'weighted_votes': weighted_votes,
                'total_models': predictions_made
            }
            
        except Exception as e:
            print(f"Error in weighted voting ensemble prediction: {e}")
            return self._default_prediction()
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when ensemble unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'WeightedVotingEnsemble',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': 'No trained models available'
        }