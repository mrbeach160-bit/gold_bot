# models/advanced_ensemble.py
"""
Advanced ensemble methods and meta-learning for Phase 3 implementation.
Dynamic weight ensemble, regime-aware selection, and confidence-weighted voting.
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Try to import required libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    import lightgbm as lgb
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy import stats
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .base import BaseModel


class PerformanceTracker:
    """Track model performance over time for dynamic weighting."""
    
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.performance_history = defaultdict(lambda: deque(maxlen=lookback_window))
        self.prediction_history = defaultdict(lambda: deque(maxlen=lookback_window))
        self.recent_weights = {}
        
    def update_performance(self, model_name: str, prediction: Dict[str, Any], 
                          actual_outcome: str, time_decay_factor: float = 0.95):
        """Update performance metrics for a model."""
        # Calculate accuracy (1 for correct, 0 for incorrect)
        accuracy = 1.0 if prediction['direction'] == actual_outcome else 0.0
        
        # Weight by confidence and time decay
        confidence_weight = prediction.get('confidence', 0.5)
        weighted_score = accuracy * confidence_weight * time_decay_factor
        
        self.performance_history[model_name].append(weighted_score)
        self.prediction_history[model_name].append({
            'prediction': prediction,
            'actual': actual_outcome,
            'timestamp': datetime.now(),
            'score': weighted_score
        })
    
    def get_recent_performance(self, model_name: str, window: int = 20) -> float:
        """Get recent performance score for a model."""
        if model_name not in self.performance_history:
            return 0.5  # Neutral performance for new models
        
        recent_scores = list(self.performance_history[model_name])[-window:]
        if not recent_scores:
            return 0.5
        
        return np.mean(recent_scores)
    
    def get_performance_trend(self, model_name: str) -> float:
        """Get performance trend (positive = improving, negative = declining)."""
        if model_name not in self.performance_history:
            return 0.0
        
        scores = list(self.performance_history[model_name])
        if len(scores) < 10:
            return 0.0
        
        # Calculate trend using linear regression
        x = np.arange(len(scores))
        if HAS_SCIPY:
            slope, _, _, _, _ = stats.linregress(x, scores)
            return slope
        else:
            # Simple trend calculation
            recent_avg = np.mean(scores[-10:])
            older_avg = np.mean(scores[-20:-10]) if len(scores) >= 20 else np.mean(scores[:-10])
            return recent_avg - older_avg
    
    def calculate_dynamic_weights(self, model_names: List[str]) -> Dict[str, float]:
        """Calculate dynamic weights based on recent performance."""
        weights = {}
        
        for model_name in model_names:
            # Base performance score
            performance = self.get_recent_performance(model_name)
            
            # Performance trend adjustment
            trend = self.get_performance_trend(model_name)
            trend_adjustment = max(-0.2, min(0.2, trend * 10))  # Cap trend impact
            
            # Stability adjustment (lower volatility = higher weight)
            scores = list(self.performance_history[model_name])
            stability = 1.0 - (np.std(scores) if len(scores) > 5 else 0.3)
            
            # Combine factors
            weight = performance + trend_adjustment + stability * 0.1
            weights[model_name] = max(0.01, weight)  # Minimum weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Equal weights if no performance data
            equal_weight = 1.0 / len(model_names)
            weights = {name: equal_weight for name in model_names}
        
        self.recent_weights = weights
        return weights


class RegimeDetector:
    """Detect market regimes for regime-aware model selection."""
    
    def __init__(self, lookback_window: int = 50):
        self.lookback_window = lookback_window
        self.current_regime = 'neutral'
        self.regime_history = deque(maxlen=100)
        
    def detect_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime."""
        if len(data) < self.lookback_window:
            return 'neutral'
        
        recent_data = data.tail(self.lookback_window)
        returns = recent_data['close'].pct_change().dropna()
        
        if len(returns) < 10:
            return 'neutral'
        
        # Volatility measurement
        volatility = returns.std()
        vol_threshold_high = returns.rolling(window=20).std().quantile(0.8)
        vol_threshold_low = returns.rolling(window=20).std().quantile(0.2)
        
        # Trend measurement
        price_trend = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        trend_threshold = 0.02  # 2% threshold
        
        # Range measurement
        high_low_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].mean()
        range_threshold = 0.05  # 5% threshold
        
        # Regime classification
        if volatility > vol_threshold_high:
            if abs(price_trend) > trend_threshold:
                regime = 'trending_volatile'
            else:
                regime = 'sideways_volatile'
        elif volatility < vol_threshold_low:
            if abs(price_trend) > trend_threshold:
                regime = 'trending_calm'
            else:
                regime = 'sideways_calm'
        else:
            if price_trend > trend_threshold:
                regime = 'bullish_moderate'
            elif price_trend < -trend_threshold:
                regime = 'bearish_moderate'
            else:
                regime = 'neutral'
        
        self.current_regime = regime
        self.regime_history.append({
            'regime': regime,
            'timestamp': datetime.now(),
            'volatility': volatility,
            'trend': price_trend,
            'range': high_low_range
        })
        
        return regime
    
    def get_regime_model_preferences(self, regime: str) -> Dict[str, float]:
        """Get model preference weights for different regimes."""
        preferences = {
            'trending_volatile': {
                'AdvancedLSTM': 1.2,  # Neural networks good for complex patterns
                'RandomForest': 1.1,  # Tree models handle volatility well
                'SVM': 0.8,           # SVM struggles with high volatility
                'LightGBM': 1.0,
                'XGBoost': 1.0
            },
            'trending_calm': {
                'AdvancedLSTM': 1.3,  # LSTM excels in clear trends
                'SVM': 1.2,           # SVM good for clear patterns
                'RandomForest': 0.9,
                'LightGBM': 1.1,
                'XGBoost': 1.1
            },
            'sideways_volatile': {
                'RandomForest': 1.3,  # Tree models handle range-bound markets
                'LightGBM': 1.2,
                'XGBoost': 1.1,
                'AdvancedLSTM': 0.8,  # Neural networks may overfit in sideways
                'SVM': 0.9
            },
            'sideways_calm': {
                'LightGBM': 1.2,      # Gradient boosting good for subtle patterns
                'XGBoost': 1.2,
                'RandomForest': 1.1,
                'SVM': 1.0,
                'AdvancedLSTM': 0.9
            },
            'bullish_moderate': {
                'AdvancedLSTM': 1.2,
                'LightGBM': 1.1,
                'XGBoost': 1.1,
                'SVM': 1.0,
                'RandomForest': 1.0
            },
            'bearish_moderate': {
                'AdvancedLSTM': 1.2,
                'SVM': 1.1,
                'LightGBM': 1.1,
                'XGBoost': 1.0,
                'RandomForest': 1.0
            },
            'neutral': {
                'AdvancedLSTM': 1.0,
                'LightGBM': 1.0,
                'XGBoost': 1.0,
                'RandomForest': 1.0,
                'SVM': 1.0
            }
        }
        
        return preferences.get(regime, preferences['neutral'])


class ConsensusAnalyzer:
    """Analyze model consensus and disagreement for uncertainty quantification."""
    
    def __init__(self):
        self.consensus_history = deque(maxlen=100)
        
    def analyze_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consensus among model predictions."""
        if not predictions:
            return {'agreement_score': 0.0, 'uncertainty': 1.0}
        
        # Extract directions and confidences
        directions = [pred['direction'] for pred in predictions]
        confidences = [pred.get('confidence', 0.5) for pred in predictions]
        
        # Calculate agreement score
        direction_counts = pd.Series(directions).value_counts()
        max_agreement = direction_counts.iloc[0] if len(direction_counts) > 0 else 0
        agreement_score = max_agreement / len(predictions)
        
        # Calculate uncertainty (inverse of consensus strength)
        if len(set(directions)) == 1:
            # All models agree
            avg_confidence = np.mean(confidences)
            uncertainty = 1.0 - avg_confidence
        else:
            # Models disagree
            uncertainty = 1.0 - agreement_score
        
        # Weight-based consensus (weighted by confidence)
        weighted_votes = defaultdict(float)
        total_weight = 0
        
        for pred in predictions:
            direction = pred['direction']
            confidence = pred.get('confidence', 0.5)
            weighted_votes[direction] += confidence
            total_weight += confidence
        
        if total_weight > 0:
            weighted_consensus = max(weighted_votes.values()) / total_weight
        else:
            weighted_consensus = agreement_score
        
        # Detect unusual predictions (outliers)
        confidence_mean = np.mean(confidences)
        confidence_std = np.std(confidences)
        outlier_threshold = confidence_mean + 2 * confidence_std
        
        outliers = [pred for pred in predictions 
                   if pred.get('confidence', 0.5) > outlier_threshold]
        
        consensus_info = {
            'agreement_score': agreement_score,
            'weighted_consensus': weighted_consensus,
            'uncertainty': uncertainty,
            'num_models': len(predictions),
            'unique_directions': len(set(directions)),
            'avg_confidence': confidence_mean,
            'confidence_std': confidence_std,
            'outliers': len(outliers),
            'direction_distribution': dict(direction_counts),
            'dominant_direction': direction_counts.index[0] if len(direction_counts) > 0 else 'HOLD'
        }
        
        self.consensus_history.append({
            'consensus_info': consensus_info,
            'timestamp': datetime.now()
        })
        
        return consensus_info


class DynamicEnsemble(BaseModel):
    """Dynamic ensemble with performance-based weighting and regime awareness."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.performance_tracker = PerformanceTracker()
        self.regime_detector = RegimeDetector()
        self.consensus_analyzer = ConsensusAnalyzer()
        self.base_models = {}
        self.model_names = []
        
    def add_model(self, model: BaseModel):
        """Add a base model to the ensemble."""
        model_name = model.get_model_info()['name']
        self.base_models[model_name] = model
        if model_name not in self.model_names:
            self.model_names.append(model_name)
    
    def remove_model(self, model_name: str):
        """Remove a model from the ensemble."""
        if model_name in self.base_models:
            del self.base_models[model_name]
            if model_name in self.model_names:
                self.model_names.remove(model_name)
    
    def train(self, data: pd.DataFrame) -> bool:
        """Train all base models in the ensemble."""
        print(f"Training dynamic ensemble for {self.symbol} {self.timeframe}...")
        
        successful_models = 0
        for model_name, model in self.base_models.items():
            try:
                print(f"Training {model_name}...")
                if model.train(data):
                    successful_models += 1
                    print(f"✅ {model_name} trained successfully")
                else:
                    print(f"❌ {model_name} training failed")
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
        
        self._trained = successful_models > 0
        print(f"Dynamic ensemble training completed: {successful_models}/{len(self.base_models)} models trained")
        return self._trained
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate ensemble prediction with dynamic weighting."""
        if not self._trained:
            return self._default_prediction()
        
        try:
            # Get predictions from all base models
            base_predictions = []
            model_predictions = {}
            
            for model_name, model in self.base_models.items():
                if model.is_trained():
                    try:
                        pred = model.predict(data)
                        base_predictions.append(pred)
                        model_predictions[model_name] = pred
                    except Exception as e:
                        print(f"Warning: Error getting prediction from {model_name}: {e}")
            
            if not base_predictions:
                return self._default_prediction()
            
            # Detect current market regime
            current_regime = self.regime_detector.detect_regime(data)
            
            # Get dynamic weights based on performance
            performance_weights = self.performance_tracker.calculate_dynamic_weights(
                list(model_predictions.keys())
            )
            
            # Get regime-based preferences
            regime_preferences = self.regime_detector.get_regime_model_preferences(current_regime)
            
            # Combine weights (performance + regime awareness)
            final_weights = {}
            total_weight = 0
            
            for model_name in model_predictions.keys():
                performance_weight = performance_weights.get(model_name, 0.2)
                regime_preference = regime_preferences.get(model_name, 1.0)
                
                # Combine weights with regime preference
                combined_weight = performance_weight * regime_preference
                final_weights[model_name] = combined_weight
                total_weight += combined_weight
            
            # Normalize weights
            if total_weight > 0:
                final_weights = {k: v / total_weight for k, v in final_weights.items()}
            
            # Analyze consensus
            consensus_info = self.consensus_analyzer.analyze_predictions(base_predictions)
            
            # Generate ensemble prediction
            ensemble_pred = self._generate_ensemble_prediction(
                model_predictions, final_weights, consensus_info
            )
            
            # Add ensemble metadata
            ensemble_pred.update({
                'model_name': 'DynamicEnsemble',
                'regime': current_regime,
                'consensus_info': consensus_info,
                'model_weights': final_weights,
                'base_predictions': {k: v['direction'] for k, v in model_predictions.items()},
                'ensemble_method': 'dynamic_weighted'
            })
            
            return ensemble_pred
            
        except Exception as e:
            print(f"Error generating dynamic ensemble prediction: {e}")
            return self._default_prediction()
    
    def _generate_ensemble_prediction(self, model_predictions: Dict[str, Dict], 
                                    weights: Dict[str, float], 
                                    consensus_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final ensemble prediction from weighted model outputs."""
        
        # Weighted voting for direction
        direction_votes = defaultdict(float)
        total_confidence = 0
        
        for model_name, prediction in model_predictions.items():
            weight = weights.get(model_name, 0.2)
            direction = prediction['direction']
            confidence = prediction.get('confidence', 0.5)
            
            direction_votes[direction] += weight * confidence
            total_confidence += weight * confidence
        
        # Determine final direction
        if direction_votes:
            final_direction = max(direction_votes.keys(), key=lambda k: direction_votes[k])
            direction_confidence = direction_votes[final_direction] / max(total_confidence, 0.001)
        else:
            final_direction = 'HOLD'
            direction_confidence = 0.5
        
        # Adjust confidence based on consensus
        consensus_adjustment = consensus_info['agreement_score']
        uncertainty_penalty = consensus_info['uncertainty'] * 0.3
        
        final_confidence = direction_confidence * consensus_adjustment - uncertainty_penalty
        final_confidence = max(0.1, min(0.99, final_confidence))  # Bound confidence
        
        # Apply confidence threshold for HOLD bias
        if final_confidence < 0.55 and final_direction != 'HOLD':
            final_direction = 'HOLD'
            final_confidence = 0.5
        
        return {
            'direction': final_direction,
            'confidence': float(final_confidence),
            'probability': float(final_confidence),
            'timestamp': datetime.now(),
            'features_used': ['ensemble_prediction'],
            'direction_votes': dict(direction_votes),
            'raw_confidence': float(direction_confidence)
        }
    
    def update_performance(self, prediction: Dict[str, Any], actual_outcome: str):
        """Update performance tracking for ensemble and base models."""
        # Update ensemble performance
        self.performance_tracker.update_performance('DynamicEnsemble', prediction, actual_outcome)
        
        # Update base model performances if available
        if 'base_predictions' in prediction:
            for model_name, model_direction in prediction['base_predictions'].items():
                # Create a mock prediction dict for the base model
                base_pred = {
                    'direction': model_direction,
                    'confidence': 0.6,  # Default confidence if not available
                    'model_name': model_name
                }
                self.performance_tracker.update_performance(model_name, base_pred, actual_outcome)
    
    def save(self) -> bool:
        """Save ensemble state."""
        try:
            os.makedirs('model', exist_ok=True)
            
            # Save performance tracker
            perf_path = self.get_model_path('_performance.pkl')
            joblib.dump(self.performance_tracker, perf_path)
            
            # Save regime detector
            regime_path = self.get_model_path('_regime.pkl')
            joblib.dump(self.regime_detector, regime_path)
            
            # Save model list
            models_path = self.get_model_path('_models.pkl')
            joblib.dump(self.model_names, models_path)
            
            print(f"Dynamic ensemble state saved")
            return True
            
        except Exception as e:
            print(f"Error saving dynamic ensemble: {e}")
            return False
    
    def load(self) -> bool:
        """Load ensemble state."""
        try:
            # Load performance tracker
            perf_path = self.get_model_path('_performance.pkl')
            if os.path.exists(perf_path):
                self.performance_tracker = joblib.load(perf_path)
            
            # Load regime detector
            regime_path = self.get_model_path('_regime.pkl')
            if os.path.exists(regime_path):
                self.regime_detector = joblib.load(regime_path)
            
            # Load model list
            models_path = self.get_model_path('_models.pkl')
            if os.path.exists(models_path):
                self.model_names = joblib.load(models_path)
            
            self._trained = True
            print(f"Dynamic ensemble state loaded")
            return True
            
        except Exception as e:
            print(f"Error loading dynamic ensemble: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return ensemble model information."""
        return {
            'name': 'DynamicEnsemble',
            'type': 'advanced_ensemble',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'trained': self._trained,
            'base_models': list(self.base_models.keys()),
            'features': ['dynamic_weighting', 'regime_awareness', 'consensus_analysis'],
            'current_regime': self.regime_detector.current_regime,
            'recent_weights': getattr(self.performance_tracker, 'recent_weights', {})
        }
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when ensemble unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'DynamicEnsemble',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': 'Ensemble not trained or no models available'
        }


class AdvancedMetaLearner(BaseModel):
    """Advanced meta-learner with sophisticated feature engineering."""
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(symbol, timeframe)
        self.meta_model = None
        self.feature_scaler = None
        self.feature_columns = None
        
    def prepare_meta_features(self, data: pd.DataFrame, 
                             model_predictions: Dict[str, Dict]) -> pd.DataFrame:
        """Prepare meta-features from base model predictions and market data."""
        meta_features = []
        
        # Model prediction features
        for model_name, pred in model_predictions.items():
            direction = pred['direction']
            confidence = pred.get('confidence', 0.5)
            
            # One-hot encode directions
            meta_features.extend([
                1 if direction == 'BUY' else 0,
                1 if direction == 'HOLD' else 0,
                1 if direction == 'SELL' else 0
            ])
            
            # Confidence scores
            meta_features.append(confidence)
            
            # Confidence-weighted direction scores
            if direction == 'BUY':
                meta_features.extend([confidence, 0, 0])
            elif direction == 'SELL':
                meta_features.extend([0, 0, confidence])
            else:  # HOLD
                meta_features.extend([0, confidence, 0])
        
        # Agreement features
        directions = [pred['direction'] for pred in model_predictions.values()]
        confidences = [pred.get('confidence', 0.5) for pred in model_predictions.values()]
        
        # Model agreement metrics
        direction_counts = pd.Series(directions).value_counts()
        agreement_score = direction_counts.iloc[0] / len(directions) if len(directions) > 0 else 0
        confidence_std = np.std(confidences) if len(confidences) > 1 else 0
        
        meta_features.extend([
            agreement_score,
            np.mean(confidences),
            confidence_std,
            len(set(directions)),  # Number of unique predictions
        ])
        
        # Market context features
        if len(data) >= 20:
            recent_data = data.tail(20)
            returns = recent_data['close'].pct_change().dropna()
            
            market_features = [
                returns.mean(),  # Recent return trend
                returns.std(),   # Recent volatility
                (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20] if len(data) >= 20 else 0,  # 20-period return
                data['volume'].iloc[-5:].mean() / data['volume'].iloc[-20:-5].mean() if len(data) >= 20 else 1,  # Volume ratio
            ]
            meta_features.extend(market_features)
        else:
            # Default market features for insufficient data
            meta_features.extend([0, 0.01, 0, 1])
        
        return pd.DataFrame([meta_features])
    
    def train(self, data: pd.DataFrame, model_predictions_history: List[Dict]) -> bool:
        """Train meta-learner using historical model predictions and outcomes."""
        if not HAS_SKLEARN:
            print("Cannot train Advanced Meta-learner: sklearn not available")
            return False
        
        try:
            print(f"Training Advanced Meta-learner for {self.symbol} {self.timeframe}...")
            
            if len(model_predictions_history) < 50:
                print(f"Insufficient prediction history for meta-learner: {len(model_predictions_history)} samples")
                return False
            
            # Prepare training data
            X_list = []
            y_list = []
            
            for entry in model_predictions_history:
                if 'model_predictions' in entry and 'actual_outcome' in entry and 'data' in entry:
                    meta_features = self.prepare_meta_features(entry['data'], entry['model_predictions'])
                    X_list.append(meta_features.iloc[0].values)
                    
                    # Target encoding
                    actual = entry['actual_outcome']
                    target = 2 if actual == 'BUY' else 0 if actual == 'SELL' else 1  # BUY=2, HOLD=1, SELL=0
                    y_list.append(target)
            
            if len(X_list) < 30:
                print("Insufficient valid training samples for meta-learner")
                return False
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            # Feature scaling
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train-test split
            split_idx = int(0.8 * len(X_scaled))
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train meta-model (ensemble of different algorithms)
            self.meta_model = {
                'lgb': lgb.LGBMClassifier(
                    n_estimators=100,
                    num_leaves=31,
                    learning_rate=0.1,
                    random_state=42
                ),
                'rf': RandomForestClassifier(
                    n_estimators=50,
                    max_depth=8,
                    random_state=42
                ),
                'lr': LogisticRegression(
                    random_state=42,
                    max_iter=1000
                )
            }
            
            # Train all meta-models
            for name, model in self.meta_model.items():
                model.fit(X_train, y_train)
                train_acc = model.score(X_train, y_train)
                test_acc = model.score(X_test, y_test)
                print(f"Meta-model {name}: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")
            
            self.feature_columns = [f'feature_{i}' for i in range(X.shape[1])]
            self._trained = True
            
            print("Advanced Meta-learner training completed")
            return True
            
        except Exception as e:
            print(f"Error training Advanced Meta-learner: {e}")
            return False
    
    def predict(self, data: pd.DataFrame, model_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate meta-learner prediction."""
        if not self._trained or not self.meta_model:
            return self._default_prediction()
        
        try:
            # Prepare meta-features
            meta_features = self.prepare_meta_features(data, model_predictions)
            X = self.feature_scaler.transform(meta_features)
            
            # Get predictions from all meta-models
            predictions = {}
            probabilities = {}
            
            for name, model in self.meta_model.items():
                pred_proba = model.predict_proba(X)[0]
                pred_class = np.argmax(pred_proba)
                predictions[name] = pred_class
                probabilities[name] = pred_proba
            
            # Ensemble meta-model predictions (majority vote with confidence weighting)
            final_proba = np.mean([probabilities[name] for name in self.meta_model.keys()], axis=0)
            final_class = np.argmax(final_proba)
            confidence = np.max(final_proba)
            
            # Map class to direction
            class_to_direction = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            direction = class_to_direction[final_class]
            
            # Apply confidence threshold
            if confidence < 0.6:
                direction = 'HOLD'
                confidence = final_proba[1]  # Use HOLD probability
            
            return {
                'direction': direction,
                'confidence': float(confidence),
                'probability': float(confidence),
                'model_name': 'AdvancedMetaLearner',
                'timestamp': datetime.now(),
                'features_used': self.feature_columns or [],
                'class_probabilities': {
                    'SELL': float(final_proba[0]),
                    'HOLD': float(final_proba[1]),
                    'BUY': float(final_proba[2])
                },
                'meta_model_predictions': {
                    name: class_to_direction[pred] for name, pred in predictions.items()
                }
            }
            
        except Exception as e:
            print(f"Error making Advanced Meta-learner prediction: {e}")
            return self._default_prediction()
    
    def save(self) -> bool:
        """Save meta-learner model."""
        try:
            os.makedirs('model', exist_ok=True)
            
            model_path = self.get_model_path('.pkl')
            scaler_path = self.get_model_path('_scaler.pkl')
            features_path = self.get_model_path('_features.pkl')
            
            joblib.dump(self.meta_model, model_path)
            if self.feature_scaler:
                joblib.dump(self.feature_scaler, scaler_path)
            if self.feature_columns:
                joblib.dump(self.feature_columns, features_path)
            
            print(f"Advanced Meta-learner saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving Advanced Meta-learner: {e}")
            return False
    
    def load(self) -> bool:
        """Load meta-learner model."""
        try:
            model_path = self.get_model_path('.pkl')
            scaler_path = self.get_model_path('_scaler.pkl')
            features_path = self.get_model_path('_features.pkl')
            
            if not os.path.exists(model_path):
                print(f"Advanced Meta-learner model file not found: {model_path}")
                return False
            
            self.meta_model = joblib.load(model_path)
            
            if os.path.exists(scaler_path):
                self.feature_scaler = joblib.load(scaler_path)
            
            if os.path.exists(features_path):
                self.feature_columns = joblib.load(features_path)
            
            self._trained = True
            print(f"Advanced Meta-learner loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading Advanced Meta-learner: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return meta-learner model information."""
        return {
            'name': 'AdvancedMetaLearner',
            'type': 'meta_learning_ensemble',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'trained': self._trained,
            'meta_models': list(self.meta_model.keys()) if self.meta_model else [],
            'features': self.feature_columns or [],
            'available': HAS_SKLEARN
        }
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when meta-learner unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'AdvancedMetaLearner',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': 'Meta-learner not trained or unavailable'
        }