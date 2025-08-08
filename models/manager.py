# models/manager.py
"""
Centralized model management for loading, training, and prediction coordination.
"""

import pandas as pd
import os
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .base import BaseModel, create_model
from .ml_models import LSTMModel, LightGBMModel, XGBoostModel, RandomForestModel, SVMModel
from .ensemble import MetaLearner, VotingEnsemble, WeightedVotingEnsemble
from .advanced_ensemble import DynamicEnsemble, AdvancedMetaLearner
from .evaluation import ComprehensiveEvaluator

# Try to import config system, fall back to defaults if not available
try:
    from config import ConfigManager, ModelConfig
    CONFIG_AVAILABLE = True
except ImportError:
    try:
        from config.manager import ConfigManager
        from config.settings import ModelConfig
        CONFIG_AVAILABLE = True
    except ImportError:
        CONFIG_AVAILABLE = False
        print("Warning: Configuration system not available, using defaults")

# Define ModelConfig for type hints
if not CONFIG_AVAILABLE:
    class ModelConfig:
        def __init__(self):
            self.models_to_use = ["lstm", "lightgbm", "xgboost"]
            self.ensemble_method = "meta_learner"
            self.confidence_threshold = 0.6


class ModelManager:
    """Centralized manager for all ML models."""
    
    def __init__(self, symbol: str, timeframe: str, config: Optional[Union[ModelConfig, Any]] = None):
        """Initialize ModelManager with symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., 'XAU/USD')
            timeframe: Timeframe (e.g., '5m', '1h')
            config: Optional ModelConfig, will load from ConfigManager if available
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.models = {}
        self.ensemble_models = {}
        
        # Load configuration
        if config:
            self.config = config
        elif CONFIG_AVAILABLE:
            try:
                config_manager = ConfigManager()
                app_config = config_manager.get_config()
                self.config = app_config.model if app_config else ModelConfig()
            except:
                self.config = ModelConfig()
        else:
            # Default configuration
            self.config = self._default_config()
        
        self._initialize_models()
    
    def _default_config(self) -> Any:
        """Create default configuration when config system unavailable."""
        class DefaultModelConfig:
            def __init__(self):
                self.models_to_use = ["lstm", "lightgbm", "xgboost", "randomforest", "svm"]
                self.ensemble_method = "dynamic_ensemble"
                self.confidence_threshold = 0.6
                self.use_advanced_features = True
                self.enable_regime_detection = True
        
        return DefaultModelConfig()
    
    def _initialize_models(self):
        """Initialize models based on configuration."""
        print(f"Initializing models for {self.symbol} {self.timeframe}")
        
        # Initialize individual models - expanded to include new models
        for model_type in self.config.models_to_use:
            if model_type.lower() in ['lstm', 'lightgbm', 'lgb', 'xgboost', 'xgb', 'randomforest', 'rf', 'svm', 'svc']:
                try:
                    model = create_model(model_type, self.symbol, self.timeframe)
                    self.models[model_type.lower()] = model
                    print(f"Initialized {model_type} model")
                except Exception as e:
                    print(f"Error initializing {model_type} model: {e}")
            elif model_type.lower() in ['meta', 'ensemble', 'dynamic']:
                # Ensemble models will be initialized separately
                continue
        
        # Initialize ensemble models based on configuration
        ensemble_method = getattr(self.config, 'ensemble_method', 'dynamic_ensemble')
        
        if ensemble_method == 'dynamic_ensemble' or 'dynamic' in [m.lower() for m in self.config.models_to_use]:
            self.ensemble_models['dynamic_ensemble'] = DynamicEnsemble(self.symbol, self.timeframe)
            print("Initialized Dynamic Ensemble")
        
        if ensemble_method == 'advanced_meta' or 'advanced_meta' in [m.lower() for m in self.config.models_to_use]:
            self.ensemble_models['advanced_meta'] = AdvancedMetaLearner(self.symbol, self.timeframe)
            print("Initialized Advanced Meta Learner")
        
        # Legacy meta learner for backward compatibility
        if 'meta' in [m.lower() for m in self.config.models_to_use]:
            self.ensemble_models['meta'] = MetaLearner(self.symbol, self.timeframe)
            print("Initialized Meta Learner")
        
        # Initialize comprehensive evaluator
        self.evaluator = ComprehensiveEvaluator()
        print("Initialized Comprehensive Evaluator")
    
    def load_all_models(self) -> bool:
        """Load all available models from disk.
        
        Returns:
            bool: True if at least one model loaded successfully
        """
        print(f"Loading models for {self.symbol} {self.timeframe}...")
        loaded_count = 0
        
        # Load individual models
        for model_name, model in self.models.items():
            try:
                if model.load():
                    loaded_count += 1
                    print(f"✅ {model_name} model loaded")
                else:
                    print(f"❌ {model_name} model not found or failed to load")
            except Exception as e:
                print(f"❌ Error loading {model_name} model: {e}")
        
        # Load ensemble models
        for ensemble_name, ensemble in self.ensemble_models.items():
            try:
                if ensemble.load():
                    loaded_count += 1
                    print(f"✅ {ensemble_name} ensemble loaded")
                else:
                    print(f"❌ {ensemble_name} ensemble not found or failed to load")
            except Exception as e:
                print(f"❌ Error loading {ensemble_name} ensemble: {e}")
        
        success = loaded_count > 0
        print(f"Loaded {loaded_count} models successfully")
        return success
    
    def train_all_models(self, data: pd.DataFrame) -> Dict[str, bool]:
        """Train all models with provided data.
        
        Args:
            data: Training data as pandas DataFrame
            
        Returns:
            Dict[str, bool]: Training results for each model
        """
        print(f"Training all models for {self.symbol} {self.timeframe}...")
        results = {}
        
        # Train individual models
        for model_name, model in self.models.items():
            try:
                print(f"\nTraining {model_name} model...")
                success = model.train(data)
                results[model_name] = success
                
                if success:
                    # Save model after successful training
                    model.save()
                    print(f"✅ {model_name} model trained and saved")
                else:
                    print(f"❌ {model_name} model training failed")
                    
            except Exception as e:
                print(f"❌ Error training {model_name} model: {e}")
                results[model_name] = False
        
        # Note: Meta learner training requires predictions from other models
        # This would be handled separately in a full training pipeline
        
        return results
    
    def get_predictions(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get predictions from all trained models.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Dict[str, Dict]: Predictions from each model
        """
        predictions = {}
        
        # Get predictions from individual models
        for model_name, model in self.models.items():
            try:
                if model.is_trained():
                    pred = model.predict(data)
                    predictions[model_name] = pred
                    print(f"{model_name}: {pred['direction']} (confidence: {pred['confidence']:.3f})")
                else:
                    print(f"{model_name}: Model not trained")
                    predictions[model_name] = {
                        'direction': 'HOLD',
                        'confidence': 0.0,
                        'error': 'Model not trained'
                    }
            except Exception as e:
                print(f"Error getting {model_name} prediction: {e}")
                predictions[model_name] = {
                    'direction': 'HOLD',
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        return predictions
    
    def get_ensemble_prediction(self, data: pd.DataFrame, method: Optional[str] = None) -> Dict[str, Any]:
        """Get ensemble prediction using specified method.
        
        Args:
            data: Input data for prediction
            method: Ensemble method ('dynamic_ensemble', 'advanced_meta', 'meta_learner', 'voting', 'weighted_voting')
                   Uses config default if not specified
            
        Returns:
            Dict: Ensemble prediction
        """
        if method is None:
            method = getattr(self.config, 'ensemble_method', 'dynamic_ensemble')
        
        try:
            # Advanced Dynamic Ensemble (Phase 3)
            if method == 'dynamic_ensemble' and 'dynamic_ensemble' in self.ensemble_models:
                dynamic_ensemble = self.ensemble_models['dynamic_ensemble']
                if dynamic_ensemble.is_trained():
                    return dynamic_ensemble.predict(data)
                else:
                    print("Dynamic ensemble not trained, falling back to advanced meta")
                    method = 'advanced_meta'
            
            # Advanced Meta Learner (Phase 3)
            if method == 'advanced_meta' and 'advanced_meta' in self.ensemble_models:
                advanced_meta = self.ensemble_models['advanced_meta']
                if advanced_meta.is_trained():
                    # Get individual model predictions for meta learner
                    individual_preds = self.get_predictions(data)
                    trained_preds = {k: v for k, v in individual_preds.items() 
                                   if 'error' not in v and v.get('confidence', 0) > 0}
                    
                    if trained_preds:
                        return advanced_meta.predict(data, trained_preds)
                    else:
                        print("No trained models available for advanced meta learner")
                        method = 'voting'
                else:
                    print("Advanced meta learner not trained, falling back to meta learner")
                    method = 'meta_learner'
            
            # Legacy Meta Learner
            if method == 'meta_learner' and 'meta' in self.ensemble_models:
                meta_learner = self.ensemble_models['meta']
                if meta_learner.is_trained():
                    return meta_learner.predict(data)
                else:
                    print("Meta learner not trained, falling back to voting")
                    method = 'voting'
            
            if method == 'voting':
                trained_models = [model for model in self.models.values() if model.is_trained()]
                if trained_models:
                    ensemble = VotingEnsemble(trained_models)
                    return ensemble.predict(data)
                else:
                    return self._default_ensemble_prediction("No trained models for voting")
            
            elif method == 'weighted_voting':
                trained_models = [model for model in self.models.values() if model.is_trained()]
                if trained_models:
                    ensemble = WeightedVotingEnsemble(trained_models)
                    return ensemble.predict(data)
                else:
                    return self._default_ensemble_prediction("No trained models for weighted voting")
            
            else:
                return self._default_ensemble_prediction(f"Unknown ensemble method: {method}")
                
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            return self._default_ensemble_prediction(str(e))
    
    def setup_dynamic_ensemble(self) -> bool:
        """Set up dynamic ensemble by adding all trained base models."""
        if 'dynamic_ensemble' not in self.ensemble_models:
            self.ensemble_models['dynamic_ensemble'] = DynamicEnsemble(self.symbol, self.timeframe)
        
        dynamic_ensemble = self.ensemble_models['dynamic_ensemble']
        
        # Add all trained base models to the ensemble
        added_models = 0
        for model_name, model in self.models.items():
            if model.is_trained():
                dynamic_ensemble.add_model(model)
                added_models += 1
                print(f"Added {model_name} to dynamic ensemble")
        
        if added_models > 0:
            dynamic_ensemble._trained = True
            print(f"Dynamic ensemble ready with {added_models} base models")
            return True
        else:
            print("No trained models available for dynamic ensemble")
            return False
    
    def update_ensemble_performance(self, prediction: Dict[str, Any], actual_outcome: str):
        """Update ensemble performance tracking."""
        # Update dynamic ensemble performance if available
        if 'dynamic_ensemble' in self.ensemble_models:
            dynamic_ensemble = self.ensemble_models['dynamic_ensemble']
            dynamic_ensemble.update_performance(prediction, actual_outcome)
    
    def evaluate_all_models(self, data: pd.DataFrame, comprehensive: bool = True) -> Dict[str, Any]:
        """Evaluate all models using the comprehensive evaluation framework."""
        print("Starting comprehensive evaluation of all models...")
        
        evaluation_results = {
            'evaluation_timestamp': datetime.now(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'individual_evaluations': {},
            'ensemble_evaluations': {},
            'model_comparison': {}
        }
        
        # Evaluate individual models
        for model_name, model in self.models.items():
            if model.is_trained():
                print(f"Evaluating {model_name}...")
                try:
                    eval_result = self.evaluator.evaluate_model(model, data, comprehensive)
                    evaluation_results['individual_evaluations'][model_name] = eval_result
                except Exception as e:
                    print(f"Error evaluating {model_name}: {e}")
                    evaluation_results['individual_evaluations'][model_name] = {'error': str(e)}
        
        # Evaluate ensemble models
        for ensemble_name, ensemble in self.ensemble_models.items():
            if ensemble.is_trained():
                print(f"Evaluating {ensemble_name}...")
                try:
                    eval_result = self.evaluator.evaluate_model(ensemble, data, comprehensive)
                    evaluation_results['ensemble_evaluations'][ensemble_name] = eval_result
                except Exception as e:
                    print(f"Error evaluating {ensemble_name}: {e}")
                    evaluation_results['ensemble_evaluations'][ensemble_name] = {'error': str(e)}
        
        # Compare all models
        all_models = []
        for model in self.models.values():
            if model.is_trained():
                all_models.append(model)
        for ensemble in self.ensemble_models.values():
            if ensemble.is_trained():
                all_models.append(ensemble)
        
        if all_models:
            try:
                comparison_result = self.evaluator.compare_models(all_models, data)
                evaluation_results['model_comparison'] = comparison_result
            except Exception as e:
                print(f"Error in model comparison: {e}")
                evaluation_results['model_comparison'] = {'error': str(e)}
        
        return evaluation_results
    
    def get_multi_horizon_predictions(self, data: pd.DataFrame, 
                                     horizons: List[int] = [5, 20, 100]) -> Dict[str, Dict]:
        """Get predictions for multiple time horizons."""
        multi_horizon_preds = {}
        
        for horizon in horizons:
            print(f"Generating {horizon}-period ahead predictions...")
            
            # For multi-horizon, we would ideally train models for each horizon
            # For now, we'll use the current models and adjust confidence based on horizon
            predictions = self.get_predictions(data)
            
            # Adjust confidence for longer horizons (typically less reliable)
            horizon_factor = max(0.3, 1.0 - (horizon - 5) * 0.01)  # Decrease confidence for longer horizons
            
            adjusted_predictions = {}
            for model_name, pred in predictions.items():
                if 'error' not in pred:
                    adjusted_pred = pred.copy()
                    adjusted_pred['confidence'] *= horizon_factor
                    adjusted_pred['horizon'] = horizon
                    adjusted_predictions[model_name] = adjusted_pred
            
            # Get ensemble prediction for this horizon
            if adjusted_predictions:
                ensemble_pred = self.get_ensemble_prediction(data)
                if 'error' not in ensemble_pred:
                    ensemble_pred['confidence'] *= horizon_factor
                    ensemble_pred['horizon'] = horizon
                    adjusted_predictions['ensemble'] = ensemble_pred
            
            multi_horizon_preds[f'{horizon}_period'] = adjusted_predictions
        
        return multi_horizon_preds
    
    def save_all_models(self) -> Dict[str, bool]:
        """Save all trained models.
        
        Returns:
            Dict[str, bool]: Save results for each model
        """
        results = {}
        
        # Save individual models
        for model_name, model in self.models.items():
            try:
                if model.is_trained():
                    success = model.save()
                    results[model_name] = success
                    if success:
                        print(f"✅ {model_name} model saved")
                    else:
                        print(f"❌ {model_name} model save failed")
                else:
                    results[model_name] = False
                    print(f"❌ {model_name} model not trained, skipping save")
            except Exception as e:
                print(f"❌ Error saving {model_name} model: {e}")
                results[model_name] = False
        
        # Save ensemble models
        for ensemble_name, ensemble in self.ensemble_models.items():
            try:
                if ensemble.is_trained():
                    success = ensemble.save()
                    results[ensemble_name] = success
                    if success:
                        print(f"✅ {ensemble_name} ensemble saved")
                    else:
                        print(f"❌ {ensemble_name} ensemble save failed")
                else:
                    results[ensemble_name] = False
                    print(f"❌ {ensemble_name} ensemble not trained, skipping save")
            except Exception as e:
                print(f"❌ Error saving {ensemble_name} ensemble: {e}")
                results[ensemble_name] = False
        
        return results
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models.
        
        Returns:
            Dict: Status information for all models
        """
        status = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'ensembles': {},
            'summary': {
                'total_models': len(self.models) + len(self.ensemble_models),
                'trained_models': 0,
                'available_models': 0
            }
        }
        
        # Check individual models
        for model_name, model in self.models.items():
            model_info = model.get_model_info()
            model_path = model_info.get('model_path', '')
            file_exists = os.path.exists(model_path) if model_path else False
            
            status['models'][model_name] = {
                'trained': model.is_trained(),
                'file_exists': file_exists,
                'model_path': model_path,
                'info': model_info
            }
            
            if model.is_trained():
                status['summary']['trained_models'] += 1
            if file_exists:
                status['summary']['available_models'] += 1
        
        # Check ensemble models
        for ensemble_name, ensemble in self.ensemble_models.items():
            ensemble_info = ensemble.get_model_info()
            ensemble_path = ensemble_info.get('model_path', '')
            file_exists = os.path.exists(ensemble_path) if ensemble_path else False
            
            status['ensembles'][ensemble_name] = {
                'trained': ensemble.is_trained(),
                'file_exists': file_exists,
                'model_path': ensemble_path,
                'info': ensemble_info
            }
            
            if ensemble.is_trained():
                status['summary']['trained_models'] += 1
            if file_exists:
                status['summary']['available_models'] += 1
        
        return status
    
    def _default_ensemble_prediction(self, error_msg: str = "") -> Dict[str, Any]:
        """Return default ensemble prediction when unavailable."""
        return {
            'direction': 'HOLD',
            'confidence': 0.0,
            'probability': 0.0,
            'model_name': 'Ensemble',
            'timestamp': datetime.now(),
            'features_used': [],
            'error': error_msg or 'Ensemble unavailable'
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return list(self.models.keys()) + list(self.ensemble_models.keys())
    
    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """Get specific model instance by name."""
        if model_name in self.models:
            return self.models[model_name]
        elif model_name in self.ensemble_models:
            return self.ensemble_models[model_name]
        else:
            return None