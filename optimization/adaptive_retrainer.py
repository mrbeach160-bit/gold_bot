"""
Adaptive Model Retraining System

Continuous model optimization with performance monitoring, drift detection,
and automated retraining for evolving market conditions.
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass
import joblib

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class RetrainingConfig:
    """Configuration for adaptive retraining"""
    min_accuracy_threshold: float = 0.55
    drift_threshold: float = 0.7
    min_samples_for_retrain: int = 1000
    max_retrain_frequency_hours: int = 6
    validation_split: float = 0.2
    max_model_age_days: int = 30
    performance_window_hours: int = 24
    enable_incremental_learning: bool = True
    enable_online_validation: bool = True

class PerformanceMonitor:
    """Monitor model performance for retraining decisions"""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.performance_history = []
        self.validation_results = []
        self.drift_scores = []
        
    def record_performance(self, 
                          prediction: Dict[str, Any], 
                          actual_result: Optional[bool] = None,
                          market_data: Optional[Dict[str, Any]] = None):
        """Record prediction performance"""
        try:
            timestamp = datetime.now()
            
            record = {
                'timestamp': timestamp,
                'prediction': prediction,
                'actual_result': actual_result,
                'market_data': market_data
            }
            
            self.performance_history.append(record)
            
            # Keep only recent history
            cutoff_time = timestamp - timedelta(hours=self.config.performance_window_hours * 2)
            self.performance_history = [
                r for r in self.performance_history 
                if r['timestamp'] > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error recording performance: {e}")
    
    def get_recent_metrics(self) -> Dict[str, Any]:
        """Get recent performance metrics"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.config.performance_window_hours)
            recent_records = [
                r for r in self.performance_history 
                if r['timestamp'] > cutoff_time and r['actual_result'] is not None
            ]
            
            if not recent_records:
                return {'accuracy': 0.0, 'sample_count': 0, 'coverage': 0.0}
            
            # Calculate accuracy
            correct_predictions = sum(
                1 for r in recent_records 
                if (r['prediction'].get('direction') == 'BUY' and r['actual_result']) or
                   (r['prediction'].get('direction') == 'SELL' and not r['actual_result']) or
                   (r['prediction'].get('direction') == 'HOLD')
            )
            
            accuracy = correct_predictions / len(recent_records)
            
            # Calculate confidence distribution
            confidences = [r['prediction'].get('confidence', 0) for r in recent_records]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                'accuracy': accuracy,
                'sample_count': len(recent_records),
                'avg_confidence': avg_confidence,
                'coverage': len(recent_records) / (self.config.performance_window_hours * 60),  # predictions per hour
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error calculating recent metrics: {e}")
            return {'accuracy': 0.0, 'sample_count': 0, 'coverage': 0.0}

class DriftDetector:
    """Detect concept drift in market data and features"""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.reference_distribution = {}
        self.recent_distribution = {}
        self.drift_history = []
        self.feature_importance_history = []
        
    def update_distributions(self, features: Dict[str, Any], is_reference: bool = False):
        """Update feature distributions for drift detection"""
        try:
            target_dist = self.reference_distribution if is_reference else self.recent_distribution
            
            for feature_name, value in features.items():
                if not isinstance(value, (int, float)):
                    continue
                
                if feature_name not in target_dist:
                    target_dist[feature_name] = []
                
                target_dist[feature_name].append(value)
                
                # Keep limited history
                if len(target_dist[feature_name]) > 1000:
                    target_dist[feature_name] = target_dist[feature_name][-1000:]
                    
        except Exception as e:
            logger.error(f"Error updating distributions: {e}")
    
    def detect_drift(self) -> Dict[str, Any]:
        """Detect concept drift using statistical tests"""
        try:
            drift_results = {
                'drift_detected': False,
                'drift_score': 0.0,
                'affected_features': [],
                'timestamp': datetime.now()
            }
            
            if not self.reference_distribution or not self.recent_distribution:
                return drift_results
            
            drift_scores = []
            affected_features = []
            
            # Compare distributions for each feature
            for feature_name in self.reference_distribution.keys():
                if feature_name not in self.recent_distribution:
                    continue
                
                ref_data = self.reference_distribution[feature_name]
                recent_data = self.recent_distribution[feature_name]
                
                if len(ref_data) < 50 or len(recent_data) < 50:
                    continue
                
                # Simple drift detection using mean and variance comparison
                ref_mean = np.mean(ref_data)
                recent_mean = np.mean(recent_data)
                ref_std = np.std(ref_data)
                recent_std = np.std(recent_data)
                
                # Normalized difference in means
                if ref_std > 0:
                    mean_drift = abs(recent_mean - ref_mean) / ref_std
                else:
                    mean_drift = 0
                
                # Variance ratio
                if ref_std > 0 and recent_std > 0:
                    var_ratio = max(recent_std / ref_std, ref_std / recent_std)
                else:
                    var_ratio = 1
                
                # Combined drift score
                feature_drift_score = mean_drift + (var_ratio - 1)
                
                if feature_drift_score > 1.0:  # Threshold for significant drift
                    affected_features.append({
                        'feature': feature_name,
                        'drift_score': feature_drift_score,
                        'mean_shift': mean_drift,
                        'variance_ratio': var_ratio
                    })
                
                drift_scores.append(feature_drift_score)
            
            # Overall drift score
            overall_drift_score = np.mean(drift_scores) if drift_scores else 0
            
            # Check if drift threshold exceeded
            if overall_drift_score > self.config.drift_threshold:
                drift_results['drift_detected'] = True
                drift_results['drift_score'] = overall_drift_score
                drift_results['affected_features'] = affected_features
                
                logger.warning(f"Concept drift detected: score={overall_drift_score:.3f}, "
                             f"features affected: {len(affected_features)}")
            
            # Record drift history
            self.drift_history.append(drift_results)
            if len(self.drift_history) > 100:
                self.drift_history = self.drift_history[-100:]
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return {'drift_detected': False, 'drift_score': 0.0, 'affected_features': []}

class IncrementalLearner:
    """Incremental learning for continuous model updates"""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.training_buffer = []
        self.last_update = datetime.now()
        self.update_count = 0
        
    def add_sample(self, features: Dict[str, Any], target: Optional[bool] = None):
        """Add new training sample to buffer"""
        try:
            if target is None:
                return
            
            sample = {
                'features': features,
                'target': target,
                'timestamp': datetime.now()
            }
            
            self.training_buffer.append(sample)
            
            # Keep buffer size manageable
            if len(self.training_buffer) > self.config.min_samples_for_retrain * 2:
                self.training_buffer = self.training_buffer[-self.config.min_samples_for_retrain:]
                
        except Exception as e:
            logger.error(f"Error adding training sample: {e}")
    
    def should_update(self) -> bool:
        """Check if incremental update should be performed"""
        try:
            # Check buffer size
            if len(self.training_buffer) < 100:  # Minimum batch size
                return False
            
            # Check time since last update
            time_since_update = datetime.now() - self.last_update
            if time_since_update.total_seconds() < 3600:  # Wait at least 1 hour
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking update condition: {e}")
            return False
    
    def get_training_data(self) -> Optional[pd.DataFrame]:
        """Get training data from buffer"""
        try:
            if not self.training_buffer:
                return None
            
            # Convert buffer to DataFrame
            features_list = []
            targets = []
            
            for sample in self.training_buffer:
                features_list.append(sample['features'])
                targets.append(sample['target'])
            
            df = pd.DataFrame(features_list)
            df['target'] = targets
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return None
    
    def clear_buffer(self):
        """Clear training buffer after successful update"""
        self.training_buffer = []
        self.last_update = datetime.now()
        self.update_count += 1

class ModelSelector:
    """Select best performing models for ensemble"""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.model_performance = {}
        self.model_registry = {}
        
    def register_model(self, model_name: str, model_instance: Any, metadata: Dict[str, Any]):
        """Register a model for selection"""
        self.model_registry[model_name] = {
            'model': model_instance,
            'metadata': metadata,
            'creation_time': datetime.now(),
            'last_evaluation': None,
            'performance_history': []
        }
        
        logger.info(f"Model registered: {model_name}")
    
    def evaluate_model(self, model_name: str, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            if model_name not in self.model_registry:
                return {'error': 'Model not found'}
            
            model_info = self.model_registry[model_name]
            model = model_info['model']
            
            # Mock evaluation (replace with actual model evaluation)
            accuracy = np.random.uniform(0.5, 0.8)
            precision = np.random.uniform(0.5, 0.8)
            recall = np.random.uniform(0.5, 0.8)
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            evaluation_result = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'sample_count': len(test_data),
                'evaluation_time': datetime.now()
            }
            
            # Update model performance history
            model_info['last_evaluation'] = evaluation_result
            model_info['performance_history'].append(evaluation_result)
            
            # Keep limited history
            if len(model_info['performance_history']) > 50:
                model_info['performance_history'] = model_info['performance_history'][-50:]
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return {'error': str(e)}
    
    def select_best_models(self, max_models: int = 5) -> List[str]:
        """Select best performing models"""
        try:
            # Get models with recent evaluations
            evaluated_models = [
                (name, info) for name, info in self.model_registry.items()
                if info['last_evaluation'] is not None
            ]
            
            if not evaluated_models:
                return list(self.model_registry.keys())[:max_models]
            
            # Sort by performance (accuracy * f1_score)
            def performance_score(model_info):
                eval_result = model_info[1]['last_evaluation']
                return eval_result['accuracy'] * eval_result.get('f1_score', 0.5)
            
            evaluated_models.sort(key=performance_score, reverse=True)
            
            # Return top models
            selected_models = [name for name, _ in evaluated_models[:max_models]]
            
            logger.info(f"Selected models: {selected_models}")
            return selected_models
            
        except Exception as e:
            logger.error(f"Error selecting models: {e}")
            return list(self.model_registry.keys())[:max_models]

class AdaptiveRetrainer:
    """Main adaptive retraining system"""
    
    def __init__(self, 
                 config: Optional[RetrainingConfig] = None,
                 model_manager: Optional[Any] = None):
        
        self.config = config or RetrainingConfig()
        self.model_manager = model_manager
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor(self.config)
        self.drift_detector = DriftDetector(self.config)
        self.incremental_learner = IncrementalLearner(self.config)
        self.model_selector = ModelSelector(self.config)
        
        # State management
        self.last_retrain_time = datetime.now()
        self.retrain_count = 0
        self.running = False
        self.optimization_thread = None
        
        # Callbacks
        self.retrain_callbacks = []
        
        logger.info("Adaptive Retrainer initialized")
    
    def add_retrain_callback(self, callback: Callable):
        """Add callback for retraining events"""
        self.retrain_callbacks.append(callback)
    
    def start_continuous_optimization(self):
        """Start continuous optimization loop"""
        if self.running:
            return
        
        self.running = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()
        
        logger.info("Continuous optimization started")
    
    def stop_optimization(self):
        """Stop continuous optimization"""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=10)
        
        logger.info("Continuous optimization stopped")
    
    def _optimization_loop(self):
        """Main optimization loop"""
        while self.running:
            try:
                asyncio.run(self._optimization_cycle())
                time.sleep(3600)  # Check every hour
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    async def _optimization_cycle(self):
        """Single optimization cycle"""
        try:
            logger.debug("Running optimization cycle")
            
            # Get recent performance metrics
            metrics = self.performance_monitor.get_recent_metrics()
            
            # Check if retraining is needed
            should_retrain = await self._should_retrain(metrics)
            
            if should_retrain['retrain']:
                logger.info(f"Triggering retrain: {should_retrain['reason']}")
                await self._perform_retrain()
            
            # Check for incremental updates
            if self.config.enable_incremental_learning:
                if self.incremental_learner.should_update():
                    await self._perform_incremental_update()
            
            # Evaluate models periodically
            await self._evaluate_models()
            
        except Exception as e:
            logger.error(f"Optimization cycle error: {e}")
    
    async def _should_retrain(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if full retraining is needed"""
        try:
            reasons = []
            
            # Check accuracy threshold
            if metrics['accuracy'] < self.config.min_accuracy_threshold:
                reasons.append(f"Low accuracy: {metrics['accuracy']:.3f}")
            
            # Check drift
            drift_result = self.drift_detector.detect_drift()
            if drift_result['drift_detected']:
                reasons.append(f"Concept drift detected: {drift_result['drift_score']:.3f}")
            
            # Check time since last retrain
            time_since_retrain = datetime.now() - self.last_retrain_time
            if time_since_retrain.total_seconds() > (self.config.max_model_age_days * 24 * 3600):
                reasons.append("Model age exceeded maximum")
            
            # Check minimum frequency
            if time_since_retrain.total_seconds() < (self.config.max_retrain_frequency_hours * 3600):
                return {'retrain': False, 'reason': 'Too soon since last retrain'}
            
            # Check sample count
            if metrics['sample_count'] < self.config.min_samples_for_retrain:
                return {'retrain': False, 'reason': 'Insufficient samples'}
            
            should_retrain = len(reasons) > 0
            return {
                'retrain': should_retrain,
                'reason': '; '.join(reasons) if should_retrain else 'No retrain needed'
            }
            
        except Exception as e:
            logger.error(f"Error checking retrain conditions: {e}")
            return {'retrain': False, 'reason': f'Error: {str(e)}'}
    
    async def _perform_retrain(self):
        """Perform full model retraining"""
        try:
            logger.info("Starting model retraining")
            start_time = time.time()
            
            # Get training data
            training_data = self.incremental_learner.get_training_data()
            if training_data is None or len(training_data) < self.config.min_samples_for_retrain:
                logger.warning("Insufficient training data for retrain")
                return
            
            # Split data
            split_idx = int(len(training_data) * (1 - self.config.validation_split))
            train_data = training_data[:split_idx]
            val_data = training_data[split_idx:]
            
            # Mock retraining (replace with actual implementation)
            await asyncio.sleep(1)  # Simulate training time
            
            # Update state
            self.last_retrain_time = datetime.now()
            self.retrain_count += 1
            
            # Clear incremental learning buffer
            self.incremental_learner.clear_buffer()
            
            # Reset drift detector reference
            if not val_data.empty:
                for _, row in val_data.iterrows():
                    features = row.drop('target').to_dict()
                    self.drift_detector.update_distributions(features, is_reference=True)
            
            training_time = time.time() - start_time
            
            logger.info(f"Model retraining completed in {training_time:.2f}s, "
                       f"total retrains: {self.retrain_count}")
            
            # Notify callbacks
            for callback in self.retrain_callbacks:
                try:
                    callback({
                        'type': 'retrain_completed',
                        'training_time': training_time,
                        'sample_count': len(training_data),
                        'retrain_count': self.retrain_count
                    })
                except Exception as e:
                    logger.error(f"Retrain callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Retraining error: {e}")
    
    async def _perform_incremental_update(self):
        """Perform incremental model update"""
        try:
            logger.info("Performing incremental update")
            
            # Get recent data
            training_data = self.incremental_learner.get_training_data()
            if training_data is None or len(training_data) < 100:
                return
            
            # Mock incremental update
            await asyncio.sleep(0.5)  # Simulate update time
            
            # Clear buffer
            self.incremental_learner.clear_buffer()
            
            logger.info("Incremental update completed")
            
        except Exception as e:
            logger.error(f"Incremental update error: {e}")
    
    async def _evaluate_models(self):
        """Evaluate registered models"""
        try:
            # Get validation data
            training_data = self.incremental_learner.get_training_data()
            if training_data is None or len(training_data) < 100:
                return
            
            # Evaluate each registered model
            for model_name in self.model_selector.model_registry.keys():
                evaluation = self.model_selector.evaluate_model(model_name, training_data)
                if 'error' not in evaluation:
                    logger.debug(f"Model {model_name} accuracy: {evaluation['accuracy']:.3f}")
            
        except Exception as e:
            logger.error(f"Model evaluation error: {e}")
    
    def record_prediction(self, 
                         prediction: Dict[str, Any], 
                         actual_result: Optional[bool] = None,
                         features: Optional[Dict[str, Any]] = None):
        """Record prediction for performance monitoring and learning"""
        try:
            # Record for performance monitoring
            self.performance_monitor.record_performance(prediction, actual_result)
            
            # Update drift detection
            if features:
                self.drift_detector.update_distributions(features, is_reference=False)
            
            # Add to incremental learning buffer
            if features and actual_result is not None:
                self.incremental_learner.add_sample(features, actual_result)
                
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current retrainer status"""
        try:
            metrics = self.performance_monitor.get_recent_metrics()
            drift_status = self.drift_detector.detect_drift()
            
            return {
                'running': self.running,
                'last_retrain': self.last_retrain_time.isoformat(),
                'retrain_count': self.retrain_count,
                'recent_performance': metrics,
                'drift_status': drift_status,
                'buffer_size': len(self.incremental_learner.training_buffer),
                'registered_models': len(self.model_selector.model_registry),
                'config': {
                    'min_accuracy': self.config.min_accuracy_threshold,
                    'drift_threshold': self.config.drift_threshold,
                    'max_age_days': self.config.max_model_age_days
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    async def demo():
        # Create retrainer
        config = RetrainingConfig(
            min_accuracy_threshold=0.60,
            drift_threshold=0.5,
            min_samples_for_retrain=500
        )
        
        retrainer = AdaptiveRetrainer(config)
        
        # Add callback
        def on_retrain(event):
            print(f"Retrain event: {event}")
        
        retrainer.add_retrain_callback(on_retrain)
        
        # Start optimization
        retrainer.start_continuous_optimization()
        
        # Simulate some predictions
        for i in range(10):
            prediction = {
                'direction': 'BUY' if i % 2 == 0 else 'SELL',
                'confidence': 0.7 - (i * 0.05),
                'timestamp': datetime.now()
            }
            
            features = {
                'rsi': 50 + np.random.normal(0, 10),
                'macd': np.random.normal(0, 1),
                'price': 2000 + np.random.normal(0, 10)
            }
            
            actual_result = np.random.choice([True, False])
            
            retrainer.record_prediction(prediction, actual_result, features)
            
            await asyncio.sleep(1)
        
        # Get status
        status = retrainer.get_status()
        print("Retrainer status:", status)
        
        # Stop optimization
        retrainer.stop_optimization()
    
    # Run demo
    asyncio.run(demo())