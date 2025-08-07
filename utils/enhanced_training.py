"""
Enhanced Model Training Pipeline with Robustness and Validation

Provides comprehensive data quality validation, model training with failure recovery,
and training metrics tracking for production-ready model training.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Import enhanced utilities
try:
    from utils.logging_system import get_logger, get_performance_logger
    from utils.dependency_manager import get_dependency_status, is_available
    from utils.path_manager import get_model_path, ensure_path_exists
    from config.validation_enhanced import validate_complete_configuration
    ENHANCED_UTILS_AVAILABLE = True
except ImportError:
    ENHANCED_UTILS_AVAILABLE = False
    import logging
    logging.basicConfig(level=logging.INFO)

# Setup logging
if ENHANCED_UTILS_AVAILABLE:
    logger = get_logger("enhanced_training")
    perf_logger = get_performance_logger("training_performance")
else:
    logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Validates data quality before model training."""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_data_quality(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Comprehensive data quality validation.
        
        Args:
            data: OHLCV data to validate
            symbol: Trading symbol for context
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {},
            'recommendations': []
        }
        
        try:
            # Basic structure validation
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                results['errors'].append(f"Missing required columns: {missing_columns}")
                results['is_valid'] = False
                return results
            
            # Data size validation
            min_required_rows = 200
            if len(data) < min_required_rows:
                results['errors'].append(f"Insufficient data: {len(data)} rows (minimum {min_required_rows})")
                results['is_valid'] = False
                return results
            
            results['metrics']['total_rows'] = len(data)
            
            # Missing values validation
            missing_pct = (data[required_columns].isnull().sum() / len(data) * 100)
            max_missing_pct = 5.0  # 5% threshold
            
            for col, pct in missing_pct.items():
                if pct > max_missing_pct:
                    results['errors'].append(f"Too many missing values in {col}: {pct:.1f}%")
                    results['is_valid'] = False
                elif pct > 1.0:
                    results['warnings'].append(f"Missing values in {col}: {pct:.1f}%")
            
            results['metrics']['missing_values_pct'] = missing_pct.to_dict()
            
            # OHLC consistency validation
            ohlc_issues = 0
            
            # High should be >= max(open, close)
            high_invalid = data['high'] < data[['open', 'close']].max(axis=1)
            ohlc_issues += high_invalid.sum()
            
            # Low should be <= min(open, close)
            low_invalid = data['low'] > data[['open', 'close']].min(axis=1)
            ohlc_issues += low_invalid.sum()
            
            if ohlc_issues > 0:
                ohlc_error_pct = (ohlc_issues / len(data)) * 100
                if ohlc_error_pct > 1.0:
                    results['errors'].append(f"OHLC consistency errors: {ohlc_error_pct:.1f}%")
                    results['is_valid'] = False
                else:
                    results['warnings'].append(f"Minor OHLC inconsistencies: {ohlc_error_pct:.1f}%")
            
            results['metrics']['ohlc_consistency_errors'] = ohlc_issues
            
            # Price movement validation
            price_changes = data['close'].pct_change().dropna()
            
            # Check for suspicious price movements
            extreme_moves = abs(price_changes) > 0.1  # 10% moves
            extreme_count = extreme_moves.sum()
            
            if extreme_count > len(data) * 0.01:  # More than 1% of data
                results['warnings'].append(f"High number of extreme price movements: {extreme_count}")
            
            # Check for flat periods (no price movement)
            flat_periods = (price_changes == 0).sum()
            if flat_periods > len(data) * 0.1:  # More than 10% flat
                results['warnings'].append(f"High number of flat price periods: {flat_periods}")
            
            results['metrics']['extreme_movements'] = extreme_count
            results['metrics']['flat_periods'] = flat_periods
            results['metrics']['price_volatility'] = price_changes.std()
            
            # Volume validation
            if 'volume' in data.columns:
                zero_volume = (data['volume'] <= 0).sum()
                if zero_volume > len(data) * 0.1:
                    results['warnings'].append(f"High number of zero volume periods: {zero_volume}")
                
                results['metrics']['zero_volume_periods'] = zero_volume
                results['metrics']['avg_volume'] = data['volume'].mean()
            
            # Time series validation
            if hasattr(data.index, 'to_series'):
                time_diffs = data.index.to_series().diff().dropna()
                irregular_intervals = len(time_diffs.value_counts())
                
                if irregular_intervals > 3:  # More than 3 different intervals
                    results['warnings'].append("Irregular time intervals detected")
                
                results['metrics']['time_intervals'] = irregular_intervals
            
            # Add recommendations based on findings
            if results['warnings']:
                results['recommendations'].append("Consider data cleaning before training")
            
            if len(data) < 500:
                results['recommendations'].append("More historical data would improve model quality")
            
            if results['metrics'].get('price_volatility', 0) < 0.001:
                results['recommendations'].append("Low price volatility may reduce model effectiveness")
            
            logger.info(f"Data quality validation for {symbol}: {'PASSED' if results['is_valid'] else 'FAILED'}")
            
            return results
            
        except Exception as e:
            logger.error(f"Data quality validation error: {e}")
            results['errors'].append(f"Validation error: {str(e)}")
            results['is_valid'] = False
            return results


class TrainingMetricsTracker:
    """Tracks and manages training metrics and performance."""
    
    def __init__(self):
        self.metrics = {}
        self.training_start_time = None
        self.model_training_times = {}
    
    def start_training_session(self, session_info: Dict[str, Any]):
        """Start a new training session."""
        self.training_start_time = time.time()
        self.metrics = {
            'session_start': datetime.now().isoformat(),
            'session_info': session_info,
            'models': {},
            'overall_status': 'in_progress'
        }
        
        logger.info(f"Training session started: {session_info}")
    
    def start_model_training(self, model_name: str):
        """Start training for a specific model."""
        self.model_training_times[model_name] = time.time()
        self.metrics['models'][model_name] = {
            'status': 'training',
            'start_time': datetime.now().isoformat(),
            'errors': [],
            'warnings': []
        }
        
        if ENHANCED_UTILS_AVAILABLE:
            perf_logger.start_timer(f"model_training_{model_name}")
    
    def end_model_training(self, model_name: str, success: bool, 
                          validation_metrics: Dict[str, Any] = None,
                          error_message: str = None):
        """End training for a specific model."""
        if model_name in self.model_training_times:
            training_time = time.time() - self.model_training_times[model_name]
            del self.model_training_times[model_name]
        else:
            training_time = 0
        
        self.metrics['models'][model_name].update({
            'status': 'completed' if success else 'failed',
            'end_time': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'success': success
        })
        
        if validation_metrics:
            self.metrics['models'][model_name]['validation_metrics'] = validation_metrics
        
        if error_message:
            self.metrics['models'][model_name]['error_message'] = error_message
        
        if ENHANCED_UTILS_AVAILABLE:
            perf_logger.end_timer(f"model_training_{model_name}")
            perf_logger.log_metric(f"{model_name}_training_time", training_time)
            if validation_metrics and 'accuracy' in validation_metrics:
                perf_logger.log_metric(f"{model_name}_accuracy", validation_metrics['accuracy'])
        
        logger.info(f"Model {model_name} training {'completed' if success else 'failed'} in {training_time:.2f}s")
    
    def end_training_session(self):
        """End the training session."""
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            self.metrics['total_training_time_seconds'] = total_time
        
        # Calculate summary statistics
        successful_models = [name for name, info in self.metrics['models'].items() 
                           if info.get('success', False)]
        failed_models = [name for name, info in self.metrics['models'].items() 
                        if not info.get('success', False)]
        
        self.metrics['summary'] = {
            'total_models': len(self.metrics['models']),
            'successful_models': len(successful_models),
            'failed_models': len(failed_models),
            'success_rate': len(successful_models) / len(self.metrics['models']) if self.metrics['models'] else 0,
            'successful_model_names': successful_models,
            'failed_model_names': failed_models
        }
        
        self.metrics['overall_status'] = 'completed'
        self.metrics['session_end'] = datetime.now().isoformat()
        
        logger.info(f"Training session completed: {len(successful_models)}/{len(self.metrics['models'])} models successful")
    
    def save_metrics(self, filepath: str):
        """Save training metrics to file."""
        try:
            if ENHANCED_UTILS_AVAILABLE:
                filepath = ensure_path_exists(Path(filepath))
            
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            
            logger.info(f"Training metrics saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save training metrics: {e}")


class EnhancedTrainingPipeline:
    """
    Enhanced training pipeline with robustness features.
    """
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_validator = DataQualityValidator()
        self.metrics_tracker = TrainingMetricsTracker()
        self.trained_models = {}
        self.failed_models = {}
    
    def validate_environment(self) -> bool:
        """Validate that the environment is ready for training."""
        logger.info("Validating training environment...")
        
        if ENHANCED_UTILS_AVAILABLE:
            # Use enhanced configuration validation
            config_result = validate_complete_configuration()
            
            if not config_result.is_valid:
                logger.error("Environment validation failed:")
                for error in config_result.errors:
                    logger.error(f"  - {error}")
                return False
            
            if config_result.warnings:
                logger.warning("Environment validation warnings:")
                for warning in config_result.warnings:
                    logger.warning(f"  - {warning}")
            
            # Check dependencies
            dep_status = get_dependency_status()
            critical_missing = dep_status.get('critical_missing', [])
            
            if critical_missing:
                logger.error(f"Critical dependencies missing: {critical_missing}")
                return False
            
            logger.info("Environment validation passed ✓")
            return True
        else:
            # Basic validation
            logger.warning("Enhanced validation not available, performing basic checks")
            
            # Check if we can create model directory
            try:
                model_dir = Path('model')
                model_dir.mkdir(exist_ok=True)
                test_file = model_dir / 'test_write.tmp'
                test_file.write_text('test')
                test_file.unlink()
                logger.info("Basic environment validation passed ✓")
                return True
            except Exception as e:
                logger.error(f"Basic environment validation failed: {e}")
                return False
    
    def train_models_with_recovery(self, data: pd.DataFrame, 
                                  retry_failed: bool = True) -> Dict[str, bool]:
        """
        Train models with failure recovery and comprehensive tracking.
        
        Args:
            data: Training data
            retry_failed: Whether to retry failed models
            
        Returns:
            Dictionary of model training results
        """
        # Start training session
        session_info = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'data_rows': len(data),
            'retry_enabled': retry_failed
        }
        
        self.metrics_tracker.start_training_session(session_info)
        
        results = {}
        
        try:
            # Import model manager
            from models.manager import ModelManager
            
            # Initialize model manager
            model_manager = ModelManager(self.symbol, self.timeframe)
            available_models = model_manager.get_available_models()
            
            logger.info(f"Training {len(available_models)} models...")
            
            # Train each model individually with error handling
            for model_name in available_models:
                results[model_name] = self._train_single_model_with_recovery(
                    model_manager, model_name, data, retry_failed
                )
            
            # End training session
            self.metrics_tracker.end_training_session()
            
            # Save training metrics
            if ENHANCED_UTILS_AVAILABLE:
                metrics_file = get_model_path(f"training_metrics_{self.symbol}_{self.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            else:
                metrics_file = f"model/training_metrics_{self.symbol}_{self.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            self.metrics_tracker.save_metrics(str(metrics_file))
            
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline error: {e}")
            self.metrics_tracker.end_training_session()
            return {}
    
    def _train_single_model_with_recovery(self, model_manager, model_name: str, 
                                        data: pd.DataFrame, retry_failed: bool) -> bool:
        """Train a single model with recovery logic."""
        max_retries = 2 if retry_failed else 0
        
        for attempt in range(max_retries + 1):
            try:
                self.metrics_tracker.start_model_training(model_name)
                
                logger.info(f"Training {model_name} (attempt {attempt + 1}/{max_retries + 1})")
                
                # Get the model instance
                model = model_manager.get_model(model_name)
                if not model:
                    raise ValueError(f"Model {model_name} not available")
                
                # Train the model
                training_success = model.train(data)
                
                if training_success:
                    # Validate the trained model
                    validation_metrics = self._validate_trained_model(model, data)
                    
                    self.metrics_tracker.end_model_training(
                        model_name, True, validation_metrics
                    )
                    
                    self.trained_models[model_name] = model
                    logger.info(f"✅ {model_name} training successful")
                    return True
                else:
                    raise RuntimeError(f"{model_name} training returned False")
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"⚠️ {model_name} training attempt {attempt + 1} failed: {error_msg}")
                
                if attempt == max_retries:
                    # Final failure
                    self.metrics_tracker.end_model_training(
                        model_name, False, error_message=error_msg
                    )
                    self.failed_models[model_name] = error_msg
                    logger.error(f"❌ {model_name} training failed after {max_retries + 1} attempts")
                    return False
                else:
                    # Wait before retry
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    def _validate_trained_model(self, model, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate a trained model's performance."""
        validation_metrics = {}
        
        try:
            # Basic validation - can the model make predictions?
            test_data = data.tail(10)
            
            if hasattr(model, 'predict'):
                prediction = model.predict(test_data)
                
                if prediction and isinstance(prediction, dict):
                    confidence = prediction.get('confidence', 0)
                    direction = prediction.get('direction', 'UNKNOWN')
                    
                    validation_metrics['prediction_test'] = 'passed'
                    validation_metrics['sample_confidence'] = confidence
                    validation_metrics['sample_direction'] = direction
                else:
                    validation_metrics['prediction_test'] = 'failed'
            
            # Model info validation
            if hasattr(model, 'get_model_info'):
                model_info = model.get_model_info()
                validation_metrics['model_info'] = model_info
            
        except Exception as e:
            validation_metrics['validation_error'] = str(e)
            logger.warning(f"Model validation error: {e}")
        
        return validation_metrics
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'trained_models': list(self.trained_models.keys()),
            'failed_models': self.failed_models,
            'success_rate': len(self.trained_models) / (len(self.trained_models) + len(self.failed_models)) if (self.trained_models or self.failed_models) else 0,
            'metrics': self.metrics_tracker.metrics
        }


def run_enhanced_training(symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
    """
    Run enhanced training pipeline with full robustness features.
    
    Args:
        symbol: Trading symbol
        timeframe: Trading timeframe
        data: Training data
        
    Returns:
        True if training was successful, False otherwise
    """
    logger.info(f"🚀 Starting enhanced training pipeline for {symbol} @ {timeframe}")
    
    # Initialize pipeline
    pipeline = EnhancedTrainingPipeline(symbol, timeframe)
    
    # 1. Validate environment
    if not pipeline.validate_environment():
        logger.error("❌ Environment validation failed")
        return False
    
    # 2. Validate data quality
    logger.info("📊 Validating data quality...")
    data_validation = pipeline.data_validator.validate_data_quality(data, symbol)
    
    if not data_validation['is_valid']:
        logger.error("❌ Data quality validation failed:")
        for error in data_validation['errors']:
            logger.error(f"  - {error}")
        return False
    
    if data_validation['warnings']:
        logger.warning("⚠️ Data quality warnings:")
        for warning in data_validation['warnings']:
            logger.warning(f"  - {warning}")
    
    logger.info("✅ Data quality validation passed")
    
    # 3. Run training with recovery
    logger.info("🧠 Starting model training with recovery...")
    training_results = pipeline.train_models_with_recovery(data, retry_failed=True)
    
    # 4. Summary
    summary = pipeline.get_training_summary()
    
    success_count = len(summary['trained_models'])
    total_count = success_count + len(summary['failed_models'])
    
    logger.info(f"📊 Training completed: {success_count}/{total_count} models successful")
    
    if summary['trained_models']:
        logger.info(f"✅ Successful models: {', '.join(summary['trained_models'])}")
    
    if summary['failed_models']:
        logger.warning(f"❌ Failed models: {list(summary['failed_models'].keys())}")
    
    # Return True if at least one model was trained successfully
    return success_count > 0


# Export main functions
__all__ = [
    'DataQualityValidator',
    'TrainingMetricsTracker', 
    'EnhancedTrainingPipeline',
    'run_enhanced_training'
]