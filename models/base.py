# models/base.py
"""
Base model interface and factory function for unified model architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import os


class BaseModel(ABC):
    """Abstract base class for all ML models with standardized interface."""
    
    def __init__(self, symbol: str, timeframe: str):
        """Initialize base model with symbol and timeframe."""
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = None
        self._trained = False
        
    @abstractmethod
    def train(self, data: pd.DataFrame) -> bool:
        """Train the model with provided data.
        
        Args:
            data: Training data as pandas DataFrame
            
        Returns:
            bool: True if training successful, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with the model.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Dict with standardized prediction format:
            {
                'direction': str,      # 'BUY' | 'SELL' | 'HOLD'
                'confidence': float,   # 0.0 to 1.0
                'probability': float,  # Raw model probability
                'model_name': str,     # Name of the model
                'timestamp': datetime, # Prediction timestamp
                'features_used': List[str]  # List of features used
            }
        """
        pass
    
    @abstractmethod
    def save(self) -> bool:
        """Save model to file.
        
        Returns:
            bool: True if save successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load(self) -> bool:
        """Load model from file.
        
        Returns:
            bool: True if load successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata.
        
        Returns:
            Dict with model information including name, version, parameters, etc.
        """
        pass
    
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._trained
    
    def get_model_path(self, file_extension: str = '.pkl') -> str:
        """Generate standardized model file path."""
        model_name = self.__class__.__name__.replace('Model', '')
        sanitized_symbol = self._sanitize_filename(self.symbol)
        return os.path.join('model', f'{model_name}_{sanitized_symbol}_{self.timeframe}{file_extension}')
    
    def _sanitize_filename(self, name: str) -> str:
        """Clean string for safe filename usage."""
        import re
        return re.sub(r'[^a-zA-Z0-9_-]', '', name)


def create_model(model_type: str, symbol: str, timeframe: str) -> BaseModel:
    """Factory function to create model instances.
    
    Args:
        model_type: Type of model ('lstm', 'lightgbm', 'xgboost', 'svc', 'nb')
        symbol: Trading symbol (e.g., 'XAU/USD')
        timeframe: Timeframe (e.g., '5m', '1h')
        
    Returns:
        BaseModel: Instance of the requested model type
        
    Raises:
        ValueError: If model_type is not supported or dependencies unavailable
    """
    from .ml_models import (LSTMModel, LightGBMModel, XGBoostModel, SVCModel, NaiveBayesModel,
                           TENSORFLOW_AVAILABLE, LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE)
    
    model_mapping = {
        'lstm': (LSTMModel, TENSORFLOW_AVAILABLE, "TensorFlow"),
        'lightgbm': (LightGBMModel, LIGHTGBM_AVAILABLE, "LightGBM"),
        'lgb': (LightGBMModel, LIGHTGBM_AVAILABLE, "LightGBM"),  # Alias
        'xgboost': (XGBoostModel, XGBOOST_AVAILABLE, "XGBoost"),
        'xgb': (XGBoostModel, XGBOOST_AVAILABLE, "XGBoost"),  # Alias
        'svc': (SVCModel, True, "scikit-learn"),  # SVC is part of sklearn which is always available
        'naivebayes': (NaiveBayesModel, True, "scikit-learn"),
        'nb': (NaiveBayesModel, True, "scikit-learn"),  # Alias
    }
    
    model_type_lower = model_type.lower()
    if model_type_lower not in model_mapping:
        available_types = ', '.join(model_mapping.keys())
        raise ValueError(f"Unsupported model type '{model_type}'. Available types: {available_types}")
    
    model_class, dependency_available, dependency_name = model_mapping[model_type_lower]
    
    if not dependency_available:
        print(f"Warning: {dependency_name} not available for {model_type} model")
        # Still create the model instance, but it will handle unavailability gracefully
    
    return model_class(symbol, timeframe)