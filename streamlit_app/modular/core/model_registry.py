"""
Model Registry for the modular application.
Handles model metadata, loading, and file management.
"""

import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import re

try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class ModelRegistry:
    """Registry for managing trained models and their metadata."""
    
    def __init__(self, model_dir: str = 'model'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Define model file patterns (matching legacy app.py)
        self.model_patterns = {
            'lstm': 'lstm_model_{symbol}_{timeframe}.keras',
            'lstm_scaler': 'scaler_{symbol}_{timeframe}.pkl',
            'xgb': 'xgboost_model_{symbol}_{timeframe}.json',
            'cnn': 'cnn_model_{symbol}_{timeframe}.keras',
            'cnn_scaler': 'cnn_scaler_{symbol}_{timeframe}.pkl',
            'svc': 'svc_model_{symbol}_{timeframe}.pkl',
            'svc_scaler': 'svc_scaler_{symbol}_{timeframe}.pkl',
            'nb': 'nb_model_{symbol}_{timeframe}.pkl',
            'meta': 'meta_learner_randomforest_{timeframe}.pkl'
        }
    
    def sanitize_filename(self, symbol: str) -> str:
        """Sanitize symbol for filename use (matching legacy logic)."""
        return re.sub(r'[^\w\-_\.]', '_', symbol)
    
    def get_model_path(self, model_type: str, symbol: str, timeframe: str) -> str:
        """Get the file path for a specific model."""
        symbol_fn = self.sanitize_filename(symbol)
        if model_type in self.model_patterns:
            filename = self.model_patterns[model_type].format(
                symbol=symbol_fn, 
                timeframe=timeframe
            )
            return os.path.join(self.model_dir, filename)
        return ""
    
    def scan(self, symbol: str, timeframe: str) -> Dict[str, Dict[str, Any]]:
        """
        Scan for available models and return metadata.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Dictionary with model metadata
        """
        metadata = {}
        
        for model_type in self.model_patterns.keys():
            model_path = self.get_model_path(model_type, symbol, timeframe)
            
            if os.path.exists(model_path):
                try:
                    stat = os.stat(model_path)
                    metadata[model_type] = {
                        'exists': True,
                        'path': model_path,
                        'size_mb': round(stat.st_size / (1024 * 1024), 2),
                        'modified': datetime.fromtimestamp(stat.st_mtime),
                        'error': None
                    }
                except Exception as e:
                    metadata[model_type] = {
                        'exists': True,
                        'path': model_path,
                        'size_mb': 0,
                        'modified': None,
                        'error': str(e)
                    }
            else:
                metadata[model_type] = {
                    'exists': False,
                    'path': model_path,
                    'size_mb': 0,
                    'modified': None,
                    'error': 'File not found'
                }
        
        return metadata
    
    def load_for_prediction(self, symbol: str, timeframe: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Load all available models for prediction.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Tuple of (loaded_models_dict, load_errors_dict)
        """
        models = {}
        load_errors = {}
        
        # Load LSTM model
        try:
            lstm_path = self.get_model_path('lstm', symbol, timeframe)
            if os.path.exists(lstm_path) and TENSORFLOW_AVAILABLE:
                models['lstm'] = load_model(lstm_path, compile=False)
            else:
                load_errors['lstm'] = 'File not found or TensorFlow not available'
        except Exception as e:
            load_errors['lstm'] = str(e)
        
        # Load LSTM scaler
        try:
            scaler_path = self.get_model_path('lstm_scaler', symbol, timeframe)
            if os.path.exists(scaler_path):
                models['lstm_scaler'] = joblib.load(scaler_path)
            else:
                load_errors['lstm_scaler'] = 'File not found'
        except Exception as e:
            load_errors['lstm_scaler'] = str(e)
        
        # Load XGBoost model
        try:
            xgb_path = self.get_model_path('xgb', symbol, timeframe)
            if os.path.exists(xgb_path) and XGBOOST_AVAILABLE:
                xgb_model = XGBClassifier()
                xgb_model.load_model(xgb_path)
                models['xgb'] = xgb_model
            else:
                load_errors['xgb'] = 'File not found or XGBoost not available'
        except Exception as e:
            load_errors['xgb'] = str(e)
        
        # Load CNN model
        try:
            cnn_path = self.get_model_path('cnn', symbol, timeframe)
            if os.path.exists(cnn_path) and TENSORFLOW_AVAILABLE:
                models['cnn'] = load_model(cnn_path, compile=False)
            else:
                load_errors['cnn'] = 'File not found or TensorFlow not available'
        except Exception as e:
            load_errors['cnn'] = str(e)
        
        # Load CNN scaler
        try:
            cnn_scaler_path = self.get_model_path('cnn_scaler', symbol, timeframe)
            if os.path.exists(cnn_scaler_path):
                models['cnn_scaler'] = joblib.load(cnn_scaler_path)
            else:
                load_errors['cnn_scaler'] = 'File not found'
        except Exception as e:
            load_errors['cnn_scaler'] = str(e)
        
        # Load SVC model
        try:
            svc_path = self.get_model_path('svc', symbol, timeframe)
            if os.path.exists(svc_path):
                models['svc'] = joblib.load(svc_path)
            else:
                load_errors['svc'] = 'File not found'
        except Exception as e:
            load_errors['svc'] = str(e)
        
        # Load SVC scaler
        try:
            svc_scaler_path = self.get_model_path('svc_scaler', symbol, timeframe)
            if os.path.exists(svc_scaler_path):
                models['svc_scaler'] = joblib.load(svc_scaler_path)
            else:
                load_errors['svc_scaler'] = 'File not found'
        except Exception as e:
            load_errors['svc_scaler'] = str(e)
        
        # Load Naive Bayes model
        try:
            nb_path = self.get_model_path('nb', symbol, timeframe)
            if os.path.exists(nb_path):
                models['nb'] = joblib.load(nb_path)
            else:
                load_errors['nb'] = 'File not found'
        except Exception as e:
            load_errors['nb'] = str(e)
        
        # Load Meta learner (optional)
        try:
            meta_path = self.get_model_path('meta', symbol, timeframe)
            if os.path.exists(meta_path):
                models['meta'] = joblib.load(meta_path)
            else:
                load_errors['meta'] = 'File not found (meta learner is optional)'
        except Exception as e:
            load_errors['meta'] = str(e)
        
        return models, load_errors
    
    def get_summary_table(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Get a summary table of model availability.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            DataFrame with model summary
        """
        metadata = self.scan(symbol, timeframe)
        
        summary_data = []
        for model_type, info in metadata.items():
            summary_data.append({
                'Model': model_type.upper().replace('_', ' '),
                'Available': '✅' if info['exists'] else '❌',
                'Size (MB)': info['size_mb'] if info['exists'] else 0,
                'Last Modified': info['modified'].strftime('%Y-%m-%d %H:%M:%S') if info['modified'] else 'N/A',
                'Status': 'Ready' if info['exists'] and not info['error'] else (info['error'] or 'Missing')
            })
        
        return pd.DataFrame(summary_data)
    
    def count_available_models(self, symbol: str, timeframe: str) -> Tuple[int, int]:
        """
        Count available models.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Tuple of (available_count, total_count)
        """
        metadata = self.scan(symbol, timeframe)
        
        # Don't count scalers and meta learner in main count
        main_models = ['lstm', 'xgb', 'cnn', 'svc', 'nb']
        available = sum(1 for model in main_models if metadata.get(model, {}).get('exists', False))
        total = len(main_models)
        
        return available, total