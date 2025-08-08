"""
Legacy Bridge for the modular application.
Provides access to legacy functions when needed.
"""

import os
import sys
import pandas as pd
import streamlit as st
from typing import Optional, Any, Dict

# Add paths for legacy imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

streamlit_app_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if streamlit_app_path not in sys.path:
    sys.path.insert(0, streamlit_app_path)


class LegacyBridge:
    """Bridge to access legacy functions from the original app.py."""
    
    def __init__(self):
        self._legacy_functions_cache = {}
        self._import_errors = {}
    
    def get_legacy_training_function(self):
        """Get the legacy train_and_save_all_models function."""
        try:
            if 'train_and_save_all_models' not in self._legacy_functions_cache:
                from app import train_and_save_all_models
                self._legacy_functions_cache['train_and_save_all_models'] = train_and_save_all_models
            
            return self._legacy_functions_cache['train_and_save_all_models']
            
        except ImportError as e:
            self._import_errors['train_and_save_all_models'] = str(e)
            st.error(f"Cannot import legacy training function: {str(e)}")
            return None
        except Exception as e:
            self._import_errors['train_and_save_all_models'] = str(e)
            st.error(f"Error accessing legacy training function: {str(e)}")
            return None
    
    def get_legacy_prediction_function(self):
        """Get the legacy predict_with_models function."""
        try:
            if 'predict_with_models' not in self._legacy_functions_cache:
                from app import predict_with_models
                self._legacy_functions_cache['predict_with_models'] = predict_with_models
            
            return self._legacy_functions_cache['predict_with_models']
            
        except ImportError as e:
            self._import_errors['predict_with_models'] = str(e)
            st.warning(f"Cannot import legacy prediction function: {str(e)}")
            return None
        except Exception as e:
            self._import_errors['predict_with_models'] = str(e)
            st.warning(f"Error accessing legacy prediction function: {str(e)}")
            return None
    
    def get_legacy_model_loading_function(self):
        """Get the legacy load_all_models function."""
        try:
            if 'load_all_models' not in self._legacy_functions_cache:
                from app import load_all_models
                self._legacy_functions_cache['load_all_models'] = load_all_models
            
            return self._legacy_functions_cache['load_all_models']
            
        except ImportError as e:
            self._import_errors['load_all_models'] = str(e)
            st.warning(f"Cannot import legacy model loading function: {str(e)}")
            return None
        except Exception as e:
            self._import_errors['load_all_models'] = str(e)
            st.warning(f"Error accessing legacy model loading function: {str(e)}")
            return None
    
    def get_legacy_backtest_function(self):
        """Get the legacy run_backtest function."""
        try:
            if 'run_backtest' not in self._legacy_functions_cache:
                from app import run_backtest
                self._legacy_functions_cache['run_backtest'] = run_backtest
            
            return self._legacy_functions_cache['run_backtest']
            
        except ImportError as e:
            self._import_errors['run_backtest'] = str(e)
            st.warning(f"Cannot import legacy backtest function: {str(e)}")
            return None
        except Exception as e:
            self._import_errors['run_backtest'] = str(e)
            st.warning(f"Error accessing legacy backtest function: {str(e)}")
            return None
    
    def get_legacy_data_function(self):
        """Get the legacy data fetching utilities."""
        try:
            if 'get_gold_data' not in self._legacy_functions_cache:
                from utils.data import get_gold_data
                self._legacy_functions_cache['get_gold_data'] = get_gold_data
            
            return self._legacy_functions_cache['get_gold_data']
            
        except ImportError as e:
            self._import_errors['get_gold_data'] = str(e)
            st.error(f"Cannot import legacy data function: {str(e)}")
            return None
        except Exception as e:
            self._import_errors['get_gold_data'] = str(e)
            st.error(f"Error accessing legacy data function: {str(e)}")
            return None
    
    def get_legacy_indicators_function(self):
        """Get the legacy indicators function."""
        try:
            if 'add_indicators' not in self._legacy_functions_cache:
                from utils.indicators import add_indicators
                self._legacy_functions_cache['add_indicators'] = add_indicators
            
            return self._legacy_functions_cache['add_indicators']
            
        except ImportError as e:
            self._import_errors['add_indicators'] = str(e)
            st.warning(f"Cannot import legacy indicators function: {str(e)}")
            return None
        except Exception as e:
            self._import_errors['add_indicators'] = str(e)
            st.warning(f"Error accessing legacy indicators function: {str(e)}")
            return None
    
    def train_with_legacy(self, data: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Train models using the legacy training function.
        
        Args:
            data: Training data
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            True if successful, False otherwise
        """
        try:
            train_function = self.get_legacy_training_function()
            if train_function is None:
                return False
            
            # Call legacy training function
            train_function(data.copy(), symbol, timeframe)
            return True
            
        except Exception as e:
            st.error(f"Legacy training failed: {str(e)}")
            return False
    
    def predict_with_legacy(self, models: Dict[str, Any], data: pd.DataFrame) -> Optional[tuple]:
        """
        Make prediction using legacy prediction function.
        
        Args:
            models: Dictionary of loaded models
            data: Historical data
            
        Returns:
            Tuple of (direction, confidence, predicted_price) or None if failed
        """
        try:
            predict_function = self.get_legacy_prediction_function()
            if predict_function is None:
                return None
            
            # Call legacy prediction function
            result = predict_function(models, data)
            return result
            
        except Exception as e:
            st.warning(f"Legacy prediction failed: {str(e)}")
            return None
    
    def load_models_with_legacy(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Load models using legacy loading function.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Dictionary of loaded models or None if failed
        """
        try:
            load_function = self.get_legacy_model_loading_function()
            if load_function is None:
                return None
            
            # Call legacy model loading function
            models = load_function(symbol, timeframe)
            return models
            
        except Exception as e:
            st.warning(f"Legacy model loading failed: {str(e)}")
            return None
    
    def get_import_errors(self) -> Dict[str, str]:
        """Get any import errors that occurred."""
        return self._import_errors.copy()
    
    def is_legacy_available(self) -> bool:
        """Check if legacy functions are available."""
        # Try to import key legacy functions
        try:
            from app import train_and_save_all_models
            return True
        except ImportError:
            return False
    
    def get_legacy_status(self) -> Dict[str, Any]:
        """Get status of legacy system availability."""
        functions_to_check = [
            'train_and_save_all_models',
            'predict_with_models', 
            'load_all_models',
            'run_backtest'
        ]
        
        status = {
            'available': self.is_legacy_available(),
            'functions': {},
            'import_errors': self._import_errors.copy()
        }
        
        for func_name in functions_to_check:
            try:
                if func_name == 'train_and_save_all_models':
                    func = self.get_legacy_training_function()
                elif func_name == 'predict_with_models':
                    func = self.get_legacy_prediction_function()
                elif func_name == 'load_all_models':
                    func = self.get_legacy_model_loading_function()
                elif func_name == 'run_backtest':
                    func = self.get_legacy_backtest_function()
                else:
                    func = None
                
                status['functions'][func_name] = func is not None
                
            except:
                status['functions'][func_name] = False
        
        return status