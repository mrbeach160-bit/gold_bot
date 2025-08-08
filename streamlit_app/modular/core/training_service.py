"""
Training Service for the modular application.
Handles model training with unified and legacy fallback.
"""

import pandas as pd
import streamlit as st
import os
import sys
from typing import Optional, Dict, Any

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TrainingService:
    """Service for training models with unified and legacy fallback."""
    
    def __init__(self):
        self.model_dir = 'model'
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train_models(self, data: pd.DataFrame, symbol: str, timeframe: str, 
                    prefer_unified: bool = True, evaluate_after: bool = True) -> Dict[str, Any]:
        """
        Train models using unified ModelManager or legacy fallback.
        
        Args:
            data: Training data
            symbol: Trading symbol
            timeframe: Timeframe
            prefer_unified: Whether to try unified training first
            evaluate_after: Whether to evaluate models after training
            
        Returns:
            Dictionary with training results
        """
        results = {
            'success': False,
            'method_used': None,
            'errors': [],
            'models_trained': [],
            'evaluation': None
        }
        
        # Validate input data
        if data is None or data.empty:
            results['errors'].append('No training data provided')
            return results
        
        if len(data) < 100:
            results['errors'].append(f'Insufficient data: {len(data)} rows (minimum 100 required)')
            return results
        
        # Try unified training first if preferred
        if prefer_unified:
            try:
                success = self._try_unified_training(data, symbol, timeframe)
                if success:
                    results['success'] = True
                    results['method_used'] = 'unified'
                    results['models_trained'] = ['unified_models']
                    st.success("✅ Unified training completed successfully")
                    
                    if evaluate_after:
                        results['evaluation'] = self._evaluate_models(data, symbol, timeframe)
                    
                    return results
                else:
                    st.warning("⚠️ Unified training failed, falling back to legacy method")
                    
            except Exception as e:
                st.warning(f"⚠️ Unified training error: {str(e)}, falling back to legacy method")
                results['errors'].append(f'Unified training failed: {str(e)}')
        
        # Fallback to legacy training
        try:
            success = self._fallback_legacy_training(data, symbol, timeframe)
            if success:
                results['success'] = True
                results['method_used'] = 'legacy'
                results['models_trained'] = ['lstm', 'xgboost', 'cnn', 'svc', 'naive_bayes', 'meta_learner']
                st.success("✅ Legacy training completed successfully")
                
                if evaluate_after:
                    results['evaluation'] = self._evaluate_models(data, symbol, timeframe)
                
                return results
            else:
                results['errors'].append('Legacy training failed')
                
        except Exception as e:
            st.error(f"❌ Legacy training error: {str(e)}")
            results['errors'].append(f'Legacy training failed: {str(e)}')
        
        return results
    
    def _try_unified_training(self, data: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Try unified training using ModelManager.
        
        Args:
            data: Training data
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import unified systems
            from utils.model_manager import ModelManager
            
            # Check if ModelManager is available and has required methods
            model_manager = ModelManager()
            
            if hasattr(model_manager, 'train_all_models') and hasattr(model_manager, 'save_all_models'):
                with st.status("🏗️ Training models with unified system...", expanded=True) as status:
                    # Train all models
                    status.update(label="Training models...")
                    trained_models = model_manager.train_all_models(data, symbol, timeframe)
                    
                    if trained_models:
                        # Save all models
                        status.update(label="Saving models...")
                        save_success = model_manager.save_all_models(trained_models, symbol, timeframe)
                        
                        if save_success:
                            status.update(label="Unified training complete!", state="complete", expanded=False)
                            return True
                        else:
                            st.warning("Model saving failed in unified system")
                            return False
                    else:
                        st.warning("No models were trained in unified system")
                        return False
            else:
                st.warning("ModelManager does not have required methods")
                return False
                
        except ImportError:
            st.warning("Unified ModelManager not available")
            return False
        except Exception as e:
            st.warning(f"Unified training failed: {str(e)}")
            return False
    
    def _fallback_legacy_training(self, data: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Fallback to legacy training function.
        
        Args:
            data: Training data
            symbol: Trading symbol 
            timeframe: Timeframe
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import legacy training function
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from app import train_and_save_all_models
            
            # Call legacy training function
            train_and_save_all_models(data.copy(), symbol, timeframe)
            
            return True
            
        except ImportError as e:
            st.error(f"Cannot import legacy training function: {str(e)}")
            return False
        except Exception as e:
            st.error(f"Legacy training failed: {str(e)}")
            return False
    
    def _evaluate_models(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Evaluate trained models on the training data.
        
        Args:
            data: Training data
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Dictionary with evaluation results or None if failed
        """
        try:
            from .model_registry import ModelRegistry
            
            registry = ModelRegistry()
            available, total = registry.count_available_models(symbol, timeframe)
            
            evaluation = {
                'models_available': available,
                'total_models': total,
                'availability_rate': available / total if total > 0 else 0,
                'data_size': len(data),
                'symbol': symbol,
                'timeframe': timeframe
            }
            
            # Try to get model sizes
            try:
                metadata = registry.scan(symbol, timeframe)
                total_size_mb = sum(
                    info.get('size_mb', 0) for info in metadata.values() 
                    if info.get('exists', False)
                )
                evaluation['total_size_mb'] = round(total_size_mb, 2)
            except:
                evaluation['total_size_mb'] = 0
            
            return evaluation
            
        except Exception as e:
            st.warning(f"Model evaluation failed: {str(e)}")
            return None
    
    def get_training_parameters(self) -> Dict[str, Any]:
        """
        Get default training parameters that can be customized in the UI.
        
        Returns:
            Dictionary with training parameters
        """
        return {
            'data_size': 1000,
            'prefer_unified': True,
            'evaluate_after': True,
            'min_data_required': 100,
            'supported_timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
            'model_types': ['LSTM', 'XGBoost', 'CNN', 'SVC', 'Naive Bayes', 'Meta Learner']
        }