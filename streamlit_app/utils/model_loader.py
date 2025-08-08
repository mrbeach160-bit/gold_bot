# model_loader.py - Utility functions for loading trained models
import os
import streamlit as st
import joblib

from .formatters import sanitize_filename

# Import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def load_all_models(symbol, timeframe_key):
    """Load all trained models for a given symbol and timeframe"""
    model_dir = 'model'
    symbol_fn = sanitize_filename(symbol)
    
    models = {}
    
    try:
        # LSTM Model and Scaler
        lstm_path = os.path.join(model_dir, f'lstm_model_{symbol_fn}_{timeframe_key}.keras')
        scaler_path = os.path.join(model_dir, f'scaler_{symbol_fn}_{timeframe_key}.pkl')
        
        if TF_AVAILABLE and os.path.exists(lstm_path) and os.path.exists(scaler_path):
            models['lstm'] = load_model(lstm_path)
            models['scaler'] = joblib.load(scaler_path)
        else:
            models['lstm'] = None
            models['scaler'] = None
        
        # XGBoost Model
        xgb_path = os.path.join(model_dir, f'xgb_model_{symbol_fn}_{timeframe_key}.pkl')
        if os.path.exists(xgb_path):
            models['xgb'] = joblib.load(xgb_path)
        else:
            models['xgb'] = None
        
        # CNN Model and Scaler
        cnn_path = os.path.join(model_dir, f'cnn_model_{symbol_fn}_{timeframe_key}.keras')
        cnn_scaler_path = os.path.join(model_dir, f'cnn_scaler_{symbol_fn}_{timeframe_key}.pkl')
        
        if TF_AVAILABLE and os.path.exists(cnn_path) and os.path.exists(cnn_scaler_path):
            models['cnn'] = load_model(cnn_path)
            models['cnn_scaler'] = joblib.load(cnn_scaler_path)
        else:
            models['cnn'] = None
            models['cnn_scaler'] = None
        
        # SVC Model and Scaler
        svc_path = os.path.join(model_dir, f'svc_model_{symbol_fn}_{timeframe_key}.pkl')
        svc_scaler_path = os.path.join(model_dir, f'svc_scaler_{symbol_fn}_{timeframe_key}.pkl')
        
        if os.path.exists(svc_path) and os.path.exists(svc_scaler_path):
            models['svc'] = joblib.load(svc_path)
            models['svc_scaler'] = joblib.load(svc_scaler_path)
        else:
            models['svc'] = None
            models['svc_scaler'] = None
        
        # Naive Bayes Model
        nb_path = os.path.join(model_dir, f'nb_model_{symbol_fn}_{timeframe_key}.pkl')
        if os.path.exists(nb_path):
            models['nb'] = joblib.load(nb_path)
        else:
            models['nb'] = None
        
        # Meta Learner (if available)
        meta_path = os.path.join(model_dir, f'meta_model_{symbol_fn}_{timeframe_key}.pkl')
        if os.path.exists(meta_path):
            models['meta'] = joblib.load(meta_path)
        else:
            models['meta'] = None
            # Try to load default meta learner
            try:
                from utils.meta_learner import train_meta_learner
                st.info("Meta learner not found, will need to train one")
            except ImportError:
                pass
        
        # Check if essential models are loaded
        essential_models = ['lstm', 'scaler', 'xgb']
        missing_models = [name for name in essential_models if models.get(name) is None]
        
        if missing_models:
            st.warning(f"Missing essential models: {missing_models}")
            return None
        
        return models
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None


def check_model_availability(symbol, timeframe_key):
    """Check which models are available for a symbol and timeframe"""
    model_dir = 'model'
    symbol_fn = sanitize_filename(symbol)
    
    model_files = {
        'lstm': f'lstm_model_{symbol_fn}_{timeframe_key}.keras',
        'scaler': f'scaler_{symbol_fn}_{timeframe_key}.pkl',
        'xgb': f'xgb_model_{symbol_fn}_{timeframe_key}.pkl',
        'cnn': f'cnn_model_{symbol_fn}_{timeframe_key}.keras',
        'cnn_scaler': f'cnn_scaler_{symbol_fn}_{timeframe_key}.pkl',
        'svc': f'svc_model_{symbol_fn}_{timeframe_key}.pkl',
        'svc_scaler': f'svc_scaler_{symbol_fn}_{timeframe_key}.pkl',
        'nb': f'nb_model_{symbol_fn}_{timeframe_key}.pkl',
        'meta': f'meta_model_{symbol_fn}_{timeframe_key}.pkl'
    }
    
    availability = {}
    for model_name, filename in model_files.items():
        file_path = os.path.join(model_dir, filename)
        availability[model_name] = os.path.exists(file_path)
    
    return availability


def get_model_info(symbol, timeframe_key):
    """Get information about saved models"""
    model_dir = 'model'
    symbol_fn = sanitize_filename(symbol)
    
    availability = check_model_availability(symbol, timeframe_key)
    
    info = {
        'total_models': sum(availability.values()),
        'availability': availability,
        'essential_complete': all(availability.get(model) for model in ['lstm', 'scaler', 'xgb']),
        'all_complete': all(availability.values())
    }
    
    return info