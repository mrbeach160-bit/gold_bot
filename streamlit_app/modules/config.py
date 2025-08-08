"""
Configuration module for the trading bot.

This module handles environment setup, feature flags, paths configuration,
and initial dependency imports with proper error handling.
"""

import os
import sys
import warnings
from typing import Dict, Any

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow to avoid GPU issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

# --- PATH CONFIGURATION ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Feature flags for optional dependencies
FEATURE_FLAGS = {
    'WEBSOCKET_AVAILABLE': False,
    'VALIDATION_UTILS_AVAILABLE': False,
    'BINANCE_AVAILABLE': False,
    'NEW_SYSTEM_AVAILABLE': False,
    'UTILS_AVAILABLE': False
}

def initialize_feature_flags() -> Dict[str, bool]:
    """Initialize feature flags based on available dependencies."""
    global FEATURE_FLAGS
    
    # WebSocket imports
    try:
        import websocket
        import json
        import threading
        from collections import deque
        FEATURE_FLAGS['WEBSOCKET_AVAILABLE'] = True
    except ImportError:
        FEATURE_FLAGS['WEBSOCKET_AVAILABLE'] = False
    
    # Validation utilities
    try:
        from ..validation_utils import (
            validate_signal_realtime, validate_take_profit_realtime, 
            is_signal_expired, get_signal_quality_score,
            get_price_staleness_indicator, format_quality_indicator
        )
        FEATURE_FLAGS['VALIDATION_UTILS_AVAILABLE'] = True
    except ImportError:
        FEATURE_FLAGS['VALIDATION_UTILS_AVAILABLE'] = False
    
    # Binance API
    try:
        from binance.client import Client
        FEATURE_FLAGS['BINANCE_AVAILABLE'] = True
    except ImportError:
        FEATURE_FLAGS['BINANCE_AVAILABLE'] = False
    
    # Phase 3 unified systems
    try:
        from data import DataManager
        from trading import TradingManager
        FEATURE_FLAGS['NEW_SYSTEM_AVAILABLE'] = True
    except ImportError:
        FEATURE_FLAGS['NEW_SYSTEM_AVAILABLE'] = False
    
    # Utils modules
    try:
        from utils.data import get_gold_data
        from utils.indicators import add_indicators, get_support_resistance, compute_rsi
        from utils.meta_learner import train_meta_learner, prepare_data_for_meta_learner, get_meta_signal
        FEATURE_FLAGS['UTILS_AVAILABLE'] = True
    except ImportError:
        FEATURE_FLAGS['UTILS_AVAILABLE'] = False
    
    return FEATURE_FLAGS

def configure_tensorflow():
    """Configure TensorFlow settings."""
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')  # Disable GPU
        return True
    except ImportError:
        return False

def get_project_root() -> str:
    """Get the project root directory."""
    return project_root

def get_model_directory() -> str:
    """Get the model directory path."""
    return os.path.join(project_root, 'model')

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled."""
    return FEATURE_FLAGS.get(feature_name, False)

# Initialize feature flags on module import
FEATURE_FLAGS = initialize_feature_flags()

# Configure TensorFlow
TF_AVAILABLE = configure_tensorflow()

# Label map for model predictions
LABEL_MAP = {1: "BUY", -1: "SELL", 0: "HOLD"}

# Timeframe mapping
TIMEFRAME_MAPPING = {
    '1m': '1min', 
    '5m': '5min', 
    '15m': '15min', 
    '1h': '1h', 
    '4h': '4h', 
    '1d': '1day'
}