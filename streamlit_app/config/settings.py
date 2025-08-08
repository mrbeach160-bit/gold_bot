# settings.py - Configuration settings and constants
import os
import sys

# --- PATH CONFIGURATION ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- FEATURE FLAGS ---
NEW_SYSTEM_AVAILABLE = False
UTILS_AVAILABLE = False
WEBSOCKET_AVAILABLE = False
BINANCE_AVAILABLE = False
VALIDATION_UTILS_AVAILABLE = False
TF_AVAILABLE = False

# Check for WebSocket availability
try:
    import websocket
    import json
    import threading
    from collections import deque
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Check for Binance availability
try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

# Check for TensorFlow availability
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Force CPU usage
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Check for validation utilities
try:
    from streamlit_app.validation_utils import (
        validate_signal_realtime, validate_take_profit_realtime, 
        is_signal_expired, get_signal_quality_score,
        get_price_staleness_indicator, format_quality_indicator
    )
    VALIDATION_UTILS_AVAILABLE = True
except ImportError:
    VALIDATION_UTILS_AVAILABLE = False

# Check for Phase 3 systems
try:
    from data import DataManager
    from trading import TradingManager
    NEW_SYSTEM_AVAILABLE = True
except ImportError:
    NEW_SYSTEM_AVAILABLE = False

# Check for utils availability
try:
    from utils.data import get_gold_data
    from utils.indicators import add_indicators, get_support_resistance, compute_rsi
    from utils.meta_learner import train_meta_learner, prepare_data_for_meta_learner, get_meta_signal
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# --- TRADING CONSTANTS ---
SYMBOL_OPTIONS = {
    "Twelve Data": ["XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD", "ETH/USD"],
    "Binance": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
}

INTERVAL_MAPPING = {
    'Twelve Data': {
        '1min': '1min',
        '5min': '5min', 
        '15min': '15min',
        '30min': '30min',
        '1h': '1h',
        '4h': '4h',
        '1day': '1day'
    },
    'Binance': {
        '1min': '1m',
        '5min': '5m',
        '15min': '15m', 
        '30min': '30m',
        '1h': '1h',
        '4h': '4h',
        '1day': '1d'
    }
}

# --- MODEL CONFIGURATION ---
MODEL_DIRECTORY = 'model'
SEQUENCE_LENGTH_LSTM = 60
SEQUENCE_LENGTH_CNN = 20

# --- FEATURE CONFIGURATIONS ---
XGB_FEATURE_NAMES = [
    'open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 
    'ATR_14', 'STOCHk_14_3_3', 'price_change', 'high_low_ratio', 'open_close_ratio'
]

CNN_FEATURE_NAMES = [
    'open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20'
]

SVC_FEATURE_NAMES = [
    'open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 
    'ATR_14', 'STOCHk_14_3_3'
]

NB_FEATURE_NAMES = [
    'open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 
    'ATR_14', 'STOCHk_14_3_3'
]

# --- TRADING DEFAULTS ---
DEFAULT_BALANCE = 1000.0
DEFAULT_RISK_PERCENT = 2.0
DEFAULT_SL_PIPS = 20
DEFAULT_TP_PIPS = 40
DEFAULT_LEVERAGE = 20

# --- VALIDATION SETTINGS ---
MIN_DATA_NEEDED = 60
MAX_PRICE_DEVIATION = 0.005  # 0.5%
MIN_FILL_PROBABILITY = 0.7  # 70%

# --- LABEL MAPPING ---
LABEL_MAP = {1: "BUY", -1: "SELL", 0: "HOLD"}

# --- ERROR MESSAGES ---
ERROR_MESSAGES = {
    'twelve_data_api': "API Key Twelve Data diperlukan.",
    'binance_api': "API Key & Secret Binance diperlukan.",
    'binance_not_available': "Modul 'python-binance' tidak terinstall. Install dengan: pip install python-binance",
    'websocket_not_available': "⚠️ WebSocket not available. Install with: pip install websocket-client",
    'tensorflow_not_available': "TensorFlow tidak terinstall. Install dengan: pip install tensorflow",
    'xgboost_not_available': "XGBoost tidak terinstall. Install dengan: pip install xgboost"
}

# --- UI CONFIGURATION ---
PAGE_CONFIG = {
    'page_title': "Multi-Source Trading AI v8.3",
    'layout': "wide"
}

APP_TITLE = "🤖 Multi-Asset Trading Bot (Master AI) v8.3 - Twelve Data WebSocket Enhanced"
APP_VERSION = "v8.3"