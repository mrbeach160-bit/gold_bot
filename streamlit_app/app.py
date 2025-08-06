# app.py (Complete Version v8.3 - Twelve Data WebSocket)
# --- IMPLEMENTASI TWELVE DATA WEBSOCKET (04/08/2025 v8.3):
# - FEAT: Menambahkan Twelve Data WebSocket support untuk real-time streaming
# - FEAT: Menghapus Finnhub integration sepenuhnya
# - FEAT: Enhanced WebSocket streaming untuk Forex, Stocks, dan Crypto via Twelve Data
# - FEAT: Auto fallback dari WebSocket ke polling untuk reliability
# - FEAT: Backward compatibility dengan Binance WebSocket
# - REMOVED: Live Chart tab sesuai request
#
# --- IMPLEMENTASI WEBSOCKET (03/08/2025 v8.1):
# - FEAT: Menambahkan WebSocket support untuk real-time trading signals
# - FEAT: Live price streaming dari Binance WebSocket
# - FEAT: Real-time signal generation dengan data streaming
# - FEAT: Connection management dan error handling yang robust
# - FEAT: Backward compatibility - tidak akan error jika WebSocket tidak tersedia
#
# --- IMPLEMENTASI PATCH (01/08/2025 v8.0 - Smart AI Entry Patch Integrated):
# - FEAT: Mengintegrasikan `smart_ai_entry_patch_Version1.py`.
# - LIVE TRADING: Mengganti kalkulasi entry price dengan fungsi `calculate_smart_entry_price` yang menganalisis multi-faktor (S/R, RSI, MACD, ATR).
# - BACKTESTING: Memodifikasi `run_backtest` untuk menggunakan logika `calculate_smart_entry_price` yang sama, memastikan konsistensi dan menambahkan simulasi probabilitas eksekusi order.
# - UI: Mengimplementasikan `display_smart_signal_results` untuk menampilkan sinyal dengan alasan strategi yang mendetail dan penilaian risiko.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta
import joblib
import time
import re
import warnings
from collections import Counter
import requests
warnings.filterwarnings('ignore')

# WebSocket imports (AMAN dengan error handling)
try:
    import websocket
    import json
    import threading
    from collections import deque
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# NEW IMPORTS for fixing previous errors and ensuring functionality
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pandas_ta as ta

# --- NEW: Import for Binance API ---
try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

# Configure TensorFlow to avoid GPU issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Disable GPU
    from tensorflow.keras.models import load_model
except ImportError:
    st.error("TensorFlow tidak terinstall. Install dengan: pip install tensorflow")
    st.stop()

try:
    from xgboost import XGBClassifier
except ImportError:
    st.error("XGBoost tidak terinstall. Install dengan: pip install xgboost")
    st.stop()

# --- PATH CONFIGURATION ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import data utilities from utils folder
try:
    from utils.data import get_gold_data # Untuk Twelve Data
    from utils.indicators import add_indicators, get_support_resistance, compute_rsi
    label_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
    from utils.meta_learner import train_meta_learner, prepare_data_for_meta_learner, get_meta_signal

    UTILS_AVAILABLE = True
except ImportError as e:
    st.error("Utils modules tidak ditemukan. Pastikan folder 'utils' tersedia dengan file:")
    st.error("- utils/data.py (untuk get_gold_data)")
    st.error("- utils/indicators.py (untuk add_indicators, get_support_resistance, compute_rsi)")
    st.error("- utils/meta_learner.py (untuk train_meta_learner, prepare_data_for_meta_learner, get_meta_signal)")
    st.error(f"Error: {e}")
    st.stop()

# ==============================================================================
# TWELVE DATA WEBSOCKET INTEGRATION v8.3 (NEW)
# ==============================================================================

class TwelveDataWebSocketClient:
    """Twelve Data WebSocket client untuk real-time streaming"""
    
    def __init__(self, api_key, on_message_callback):
        self.api_key = api_key
        self.on_message_callback = on_message_callback
        self.ws = None
        self.is_connected = False
        self.subscribed_symbols = set()
        
    def connect(self):
        """Connect ke Twelve Data WebSocket"""
        try:
            # Twelve Data WebSocket endpoint
            url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={self.api_key}"
            
            self.ws = websocket.WebSocketApp(
                url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Start in thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            return True
            
        except Exception as e:
            st.error(f"Twelve Data WebSocket connection error: {e}")
            return False
            
    def on_open(self, ws):
        """WebSocket opened"""
        self.is_connected = True
        
    def on_message(self, ws, message):
        """Handle incoming real-time price messages"""
        try:
            data = json.loads(message)
            
            # Twelve Data WebSocket format
            if data.get('event') == 'price':
                processed_data = {
                    'symbol': data.get('symbol'),
                    'price': float(data.get('price', 0)),
                    'timestamp': datetime.now(),
                    'volume': data.get('volume', 0)
                }
                
                if self.on_message_callback:
                    self.on_message_callback(processed_data)
                    
        except Exception as e:
            pass  # Ignore message parsing errors
            
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.is_connected = False
        
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        self.is_connected = False
        
    def subscribe_symbol(self, symbol):
        """Subscribe to symbol updates"""
        if self.ws and self.is_connected:
            subscribe_msg = {
                "action": "subscribe",
                "params": {
                    "symbols": symbol
                }
            }
            self.ws.send(json.dumps(subscribe_msg))
            self.subscribed_symbols.add(symbol)
            
    def unsubscribe_symbol(self, symbol):
        """Unsubscribe from symbol"""
        if self.ws and self.is_connected:
            unsubscribe_msg = {
                "action": "unsubscribe",
                "params": {
                    "symbols": symbol
                }
            }
            self.ws.send(json.dumps(unsubscribe_msg))
            self.subscribed_symbols.discard(symbol)
            
    def disconnect(self):
        """Disconnect WebSocket"""
        if self.ws:
            self.ws.close()
            self.is_connected = False

# ==============================================================================
# ENHANCED WEBSOCKET INTEGRATION v8.3 (UPDATED)
# ==============================================================================

class EnhancedWebSocketManager:
    """Enhanced WebSocket Manager untuk Twelve Data dan Binance"""
    
    def __init__(self):
        self.latest_price = {}
        self.connection_status = {}
        self.clients = {}
        self.price_history = {}
        
    def start_stream(self, api_source, symbol, api_key_1=None, api_key_2=None):
        """Start streaming berdasarkan API source"""
        
        if api_source == "Binance" and BINANCE_AVAILABLE:
            return self._start_binance_stream(symbol)
            
        elif api_source == "Twelve Data":
            # Try WebSocket first, fallback to polling
            if WEBSOCKET_AVAILABLE:
                websocket_success = self._start_twelve_data_websocket_stream(symbol, api_key_1)
                if websocket_success:
                    return True
                else:
                    st.warning("⚠️ WebSocket failed, falling back to polling...")
                    return self._start_polling_stream(symbol, api_key_1)
            else:
                return self._start_polling_stream(symbol, api_key_1)
            
        return False
    
    def _start_twelve_data_websocket_stream(self, symbol, api_key):
        """Start Twelve Data WebSocket stream"""
        if not WEBSOCKET_AVAILABLE:
            self.connection_status[symbol] = "WebSocket tidak tersedia"
            return False
            
        try:
            def on_data_received(data):
                if data and 'price' in data:
                    self.latest_price[symbol] = data['price']
                    if symbol not in self.price_history:
                        self.price_history[symbol] = deque(maxlen=100)
                    self.price_history[symbol].append({
                        'price': data['price'],
                        'timestamp': data.get('timestamp', datetime.now()),
                        'volume': data.get('volume', 0)
                    })
                    self.connection_status[symbol] = "Connected ✅"
                    
            client = TwelveDataWebSocketClient(
                api_key=api_key,
                on_message_callback=on_data_received
            )
            
            if client.connect():
                time.sleep(2)  # Wait for connection
                client.subscribe_symbol(symbol)
                
                self.clients[symbol] = client
                self.connection_status[symbol] = "Connecting... 🔄"
                return True
            else:
                self.connection_status[symbol] = "Connection Failed"
                return False
                
        except Exception as e:
            self.connection_status[symbol] = f"Twelve Data WS Error: {str(e)[:30]}"
            return False
        
    def _start_binance_stream(self, symbol):
        """Start Binance WebSocket stream"""
        if not WEBSOCKET_AVAILABLE:
            self.connection_status[symbol] = "WebSocket tidak tersedia"
            return False
            
        try:
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    price = float(data.get('c', 0))
                    if price > 0:
                        self.latest_price[symbol] = price
                        if symbol not in self.price_history:
                            self.price_history[symbol] = deque(maxlen=100)
                        self.price_history[symbol].append({
                            'price': price,
                            'timestamp': datetime.now(),
                            'volume': float(data.get('v', 0))
                        })
                        self.connection_status[symbol] = "Connected ✅"
                except Exception as e:
                    pass
                    
            def on_error(ws, error):
                self.connection_status[symbol] = f"Error: {str(error)[:30]}"
                
            def on_close(ws, close_status_code, close_msg):
                self.connection_status[symbol] = "Disconnected"
                
            def on_open(ws):
                self.connection_status[symbol] = "Connected ✅"
                
            # Buat WebSocket connection
            symbol_clean = symbol.replace('/', '').lower()
            url = f"wss://stream.binance.com:9443/ws/{symbol_clean}@ticker"
            
            ws = websocket.WebSocketApp(url,
                on_message=on_message,
                on_error=on_error, 
                on_close=on_close,
                on_open=on_open)
            
            # Start di thread terpisah
            def run_ws():
                try:
                    ws.run_forever()
                except Exception as e:
                    self.connection_status[symbol] = f"Thread Error: {str(e)[:30]}"
                    
            thread = threading.Thread(target=run_ws, daemon=True)
            thread.start()
            
            self.clients[symbol] = ws
            self.connection_status[symbol] = "Connecting... 🔄"
            return True
            
        except Exception as e:
            self.connection_status[symbol] = f"Start Error: {str(e)[:30]}"
            return False
    
    def _start_polling_stream(self, symbol, api_key):
        """Start polling untuk Twelve Data (fallback)"""
        try:
            def poll_data():
                while symbol in self.clients:
                    try:
                        # Polling setiap 30 detik
                        data = get_gold_data(api_key, interval='1min', symbol=symbol, outputsize=1)
                        if data is not None and not data.empty:
                            price = data['close'].iloc[-1]
                            self.latest_price[symbol] = price
                            if symbol not in self.price_history:
                                self.price_history[symbol] = deque(maxlen=100)
                            self.price_history[symbol].append({
                                'price': price,
                                'timestamp': datetime.now(),
                                'volume': data['volume'].iloc[-1] if 'volume' in data.columns else 0
                            })
                            self.connection_status[symbol] = "Polling ✅"
                        time.sleep(30)  # 30 second interval
                    except Exception as e:
                        self.connection_status[symbol] = f"Polling Error: {str(e)[:30]}"
                        time.sleep(60)  # Wait longer on error
            
            # Start polling thread
            thread = threading.Thread(target=poll_data, daemon=True)
            thread.start()
            
            self.clients[symbol] = thread
            self.connection_status[symbol] = "Starting Polling... 🔄"
            return True
            
        except Exception as e:
            self.connection_status[symbol] = f"Polling Start Error: {str(e)[:30]}"
            return False
    
    def get_price(self, symbol):
        """Get latest price"""
        return self.latest_price.get(symbol)
        
    def get_status(self, symbol):
        """Get connection status"""
        return self.connection_status.get(symbol, "Not Connected")
        
    def get_price_history(self, symbol, limit=10):
        """Get recent price history"""
        history = self.price_history.get(symbol, [])
        return list(history)[-limit:] if history else []
        
    def is_connected(self, symbol):
        """Check if connected"""
        status = self.get_status(symbol)
        return "✅" in status
        
    def disconnect(self, symbol):
        """Disconnect stream"""
        try:
            if symbol in self.clients:
                client = self.clients[symbol]
                if hasattr(client, 'disconnect'):
                    client.disconnect()
                del self.clients[symbol]
            self.connection_status[symbol] = "Disconnected"
            if symbol in self.latest_price:
                del self.latest_price[symbol]
        except Exception as e:
            pass

# Initialize Enhanced WebSocket manager
if 'ws_manager' not in st.session_state:
    st.session_state.ws_manager = EnhancedWebSocketManager()

# ==============================================================================
# SMART AI ENTRY PATCH v8.0 FUNCTIONS
# ==============================================================================
def calculate_smart_entry_price(signal, recent_data, predicted_price, confidence, symbol="XAUUSD"):
    """
    🧠 SMART AI ENTRY PRICE CALCULATION
    
    Menghitung entry price berdasarkan multiple factors:
    - Support/Resistance levels
    - RSI conditions (oversold/overbought)
    - MACD momentum
    - ATR volatility buffer
    - AI confidence adjustment
    
    Args:
        signal: 'BUY' atau 'SELL'
        recent_data: DataFrame dengan OHLC dan indicators
        predicted_price: LSTM prediction price
        confidence: AI confidence level (0-1)
        symbol: Trading symbol untuk pip calculation
        
    Returns:
        dict: {
            'entry_price': float,
            'strategy_reasons': list,
            'risk_level': str,
            'expected_fill_probability': float
        }
    """
    try:
        current_price = recent_data['close'].iloc[-1]
        
        # Get technical indicators
        rsi = recent_data['rsi'].iloc[-1] if 'rsi' in recent_data.columns else 50
        atr = recent_data['ATR_14'].iloc[-1] if 'ATR_14' in recent_data.columns else current_price * 0.01
        macd = recent_data['MACD_12_26_9'].iloc[-1] if 'MACD_12_26_9' in recent_data.columns else 0
        macd_signal = recent_data['MACDs_12_26_9'].iloc[-1] if 'MACDs_12_26_9' in recent_data.columns else 0
        
        # Get Support/Resistance levels
        supports, resistances = get_support_resistance(recent_data)
        
        # Initialize strategy reasons
        strategy_reasons = []
        risk_level = "MEDIUM"
        
        if signal == "BUY":
            entry_price, reasons, risk = _calculate_buy_entry(
                current_price, predicted_price, supports, resistances,
                rsi, atr, macd, macd_signal, confidence
            )
        elif signal == "SELL":
            entry_price, reasons, risk = _calculate_sell_entry(
                current_price, predicted_price, supports, resistances,
                rsi, atr, macd, macd_signal, confidence
            )
        else:
            return {
                'entry_price': current_price,
                'strategy_reasons': ['HOLD signal - no entry'],
                'risk_level': 'LOW',
                'expected_fill_probability': 0.0
            }
        
        strategy_reasons.extend(reasons)
        risk_level = risk
        
        # Calculate expected fill probability
        price_distance = abs(entry_price - current_price) / current_price
        fill_probability = max(0.1, 1.0 - (price_distance * 20))  # Higher chance if closer to current price
        
        # Validate entry price reasonableness
        max_deviation = 0.02  # 2% max deviation from current price
        if price_distance > max_deviation:
            st.warning(f"⚠️ Smart entry price terlalu jauh dari current price ({price_distance:.2%}). Adjusting...")
            if signal == "BUY":
                entry_price = current_price + (current_price * 0.001)  # Small premium
            else:
                entry_price = current_price - (current_price * 0.001)  # Small discount
            strategy_reasons.append("Entry adjusted for realism")
            fill_probability = 0.9
        
        return {
            'entry_price': entry_price,
            'strategy_reasons': strategy_reasons,
            'risk_level': risk_level,
            'expected_fill_probability': fill_probability
        }
        
    except Exception as e:
        st.error(f"❌ Error in smart entry calculation: {e}")
        return {
            'entry_price': current_price,
            'strategy_reasons': [f'Fallback to current price due to error: {str(e)}'],
            'risk_level': 'HIGH',
            'expected_fill_probability': 0.5
        }

def _calculate_buy_entry(current_price, predicted_price, supports, resistances, 
                        rsi, atr, macd, macd_signal, confidence):
    """
    Calculate optimal BUY entry price
    """
    strategy_reasons = []
    risk_level = "MEDIUM"
    
    # Base entry calculation
    base_entry = current_price
    
    # 1. Support level analysis
    valid_supports = supports[supports <= current_price * 1.005]  # Within 0.5%
    if not valid_supports.empty:
        nearest_support = valid_supports.max()
        support_buffer = atr * 0.1  # Small buffer above support
        base_entry = nearest_support + support_buffer
        strategy_reasons.append(f"Entry above support: ${nearest_support:.2f}")
        risk_level = "LOW"
    else:
        # No nearby support, use current price with small premium
        base_entry = current_price + (atr * 0.05)
        strategy_reasons.append("Entry at market with small premium")
        risk_level = "MEDIUM"
    
    # 2. RSI condition adjustment
    rsi_adjustment = 0
    if rsi < 30:  # Oversold - great for BUY
        rsi_adjustment = -atr * 0.2  # Better entry price
        strategy_reasons.append(f"RSI oversold advantage ({rsi:.1f})")
        risk_level = "LOW"
    elif rsi < 40:  # Mild oversold
        rsi_adjustment = -atr * 0.1
        strategy_reasons.append(f"RSI favorable ({rsi:.1f})")
    elif rsi > 70:  # Overbought - risky for BUY
        rsi_adjustment = atr * 0.1  # Premium for risky entry
        strategy_reasons.append(f"RSI overbought risk ({rsi:.1f})")
        risk_level = "HIGH"
    
    # 3. MACD momentum check
    macd_adjustment = 0
    if macd > macd_signal:  # Bullish momentum
        macd_adjustment = atr * 0.05  # Small premium for momentum
        strategy_reasons.append("MACD bullish momentum")
    else:
        strategy_reasons.append("MACD neutral/bearish")
    
    # 4. Confidence factor
    confidence_adjustment = (confidence - 0.5) * atr * 0.1  # -0.05 to +0.05 ATR
    if confidence > 0.8:
        strategy_reasons.append(f"High confidence ({confidence:.1%})")
    elif confidence < 0.6:
        strategy_reasons.append(f"Low confidence ({confidence:.1%})")
        risk_level = "HIGH"
    
    # 5. Predicted price consideration
    pred_adjustment = 0
    if predicted_price > current_price:
        pred_factor = min(0.3, (predicted_price - current_price) / current_price)
        pred_adjustment = pred_factor * atr * 0.5
        strategy_reasons.append("AI predicts price increase")
    
    # Final entry price calculation
    final_entry = base_entry + rsi_adjustment + macd_adjustment + confidence_adjustment + pred_adjustment
    
    return final_entry, strategy_reasons, risk_level

def _calculate_sell_entry(current_price, predicted_price, supports, resistances,
                         rsi, atr, macd, macd_signal, confidence):
    """
    Calculate optimal SELL entry price
    """
    strategy_reasons = []
    risk_level = "MEDIUM"
    
    # Base entry calculation
    base_entry = current_price
    
    # 1. Resistance level analysis
    valid_resistances = resistances[resistances >= current_price * 0.995]  # Within 0.5%
    if not valid_resistances.empty:
        nearest_resistance = valid_resistances.min()
        resistance_buffer = atr * 0.1  # Small buffer below resistance
        base_entry = nearest_resistance - resistance_buffer
        strategy_reasons.append(f"Entry below resistance: ${nearest_resistance:.2f}")
        risk_level = "LOW"
    else:
        # No nearby resistance, use current price with small discount
        base_entry = current_price - (atr * 0.05)
        strategy_reasons.append("Entry at market with small discount")
        risk_level = "MEDIUM"
    
    # 2. RSI condition adjustment
    rsi_adjustment = 0
    if rsi > 70:  # Overbought - great for SELL
        rsi_adjustment = atr * 0.2  # Better entry price (higher)
        strategy_reasons.append(f"RSI overbought advantage ({rsi:.1f})")
        risk_level = "LOW"
    elif rsi > 60:  # Mild overbought
        rsi_adjustment = atr * 0.1
        strategy_reasons.append(f"RSI favorable ({rsi:.1f})")
    elif rsi < 30:  # Oversold - risky for SELL
        rsi_adjustment = -atr * 0.1  # Discount for risky entry
        strategy_reasons.append(f"RSI oversold risk ({rsi:.1f})")
        risk_level = "HIGH"
    
    # 3. MACD momentum check
    macd_adjustment = 0
    if macd < macd_signal:  # Bearish momentum
        macd_adjustment = atr * 0.05  # Small premium (higher entry)
        strategy_reasons.append("MACD bearish momentum")
    else:
        strategy_reasons.append("MACD neutral/bullish")
    
    # 4. Confidence factor
    confidence_adjustment = (confidence - 0.5) * atr * 0.1
    if confidence > 0.8:
        strategy_reasons.append(f"High confidence ({confidence:.1%})")
    elif confidence < 0.6:
        strategy_reasons.append(f"Low confidence ({confidence:.1%})")
        risk_level = "HIGH"
    
    # 5. Predicted price consideration
    pred_adjustment = 0
    if predicted_price < current_price:
        pred_factor = min(0.3, (current_price - predicted_price) / current_price)
        pred_adjustment = pred_factor * atr * 0.5
        strategy_reasons.append("AI predicts price decrease")
    
    # Final entry price calculation
    final_entry = base_entry + rsi_adjustment + macd_adjustment + confidence_adjustment + pred_adjustment
    
    return final_entry, strategy_reasons, risk_level

def display_smart_signal_results(signal, confidence, smart_entry_result, position_info, symbol):
    """
    Enhanced UI display dengan Smart AI strategy reasoning
    """
    if signal == "HOLD":
        st.info("🔄 **HOLD** - Menunggu opportunity yang lebih baik")
        return
    
    # Main signal display
    signal_color = "🟢" if signal == "BUY" else "🔴"
    confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
    
    st.markdown(f"""
    ## {signal_color} **{signal} SIGNAL**
    **Confidence:** {confidence:.1%} `{confidence_bar}`
    """)
    
    # Smart Entry Information
    entry_price = smart_entry_result['entry_price']
    fill_probability = smart_entry_result['expected_fill_probability']
    risk_level = smart_entry_result['risk_level']
    
    # Color coding for risk level
    risk_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
    risk_color = risk_colors.get(risk_level, "⚪")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Smart Entry Price",
            format_price(symbol, entry_price),
            help="AI-optimized entry price based on multiple factors"
        )
    
    with col2:
        st.metric(
            "Fill Probability",
            f"{fill_probability:.1%}",
            help="Estimated probability of order execution"
        )
    
    with col3:
        st.metric(
            f"{risk_color} Risk Level",
            risk_level,
            help="Entry risk assessment based on market conditions"
        )
    
    # Strategy Reasoning
    st.markdown("### 🧠 **Smart Entry Strategy**")
    
    for i, reason in enumerate(smart_entry_result['strategy_reasons'], 1):
        st.markdown(f"**{i}.** {reason}")
    
    # Position Details
    if position_info:
        st.markdown("### 📊 **Position Details**")
        
        detail_cols = st.columns(4)
        detail_cols[0].metric("Position Size", f"{position_info['position_size']:.4f} {symbol.split('/')[0] if '/' in symbol else 'units'}")
        detail_cols[1].metric("Stop Loss", format_price(symbol, position_info['stop_loss']))
        detail_cols[2].metric("Take Profit", format_price(symbol, position_info['take_profit']))
        detail_cols[3].metric("Risk Amount", f"${position_info['risk_amount']:.2f}")
        
        # Risk-Reward Ratio
        sl_dist = abs(position_info['entry_price'] - position_info['stop_loss'])
        tp_dist = abs(position_info['take_profit'] - position_info['entry_price'])
        rr_ratio = (tp_dist / sl_dist) if sl_dist > 0 else 0
        rr_color = "🟢" if rr_ratio >= 2 else "🟡" if rr_ratio >= 1.5 else "🔴"
        st.metric(f"{rr_color} Risk:Reward Ratio", f"1:{rr_ratio:.2f}")

# ==============================================================================
# PATCHED FUNCTIONS from trading_bot_patches.py (v7.4)
# ==============================================================================

def validate_trading_inputs(symbol, balance, risk_percent, sl_pips, tp_pips):
    """
    Validasi input trading untuk mencegah error dan memberikan warning
    """
    issues = []
    
    if balance < 100:
        issues.append("⚠️ Balance terlalu kecil (< $100). Minimal $500 direkomendasikan.")
    
    if risk_percent > 10:
        issues.append("🚨 Risk per trade > 10% sangat berbahaya! Recommended: 1-3%")
    elif risk_percent > 5:
        issues.append("⚠️ Risk per trade > 5% cukup tinggi. Pertimbangkan untuk menurunkan.")
    
    if sl_pips < 5:
        issues.append("⚠️ Stop Loss terlalu ketat (< 5 pips). Mungkin sering terkena noise.")
    
    if tp_pips < sl_pips:
        issues.append("⚠️ Take Profit lebih kecil dari Stop Loss. R:R ratio tidak optimal.")
    
    if tp_pips > sl_pips * 5:
        issues.append("⚠️ Take Profit terlalu jauh. Mungkin jarang tercapai.")
    
    return issues

def calculate_position_info(signal, symbol, entry_price, sl_pips, tp_pips, balance, risk_percent, conversion_rate_to_usd, take_profit_price=None, leverage=20):
    """
    IMPROVED: Menghitung informasi posisi dengan validasi yang lebih baik dan error handling
    """
    if signal == "HOLD":
        return None
        
    if conversion_rate_to_usd is None or conversion_rate_to_usd <= 0:
        st.error(f"❌ Conversion rate invalid: {conversion_rate_to_usd}")
        return None
    
    if entry_price <= 0:
        st.error(f"❌ Entry price invalid: {entry_price}")
        return None

    try:
        # Tentukan pip_size
        def get_pip_value_improved(symbol, price):
            symbol_upper = symbol.replace('/', '').upper()
            if symbol_upper == "BTCUSDT":
                return 0.1
            elif 'JPY' in symbol_upper:
                return 0.01
            elif 'XAU' in symbol_upper:
                return 0.01
            elif 'ETH' in symbol_upper:
                return 0.01
            else:
                return 0.0001

        pip_size = get_pip_value_improved(symbol, entry_price)
        sl_distance = sl_pips * pip_size

        # Stop loss price calculation dengan validasi
        if signal == "BUY":
            stop_loss_price = entry_price - sl_distance
            if stop_loss_price <= 0:
                st.error("❌ Stop loss price tidak valid (≤ 0)")
                return None
        else:
            stop_loss_price = entry_price + sl_distance

        # Take profit price calculation
        if take_profit_price:
            tp_price = take_profit_price
            # Validasi TP price masuk akal
            if signal == "BUY" and tp_price <= entry_price:
                st.warning("⚠️ TP price tidak valid untuk BUY signal, menggunakan default")
                tp_price = entry_price + (tp_pips * pip_size)
            elif signal == "SELL" and tp_price >= entry_price:
                st.warning("⚠️ TP price tidak valid untuk SELL signal, menggunakan default")
                tp_price = entry_price - (tp_pips * pip_size)
        else:
            tp_price = entry_price + (tp_pips * pip_size) if signal == "BUY" else entry_price - (tp_pips * pip_size)

        # Position size calculation dengan validasi
        risk_amount = balance * (risk_percent / 100)
        sl_distance_price = abs(entry_price - stop_loss_price)
        
        if sl_distance_price == 0:
            st.error("❌ Stop loss distance tidak boleh 0")
            return None

        # BTCUSDT perpetual futures calculation
        if symbol.replace('/', '').upper() == "BTCUSDT":
            position_size = risk_amount / sl_distance_price
            max_position_size = (balance * leverage) / entry_price
            if position_size > max_position_size:
                position_size = max_position_size
                st.warning(f"⚠️ Position size dikurangi karena leverage limit: {position_size:.4f}")
        else:
            # Forex/Spot calculation
            risk_amount_usd = risk_amount
            position_size = risk_amount_usd / (sl_distance_price * conversion_rate_to_usd)
            max_position_value_usd = balance * leverage
            position_value_usd = position_size * entry_price * conversion_rate_to_usd
            
            if position_value_usd > max_position_value_usd:
                position_size = max_position_value_usd / (entry_price * conversion_rate_to_usd)
                st.warning(f"⚠️ Position size dikurangi karena leverage limit: {position_size:.4f}")

        # Final validation
        if position_size <= 0:
            st.error("❌ Position size tidak valid (≤ 0)")
            return None

        # Minimum position size check
        min_position_value = 10  # $10 minimum
        if position_size * entry_price * conversion_rate_to_usd < min_position_value:
            st.warning(f"⚠️ Position size sangat kecil (< ${min_position_value})")

        return {
            'signal': signal,
            'entry_price': entry_price,
            'stop_loss': stop_loss_price,
            'take_profit': tp_price,
            'position_size': position_size,
            'risk_amount': risk_amount,
            'pip_size': pip_size,
            'conversion_rate': conversion_rate_to_usd
        }

    except Exception as e:
        st.error(f"❌ Error dalam perhitungan posisi: {str(e)}")
        return None

def calculate_ai_take_profit(signal, entry_price, supports, resistances, atr_value):
    """
    Calculate intelligent take profit based on S/R levels with proper validation
    """
    try:
        buffer_percent = 0.001  # 0.1% buffer untuk avoid false breakouts
        
        if signal == "BUY":
            # Cari resistance terdekat di atas entry + buffer
            valid_resistances = resistances[resistances > entry_price * (1 + buffer_percent)]
            if not valid_resistances.empty:
                nearest_resistance = valid_resistances.min()
                # Ensure reasonable distance (at least 1.5x ATR)
                min_tp_distance = entry_price + (1.5 * atr_value)
                return max(nearest_resistance, min_tp_distance)
            else:
                # Fallback: 2x ATR
                return entry_price + (2.0 * atr_value)
        
        elif signal == "SELL":
            # Cari support terdekat di bawah entry - buffer
            valid_supports = supports[supports < entry_price * (1 - buffer_percent)]
            if not valid_supports.empty:
                nearest_support = valid_supports.max()
                # Ensure reasonable distance
                max_tp_distance = entry_price - (1.5 * atr_value)
                return min(nearest_support, max_tp_distance)
            else:
                return entry_price - (2.0 * atr_value)
    
    except Exception as e:
        st.warning(f"AI TP calculation failed: {e}")
        return None


def execute_trade_exit_realistic(current_candle, active_trade, slippage=0.001):
    """
    Realistic trade exit dengan proper priority dan slippage simulation
    """
    try:
        high_price = current_candle['high']
        low_price = current_candle['low']
        
        if active_trade['type'] == 'BUY':
            # Check SL first (more likely dalam volatile market)
            if low_price <= active_trade['sl']:
                # Simulate slippage - worse fill
                exit_price = max(active_trade['sl'] * (1 - slippage), low_price)
                return exit_price, 'SL'
            # Then check TP
            elif high_price >= active_trade['tp']:
                # Simulate slippage - worse fill
                exit_price = min(active_trade['tp'] * (1 - slippage), high_price) 
                return exit_price, 'TP'
        
        elif active_trade['type'] == 'SELL':
            # Check SL first
            if high_price >= active_trade['sl']:
                exit_price = min(active_trade['sl'] * (1 + slippage), high_price)
                return exit_price, 'SL'
            elif low_price <= active_trade['tp']:
                exit_price = max(active_trade['tp'] * (1 + slippage), low_price)
                return exit_price, 'TP'
        
        return None, None
        
    except Exception as e:
        st.warning(f"Error in trade exit calculation: {e}")
        return None, None

def calculate_realistic_pnl(entry_price, exit_price, position_size, trade_type, symbol):
    """
    Calculate PnL dengan realistic transaction costs
    """
    try:
        # Basic PnL calculation
        if trade_type == 'BUY':
            gross_pnl = (exit_price - entry_price) * position_size
        else:
            gross_pnl = (entry_price - exit_price) * position_size
        
        # Transaction costs
        spread_cost = entry_price * 0.0001 * position_size  # 1 pip spread average
        
        # Commission calculation based on symbol type
        if 'USDT' in symbol.upper():
            commission = position_size * entry_price * 0.001  # 0.1% for crypto
        elif any(fx in symbol.upper() for fx in ['EUR', 'GBP', 'JPY']):
            commission = max(2.0, position_size * entry_price * 0.00005)  # Forex commission
        else:
            commission = max(1.0, position_size * entry_price * 0.0001)  # Default 0.01%
        
        # Swap/overnight cost (simplified - assume intraday for now)
        swap_cost = 0
        
        total_costs = spread_cost + commission + swap_cost
        net_pnl = gross_pnl - total_costs
        
        return {
            'net_pnl': net_pnl,
            'gross_pnl': gross_pnl,
            'spread_cost': spread_cost,
            'commission': commission,
            'swap_cost': swap_cost,
            'total_costs': total_costs
        }
        
    except Exception as e:
        st.warning(f"Error calculating PnL: {e}")
        return {
            'net_pnl': 0, 'gross_pnl': 0, 'spread_cost': 0,
            'commission': 0, 'swap_cost': 0, 'total_costs': 0
        }

def predict_with_models(models, data):
    """
    IMPROVED: Make predictions dengan better error handling dan validation
    """
    try:
        min_data_needed = 60
        if len(data) < min_data_needed + 1:
            return "HOLD", 0.5, data['close'].iloc[-1] if not data.empty else 0

        if models.get('meta') is None:
            st.warning("⚠️ Meta learner model tidak tersedia")
            return "HOLD", 0.5, data['close'].iloc[-1]

        # LSTM Prediction dengan validation
        try:
            sequence_length_lstm = 60
            if len(data) < sequence_length_lstm:
                return "HOLD", 0.5, data['close'].iloc[-1]
                
            last_lstm_data = data['close'].tail(sequence_length_lstm).values
            
            if models.get('scaler') is None:
                st.warning("⚠️ LSTM scaler tidak tersedia")
                return "HOLD", 0.5, data['close'].iloc[-1]
                
            scaled_lstm_data = models['scaler'].transform(last_lstm_data.reshape(-1, 1))
            X_lstm = np.reshape(scaled_lstm_data, (1, sequence_length_lstm, 1))
            
            lstm_prediction_scaled = models['lstm'].predict(X_lstm, verbose=0)[0][0]
            lstm_prediction_price = models['scaler'].inverse_transform([[lstm_prediction_scaled]])[0][0]
            
            # Validate LSTM prediction
            current_price = data['close'].iloc[-1]
            if abs(lstm_prediction_price - current_price) / current_price > 0.1:  # >10% change tidak masuk akal
                st.warning("⚠️ LSTM prediction tidak realistis, menggunakan harga saat ini")
                lstm_prediction_price = current_price
                
            lstm_pred_diff = (lstm_prediction_price - current_price) / current_price
            
        except Exception as e:
            st.warning(f"⚠️ LSTM prediction error: {e}")
            lstm_pred_diff = 0
            lstm_prediction_price = data['close'].iloc[-1]

        # XGBoost Prediction
        try:
            xgb_feature_names = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 'ATR_14', 'STOCHk_14_3_3', 'price_change', 'high_low_ratio', 'open_close_ratio']
            temp_df_xgb = data.copy()
            
            # Add derived features if missing
            if 'price_change' not in temp_df_xgb.columns: 
                temp_df_xgb['price_change'] = temp_df_xgb['close'].pct_change()
            if 'high_low_ratio' not in temp_df_xgb.columns: 
                temp_df_xgb['high_low_ratio'] = temp_df_xgb['high'] / temp_df_xgb['low']
            if 'open_close_ratio' not in temp_df_xgb.columns: 
                temp_df_xgb['open_close_ratio'] = temp_df_xgb['open'] / temp_df_xgb['close']
            
            available_xgb_features = [f for f in xgb_feature_names if f in temp_df_xgb.columns]
            X_xgb_latest = temp_df_xgb[available_xgb_features].tail(1).fillna(0)
            
            if not X_xgb_latest.empty and models.get('xgb'):
                xgb_pred = models['xgb'].predict(X_xgb_latest)[0]
                xgb_confidence = models['xgb'].predict_proba(X_xgb_latest)[0][xgb_pred]
            else:
                xgb_pred = 0
                xgb_confidence = 0.5
                
        except Exception as e:
            st.warning(f"⚠️ XGBoost prediction error: {e}")
            xgb_pred = 0
            xgb_confidence = 0.5

        # Other models predictions with simplified error handling
        try:
            cnn_pred, svc_pred, nb_pred = 0, 0, 0
            svc_confidence, nb_confidence = 0.5, 0.5

            # CNN
            cnn_feature_names = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20']
            available_cnn_features = [f for f in cnn_feature_names if f in data.columns]
            sequence_length_cnn = 20
            if len(data) >= sequence_length_cnn and models.get('cnn') and models.get('cnn_scaler'):
                last_cnn_data_features = data[available_cnn_features].tail(sequence_length_cnn).fillna(0)
                scaled_cnn_data = models['cnn_scaler'].transform(last_cnn_data_features)
                X_cnn = np.reshape(scaled_cnn_data, (1, sequence_length_cnn, len(available_cnn_features)))
                cnn_pred_proba = models['cnn'].predict(X_cnn, verbose=0)[0][0]
                cnn_pred = 1 if cnn_pred_proba > 0.5 else 0

            # SVC
            svc_feature_names = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 'ATR_14', 'STOCHk_14_3_3']
            available_svc_features = [f for f in svc_feature_names if f in data.columns]
            if models.get('svc') and models.get('svc_scaler'):
                X_svc_latest = data[available_svc_features].tail(1).fillna(0)
                if not X_svc_latest.empty:
                    X_svc_latest_scaled = models['svc_scaler'].transform(X_svc_latest)
                    svc_pred = models['svc'].predict(X_svc_latest_scaled)[0]
                    svc_confidence = models['svc'].predict_proba(X_svc_latest_scaled)[0][svc_pred]

            # Naive Bayes
            nb_feature_names = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 'ATR_14', 'STOCHk_14_3_3']
            available_nb_features = [f for f in nb_feature_names if f in data.columns]
            if models.get('nb'):
                X_nb_latest = data[available_nb_features].tail(1).fillna(0)
                if not X_nb_latest.empty:
                    nb_pred = models['nb'].predict(X_nb_latest)[0]
                    nb_confidence = models['nb'].predict_proba(X_nb_latest)[0][nb_pred]

        except Exception as e:
            st.warning(f"⚠️ Other models prediction error: {e}")
            cnn_pred = svc_pred = nb_pred = 0
            svc_confidence = nb_confidence = 0.5

        # Meta learner prediction
        try:
            current_data_point = data.tail(1).copy()
            
            # Create prediction series
            lstm_preds_single = pd.Series(lstm_pred_diff, index=current_data_point.index)
            xgb_preds_single = pd.Series(xgb_pred, index=current_data_point.index)
            xgb_conf_single = pd.Series(xgb_confidence, index=current_data_point.index)
            cnn_preds_single = pd.Series(cnn_pred, index=current_data_point.index)
            svc_preds_single = pd.Series(svc_pred, index=current_data_point.index)
            svc_conf_single = pd.Series(svc_confidence, index=current_data_point.index)
            nb_preds_single = pd.Series(nb_pred, index=current_data_point.index)
            nb_conf_single = pd.Series(nb_confidence, index=current_data_point.index)
            
            X_meta_pred, _ = prepare_data_for_meta_learner(
                current_data_point, lstm_preds_single, xgb_preds_single, xgb_conf_single,
                cnn_preds_single, svc_preds_single, svc_conf_single, nb_preds_single, nb_conf_single
            )

            if not X_meta_pred.empty:
                meta_signal_series = get_meta_signal(
                    current_data_point, lstm_preds_single, xgb_preds_single, xgb_conf_single,
                    cnn_preds_single, svc_preds_single, svc_conf_single, nb_preds_single, nb_conf_single,
                    models['meta']
                )
                meta_signal_numeric = meta_signal_series.iloc[0]

                final_signal = "BUY" if meta_signal_numeric == 1 else "SELL" if meta_signal_numeric == -1 else "HOLD"
                
                # Calculate confidence from meta model
                final_confidence = 0.5
                if final_signal != "HOLD":
                    meta_proba = models['meta'].predict_proba(X_meta_pred)[0]
                    class_index = np.where(models['meta'].classes_ == meta_signal_numeric)[0]
                    if len(class_index) > 0:
                        final_confidence = meta_proba[class_index[0]]
                
                return final_signal, final_confidence, lstm_prediction_price
            else:
                return "HOLD", 0.5, lstm_prediction_price
                
        except Exception as e:
            st.warning(f"⚠️ Meta learner error: {e}")
            return "HOLD", 0.5, lstm_prediction_price

    except Exception as e:
        st.error(f"❌ Critical error in model prediction: {str(e)}")
        return "HOLD", 0.5, data['close'].iloc[-1] if not data.empty else 0

# ==============================================================================
# ENHANCED DATA LOADING FUNCTION (UPDATED)
# ==============================================================================

def get_binance_data(api_key, api_secret, interval, symbol, outputsize=500):
    """
    Mengambil data historis dari Binance dan memformatnya.
    Kini mendukung pengambilan data > 1000 dengan pagination.
    """
    try:
        client = Client(api_key, api_secret)
        interval_map = {
            '1min': Client.KLINE_INTERVAL_1MINUTE, 
            '5min': Client.KLINE_INTERVAL_5MINUTE,
            '15min': Client.KLINE_INTERVAL_15MINUTE, 
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR, 
            '1day': Client.KLINE_INTERVAL_1DAY,
        }
        binance_interval = interval_map.get(interval)
        if not binance_interval:
            raise ValueError(f"Interval {interval} tidak didukung oleh Binance.")

        all_klines = []
        end_time = None
        
        while len(all_klines) < outputsize:
            limit = min(outputsize - len(all_klines), 1000)
            klines = client.get_klines(
                symbol=symbol, 
                interval=binance_interval, 
                limit=limit, 
                endTime=end_time
            )
            
            if not klines:
                break
                
            all_klines = klines + all_klines
            end_time = klines[0][0]
            time.sleep(0.1)

        if not all_klines:
            return None

        cols = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        df = pd.DataFrame(all_klines, columns=cols)
        df.drop_duplicates(subset='timestamp', keep='first', inplace=True)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[numeric_cols]
        df = df.tail(outputsize)

        return df.sort_index()

    except Exception as e:
        st.error(f"Error saat mengambil data dari Binance: {e}")
        return None

@st.cache_data(ttl=300)
def load_and_process_data_enhanced(api_source, symbol, interval, api_key_1, api_key_2=None, outputsize=500):
    """Enhanced data loading v8.3 - Simplified untuk Twelve Data dan Binance saja"""
    data = None
    try:
        if api_source == 'Twelve Data':
            if not api_key_1 or api_key_1.strip() == "":
                raise ValueError("API Key Twelve Data tidak valid")
            data = get_gold_data(api_key_1, interval=interval, symbol=symbol, outputsize=outputsize)
        
        elif api_source == 'Binance':
            if not BINANCE_AVAILABLE:
                raise ValueError("Binance library tidak tersedia. Install dengan: pip install python-binance")
            if not api_key_1 or not api_key_2 or api_key_1.strip() == "" or api_key_2.strip() == "":
                raise ValueError("API Key & Secret Binance tidak valid")
            symbol_binance = symbol.replace('/', '')
            data = get_binance_data(api_key_1, api_key_2, interval, symbol_binance, outputsize)

        if data is None or data.empty:
            raise ValueError(f"Data kosong dari {api_source}. Periksa koneksi, API Key, atau format simbol.")

        # Rest of the processing remains the same
        required_cols = ['open', 'high', 'low', 'close']
        if 'volume' in data.columns:
            required_cols.append('volume')

        for col in required_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            else:
                raise ValueError(f"Data tidak lengkap, kolom '{col}' tidak ditemukan.")

        data.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

        if data.empty:
            raise ValueError("Data menjadi kosong setelah pembersihan. Kemungkinan format data dari API salah atau tidak ada data valid.")

        if 'XAU' in symbol.upper() and (data['close'] > 10000).any():
            st.warning(f"Peringatan: Terdeteksi harga Emas > $10,000. Data mungkin tidak akurat.")
        if 'EUR' in symbol.upper() and (data['close'] > 2).any():
            st.warning(f"Peringatan: Terdeteksi harga EUR/USD > 2.0. Data mungkin tidak akurat.")

        data = add_indicators(data)

        try:
            data.ta.stoch(append=True)
        except Exception as e:
            st.warning(f"Gagal menghitung Stochastic Oscillator: {e}")

        if 'BBL_20_2.0' not in data.columns or 'BBU_20_2.0' not in data.columns:
            try:
                bbands = data.ta.bbands(length=20, std=2)
                if bbands is not None and not bbands.empty:
                    data = data.join(bbands)
            except Exception:
                pass

        if 'ema_fast' not in data.columns: data["ema_fast"] = data["close"].ewm(span=8, adjust=False).mean()
        if 'ema_slow' not in data.columns: data["ema_slow"] = data["close"].ewm(span=21, adjust=False).mean()

        if 'ema_fast' in data.columns and 'ema_slow' in data.columns:
            data["ema_signal"] = data.apply(lambda row: "BUY" if row["ema_fast"] > row["ema_slow"] else "SELL" if row["ema_fast"] < row["ema_slow"] else "HOLD", axis=1)
            data['ema_signal_numeric'] = data['ema_signal'].map({"BUY": 1, "SELL": -1, "HOLD": 0}).fillna(0)
        else:
            data['ema_signal_numeric'] = 0

        if 'MACDh_12_26_9' not in data.columns:
            data.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)

        if 'ADX_14' not in data.columns:
            data.ta.adx(close='close', length=14, append=True)

        if all(col in data.columns for col in ['BBL_20_2.0', 'BBU_20_2.0']):
            data['bb_percent'] = (data['close'] - data['BBL_20_2.0']) / (data['BBU_20_2.0'] - data['BBL_20_2.0'])
            data['bb_percent'] = data['bb_percent'].fillna(0.5)
        else:
            data['bb_percent'] = 0.5

        supports, resistances = get_support_resistance(data)

        data['dist_to_support'] = np.nan
        data['dist_to_resistance'] = np.nan

        for idx in data.index:
            current_price = data.loc[idx, 'close']

            sup_below = supports[supports < current_price]
            if not sup_below.empty:
                data.loc[idx, 'dist_to_support'] = (current_price - sup_below.max()) / current_price
            else:
                data.loc[idx, 'dist_to_support'] = 1.0

            res_above = resistances[resistances > current_price]
            if not res_above.empty:
                data.loc[idx, 'dist_to_resistance'] = (res_above.min() - current_price) / current_price
            else:
                data.loc[idx, 'dist_to_resistance'] = 1.0

                data['dist_to_support'] = data['dist_to_support'].fillna(1.0)
        data['dist_to_resistance'] = data['dist_to_resistance'].fillna(1.0)

        look_forward_periods = 5
        price_future = data['close'].shift(-look_forward_periods)
        price_change_pct = (price_future - data['close']) / data['close']
        
        change_threshold = 0.0005

        conditions = [
            price_change_pct > change_threshold,
            price_change_pct < -change_threshold
        ]
        choices = [1, -1]
        data['target_meta'] = np.select(conditions, choices, default=0)

        data = data.fillna(method='ffill').fillna(method='bfill')
        data = data.fillna(0)

        return data
    except Exception as e:
        st.error(f"Gagal memuat atau memproses data: {str(e)}")
        return None

def create_simple_lstm_model(input_shape):
    """Create a simple LSTM model"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_simple_lstm(data, symbol, timeframe):
    """Train a simple LSTM model"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

    sequence_length = 60
    X, y = [], []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = create_simple_lstm_model((X.shape[1], 1))
    model.fit(X, y, batch_size=32, epochs=5, verbose=0)

    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    symbol_fn = sanitize_filename(symbol)

    model.save(os.path.join(model_dir, f'lstm_model_{symbol_fn}_{timeframe}.keras'))
    joblib.dump(scaler, os.path.join(model_dir, f'scaler_{symbol_fn}_{timeframe}.pkl'))

    return model, scaler

def train_simple_xgboost(data, symbol, timeframe):
    """Train a simple XGBoost model"""
    df = data.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()

    if len(df) < 50:
        raise ValueError(f"Insufficient data for training: {len(df)} rows")

    features = ['open', 'high', 'low', 'close']
    if 'rsi' in df.columns: features.append('rsi')
    if 'MACD_12_26_9' in df.columns: features.append('MACD_12_26_9')
    if 'EMA_10' in df.columns: features.append('EMA_10')
    if 'EMA_20' in df.columns: features.append('EMA_20')
    if 'ATR_14' in df.columns: features.append('ATR_14')
    if 'STOCHk_14_3_3' in df.columns: features.append('STOCHk_14_3_3')

    df['price_change'] = df['close'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['open_close_ratio'] = df['open'] / df['close']

    features.extend(['price_change', 'high_low_ratio', 'open_close_ratio'])

    available_features = [f for f in features if f in df.columns]
    X = df[available_features].fillna(0)
    y = df['target']

    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss'
    )
    model.fit(X, y)

    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    symbol_fn = sanitize_filename(symbol)

    model.save_model(os.path.join(model_dir, f'xgboost_model_{symbol_fn}_{timeframe}.json'))

    return model

def create_simple_cnn_model(input_shape):
    """Create a simple CNN model"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_simple_cnn(data, symbol, timeframe):
    """Train a simple CNN model"""
    df = data.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()

    if len(df) < 100:
        raise ValueError(f"Insufficient data for CNN training: {len(df)} rows")

    feature_cols = ['open', 'high', 'low', 'close']
    if 'rsi' in df.columns: feature_cols.append('rsi')
    if 'MACD_12_26_9' in df.columns: feature_cols.append('MACD_12_26_9')
    if 'EMA_10' in df.columns: feature_cols.append('EMA_10')
    if 'EMA_20' in df.columns: feature_cols.append('EMA_20')

    available_features = [f for f in feature_cols if f in df.columns]
    if len(available_features) < len(feature_cols):
        st.warning(f"Beberapa fitur CNN tidak tersedia, training mungkin tidak optimal.")

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[available_features].fillna(0))

    sequence_length = 20
    X, y = [], []

    for i in range(sequence_length, len(scaled_features)-1):
        X.append(scaled_features[i-sequence_length:i])
        y.append(df['target'].iloc[i])

    X, y = np.array(X), np.array(y)

    model = create_simple_cnn_model((X.shape[1], X.shape[2]))
    model.fit(X, y, batch_size=16, epochs=3, verbose=0, validation_split=0.2)

    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    symbol_fn = sanitize_filename(symbol)

    model.save(os.path.join(model_dir, f'cnn_model_{symbol_fn}_{timeframe}.keras'))
    
    joblib.dump(scaler, os.path.join(model_dir, f'cnn_scaler_{symbol_fn}_{timeframe}.pkl'))

    return model

def train_simple_svc(data, symbol, timeframe):
    """Train a simple Support Vector Classifier model"""
    df = data.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()

    if len(df) < 50:
        raise ValueError(f"Insufficient data for SVC training: {len(df)} rows")

    features = ['open', 'high', 'low', 'close']
    if 'rsi' in df.columns: features.append('rsi')
    if 'MACD_12_26_9' in df.columns: features.append('MACD_12_26_9')
    if 'EMA_10' in df.columns: features.append('EMA_10')
    if 'EMA_20' in df.columns: features.append('EMA_20')
    if 'ATR_14' in df.columns: features.append('ATR_14')
    if 'STOCHk_14_3_3' in df.columns: features.append('STOCHk_14_3_3')

    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features):
        st.warning(f"Beberapa fitur SVC tidak tersedia, training mungkin tidak optimal.")

    X = df[available_features].fillna(0)
    y = df['target']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_scaled, y)

    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    symbol_fn = sanitize_filename(symbol)

    joblib.dump(model, os.path.join(model_dir, f'svc_model_{symbol_fn}_{timeframe}.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, f'svc_scaler_{symbol_fn}_{timeframe}.pkl'))

    return model, scaler

def train_simple_naive_bayes(data, symbol, timeframe):
    """Train a simple Gaussian Naive Bayes model"""
    df = data.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()

    if len(df) < 50:
        raise ValueError(f"Insufficient data for Naive Bayes training: {len(df)} rows")

    features = ['open', 'high', 'low', 'close']
    if 'rsi' in df.columns: features.append('rsi')
    if 'MACD_12_26_9' in df.columns: features.append('MACD_12_26_9')
    if 'EMA_10' in df.columns: features.append('EMA_10')
    if 'EMA_20' in df.columns: features.append('EMA_20')
    if 'ATR_14' in df.columns: features.append('ATR_14')
    if 'STOCHk_14_3_3' in df.columns: features.append('STOCHk_14_3_3')

    available_features = [f for f in features if f in df.columns]

    X = df[available_features].fillna(0)
    y = df['target']

    model = GaussianNB()
    model.fit(X, y)

    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    symbol_fn = sanitize_filename(symbol)

    joblib.dump(model, os.path.join(model_dir, f'nb_model_{symbol_fn}_{timeframe}.pkl'))

    return model

def sanitize_filename(name):
    """Membersihkan string untuk digunakan sebagai nama file yang aman."""
    return re.sub(r'[^a-zA-Z0-9_-]', '', name.replace('/', '_').replace('\\', '_'))

def format_price(symbol, price):
    """
    Format price berdasarkan symbol
    """
    try:
        price = float(price)
        if 'XAU' in symbol.upper() or 'GOLD' in symbol.upper():
            return f"${price:,.2f}"
        elif 'BTC' in symbol.upper():
            return f"${price:,.1f}"
        elif 'ETH' in symbol.upper():
            return f"${price:,.2f}"
        elif 'JPY' in symbol.upper():
            return f"{price:.3f}"
        else:
            return f"{price:.5f}"
    except (ValueError, TypeError):
        return str(price)

def train_and_save_all_models(df, symbol, timeframe_key):
    """Train and save all models with proper error handling and batch prediction."""
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    symbol_fn = sanitize_filename(symbol)

    with st.status(f"🏗️ Memulai proses training untuk {symbol} ({timeframe_key})...", expanded=True) as status:
        try:
            status.update(label="Langkah 1/6: Melatih model LSTM...")
            lstm_model, scaler = train_simple_lstm(df, symbol, timeframe_key)
            st.write(f"Training LSTM untuk {symbol} selesai.")

            status.update(label="Langkah 2/6: Melatih model XGBoost...")
            xgb_model = train_simple_xgboost(df.copy(), symbol, timeframe_key)
            st.write(f"Training XGBoost untuk {symbol} selesai.")

            status.update(label="Langkah 3/6: Melatih model CNN...")
            cnn_model = train_simple_cnn(df.copy(), symbol, timeframe_key)
            st.write(f"Training CNN untuk {symbol} selesai.")

            status.update(label="Langkah 4/6: Melatih model SVC...")
            svc_model, svc_scaler = train_simple_svc(df.copy(), symbol, timeframe_key)
            st.write(f"Training SVC untuk {symbol} selesai.")

            status.update(label="Langkah 5/6: Melatih model Naive Bayes...")
            nb_model = train_simple_naive_bayes(df.copy(), symbol, timeframe_key)
            st.write(f"Training Naive Bayes untuk {symbol} selesai.")

            status.update(label="Memproses prediksi dasar untuk Master AI...")

            sequence_length_lstm_train = 60
            scaled_data_for_lstm_preds = scaler.transform(df['close'].values.reshape(-1, 1))
            X_lstm_batches = []
            for i in range(sequence_length_lstm_train, len(scaled_data_for_lstm_preds)):
                X_lstm_batches.append(scaled_data_for_lstm_preds[i-sequence_length_lstm_train:i, 0])
            X_lstm_batches = np.array(X_lstm_batches)
            X_lstm_batches = np.reshape(X_lstm_batches, (X_lstm_batches.shape[0], X_lstm_batches.shape[1], 1))
            lstm_preds_scaled = lstm_model.predict(X_lstm_batches, batch_size=32, verbose=0)
            lstm_preds_price = scaler.inverse_transform(lstm_preds_scaled).flatten()
            lstm_predictions_full = [np.nan] * sequence_length_lstm_train + list(lstm_preds_price)
            lstm_predictions_series = pd.Series(lstm_predictions_full, index=df.index)
            lstm_pred_diffs = (lstm_predictions_series - df['close']) / df['close']

            xgb_feature_names_train = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 'ATR_14', 'STOCHk_14_3_3', 'price_change', 'high_low_ratio', 'open_close_ratio']
            temp_df_xgb_train = df.copy()
            if 'price_change' not in temp_df_xgb_train.columns: temp_df_xgb_train['price_change'] = temp_df_xgb_train['close'].pct_change()
            if 'high_low_ratio' not in temp_df_xgb_train.columns: temp_df_xgb_train['high_low_ratio'] = temp_df_xgb_train['high'] / temp_df_xgb_train['low']
            if 'open_close_ratio' not in temp_df_xgb_train.columns: temp_df_xgb_train['open_close_ratio'] = temp_df_xgb_train['open'] / temp_df_xgb_train['close']
            available_xgb_features_train = [f for f in xgb_feature_names_train if f in temp_df_xgb_train.columns]
            X_xgb_train = temp_df_xgb_train[available_xgb_features_train].fillna(0)
            xgb_predictions_full = xgb_model.predict(X_xgb_train)
            xgb_confidences_full_raw = xgb_model.predict_proba(X_xgb_train)
            xgb_confidences_series = pd.Series([xgb_confidences_full_raw[i, pred] for i, pred in enumerate(xgb_predictions_full)], index=df.index)
            xgb_predictions_series = pd.Series(xgb_predictions_full, index=df.index)

            cnn_feature_names_train = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20']
            available_cnn_features_train = [f for f in cnn_feature_names_train if f in df.columns]
            sequence_length_cnn_train = 20
            cnn_scaler_train = joblib.load(os.path.join(model_dir, f'cnn_scaler_{symbol_fn}_{timeframe_key}.pkl'))
            scaled_cnn_features_train = cnn_scaler_train.transform(df[available_cnn_features_train].fillna(0))
            X_cnn_batches = []
            for i in range(sequence_length_cnn_train, len(scaled_cnn_features_train)):
                 X_cnn_batches.append(scaled_cnn_features_train[i-sequence_length_cnn_train:i])
            X_cnn_batches = np.array(X_cnn_batches)
            cnn_preds_proba = cnn_model.predict(X_cnn_batches, batch_size=32, verbose=0).flatten()
            cnn_predictions_full = (cnn_preds_proba > 0.5).astype(int)
            cnn_predictions_series = pd.Series([np.nan] * sequence_length_cnn_train + list(cnn_predictions_full), index=df.index)

            svc_feature_names_train = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 'ATR_14', 'STOCHk_14_3_3']
            available_svc_features_train = [f for f in svc_feature_names_train if f in df.columns]
            X_svc_train = df[available_svc_features_train].fillna(0)
            X_svc_train_scaled = svc_scaler.transform(X_svc_train)
            svc_predictions_full = svc_model.predict(X_svc_train_scaled)
            svc_confidences_full_raw = svc_model.predict_proba(X_svc_train_scaled)
            svc_confidences_series = pd.Series([svc_confidences_full_raw[i, pred] for i, pred in enumerate(svc_predictions_full)], index=df.index)
            svc_predictions_series = pd.Series(svc_predictions_full, index=df.index)

            nb_feature_names_train = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 'ATR_14', 'STOCHk_14_3_3']
            available_nb_features_train = [f for f in nb_feature_names_train if f in df.columns]
            X_nb_train = df[available_nb_features_train].fillna(0)
            nb_predictions_full = nb_model.predict(X_nb_train)
            nb_confidences_full_raw = nb_model.predict_proba(X_nb_train)
            nb_confidences_series = pd.Series([nb_confidences_full_raw[i, pred] for i, pred in enumerate(nb_predictions_full)], index=df.index)
            nb_predictions_series = pd.Series(nb_predictions_full, index=df.index)

            st.write("Prediksi dasar selesai.")

            status.update(label="Langkah 6/6: Melatih Master AI (RandomForestClassifier)...")
            meta_model = train_meta_learner(
                df.copy(),
                lstm_pred_diffs,
                xgb_predictions_series,
                xgb_confidences_series,
                cnn_predictions_series,
                svc_predictions_series,
                svc_confidences_series,
                nb_predictions_series,
                nb_confidences_series,
                timeframe_key
            )
            st.write(f"Training Master AI untuk {symbol} selesai.")

            status.update(label="Proses training selesai!", state="complete", expanded=False)

        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            st.exception(e)
            return

    st.balloons()
    st.success(f"🎉 Semua model untuk {symbol} timeframe {timeframe_key} berhasil dilatih!")

def load_all_models(symbol, timeframe_key):
    """Load all trained models"""
    model_dir = 'model'
    symbol_fn = sanitize_filename(symbol)
    models = {'lstm': None, 'scaler': None, 'xgb': None, 'cnn': None, 'cnn_scaler': None, 'meta': None, 'svc': None, 'svc_scaler': None, 'nb': None}
    load_errors = []

    model_paths = {
        'lstm': os.path.join(model_dir, f'lstm_model_{symbol_fn}_{timeframe_key}.keras'),
        'scaler': os.path.join(model_dir, f'scaler_{symbol_fn}_{timeframe_key}.pkl'),
        'xgb': os.path.join(model_dir, f'xgboost_model_{symbol_fn}_{timeframe_key}.json'),
        'cnn': os.path.join(model_dir, f'cnn_model_{symbol_fn}_{timeframe_key}.keras'),
        'cnn_scaler': os.path.join(model_dir, f'cnn_scaler_{symbol_fn}_{timeframe_key}.pkl'),
        'svc': os.path.join(model_dir, f'svc_model_{symbol_fn}_{timeframe_key}.pkl'),
        'svc_scaler': os.path.join(model_dir, f'svc_scaler_{symbol_fn}_{timeframe_key}.pkl'),
        'nb': os.path.join(model_dir, f'nb_model_{symbol_fn}_{timeframe_key}.pkl'),
        'meta': os.path.join(model_dir, f'meta_learner_randomforest_{timeframe_key}.pkl')
    }

    try: models['lstm'] = load_model(model_paths['lstm'], compile=False)
    except Exception as e: load_errors.append(f"LSTM model: {e}")
    try: models['scaler'] = joblib.load(model_paths['scaler'])
    except Exception as e: load_errors.append(f"LSTM scaler: {e}")
    try:
        models['xgb'] = XGBClassifier()
        models['xgb'].load_model(model_paths['xgb'])
    except Exception as e: load_errors.append(f"XGBoost model: {e}")
    try: models['cnn'] = load_model(model_paths['cnn'], compile=False)
    except Exception as e: load_errors.append(f"CNN model: {e}")
    try: models['cnn_scaler'] = joblib.load(model_paths['cnn_scaler'])
    except Exception as e: load_errors.append(f"CNN scaler: {e}")
    try: models['svc'] = joblib.load(model_paths['svc'])
    except Exception as e: load_errors.append(f"SVC model: {e}")
    try: models['svc_scaler'] = joblib.load(model_paths['svc_scaler'])
    except Exception as e: load_errors.append(f"SVC scaler: {e}")
    try: models['nb'] = joblib.load(model_paths['nb'])
    except Exception as e: load_errors.append(f"Naive Bayes model: {e}")
    try: models['meta'] = joblib.load(model_paths['meta'])
    except Exception as e: load_errors.append(f"Meta Learner model: {e}")

    if load_errors:
        st.error(f"Gagal memuat beberapa model untuk {symbol} timeframe '{timeframe_key}':")
        for error in load_errors:
            st.error(f"- {error}")
        return None

    return models

@st.cache_data(ttl=900) # Cache rate for 15 minutes
def get_conversion_rate(quote_currency, api_source, api_key_1, api_key_2=None):
    """
    Fetches the conversion rate to convert the quote currency to USD.
    Returns 1.0 if the quote currency is already USD.
    """
    quote_currency = quote_currency.upper()
    if quote_currency == 'USD':
        return 1.0
    if quote_currency == 'USDT':
        return 1.0
    
    try:
        conversion_symbol = f"{quote_currency}/USD"
        
        if api_source == 'Twelve Data':
            try:
                rate_data = get_gold_data(api_key_1, interval='1min', symbol=conversion_symbol, outputsize=1)
                if rate_data is not None and not rate_data.empty:
                    return rate_data['close'].iloc[-1]
            except Exception:
                inverse_symbol = f"USD/{quote_currency}"
                rate_data = get_gold_data(api_key_1, interval='1min', symbol=inverse_symbol, outputsize=1)
                if rate_data is not None and not rate_data.empty:
                    return 1.0 / rate_data['close'].iloc[-1]
        else:
            st.warning(f"Tidak dapat mengambil nilai tukar Forex dari {api_source}. Asumsi rate 1.0 untuk {quote_currency}.")
            return 1.0

        st.error(f"Gagal mendapatkan nilai tukar untuk {quote_currency}.")
        return None

    except Exception as e:
        st.error(f"Error saat mengambil nilai tukar untuk {quote_currency}: {e}")
        return None

def get_pip_value(symbol, price):
    symbol_upper = symbol.replace('/', '').upper()
    if symbol_upper == "BTCUSDT":
        return 0.1
    elif 'JPY' in symbol_upper:
        return 0.01
    elif 'XAU' in symbol_upper:
        return 0.01
    elif 'ETH' in symbol_upper:
        return 0.01
    else:
        return 0.0001

def run_backtest(symbol, data, initial_balance, risk_percent, sl_pips, tp_pips, predict_func, all_models, api_source, api_key_1, api_key_2=None, use_ai_tp=False):
    """
    IMPROVED with Smart AI Entry: Backtesting dengan realistic execution, smart entry, dan proper error handling
    """
    validation_issues = validate_trading_inputs(symbol, initial_balance, risk_percent, sl_pips, tp_pips)
    if validation_issues:
        for issue in validation_issues:
            st.warning(issue)
    
    balance = initial_balance
    trades = []
    equity_curve = [initial_balance]
    active_trade = None
    
    start_index = 61
    total_steps = len(data) - start_index
    
    if total_steps <= 0:
        st.error("❌ Data tidak cukup untuk backtest")
        return balance, trades, equity_curve
    
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    update_frequency = max(1, total_steps // 50)
    
    total_trades_count = 0
    winning_trades = 0
    total_pnl = 0
    total_costs = 0
    
    try:
        for i in range(start_index, len(data)):
            if i % update_frequency == 0:
                progress = (i - start_index + 1) / total_steps
                progress_placeholder.progress(progress, text=f"Backtesting: {progress:.1%} ({i-start_index+1}/{total_steps})")
                
                win_rate = (winning_trades / total_trades_count * 100) if total_trades_count > 0 else 0
                with metrics_placeholder.container():
                    met_cols = st.columns(4)
                    met_cols[0].metric("Balance", f"${balance:.2f}")
                    met_cols[1].metric("Total Trades", total_trades_count)
                    met_cols[2].metric("Win Rate", f"{win_rate:.1f}%")
                    met_cols[3].metric("Net P&L", f"${total_pnl:.2f}")

            current_candle = data.iloc[i]
            
            # --- Trade Exit Logic (Existing) ---
            if active_trade:
                exit_price, exit_type = execute_trade_exit_realistic(current_candle, active_trade)
                
                if exit_price is not None:
                    pnl_details = calculate_realistic_pnl(
                        active_trade['entry'], exit_price, active_trade['position_size'],
                        active_trade['type'], symbol
                    )
                    
                    balance += pnl_details['net_pnl']
                    total_pnl += pnl_details['net_pnl']
                    total_costs += pnl_details['total_costs']
                    
                    if pnl_details['net_pnl'] > 0:
                        winning_trades += 1
                    
                    active_trade.update({
                        'exit_date': current_candle.name,
                        'exit_price': exit_price,
                        'exit_type': exit_type,
                        'net_pnl': pnl_details['net_pnl'],
                        'gross_pnl': pnl_details['gross_pnl'],
                        'costs': pnl_details['total_costs'],
                        'balance': balance
                    })
                    trades.append(active_trade)
                    equity_curve.append(balance)
                    active_trade = None
                    total_trades_count += 1

            # --- Trade Entry Logic (Patched with Smart AI Entry) ---
            if not active_trade and balance > 50:
                data_slice = data.iloc[:i] # Use data up to the previous candle for prediction
                if len(data_slice) < 61:
                    continue

                try:
                    signal, confidence, predicted_price = predict_func(all_models, data_slice)
                    
                    if signal in ['BUY', 'SELL'] and confidence > 0.55:
                        # 🧠 SMART AI ENTRY CALCULATION
                        smart_entry_result = calculate_smart_entry_price(
                            signal=signal,
                            recent_data=data_slice,
                            predicted_price=predicted_price,
                            confidence=confidence,
                            symbol=symbol
                        )
                        
                        entry_price = smart_entry_result['entry_price']
                        fill_prob = smart_entry_result['expected_fill_probability']
                        
                        # Simulate order execution
                        order_filled = False
                        if signal == "BUY" and entry_price <= current_candle['high'] and np.random.random() < fill_prob:
                            order_filled = True
                            actual_entry_price = min(entry_price, current_candle['high'])
                        elif signal == "SELL" and entry_price >= current_candle['low'] and np.random.random() < fill_prob:
                            order_filled = True
                            actual_entry_price = max(entry_price, current_candle['low'])

                        if order_filled:
                            quote_currency = symbol.split('/')[1].upper() if '/' in symbol else 'USDT'
                            conversion_rate = get_conversion_rate(quote_currency, api_source, api_key_1, api_key_2)
                            
                            if conversion_rate is None: conversion_rate = 1.0
                            
                            take_profit_price = None
                            if use_ai_tp:
                                try:
                                    supports, resistances = get_support_resistance(data_slice)
                                    atr_val = data_slice['ATR_14'].iloc[-1]
                                    take_profit_price = calculate_ai_take_profit(signal, actual_entry_price, supports, resistances, atr_val)
                                except:
                                    pass
                            
                            position_info = calculate_position_info(
                                signal, symbol, actual_entry_price, sl_pips, tp_pips,
                                balance, risk_percent, conversion_rate, take_profit_price
                            )

                            if position_info and position_info['position_size'] > 0:
                                active_trade = {
                                    'entry_date': current_candle.name,
                                    'type': signal,
                                    'entry': position_info['entry_price'],
                                    'sl': position_info['stop_loss'],
                                    'tp': position_info['take_profit'],
                                    'position_size': position_info['position_size'],
                                    'confidence': confidence,
                                    'risk_amount': position_info['risk_amount']
                                }
                            
                except Exception as e:
                    status_placeholder.warning(f"⚠️ Error in trade entry at index {i}: {e}")
                    continue

        progress_placeholder.success("✅ Backtest completed successfully!")
        
        win_rate = (winning_trades / total_trades_count * 100) if total_trades_count > 0 else 0
        total_return = ((balance - initial_balance) / initial_balance) * 100
        
        with status_placeholder.container():
            st.success(f"🎉 Backtest Results:")
            final_cols = st.columns(5)
            final_cols[0].metric("Final Balance", f"${balance:.2f}")
            final_cols[1].metric("Total Return", f"{total_return:.2f}%")
            final_cols[2].metric("Total Trades", total_trades_count)
            final_cols[3].metric("Win Rate", f"{win_rate:.1f}%")
            final_cols[4].metric("Total Costs", f"${total_costs:.2f}")
        
        return balance, trades, equity_curve

    except KeyboardInterrupt:
        progress_placeholder.warning("⏸️ Backtest stopped by user")
        return balance, trades, equity_curve
    except Exception as e:
        progress_placeholder.error(f"❌ Backtest error: {str(e)}")
        st.exception(e)
        return balance, trades, equity_curve
    finally:
        time.sleep(2)
        progress_placeholder.empty()

# ==============================================================================
# MAIN APPLICATION LOGIC v8.3 - TWELVE DATA WEBSOCKET + NO LIVE CHART
# ==============================================================================

def main():
    st.set_page_config(page_title="Multi-Source Trading AI v8.3", layout="wide")
    st.title("🤖 Multi-Asset Trading Bot (Master AI) v8.3 - Twelve Data WebSocket Enhanced")
    
    # WebSocket availability indicator
    if WEBSOCKET_AVAILABLE:
        st.success("🌐 WebSocket support available for real-time trading")
    else:
        st.warning("⚠️ WebSocket not available. Install with: pip install websocket-client")

    with st.sidebar:
        st.header("🔧 Pengaturan Dasar")
        api_source = st.selectbox("Pilih Sumber Data", ["Twelve Data", "Binance"])

        api_key_1 = None
        api_key_2 = None
        symbol_options = []
        symbol_help_text = ""

        if api_source == "Twelve Data":
            st.info("📈 Menyediakan data Forex, Komoditas, dan Kripto dengan WebSocket real-time.")
            api_key_1 = st.text_input("Masukkan API Key Twelve Data Anda", type="password", help="Dapatkan API key gratis di https://twelvedata.com/")
            symbol_options = ["XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD", "ETH/USD"]
            symbol_help_text = "Pilih aset untuk dianalisis (Format: ASET/MATAUANG)."
            
            # Enhanced Twelve Data info
            if api_key_1:
                st.success("✅ API Key detected")
                
                with st.expander("📊 WebSocket Features", expanded=False):
                    st.markdown("**🌐 Real-time WebSocket:**")
                    st.code("XAU/USD, EUR/USD, GBP/USD, BTC/USD, ETH/USD")
                    
                    st.markdown("**🔄 Fallback Polling:**")
                    st.info("Auto fallback ke polling jika WebSocket gagal")
            
            if not api_key_1 or api_key_1.strip() == "":
                st.error("API Key Twelve Data diperlukan.")
                st.stop()

        elif api_source == "Binance":
            if not BINANCE_AVAILABLE:
                st.error("Modul 'python-binance' tidak terinstall. Install dengan: pip install python-binance")
                st.stop()
            st.info("💱 Menyediakan data Kripto dari Bursa Binance dengan WebSocket real-time.")
            api_key_1 = st.text_input("Binance API Key", type="password")
            api_key_2 = st.text_input("Binance API Secret", type="password")
            symbol_options = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
            symbol_help_text = "Masukkan simbol Kripto (Format: BTCUSDT)."
            if not api_key_1 or not api_key_2 or api_key_1.strip() == "" or api_key_2.strip() == "":
                st.error("API Key & Secret Binance diperlukan.")
                st.stop()

        symbol_input_key = f"symbol_input_{api_source.lower()}"
        symbol = st.selectbox("Pilih Simbol Trading", symbol_options, key=symbol_input_key, help=symbol_help_text)

        st.header("💰 Pengaturan Modal")
        account_balance = st.number_input("Modal Awal ($)", min_value=100.0, value=1000.0, step=100.0)
        
        # Market Status
        st.header("📊 Status Pasar")
        current_time_hour = datetime.now().hour
        if 0 <= current_time_hour < 6: 
            st.info("🌙 Pasar sedang tutup (malam)")
        elif 6 <= current_time_hour < 22: 
            st.success("💹 Pasar sedang buka")
        else: 
            st.warning("🕒 Pasar akan segera tutup")
            
        # Data source info
        st.divider()
        if api_source == "Twelve Data":
            st.success("✅ Twelve Data Active - WebSocket + Polling")
            st.caption("📡 Support: Forex, Stocks, Crypto dengan real-time WebSocket")
        elif api_source == "Binance":
            st.success("✅ Binance Active - Crypto WebSocket")
            st.caption("🪙 Support: Cryptocurrency dengan streaming")

    timeframe_mapping = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h', '1d': '1day'}
    
    # ✅ REMOVED: Live Chart tab - Only Trading and Pelatihan AI
    tab1, tab2 = st.tabs(["Trading", "Pelatihan AI"])

    with tab1:
        st.header(f"💼 Panel Trading ({api_source}) - Enhanced Real-time")
        subtab_live, subtab_backtest, subtab_signals = st.tabs(["Sinyal Live", "Backtesting", "Analisis Sinyal"])

        with subtab_live:
            st.subheader("🔍 Dapatkan Sinyal Trading Terbaru")
            
            # ============= ENHANCED PANEL WEBSOCKET (TWELVE DATA & BINANCE) =============
            if WEBSOCKET_AVAILABLE and api_source in ["Binance", "Twelve Data"]:
                with st.expander("🌐 Real-time Connection", expanded=True):
                    ws_cols = st.columns([2, 1, 1])
                    
                    # Status WebSocket
                    ws_status = st.session_state.ws_manager.get_status(symbol)
                    ws_price = st.session_state.ws_manager.get_price(symbol)
                    
                    if "✅" in ws_status:
                        if api_source == "Twelve Data":
                            ws_cols[0].success(f"🟢 Twelve Data WebSocket: {ws_status}")
                        else:
                            ws_cols[0].success(f"🟢 Binance WebSocket: {ws_status}")
                    elif "Error" in ws_status:
                        ws_cols[0].error(f"🔴 Status: {ws_status}")
                    else:
                        ws_cols[0].warning(f"🟡 Status: {ws_status}")
                    
                    # Live Price Display
                    if ws_price and ws_price > 0:
                        price_history = st.session_state.ws_manager.get_price_history(symbol, 5)
                        if len(price_history) >= 2:
                            prev_price = price_history[-2]['price']
                            price_change = ws_price - prev_price
                            ws_cols[1].metric(
                                "🔴 Live Price", 
                                format_price(symbol, ws_price),
                                delta=f"{format_price(symbol, price_change)}"
                            )
                        else:
                            ws_cols[1].metric("🔴 Live Price", format_price(symbol, ws_price))
                    else:
                        ws_cols[1].info("⏳ Waiting for data...")
                    
                    # Connect/Disconnect Button
                    if ws_cols[2].button("🔗 Connect" if not st.session_state.ws_manager.is_connected(symbol) else "🔌 Disconnect"):
                        if st.session_state.ws_manager.is_connected(symbol):
                            st.session_state.ws_manager.disconnect(symbol)
                            st.info("🔌 Stream disconnected")
                        else:
                            success = st.session_state.ws_manager.start_stream(api_source, symbol, api_key_1, api_key_2)
                            if success:
                                if api_source == "Twelve Data":
                                    st.success(f"🔗 Twelve Data WebSocket stream initiated")
                                else:
                                    st.success(f"🔗 {api_source} stream initiated")
                            else:
                                st.error("❌ Failed to start stream")
                        time.sleep(1)
                        st.rerun()
                        
                    # Price History Chart (Mini)
                    if ws_price and len(st.session_state.ws_manager.get_price_history(symbol, 20)) > 5:
                        price_history = st.session_state.ws_manager.get_price_history(symbol, 20)
                        prices = [p['price'] for p in price_history]
                        timestamps = [p['timestamp'] for p in price_history]
                        
                        import plotly.express as px
                        price_df = pd.DataFrame({
                            'Time': timestamps,
                            'Price': prices
                        })
                        fig_mini = px.line(
                            price_df, 
                            x='Time', 
                            y='Price', 
                            title=f"📊 Live Price Feed - {api_source} (Last 20 updates)",
                            height=300
                        )
                        fig_mini.update_layout(template="plotly_dark")
                        st.plotly_chart(fig_mini, use_container_width=True)
            
            # ============= PANEL TRADING (ENHANCED) =============
            live_cols = st.columns([2, 1, 1])
            live_tf_key = live_cols[0].selectbox("Pilih Timeframe", list(timeframe_mapping.keys()), index=2, key="live_tf_select", help="Timeframe untuk analisis sinyal")
            col_refresh = live_cols[1]
            col_auto = live_cols[2]
            auto_refresh_live = col_auto.checkbox("🔁 Auto Refresh", key="auto_live")
            refresh_live = col_refresh.button("🔄 Refresh", use_container_width=True)

            with st.expander("Pengaturan Risk Management", expanded=True):
                risk_cols = st.columns(3)
                live_sl = risk_cols[0].number_input("Stop Loss (pips)", min_value=5, value=20, key="live_sl")
                use_ai_tp = st.checkbox("Gunakan Take Profit dari AI (berdasarkan Support/Resistance)", value=True)
                live_tp = risk_cols[1].number_input("Take Profit (pips)", min_value=5, value=40, key="live_tp", disabled=use_ai_tp)
                live_risk = risk_cols[2].slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1, key="live_risk")

            # ============= TOMBOL SIGNAL (ENHANCED) =============
            signal_cols = st.columns(3)
            generate_signal = signal_cols[0].button("📊 Generate Signal", use_container_width=True, type="primary")
            
            # Live Stream Button (Enhanced untuk semua providers)
            if WEBSOCKET_AVAILABLE and api_source in ["Binance", "Twelve Data"]:
                start_live_stream = signal_cols[1].button("🔴 Live Stream (30s)", use_container_width=True, type="secondary")
                use_ws_price = signal_cols[2].checkbox("📡 Use Real-time Price", value=True, help="Use real-time streaming price for signals")
            else:
                start_live_stream = False
                use_ws_price = False

            # ============= LOGIC SIGNAL ENHANCED =============
            if generate_signal or refresh_live:
                try:
                    all_models_live = load_all_models(symbol, live_tf_key)
                    if all_models_live is None or any(v is None for v in all_models_live.values()):
                        st.error(f"❌ Model untuk {symbol} timeframe {live_tf_key} belum dilatih atau gagal dimuat.")
                        st.info("💡 Silakan latih semua model terlebih dahulu di tab 'Pelatihan AI'.")
                    else:
                        with st.spinner(f"🔄 Menganalisis kondisi pasar dari {api_source}..."):
                            recent_data = load_and_process_data_enhanced(api_source, symbol, timeframe_mapping[live_tf_key], api_key_1, api_key_2, outputsize=200)
                            
                            if recent_data is None or len(recent_data) <= 61:
                                st.error("❌ Data tidak cukup atau gagal dimuat. Minimal 62 candle diperlukan.")
                                st.info("💡 Coba refresh atau periksa koneksi internet dan API key.")
                            else:
                                # ENHANCED: Use real-time price if available and enabled
                                current_api_price = recent_data['close'].iloc[-1]
                                ws_price = st.session_state.ws_manager.get_price(symbol) if use_ws_price else None
                                
                                if ws_price and ws_price > 0 and use_ws_price:
                                    price_diff_pct = abs(ws_price - current_api_price) / current_api_price
                                    if price_diff_pct < 0.05:  # Within 5%
                                        st.info(f"📡 Using {api_source} Real-time Price: {format_price(symbol, ws_price)} (API: {format_price(symbol, current_api_price)})")
                                        # Update the last row with real-time price
                                        recent_data.loc[recent_data.index[-1], 'close'] = ws_price
                                    else:
                                        st.warning(f"⚠️ Real-time price deviation too high ({price_diff_pct:.2%}), using API price")
                                
                                prediction_data = recent_data.iloc[:-1]
                                signal, confidence, predicted_price = predict_with_models(all_models_live, prediction_data)
                                
                                if signal is None:
                                    st.error("❌ Model prediction gagal. Coba refresh atau latih ulang model.")
                                else:
                                    # Smart AI Entry calculation
                                    smart_entry_result = calculate_smart_entry_price(
                                        signal=signal,
                                        recent_data=recent_data,
                                        predicted_price=predicted_price,
                                        confidence=confidence,
                                        symbol=symbol
                                    )
                                    
                                    position_info = None
                                    if signal != "HOLD":
                                        quote_currency = "USD"
                                        if "/" in symbol:
                                            quote_currency = symbol.split('/')[1]
                                        elif "USDT" in symbol.upper():
                                            quote_currency = "USDT"
                                        
                                        conversion_rate = get_conversion_rate(quote_currency, api_source, api_key_1, api_key_2)

                                        if conversion_rate is None or conversion_rate <= 0:
                                            st.error(f"❌ Tidak dapat mengambil nilai tukar untuk {quote_currency}")
                                        else:
                                            ai_tp_price = None
                                            if use_ai_tp:
                                                try:
                                                    supports, resistances = get_support_resistance(recent_data)
                                                    atr_val = recent_data['ATR_14'].iloc[-1]
                                                    ai_tp_price = calculate_ai_take_profit(signal, smart_entry_result['entry_price'], supports, resistances, atr_val)
                                                except Exception as e:
                                                    st.warning(f"⚠️ AI TP calculation failed: {e}. Using manual TP.")

                                            position_info = calculate_position_info(
                                                signal, symbol, smart_entry_result['entry_price'], live_sl, live_tp, 
                                                account_balance, live_risk, conversion_rate, 
                                                take_profit_price=ai_tp_price
                                            )
                                    
                                    # Enhanced UI display
                                    display_smart_signal_results(
                                        signal, confidence, smart_entry_result, position_info, symbol
                                    )

                except Exception as e:
                    st.error(f"❌ Unexpected error dalam generate trading signal: {str(e)}")
                    st.exception(e)

            # ============= ENHANCED LIVE STREAM FEATURE =============
            if start_live_stream:
                if not st.session_state.ws_manager.is_connected(symbol):
                    st.warning("⚠️ Please connect real-time stream first for live streaming")
                else:
                    all_models_live = load_all_models(symbol, live_tf_key)
                    if all_models_live is None:
                        st.error(f"❌ Models not trained for {symbol} timeframe {live_tf_key}")
                    else:
                        st.info(f"🔴 **LIVE SIGNAL STREAM ACTIVE** ({api_source}) - Updates every 30 seconds")
                        
                        # Create placeholders for live updates
                        live_signal_placeholder = st.empty()
                        live_metrics_placeholder = st.empty()
                        live_analysis_placeholder = st.empty()
                        
                        # Control button
                        control_col1, control_col2 = st.columns(2)
                        stop_stream = control_col1.button("⏹️ Stop Stream", key="stop_live_stream")
                        iteration_count = 0
                        
                        if not stop_stream:
                            # Live stream loop
                            for i in range(60):  # Max 30 minutes (60 * 30 seconds)
                                try:
                                    iteration_count += 1
                                    
                                    # Get fresh data
                                    recent_data = load_and_process_data_enhanced(
                                        api_source, symbol, timeframe_mapping[live_tf_key], 
                                        api_key_1, api_key_2, outputsize=200
                                    )
                                    
                                    if recent_data is not None and len(recent_data) > 61:
                                        # Always use real-time price for live stream
                                        ws_price = st.session_state.ws_manager.get_price(symbol)
                                        if ws_price and ws_price > 0:
                                            recent_data.loc[recent_data.index[-1], 'close'] = ws_price
                                        
                                        # Generate signal
                                        prediction_data = recent_data.iloc[:-1]
                                        signal, confidence, predicted_price = predict_with_models(all_models_live, prediction_data)
                                        
                                        # Update live displays
                                        with live_signal_placeholder.container():
                                            signal_time = datetime.now().strftime("%H:%M:%S")
                                            if signal == "BUY":
                                                st.success(f"🟢 **LIVE BUY SIGNAL** ({api_source}) - {signal_time} - Confidence: {confidence:.1%} - #{iteration_count}")
                                            elif signal == "SELL":
                                                st.error(f"🔴 **LIVE SELL SIGNAL** ({api_source}) - {signal_time} - Confidence: {confidence:.1%} - #{iteration_count}")
                                            else:
                                                st.info(f"🟡 **HOLD** ({api_source}) - {signal_time} - Confidence: {confidence:.1%} - #{iteration_count}")
                                        
                                        # Live metrics
                                        with live_metrics_placeholder.container():
                                            met_cols = st.columns(4)
                                            if ws_price:
                                                price_change = ws_price - predicted_price if predicted_price else 0
                                                met_cols[0].metric("Live Price", format_price(symbol, ws_price), f"{format_price(symbol, price_change)}")
                                            met_cols[1].metric("AI Prediction", format_price(symbol, predicted_price))
                                            met_cols[2].metric("Confidence", f"{confidence:.1%}")
                                            met_cols[3].metric("Stream Count", f"#{iteration_count}")
                                        
                                        # Live analysis for strong signals
                                        if signal in ['BUY', 'SELL'] and confidence > 0.7:
                                            smart_entry_result = calculate_smart_entry_price(
                                                signal=signal,
                                                recent_data=recent_data,
                                                predicted_price=predicted_price,
                                                confidence=confidence,
                                                symbol=symbol
                                            )
                                            
                                            with live_analysis_placeholder.container():
                                                st.markdown("### 🧠 Live Smart Entry Analysis")
                                                entry_cols = st.columns(3)
                                                entry_cols[0].metric("Smart Entry", format_price(symbol, smart_entry_result['entry_price']))
                                                entry_cols[1].metric("Fill Probability", f"{smart_entry_result['expected_fill_probability']:.1%}")
                                                entry_cols[2].metric("Risk Level", smart_entry_result['risk_level'])
                                                
                                                # Show top 3 strategy reasons
                                                for idx, reason in enumerate(smart_entry_result['strategy_reasons'][:3], 1):
                                                    st.markdown(f"**{idx}.** {reason}")
                                    
                                    # Wait for next update
                                    time.sleep(30)
                                    
                                    # Check if user stopped (recheck button state)
                                    if control_col2.button("⏹️ Stop Stream", key=f"stop_check_{i}"):
                                        break
                                        
                                except Exception as e:
                                    st.error(f"Error in live stream iteration {i+1}: {e}")
                                    time.sleep(10)  # Wait before retry
                            
                            st.success("✅ Live signal stream completed")

            if auto_refresh_live: 
                time.sleep(30)
                st.rerun()

                # ============= REST OF THE TABS (ENHANCED) =============
        with subtab_backtest:
            st.subheader("🧪 Backtesting Strategy")
            with st.expander("Pengaturan Backtest", expanded=True):
                bt_cols = st.columns(4)
                bt_start_date = bt_cols[0].date_input("Tanggal Mulai", datetime.now() - timedelta(days=60), key="bt_start")
                bt_end_date = bt_cols[1].date_input("Tanggal Selesai", datetime.now(), key="bt_end")
                bt_tf_key = bt_cols[2].selectbox("Timeframe", list(timeframe_mapping.keys()), index=2, key="bt_tf")
                bt_initial_balance = bt_cols[3].number_input("Initial Balance", value=1000, key="bt_balance")
                risk_bt_cols = st.columns(3)
                sl_input_bt = risk_bt_cols[0].number_input("Stop Loss (pips)", min_value=10, value=25, key="bt_sl")
                tp_input_bt = risk_bt_cols[1].number_input("Take Profit (pips)", min_value=10, value=50, key="bt_tp")
                risk_bt = risk_bt_cols[2].slider("Risk per Trade (%)", 0.5, 5.0, 1.5, 0.1, key="bt_risk")
                use_ai_tp_backtest = st.checkbox("Gunakan Take Profit dari AI (S/R)", value=True, key="bt_ai_tp")

            if st.button("Run Backtest", use_container_width=True, type="primary"):
                all_models_bt = load_all_models(symbol, bt_tf_key)
                if all_models_bt is None or all_models_bt.get('meta') is None:
                    st.error(f"Model untuk {symbol} timeframe {bt_tf_key} belum dilatih.")
                else:
                    with st.spinner(f"Mengunduh data untuk backtest dari {api_source}..."):
                        days_diff = (bt_end_date - bt_start_date).days
                        multiplier = {'1m': 1440, '5m': 288, '15m': 96, '1h': 24, '4h': 6, '1d': 1}
                        
                        outputsize_bt = max(1000, days_diff * multiplier.get(bt_tf_key, 24) + 200)
                        backtest_data = load_and_process_data_enhanced(api_source, symbol, timeframe_mapping[bt_tf_key], api_key_1, api_key_2, outputsize=outputsize_bt)

                        if backtest_data is not None:
                            backtest_data = backtest_data[(backtest_data.index.date >= bt_start_date) & (backtest_data.index.date <= bt_end_date)]

                    if backtest_data is not None and len(backtest_data) > 61:
                        # Use the enhanced backtest function
                        final_balance, trades, equity_curve = run_backtest(
                            symbol, backtest_data, bt_initial_balance, risk_bt, sl_input_bt, tp_input_bt,
                            predict_with_models, all_models_bt, api_source, api_key_1, api_key_2, use_ai_tp=use_ai_tp_backtest
                        )

                        if trades:
                            trades_df = pd.DataFrame(trades)
                            total_trades = len(trades_df)
                            winning_trades_df = trades_df[trades_df['net_pnl'] > 0]
                            losing_trades_df = trades_df[trades_df['net_pnl'] <= 0]
                            
                            num_wins = len(winning_trades_df)
                            win_rate = (num_wins / total_trades) * 100 if total_trades > 0 else 0
                            
                            gross_profit = winning_trades_df['gross_pnl'].sum()
                            gross_loss = losing_trades_df['gross_pnl'].sum()
                            net_profit = trades_df['net_pnl'].sum()
                            total_costs = trades_df['costs'].sum()
                            profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')

                            st.subheader("Ringkasan Kinerja Utama")
                            metrics_cols = st.columns(4)
                            metrics_cols[0].metric("Total Return", f"{((final_balance - bt_initial_balance) / bt_initial_balance) * 100:.2f}%")
                            metrics_cols[1].metric("Win Rate", f"{win_rate:.2f}%")
                            metrics_cols[2].metric("Total Perdagangan", total_trades)
                            metrics_cols[3].metric("Final Balance", f"${final_balance:,.2f}")

                            st.subheader("Kurva Ekuitas (Equity Curve)")
                            equity_df = pd.DataFrame({'Balance': equity_curve})
                            st.line_chart(equity_df)
                            
                            st.divider()

                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Analisis Profitabilitas")
                                st.metric("Keuntungan Bersih (Net Profit)", f"${net_profit:.2f}")
                                st.metric("Keuntungan Kotor (Gross Profit)", f"${gross_profit:.2f}")
                                st.metric("Kerugian Kotor (Gross Loss)", f"${gross_loss:.2f}")
                                st.metric("Total Biaya Transaksi", f"${total_costs:.2f}")
                            
                            with col2:
                                st.subheader("Statistik Perdagangan")
                                st.metric("Profit Factor", f"{profit_factor:.2f}")
                                st.metric("Rata-rata Profit/Trade", f"${trades_df['net_pnl'].mean():.2f}")
                                st.metric("Trade Terbaik", f"${trades_df['net_pnl'].max():.2f}")
                                st.metric("Trade Terburuk", f"${trades_df['net_pnl'].min():.2f}")

                            with st.expander("Lihat Log Perdagangan Lengkap"):
                                st.dataframe(trades_df)
                        else:
                            st.warning("Tidak ada perdagangan yang dieksekusi selama periode backtest.")
                    else:
                        st.error("Data tidak cukup untuk menjalankan backtest pada rentang tanggal yang dipilih.")

        with subtab_signals:
            st.subheader("Analisis Sinyal Multi-Timeframe")
            if st.button("Analyze Multi-Timeframe Signals", use_container_width=True):
                timeframes_to_analyze = ['15m', '1h', '4h']
                signal_results = {}
                with st.spinner(f"Menganalisis sinyal multi-timeframe dari {api_source}..."):
                    for tf in timeframes_to_analyze:
                        all_models_multi = load_all_models(symbol, tf)
                        if all_models_multi is not None and all_models_multi.get('meta') is not None:
                            data = load_and_process_data_enhanced(api_source, symbol, timeframe_mapping[tf], api_key_1, api_key_2, outputsize=200)
                            if data is not None and len(data) > 61:
                                prediction_data = data.iloc[:-1]
                                signal, confidence, _ = predict_with_models(all_models_multi, prediction_data)
                                signal_results[tf] = {'signal': signal, 'confidence': confidence}
                        else:
                            signal_results[tf] = {'signal': 'NO MODEL', 'confidence': 0}

                if signal_results:
                    tf_cols = st.columns(len(signal_results))
                    for i, (tf, result) in enumerate(signal_results.items()):
                        with tf_cols[i]:
                            st.metric(f"{tf} Signal", f"{result['signal']}", f"{result['confidence']:.1%}")

    with tab2:
        st.header(f"🎓 AI Model Training Center ({api_source})")
        st.warning("PENTING: Setelah memodifikasi kode atau mengganti sumber data, Anda WAJIB melatih ulang model untuk timeframe yang ingin digunakan.")
        
        # Enhanced training info untuk different providers
        if api_source == "Twelve Data":
            st.info("📊 **Twelve Data Training Mode**: Data multi-asset dengan WebSocket real-time untuk training yang optimal")
        elif api_source == "Binance":
            st.info("🪙 **Binance Training Mode**: Data Cryptocurrency untuk training model crypto-specific")
        
        train_cols = st.columns([2, 1])
        train_tf_key = train_cols[0].selectbox("Pilih Timeframe untuk Training", list(timeframe_mapping.keys()), index=2, key="train_tf_select")
        
        data_size_options = [1000, 3000, 5000, 10000, 20000]
        data_size_help = "Pilih jumlah data. Nilai besar mungkin tidak didukung oleh semua API atau paket langganan."
        data_size = train_cols[1].selectbox("Ukuran Data Training", data_size_options, index=0, help=data_size_help)

        st.subheader("📊 Model Status Dashboard")
        model_dir = 'model'
        symbol_fn = sanitize_filename(symbol)
        model_files = [
            ('LSTM', f'lstm_model_{symbol_fn}_{train_tf_key}.keras'),
            ('Scaler', f'scaler_{symbol_fn}_{train_tf_key}.pkl'),
            ('XGBoost', f'xgboost_model_{symbol_fn}_{train_tf_key}.json'),
            ('CNN', f'cnn_model_{symbol_fn}_{train_tf_key}.keras'),
            ('Scaler CNN', f'cnn_scaler_{symbol_fn}_{train_tf_key}.pkl'),
            ('SVC', f'svc_model_{symbol_fn}_{train_tf_key}.pkl'),
            ('Scaler SVC', f'svc_scaler_{symbol_fn}_{train_tf_key}.pkl'),
            ('Naive Bayes', f'nb_model_{symbol_fn}_{train_tf_key}.pkl'),
            ('Meta AI', f'meta_learner_randomforest_{train_tf_key}.pkl')
        ]
        status_cols = st.columns(len(model_files))
        for i, (name, filename) in enumerate(model_files):
            file_path = os.path.join(model_dir, filename)
            if os.path.exists(file_path):
                status_cols[i].success(f"✅ {name}")
            else:
                status_cols[i].error(f"❌ {name}")
        
        st.divider()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button(f"🚀 Latih Ulang Semua Model untuk {symbol} ({train_tf_key}) - {api_source}", use_container_width=True, type="primary"):
                with st.spinner(f"Mengunduh {data_size} data dari {api_source} untuk {symbol}..."):
                    training_data = load_and_process_data_enhanced(api_source, symbol, timeframe_mapping[train_tf_key], api_key_1, api_key_2, outputsize=data_size)
                if training_data is not None and len(training_data) > 200:
                    st.success(f"✅ Data berhasil dimuat dari {api_source}: {len(training_data)} baris")
                    train_and_save_all_models(training_data, symbol, train_tf_key)
                    st.rerun()
                else:
                    st.error(f"❌ Gagal mengunduh data yang cukup untuk training dari {api_source}.")
        
        with col2:
            if st.button("🗑️ Hapus Model", use_container_width=True):
                if 'confirm_delete_state' not in st.session_state:
                    st.session_state.confirm_delete_state = {}
                
                if st.session_state.confirm_delete_state.get(train_tf_key, False):
                    deleted_count = 0
                    for _, filename in model_files:
                        file_path = os.path.join(model_dir, filename)
                        if os.path.exists(file_path): 
                            os.remove(file_path)
                            deleted_count += 1
                    st.success(f"✅ Berhasil menghapus {deleted_count} file model untuk timeframe {train_tf_key}.")
                    st.session_state.confirm_delete_state[train_tf_key] = False
                    st.rerun()
                else:
                    st.session_state.confirm_delete_state[train_tf_key] = True
                    st.warning(f"⚠️ Anda yakin ingin menghapus model untuk {train_tf_key}? Klik lagi untuk konfirmasi.")

        # Enhanced Training Info dengan provider-specific recommendations
        st.divider()
        with st.expander("ℹ️ Informasi Training Enhanced", expanded=False):
            st.markdown(f"""
            ### 🎯 Model Training Guide untuk {api_source}
            
            **Langkah-langkah Training:**
            1. **Pilih Timeframe** - Sesuaikan dengan strategi trading Anda
            2. **Pilih Data Size** - Lebih banyak data = model lebih akurat (tapi lebih lama)
            3. **Klik Train** - Tunggu proses selesai (5-15 menit)
            4. **Cek Status** - Pastikan semua model berhasil dilatih
            
            **Rekomendasi Data Size untuk {api_source}:**
            """)
            
            if api_source == "Twelve Data":
                st.markdown("""
                - **1000 candles**: Testing cepat (Good for most assets)
                - **3000 candles**: Trading harian (Recommended for Gold/Forex)
                - **5000+ candles**: Akurasi maksimal (Best for all assets)
                
                **📊 Twelve Data Advantages:**
                - ✅ Real-time WebSocket untuk Forex dan Crypto
                - ✅ Multi-asset support (Forex, Stocks, Crypto)
                - ✅ Historical data yang lengkap
                - ✅ Auto fallback ke polling jika WebSocket gagal
                - ✅ Free tier tersedia untuk testing
                """)
            elif api_source == "Binance":
                st.markdown("""
                - **1000 candles**: Testing cepat (Good for major cryptos)
                - **3000 candles**: Trading harian (Recommended for BTC/ETH)
                - **5000+ candles**: Akurasi maksimal (Best for all cryptos)
                
                **🪙 Binance Advantages:**
                - ✅ Real-time WebSocket untuk crypto
                - ✅ High-frequency data dengan volume akurat
                - ✅ Support untuk major cryptocurrencies
                - ✅ Low latency trading data
                """)
            
            st.markdown("""
            **Timeframe yang Disarankan:**
            - **1m-5m**: Scalping (butuh data banyak, cocok untuk WebSocket)
            - **15m-1h**: Day trading (balance speed vs accuracy)
            - **4h-1d**: Swing trading (lebih stabil, good for all providers)
            
            ⚠️ **PENTING**: Model harus dilatih ulang jika:
            - Ganti symbol trading
            - Ganti data provider (Twelve Data ↔ Binance)
            - Update kode aplikasi
            - Performa model menurun
            
            🎯 **Tips Optimasi:**
            - **XAU/USD**: Gunakan Twelve Data untuk data terlengkap
            - **BTC/ETH**: Gunakan Binance untuk data volume tinggi
            - **Forex majors**: Twelve Data WebSocket sangat recommended
            - **Altcoins**: Binance memberikan coverage terbaik
            """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Terjadi Error pada Aplikasi: {str(e)}")
        st.exception(e)
        st.markdown("""
        ### 🛠️ Enhanced Troubleshooting v8.3
        
        **Jika aplikasi error:**
        1. **Refresh halaman** (F5 atau Ctrl+R)
        2. **Cek koneksi internet**
        3. **Verifikasi API Keys** yang dimasukkan:
           - Twelve Data: https://twelvedata.com/
           - Binance: https://www.binance.com/en/support/faq/360002502072
        4. **Install dependencies** yang missing:
           ```bash
           pip install websocket-client pandas-ta plotly streamlit requests
           pip install python-binance  # untuk Binance
           pip install tensorflow xgboost scikit-learn
           ```
        5. **Restart aplikasi** jika masih bermasalah
        
        **Provider-specific Issues:**
        - **Twelve Data**: 
          * Pastikan tidak melebihi monthly quota
          * WebSocket gagal akan auto-fallback ke polling
          * Free tier: 800 calls/day, paid: unlimited
        - **Binance**: 
          * Pastikan API key memiliki permission untuk read market data
          * Rate limit: 1200 requests per minute
          * WebSocket: 5 incoming messages per second
        
        **WebSocket Troubleshooting:**
        - Jika WebSocket gagal connect, aplikasi akan otomatis fallback ke polling
        - Untuk Twelve Data: Periksa https://status.twelvedata.com/
        - Untuk Binance: Periksa https://www.binance.com/en/support/announcement
        
        **Model Training Issues:**
        - Pastikan data size >= 1000 untuk training yang stabil
        - Jika training gagal, coba kurangi data size
        - Model akan tersimpan di folder 'model/' di directory aplikasi
        
        **Performance Tips:**
        - Gunakan WebSocket untuk latency rendah
        - Cache data selama 5 menit untuk menghemat API calls
        - Training model sebaiknya dilakukan saat market tutup
        
        **Untuk dukungan teknis:**
        - Cek log error di console browser (F12)
        - Screenshot error untuk debugging
        - Pastikan semua requirements terinstall dengan benar
        - Test dengan provider berbeda jika ada masalah
        
        **Contact Info:**
        - GitHub Issues: Laporkan bug atau request fitur
        - User: @earleoshio
        - Version: v8.3 - Twelve Data WebSocket Enhanced
        """)