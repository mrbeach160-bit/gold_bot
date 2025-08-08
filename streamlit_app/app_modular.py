# app.py (Refactored Modular Version - v8.3 Twelve Data WebSocket Enhanced)
# --- MODULAR REFACTORING COMPLETE:
# - Extracted monolithic code into clean, testable modules
# - Maintains 100% functionality compatibility
# - Improved code organization for readability and future extension
# - Clear separation of concerns across modules
#
# --- ORIGINAL FEATURES PRESERVED:
# - FEAT: Twelve Data WebSocket support untuk real-time streaming
# - FEAT: Enhanced WebSocket streaming untuk Forex, Stocks, dan Crypto via Twelve Data
# - FEAT: Auto fallback dari WebSocket ke polling untuk reliability
# - FEAT: Backward compatibility dengan Binance WebSocket
# - FEAT: Smart AI Entry dengan multi-factor analysis
# - FEAT: Real-time signal generation dengan data streaming
# - FEAT: Connection management dan error handling yang robust

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from typing import Dict, Any, Optional, List

# Import our modular components
from modules.config import (
    initialize_feature_flags, is_feature_enabled, configure_tensorflow,
    get_project_root, get_model_directory, LABEL_MAP, TIMEFRAME_MAPPING
)
from modules.websocket_manager import EnhancedWebSocketManager, websocket_manager
from modules.data_utils import load_and_process_data_enhanced, get_historical_data_cached
from modules.models import (
    predict_with_models, load_all_models, train_and_save_all_models,
    sanitize_filename
)
from modules.smart_entry import calculate_smart_entry_price
from modules.trading_utils import (
    validate_trading_inputs, calculate_position_info, calculate_ai_take_profit,
    get_conversion_rate, get_pip_value
)
from modules.backtest import run_backtest, format_backtest_results, generate_backtest_summary
from modules.ui import (
    display_smart_signal_results, format_price, format_percentage, format_currency,
    create_trading_chart, display_metrics_grid, display_data_quality_info,
    create_status_indicator
)

# Initialize configuration and feature flags
FEATURE_FLAGS = initialize_feature_flags()

# Configure TensorFlow
TF_AVAILABLE = configure_tensorflow()

# Show configuration warnings if needed
if not TF_AVAILABLE:
    st.error("TensorFlow tidak terinstall. Install dengan: pip install tensorflow")
    st.stop()

if not is_feature_enabled('UTILS_AVAILABLE'):
    st.error("Utils modules tidak ditemukan. Pastikan folder 'utils' tersedia.")
    st.stop()

# --- PHASE 3: NEW DATA & TRADING SYSTEM INTEGRATION ---
NEW_SYSTEM_AVAILABLE = is_feature_enabled('NEW_SYSTEM_AVAILABLE')
if NEW_SYSTEM_AVAILABLE:
    try:
        from data import DataManager
        from trading import TradingManager
        
        # Initialize managers if available from main.py globals
        data_manager = globals().get('data_manager')
        trading_manager = globals().get('trading_manager')
        
        if data_manager is None:
            data_manager = DataManager()
        if trading_manager is None:
            trading_manager = TradingManager(data_manager=data_manager)
            
        print("✅ Phase 3 unified systems available")
    except ImportError:
        print("⚠️ Phase 3 systems not available, using legacy utils")
        data_manager = None
        trading_manager = None


def main():
    """Main application entry point."""
    st.set_page_config(page_title="Multi-Source Trading AI v8.3", layout="wide")
    st.title("🤖 Multi-Asset Trading Bot (Master AI) v8.3 - Twelve Data WebSocket Enhanced")
    
    # Phase 3 System Status
    if NEW_SYSTEM_AVAILABLE:
        st.success("🚀 Phase 3 Unified Data & Trading System Active")
        if data_manager and trading_manager:
            st.info(f"📊 DataManager: {len(data_manager.providers)} providers | 🎯 TradingManager: {trading_manager.strategy.name}")
    else:
        st.info("📦 Using Legacy Utils System")
    
    # WebSocket availability indicator
    if is_feature_enabled('WEBSOCKET_AVAILABLE'):
        st.success("🌐 WebSocket support available for real-time trading")
    else:
        st.warning("⚠️ WebSocket not available. Install with: pip install websocket-client")

    # Sidebar configuration
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
            if not is_feature_enabled('BINANCE_AVAILABLE'):
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

    # Main application tabs
    tab_trading, tab_training, tab_help = st.tabs(["🎯 Trading", "🏗️ Pelatihan AI", "❓ Help"])

    with tab_trading:
        handle_trading_tab(api_source, symbol, api_key_1, api_key_2, account_balance)

    with tab_training:
        handle_training_tab(api_source, symbol, api_key_1, api_key_2)

    with tab_help:
        handle_help_tab()


def handle_trading_tab(api_source: str, symbol: str, api_key_1: str, api_key_2: Optional[str], account_balance: float):
    """Handle the trading tab functionality."""
    
    st.header("🎯 AI Trading Signals & Analysis")
    
    # Trading parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Parameter Trading")
        timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], help="Pilih timeframe untuk analisis")
        risk_percent = st.slider("Risk per Trade (%)", 0.1, 5.0, 2.0, 0.1, help="Persentase modal yang dirisiko per trade")
        
    with col2:
        st.subheader("🎯 Risk Management")
        sl_pips = st.number_input("Stop Loss (pips)", 10, 200, 50, help="Stop loss dalam pips")
        tp_pips = st.number_input("Take Profit (pips)", 10, 500, 100, help="Take profit dalam pips")
        use_ai_tp = st.checkbox("Gunakan AI Take Profit", value=True, help="Gunakan AI untuk menentukan take profit optimal")

    # Validate inputs
    validation_issues = validate_trading_inputs(symbol, account_balance, risk_percent, sl_pips, tp_pips)
    if validation_issues.get('errors'):
        for error in validation_issues['errors']:
            st.error(f"❌ {error}")
        return
    
    if validation_issues.get('warnings'):
        for warning in validation_issues['warnings']:
            st.warning(f"⚠️ {warning}")

    # WebSocket Setup
    st.subheader("📡 Real-time Price Stream")
    
    if is_feature_enabled('WEBSOCKET_AVAILABLE'):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            ws_placeholder = st.empty()
        
        with col2:
            if st.button("🔄 Start WebSocket", key="start_ws"):
                if 'ws_manager' not in st.session_state:
                    st.session_state.ws_manager = EnhancedWebSocketManager()
                
                success = st.session_state.ws_manager.start_stream(api_source, symbol, api_key_1, api_key_2)
                if success:
                    st.success("WebSocket started!")
                else:
                    st.error("Failed to start WebSocket")
        
        # Display WebSocket status
        if 'ws_manager' in st.session_state:
            status = st.session_state.ws_manager.get_connection_status(symbol)
            ws_price = st.session_state.ws_manager.get_latest_price(symbol)
            
            with ws_placeholder.container():
                if ws_price:
                    st.metric(
                        label=f"💹 {symbol} Live Price",
                        value=format_price(symbol, ws_price),
                        help=f"Status: {status}"
                    )
                else:
                    st.info(f"WebSocket Status: {status}")

    # Load data and models
    st.subheader("📊 Data Loading & Model Prediction")
    
    if st.button("🔄 Load Data & Generate Signal", key="load_data"):
        with st.spinner("Loading data..."):
            # Load historical data
            data, success = get_historical_data_cached(
                api_source, symbol, TIMEFRAME_MAPPING.get(timeframe, '1h'), 
                api_key_1, api_key_2, cache_minutes=5
            )
            
            if not success or data is None:
                st.error("❌ Failed to load data")
                return
            
            # Display data quality info
            display_data_quality_info(data, symbol)
            
            # Load models
            timeframe_key = timeframe
            symbol_clean = sanitize_filename(symbol)
            all_models = load_all_models(symbol_clean, timeframe_key)
            
            if not all_models:
                st.warning("⚠️ No trained models found. Please train models first in the Training tab.")
                return
            
            # Generate prediction
            try:
                prediction_result = predict_with_models(all_models, data)
                
                signal = prediction_result.get('ensemble_signal', 0)
                confidence = prediction_result.get('confidence', 0)
                predicted_price = prediction_result.get('predicted_price', data['close'].iloc[-1])
                current_price = data['close'].iloc[-1]
                
                # Get WebSocket price if available
                ws_price = None
                if 'ws_manager' in st.session_state:
                    ws_price = st.session_state.ws_manager.get_latest_price(symbol)
                
                # Calculate smart entry
                smart_entry_result = calculate_smart_entry_price(
                    signal, data, predicted_price, confidence, symbol
                )
                
                # Calculate position info
                if signal != 0:
                    quote_currency = symbol.split('/')[1] if '/' in symbol else 'USD'
                    conversion_rate = get_conversion_rate(quote_currency, api_source, api_key_1, api_key_2)
                    
                    # Use AI TP if enabled
                    take_profit_price = None
                    if use_ai_tp:
                        try:
                            from utils.indicators import get_support_resistance
                            supports, resistances = get_support_resistance(data)
                            atr_value = data['ATR_14'].iloc[-1] if 'ATR_14' in data.columns else current_price * 0.001
                            take_profit_price = calculate_ai_take_profit(
                                signal, smart_entry_result['smart_entry_price'], 
                                supports, resistances, atr_value, current_price
                            )
                        except Exception as e:
                            st.warning(f"AI TP calculation failed: {e}")
                    
                    position_info = calculate_position_info(
                        signal, symbol, smart_entry_result['smart_entry_price'],
                        sl_pips, tp_pips, account_balance, risk_percent,
                        conversion_rate, take_profit_price
                    )
                else:
                    position_info = None
                
                # Display results
                st.subheader("🎯 Trading Signal Results")
                display_smart_signal_results(
                    signal, confidence, smart_entry_result, position_info, 
                    symbol, ws_price, current_price
                )
                
                # Model predictions breakdown
                individual_preds = prediction_result.get('individual_predictions', {})
                if individual_preds:
                    with st.expander("🤖 Individual Model Predictions", expanded=False):
                        for model_name, pred_info in individual_preds.items():
                            if 'error' in pred_info:
                                st.error(f"{model_name}: {pred_info['error']}")
                            else:
                                signal_val = pred_info.get('signal', 0)
                                conf_val = pred_info.get('confidence', 0)
                                signal_text = LABEL_MAP.get(signal_val, 'HOLD')
                                st.info(f"{model_name}: {signal_text} (confidence: {conf_val:.2f})")
                
                # Store results in session state for backtesting
                st.session_state.latest_data = data
                st.session_state.latest_models = all_models
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.exception(e)

    # Backtesting section
    st.subheader("📈 Strategy Backtesting")
    
    if st.button("🚀 Run Backtest", key="run_backtest"):
        if 'latest_data' not in st.session_state or 'latest_models' not in st.session_state:
            st.warning("⚠️ Please load data and generate a signal first")
            return
        
        data = st.session_state.latest_data
        all_models = st.session_state.latest_models
        
        # Run backtest
        backtest_results = run_backtest(
            symbol, data, account_balance, risk_percent, sl_pips, tp_pips,
            predict_with_models, all_models, api_source, api_key_1, api_key_2, use_ai_tp
        )
        
        if 'error' in backtest_results:
            st.error(f"❌ Backtest failed: {backtest_results['error']}")
        else:
            # Display results
            st.success("✅ Backtest completed!")
            
            # Summary metrics
            formatted_results = format_backtest_results(backtest_results)
            display_metrics_grid(formatted_results, columns=3)
            
            # Detailed summary
            summary_text = generate_backtest_summary(backtest_results)
            st.markdown(summary_text)
            
            # Chart if trades available
            trades = backtest_results.get('trades', [])
            if trades:
                fig = create_trading_chart(data, trades, f"{symbol} Backtest Results")
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade details table
                if st.checkbox("Show Trade Details"):
                    trade_df = pd.DataFrame(trades)
                    st.dataframe(trade_df)


def handle_training_tab(api_source: str, symbol: str, api_key_1: str, api_key_2: Optional[str]):
    """Handle the AI training tab functionality."""
    
    st.header("🏗️ AI Model Training")
    st.info("Train multiple AI models for trading signal generation")
    
    # Training parameters
    col1, col2 = st.columns(2)
    
    with col1:
        timeframe = st.selectbox("Timeframe for Training", ["1h", "4h", "1d"], help="Pilih timeframe untuk training")
        data_size = st.slider("Data Size", 500, 2000, 1000, 100, help="Jumlah candle untuk training")
    
    with col2:
        st.info("**Models to be trained:**")
        st.write("• LSTM (Neural Network)")
        st.write("• XGBoost (Gradient Boosting)")
        st.write("• SVM (Support Vector Machine)")
        st.write("• CNN (Convolutional Network)")
        st.write("• Naive Bayes")
        st.write("• Meta Learner (Ensemble)")
    
    if st.button("🏗️ Start Training", key="start_training"):
        with st.spinner("Loading training data..."):
            # Load training data
            data, success = load_and_process_data_enhanced(
                api_source, symbol, TIMEFRAME_MAPPING.get(timeframe, '1h'), 
                api_key_1, api_key_2, data_size
            )
            
            if not success or data is None:
                st.error("❌ Failed to load training data")
                return
            
            if len(data) < 200:
                st.error("❌ Insufficient data for training. Need at least 200 data points.")
                return
        
        # Start training
        st.info("🏗️ Starting AI model training...")
        timeframe_key = timeframe
        symbol_clean = sanitize_filename(symbol)
        
        training_results = train_and_save_all_models(data, symbol_clean, timeframe_key)
        
        # Display training results
        if training_results:
            successful_models = [k for k, v in training_results.items() if v.get('success', False)]
            failed_models = [k for k, v in training_results.items() if not v.get('success', False)]
            
            if successful_models:
                st.success(f"✅ Successfully trained {len(successful_models)} models:")
                for model in successful_models:
                    st.write(f"  • {model}")
            
            if failed_models:
                st.warning(f"⚠️ Failed to train {len(failed_models)} models:")
                for model in failed_models:
                    error = training_results[model].get('error', 'Unknown error')
                    st.write(f"  • {model}: {error}")
            
            if successful_models:
                st.balloons()
                st.success("🎉 Training completed! You can now use the models for trading signals.")


def handle_help_tab():
    """Handle the help tab with usage instructions."""
    
    st.header("❓ Help & Documentation")
    
    with st.expander("🚀 Getting Started", expanded=True):
        st.markdown("""
        **1. Setup API Keys:**
        - For Twelve Data: Get free API key from https://twelvedata.com/
        - For Binance: Create API key from your Binance account
        
        **2. Train AI Models:**
        - Go to "Pelatihan AI" tab
        - Select symbol and timeframe
        - Click "Start Training" (takes 2-5 minutes)
        
        **3. Generate Trading Signals:**
        - Go to "Trading" tab
        - Set risk parameters
        - Click "Load Data & Generate Signal"
        - Review smart entry analysis
        
        **4. Run Backtests:**
        - After generating a signal, click "Run Backtest"
        - Review performance metrics
        - Analyze trade details
        """)
    
    with st.expander("🌐 WebSocket Features"):
        st.markdown("""
        **Real-time Price Streaming:**
        - Twelve Data: Forex, Stocks, Crypto
        - Binance: Cryptocurrency pairs
        - Auto-fallback to polling if WebSocket fails
        
        **Benefits:**
        - Lower latency signals
        - Real-time validation
        - Better fill probability estimation
        """)
    
    with st.expander("🧠 Smart Entry System"):
        st.markdown("""
        **Multi-factor Analysis:**
        - Support/Resistance levels
        - RSI momentum
        - MACD trend confirmation
        - ATR volatility adjustment
        - Real-time price validation
        
        **Risk Assessment:**
        - LOW: Optimal market conditions
        - MEDIUM: Acceptable risk
        - HIGH: Avoid or reduce size
        - REJECTED: Signal fails validation
        """)
    
    with st.expander("⚙️ Technical Information"):
        st.markdown(f"""
        **System Status:**
        - WebSocket Available: {create_status_indicator('success' if is_feature_enabled('WEBSOCKET_AVAILABLE') else 'error')}
        - TensorFlow Available: {create_status_indicator('success' if TF_AVAILABLE else 'error')}
        - Validation Utils: {create_status_indicator('success' if is_feature_enabled('VALIDATION_UTILS_AVAILABLE') else 'warning')}
        - Binance Support: {create_status_indicator('success' if is_feature_enabled('BINANCE_AVAILABLE') else 'warning')}
        - Phase 3 Systems: {create_status_indicator('success' if NEW_SYSTEM_AVAILABLE else 'info')}
        
        **Model Files Location:** `{get_model_directory()}`
        
        **Performance Tips:**
        - Use WebSocket for lowest latency
        - Cache data for 5 minutes to save API calls
        - Train models during market close
        - Monitor data quality indicators
        
        **For Technical Support:**
        - Check browser console (F12) for errors
        - Screenshot any error messages
        - Verify all requirements are installed
        - Test with different data providers
        
        **Version:** v8.3 - Twelve Data WebSocket Enhanced
        **GitHub:** @earleoshio
        """)


if __name__ == "__main__":
    main()