"""
Gold Bot App - Refactored Modular Version
Main application class that orchestrates all components
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import time

# Add the streamlit_app directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import our modular components
from components.websocket_panel import WebSocketPanel, EnhancedWebSocketManager
from components.trading_panel import TradingPanel
from components.model_status import ModelStatusDisplay
from components.live_stream import LiveStreamManager
from components.backtest_runner import BacktestRunner

# Safe imports with error handling
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from binance.client import Client as BinanceClient
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Try to import utils - these would need to be available
try:
    from utils.data import get_gold_data
    from utils.indicators import add_indicators
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    st.warning("⚠️ Utils modules not available. Some features may be limited.")


class GoldBotApp:
    """Main Gold Bot Application Class - Modular Architecture"""
    
    def __init__(self):
        """Initialize the application and all components"""
        self.ws_manager = None
        self.websocket_panel = None
        self.trading_panel = None
        self.model_status = None
        self.live_stream = None
        self.backtest_runner = None
        
        # Application state
        self.api_source = None
        self.symbol = None
        self.api_key_1 = None
        self.api_key_2 = None
        self.account_balance = 1000.0
        
        self._initialize_components()
        self._setup_session_state()
    
    def _initialize_components(self):
        """Initialize all modular components"""
        # Initialize WebSocket manager
        if 'ws_manager' not in st.session_state:
            st.session_state.ws_manager = EnhancedWebSocketManager()
        self.ws_manager = st.session_state.ws_manager
        
        # Initialize components
        self.websocket_panel = WebSocketPanel(self.ws_manager)
        self.trading_panel = TradingPanel()
        self.model_status = ModelStatusDisplay()
        self.live_stream = LiveStreamManager(self.ws_manager)
        self.backtest_runner = BacktestRunner()
    
    def _setup_session_state(self):
        """Setup session state variables"""
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
            st.session_state.current_data = None
            st.session_state.models_loaded = False
            st.session_state.all_models = None
    
    def render_header(self):
        """Render application header and title"""
        st.set_page_config(
            page_title="Gold Bot v8.3 - Modular",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🤖 Gold Bot Trading AI v8.3 - Modular Architecture")
        
        # System status indicators
        status_cols = st.columns(4)
        
        with status_cols[0]:
            if TENSORFLOW_AVAILABLE and XGBOOST_AVAILABLE:
                st.success("🧠 AI Models Ready")
            else:
                st.error("❌ AI Models Missing")
        
        with status_cols[1]:
            if WEBSOCKET_AVAILABLE:
                st.success("🌐 WebSocket Available")
            else:
                st.warning("⚠️ WebSocket Disabled")
        
        with status_cols[2]:
            if UTILS_AVAILABLE:
                st.success("🔧 Utils Available")
            else:
                st.warning("⚠️ Limited Utils")
        
        with status_cols[3]:
            if BINANCE_AVAILABLE:
                st.success("🪙 Binance Ready")
            else:
                st.info("ℹ️ Binance Optional")
    
    def render_sidebar_config(self):
        """Render sidebar configuration"""
        with st.sidebar:
            st.header("🔧 Configuration")
            
            # API Source Selection
            self.api_source = st.selectbox(
                "Data Provider", 
                ["Twelve Data", "Binance"],
                help="Choose your data provider"
            )
            
            # API Key Configuration
            if self.api_source == "Twelve Data":
                st.info("📈 Forex, Commodities, and Crypto data with WebSocket")
                self.api_key_1 = st.text_input(
                    "Twelve Data API Key", 
                    type="password",
                    help="Get free API key at https://twelvedata.com/"
                )
                
                symbol_options = ["XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD", "ETH/USD"]
                
            elif self.api_source == "Binance":
                st.info("💱 Cryptocurrency data from Binance")
                self.api_key_1 = st.text_input("Binance API Key", type="password")
                self.api_key_2 = st.text_input("Binance API Secret", type="password")
                
                symbol_options = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
            
            # Symbol Selection
            self.symbol = st.selectbox(
                "Trading Symbol",
                symbol_options,
                help="Select the asset to analyze"
            )
            
            # Account Settings
            st.header("💰 Account Settings")
            self.account_balance = st.number_input(
                "Account Balance ($)",
                min_value=100.0,
                value=1000.0,
                step=100.0
            )
            
            # Market Status
            self._render_market_status()
            
            # Component Status
            self._render_component_status()
    
    def _render_market_status(self):
        """Render market status in sidebar"""
        st.header("📊 Market Status")
        current_hour = datetime.now().hour
        
        if 0 <= current_hour < 6:
            st.info("🌙 Market Closed (Night)")
        elif 6 <= current_hour < 22:
            st.success("💹 Market Open")
        else:
            st.warning("🕒 Market Closing Soon")
    
    def _render_component_status(self):
        """Render component status in sidebar"""
        st.header("🔧 Component Status")
        
        components = {
            "WebSocket Panel": self.websocket_panel is not None,
            "Trading Panel": self.trading_panel is not None,
            "Model Status": self.model_status is not None,
            "Live Stream": self.live_stream is not None,
            "Backtest Runner": self.backtest_runner is not None
        }
        
        for component, status in components.items():
            if status:
                st.success(f"✅ {component}")
            else:
                st.error(f"❌ {component}")
    
    def validate_configuration(self):
        """Validate current configuration"""
        errors = []
        
        if not self.symbol:
            errors.append("Please select a trading symbol")
        
        if self.api_source == "Twelve Data":
            if not self.api_key_1 or self.api_key_1.strip() == "":
                errors.append("Twelve Data API key is required")
        
        elif self.api_source == "Binance":
            if not self.api_key_1 or not self.api_key_2:
                errors.append("Both Binance API key and secret are required")
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
            return False
        
        return True
    
    def render_main_interface(self):
        """Render the main application interface"""
        if not self.validate_configuration():
            st.stop()
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["📊 Trading", "🤖 AI Models", "📈 Analytics"])
        
        with tab1:
            self._render_trading_tab()
        
        with tab2:
            self._render_models_tab()
        
        with tab3:
            self._render_analytics_tab()
    
    def _render_trading_tab(self):
        """Render trading tab with all trading functionality"""
        st.header("💼 Trading Dashboard")
        
        # Sub-tabs for trading features
        subtab1, subtab2, subtab3 = st.tabs(["🔴 Live Signals", "📋 Backtesting", "📡 Live Stream"])
        
        with subtab1:
            self._render_live_signals_panel()
        
        with subtab2:
            self._render_backtesting_panel()
        
        with subtab3:
            self._render_live_stream_panel()
    
    def _render_live_signals_panel(self):
        """Render live signals panel"""
        st.subheader("🔍 Live Signal Generation")
        
        # WebSocket panel if available
        if WEBSOCKET_AVAILABLE and self.api_source in ["Binance", "Twelve Data"]:
            self.websocket_panel.render_websocket_panel(
                self.api_source, self.symbol, self.api_key_1, self.api_key_2
            )
        
        # Trading controls
        trading_config = self.trading_panel.render_trading_controls(
            self.symbol, self.account_balance
        )
        
        # Signal generation buttons
        signal_controls = self.trading_panel.render_signal_buttons()
        
        if signal_controls['generate_signal']:
            self._generate_and_display_signal(trading_config)
    
    def _render_backtesting_panel(self):
        """Render backtesting panel"""
        st.subheader("🔄 Strategy Backtesting")
        
        # Backtest configuration
        backtest_config = self.backtest_runner.render_backtest_controls(
            self.symbol, self.account_balance
        )
        
        # Data configuration
        data_cols = st.columns(2)
        with data_cols[0]:
            timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)
            data_size = st.number_input("Data Size", min_value=100, value=1000, step=100)
        
        with data_cols[1]:
            if st.button("🚀 Run Backtest", type="primary"):
                self._run_backtest(backtest_config, timeframe, data_size)
        
        # Display results
        self.backtest_runner.render_backtest_results()
        self.backtest_runner.render_export_controls()
    
    def _render_live_stream_panel(self):
        """Render live stream panel"""
        st.subheader("📡 Live Stream Analysis")
        
        # Live stream controls
        self.live_stream.render_live_stream_panel(self.symbol, "15m")
        
        # Live signals feed
        self.live_stream.render_live_signals_feed()
        
        # Performance metrics
        self.live_stream.render_signal_quality_metrics()
        
        # Export and history controls
        col1, col2 = st.columns(2)
        with col1:
            self.live_stream.render_export_controls()
        with col2:
            self.live_stream.render_history_controls()
    
    def _render_models_tab(self):
        """Render AI models tab"""
        st.header("🤖 AI Model Management")
        
        # Model status display
        ensemble_readiness = self.model_status.render_model_status_panel(
            self.symbol, "15m"
        )
        
        # Model freshness check
        self.model_status.render_freshness_warning(self.symbol, "15m")
        
        # Training controls
        if st.button("🏗️ Train Models", type="primary"):
            self._train_models()
    
    def _render_analytics_tab(self):
        """Render analytics and performance tab"""
        st.header("📈 Analytics Dashboard")
        
        # Performance overview
        self._render_performance_overview()
        
        # Live metrics if available
        if self.live_stream.get_live_signals():
            self.live_stream.render_live_metrics_chart()
    
    def _generate_and_display_signal(self, trading_config):
        """Generate and display trading signal"""
        with st.spinner("🔄 Generating signal..."):
            try:
                # This would integrate with actual model prediction
                # For now, create a mock result
                signal = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4])
                confidence = np.random.uniform(0.6, 0.95)
                current_price = 2000.0  # Mock price
                
                # Get real-time price if available
                ws_price = self.websocket_panel.get_realtime_price(self.symbol)
                
                # Create mock recent data for smart entry calculation
                mock_data = pd.DataFrame({
                    'close': [current_price] * 100,
                    'high': [current_price * 1.01] * 100,
                    'low': [current_price * 0.99] * 100,
                    'rsi': [50] * 100,
                    'ATR_14': [current_price * 0.01] * 100,
                    'MACD_12_26_9': [0] * 100,
                    'MACDs_12_26_9': [0] * 100
                })
                
                # Calculate smart entry
                smart_entry_result = self.trading_panel.calculate_smart_entry_price(
                    signal, mock_data, current_price, confidence, self.symbol
                )
                
                # Calculate position info
                position_info = None
                if smart_entry_result['risk_level'] != 'REJECTED':
                    position_info = self.trading_panel.calculate_position_info(
                        signal, self.symbol, smart_entry_result['entry_price'],
                        trading_config['sl_pips'], trading_config['tp_pips'],
                        self.account_balance, trading_config['risk_percent'],
                        conversion_rate_to_usd=1.0
                    )
                
                # Display results
                self.trading_panel.display_smart_signal_results(
                    signal, confidence, smart_entry_result, position_info,
                    self.symbol, ws_price, current_price
                )
                
            except Exception as e:
                st.error(f"❌ Error generating signal: {e}")
    
    def _run_backtest(self, config, timeframe, data_size):
        """Run backtest with current configuration"""
        with st.spinner("🔄 Running backtest..."):
            try:
                # Mock data generation for demonstration
                # In real implementation, this would fetch actual historical data
                dates = pd.date_range(end=datetime.now(), periods=data_size, freq='15min')
                mock_data = pd.DataFrame({
                    'close': np.random.normal(2000, 20, data_size),
                    'high': np.random.normal(2020, 20, data_size),
                    'low': np.random.normal(1980, 20, data_size),
                    'volume': np.random.randint(1000, 10000, data_size)
                }, index=dates)
                
                # Run backtest
                results = self.backtest_runner.run_backtest(
                    symbol=self.symbol,
                    data=mock_data,
                    initial_balance=config['initial_balance'],
                    risk_percent=config['risk_percent'],
                    sl_pips=config['sl_pips'],
                    tp_pips=config['tp_pips'],
                    predict_func=None,  # Would be actual prediction function
                    all_models=None,
                    api_source=self.api_source,
                    api_key_1=self.api_key_1,
                    api_key_2=self.api_key_2,
                    use_ai_tp=config['use_ai_tp']
                )
                
                if results:
                    st.success("✅ Backtest completed successfully!")
                else:
                    st.error("❌ Backtest failed")
                    
            except Exception as e:
                st.error(f"❌ Backtest error: {e}")
    
    def _train_models(self):
        """Train AI models"""
        with st.spinner("🏗️ Training models..."):
            try:
                # Mock training process
                st.info("🔄 Training would start here...")
                time.sleep(2)  # Simulate training time
                st.success("✅ Models trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Training error: {e}")
    
    def _render_performance_overview(self):
        """Render performance overview"""
        st.subheader("📊 Performance Overview")
        
        # Mock performance data
        perf_cols = st.columns(4)
        
        with perf_cols[0]:
            st.metric("Total Trades", 150, delta="+12")
        
        with perf_cols[1]:
            st.metric("Win Rate", "68.5%", delta="+2.3%")
        
        with perf_cols[2]:
            st.metric("Profit Factor", "1.45", delta="+0.12")
        
        with perf_cols[3]:
            st.metric("Max Drawdown", "8.2%", delta="-1.1%")
    
    def run(self):
        """Main application entry point"""
        try:
            # Render header
            self.render_header()
            
            # Render sidebar configuration
            self.render_sidebar_config()
            
            # Render main interface
            self.render_main_interface()
            
        except Exception as e:
            st.error(f"🚨 Application Error: {str(e)}")
            st.exception(e)
            
            # Error recovery suggestions
            with st.expander("🛠️ Troubleshooting", expanded=False):
                st.markdown("""
                **If you encounter errors:**
                1. **Refresh the page** (F5 or Ctrl+R)
                2. **Check your API keys** are valid
                3. **Verify internet connection**
                4. **Install missing dependencies**:
                   ```bash
                   pip install streamlit pandas plotly numpy scikit-learn
                   pip install tensorflow xgboost websocket-client
                   ```
                5. **Clear browser cache** if issues persist
                
                **For technical support:**
                - Check console logs (F12)
                - Report issues with error screenshots
                - Version: Gold Bot v8.3 Modular
                """)


def main():
    """Main function to run the application"""
    app = GoldBotApp()
    app.run()


if __name__ == "__main__":
    main()