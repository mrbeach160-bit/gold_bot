# app.py - Simplified Main Application (Modular Version)
# Multi-Asset Trading Bot v8.3 - Twelve Data WebSocket Enhanced
# 
# This is the new modular version that orchestrates all components
# Original 2816-line monolithic file has been broken down into:
# - websockets/ - WebSocket clients and managers
# - ai/ - Smart entry, model training, and prediction
# - trading/ - Position management and backtesting  
# - data/ - Data loading and processing
# - ui/ - Interface components
# - utils/ - Utility functions
# - config/ - Configuration and settings

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import configuration and feature flags
from .config import (
    APP_TITLE, PAGE_CONFIG, NEW_SYSTEM_AVAILABLE, WEBSOCKET_AVAILABLE,
    BINANCE_AVAILABLE, VALIDATION_UTILS_AVAILABLE, UTILS_AVAILABLE
)

# Import modular components
from .websockets import EnhancedWebSocketManager
from .ui import (
    display_sidebar_controls, display_system_status, display_market_status,
    display_trading_interface, display_training_interface, display_model_management
)

# Configure environment
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

# Check TensorFlow
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Disable GPU
except ImportError:
    st.error("TensorFlow tidak terinstall. Install dengan: pip install tensorflow")
    st.stop()

# Check XGBoost
try:
    from xgboost import XGBClassifier
except ImportError:
    st.error("XGBoost tidak terinstall. Install dengan: pip install xgboost")
    st.stop()

# Import Phase 3 systems if available
data_manager = None
trading_manager = None
if NEW_SYSTEM_AVAILABLE:
    try:
        from data import DataManager
        from trading import TradingManager
        
        data_manager = DataManager()
        trading_manager = TradingManager(data_manager=data_manager)
        print("✅ Phase 3 unified systems loaded")
    except Exception as e:
        print(f"⚠️ Phase 3 systems failed to load: {e}")

# Validate utils availability
if not UTILS_AVAILABLE:
    st.error("Utils modules tidak ditemukan. Pastikan folder 'utils' tersedia dengan file:")
    st.error("- utils/data.py (untuk get_gold_data)")
    st.error("- utils/indicators.py (untuk add_indicators, get_support_resistance, compute_rsi)")
    st.error("- utils/meta_learner.py (untuk train_meta_learner, prepare_data_for_meta_learner, get_meta_signal)")
    st.stop()


def main():
    """Main application function"""
    # Configure page
    st.set_page_config(**PAGE_CONFIG)
    st.title(APP_TITLE)
    
    # Initialize WebSocket manager
    if 'ws_manager' not in st.session_state:
        st.session_state.ws_manager = EnhancedWebSocketManager()
    
    # Display system status
    display_system_status(NEW_SYSTEM_AVAILABLE, WEBSOCKET_AVAILABLE, data_manager, trading_manager)
    
    # Sidebar controls
    api_source, api_key_1, api_key_2, symbol, account_balance = display_sidebar_controls()
    
    # Market status
    display_market_status()
    
    # Main interface tabs
    tab1, tab2 = st.tabs(["Trading", "Pelatihan AI"])
    
    with tab1:
        display_trading_interface(api_source, symbol, api_key_1, api_key_2, account_balance)
    
    with tab2:
        display_training_interface(api_source, symbol, api_key_1, api_key_2)
        display_model_management()
    
    # Footer with help information
    _display_footer()


def _display_footer():
    """Display footer with help information"""
    st.markdown("---")
    with st.expander("📚 Help & Troubleshooting", expanded=False):
        st.markdown("""
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
        - Version: v8.3 - Twelve Data WebSocket Enhanced (Modular)
        """)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Critical application error: {str(e)}")
        st.exception(e)
        st.info("💡 Try refreshing the page or check the troubleshooting guide below.")