# main.py - Enhanced with Phase 3: Data & Trading Simplification
import sys
import os

# --- Configuration System Integration ---
# 1. Dapatkan path absolut dari direktori tempat main.py berada (yaitu, /root/gold_bot/)
project_root = os.path.dirname(os.path.abspath(__file__))

# 2. Tambahkan path ini ke daftar tempat Python mencari modul.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 3. Initialize the configuration system
try:
    from config import ConfigManager, AppConfig
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Load configuration from environment variables
    app_config = AppConfig.from_env()
    config_manager.load_config(app_config)
    
    print(f"✅ Configuration loaded successfully for environment: {app_config.environment}")
    print(f"📊 Default data source: {app_config.default_data_source}")
    print(f"📈 Default trading symbol: {app_config.trading.symbol}")
    
except Exception as e:
    print(f"⚠️  Warning: Could not load configuration system: {e}")
    print("📝 The application will continue with legacy configuration handling")

# --- Phase 3: Initialize Data & Trading Systems ---
try:
    from data import DataManager
    from trading import TradingManager
    
    # Initialize data manager
    data_manager = DataManager()
    print(f"✅ Data Manager initialized - Symbol: {data_manager.symbol}, Providers: {len(data_manager.providers)}")
    
    # Initialize trading manager
    trading_manager = TradingManager(data_manager=data_manager)
    print(f"✅ Trading Manager initialized - Strategy: {trading_manager.strategy.name}")
    
    # Initialize trading system (paper trading mode by default)
    if trading_manager.initialize_trading(enable_trading=False):
        print(f"✅ Trading system initialized in paper trading mode")
    else:
        print(f"⚠️  Trading system initialization had issues but will continue")
    
except Exception as e:
    print(f"⚠️  Warning: Could not fully initialize data/trading systems: {e}")
    print("📝 The application will continue with legacy systems as fallback")
    data_manager = None
    trading_manager = None

# --- Legacy Path Configuration (Maintained for Backward Compatibility) ---

# Make managers available globally for Streamlit app
globals()['data_manager'] = data_manager
globals()['trading_manager'] = trading_manager

# Sekarang, import Anda akan berjalan seperti biasa
from streamlit_app.app import main

if __name__ == "__main__":
    main()
