# main.py - Enhanced with Centralized Configuration Management System
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

# --- Legacy Path Configuration (Maintained for Backward Compatibility) ---

# Sekarang, import Anda akan berjalan seperti biasa
from streamlit_app.app import main

if __name__ == "__main__":
    main()
