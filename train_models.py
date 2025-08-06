# train_models.py - Enhanced with Centralized Configuration Management System
# Jalankan file ini dari terminal untuk melatih semua model AI.
# Contoh: python train_models.py --symbol "XAU/USD" --timeframe "5m" --apikey "YOUR_API_KEY"

import os
import sys
import argparse

# --- KODE PERBAIKAN PATH ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import configuration system
try:
    from config import ConfigManager, AppConfig, TradingConfig, APIConfig
    from utils.data import get_gold_data
    
    # Import training function from streamlit app
    sys.path.append(os.path.join(project_root, "streamlit_app"))
    from app import train_and_save_all_models
    
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Configuration system not available: {e}")
    print("Falling back to legacy mode...")
    
    try:
        from utils.data import get_gold_data
        sys.path.append(os.path.join(project_root, "streamlit_app"))
        from app import train_and_save_all_models
        CONFIG_AVAILABLE = False
    except ImportError as e2:
        print(f"Error: Cannot import required functions: {e2}")
        print("Please ensure all dependencies are available.")
        sys.exit(1)

def main(args):
    """Fungsi utama untuk menjalankan proses training dengan konfigurasi terpadu."""
    
    print(f"🚀 Memulai proses training untuk simbol: {args.symbol}, timeframe: {args.timeframe}")
    
    # Initialize configuration if available
    if CONFIG_AVAILABLE and not args.no_config:
        try:
            print("📊 Initializing configuration system...")
            
            # Konversi timeframe display ke format API untuk konfigurasi
            tf_map = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1day'}
            api_timeframe = tf_map.get(args.timeframe, args.timeframe)
            
            # Create configuration with provided parameters
            trading_config = TradingConfig(
                symbol=args.symbol,
                timeframe=api_timeframe
            )
            
            api_config = APIConfig(
                twelve_data_key=args.apikey
            )
            
            app_config = AppConfig(
                trading=trading_config,
                api=api_config
            )
            
            # Load configuration
            config_manager = ConfigManager()
            config_manager.load_config(app_config)
            
            print(f"✅ Configuration loaded: {app_config.trading.symbol} @ {app_config.trading.timeframe}")
            
            # Use config-aware data fetching
            print("📈 Mengunduh data historis menggunakan configuration system...")
            training_data = get_gold_data(outputsize=args.data_size)
            
        except Exception as e:
            print(f"⚠️  Configuration system error: {e}")
            print("📋 Falling back to legacy mode...")
            
            # Konversi timeframe display ke format API
            tf_map = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1day'}
            api_timeframe = tf_map.get(args.timeframe, args.timeframe)
            
            training_data = get_gold_data(args.apikey, interval=api_timeframe, symbol=args.symbol, outputsize=args.data_size)
    else:
        # Legacy mode
        print("📋 Using legacy configuration mode...")
        
        # Konversi timeframe display ke format API
        tf_map = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1day'}
        api_timeframe = tf_map.get(args.timeframe, args.timeframe)
        
        if not api_timeframe:
            print(f"Error: Timeframe '{args.timeframe}' tidak valid. Pilihan: {list(tf_map.keys())}")
            return

        print("📈 Mengunduh data historis dalam jumlah besar...")
        training_data = get_gold_data(args.apikey, interval=api_timeframe, symbol=args.symbol, outputsize=args.data_size)
    
    if training_data is not None and len(training_data) > 60:
        print(f"✅ Data berhasil diunduh: {len(training_data)} baris")
        print("🧠 Memulai training semua model...")
        
        # Timeframe key (e.g., '5m') digunakan untuk penamaan file model
        train_and_save_all_models(training_data, args.symbol, args.timeframe)
        
        print("\n" + "="*50)
        print("🎉 SEMUA MODEL BERHASIL DILATIH!")
        print("="*50)
        
        if CONFIG_AVAILABLE:
            print(f"📊 Configuration: {args.symbol} @ {args.timeframe}")
            print(f"🔑 API Source: Twelve Data")
            print(f"📈 Data Points: {len(training_data)}")
    else:
        print("❌ Error: Gagal mengunduh data yang cukup untuk training.")
        print("💡 Tips:")
        print("   - Periksa koneksi internet")
        print("   - Verifikasi API key Twelve Data")
        print("   - Pastikan simbol trading valid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Skrip untuk melatih model AI trading dengan konfigurasi terpadu.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python train_models.py --symbol "XAU/USD" --timeframe "5m" --apikey "your_api_key"
  python train_models.py --symbol "BTC/USD" --timeframe "1h" --apikey "your_api_key" --data-size 10000
  python train_models.py --symbol "EUR/USD" --timeframe "15m" --apikey "your_api_key" --no-config

Catatan:
  - API key dapat dikonfigurasi melalui environment variable TWELVE_DATA_API_KEY
  - Gunakan --no-config untuk menonaktifkan sistem konfigurasi terpadu
  - Data size default adalah 5000, lebih besar = akurasi lebih tinggi tapi training lebih lama
        """)
    
    parser.add_argument("--symbol", type=str, required=True, 
                       help="Simbol trading (e.g., 'XAU/USD', 'BTC/USD')")
    parser.add_argument("--timeframe", type=str, required=True, 
                       help="Timeframe (e.g., '5m', '1h', '15m')")
    parser.add_argument("--apikey", type=str, required=False,
                       help="API Key Twelve Data (atau gunakan TWELVE_DATA_API_KEY env var)")
    parser.add_argument("--data-size", type=int, default=5000,
                       help="Jumlah data points untuk training (default: 5000)")
    parser.add_argument("--no-config", action="store_true",
                       help="Nonaktifkan sistem konfigurasi terpadu")
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.apikey and not os.getenv("TWELVE_DATA_API_KEY"):
        print("❌ Error: API Key Twelve Data diperlukan!")
        print("💡 Solusi:")
        print("   1. Gunakan parameter --apikey 'your_api_key'")
        print("   2. Set environment variable: export TWELVE_DATA_API_KEY='your_api_key'")
        print("   3. Buat file .env dan tambahkan: TWELVE_DATA_API_KEY=your_api_key")
        sys.exit(1)
    
    # Use environment variable if apikey not provided
    if not args.apikey:
        args.apikey = os.getenv("TWELVE_DATA_API_KEY")
    
    main(args)
