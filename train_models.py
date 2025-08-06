# train_models.py - Enhanced with Unified Model Architecture Support
# Run this file from terminal to train all AI models.
# Example: python train_models.py --symbol "XAU/USD" --timeframe "5m" --apikey "YOUR_API_KEY"
# New: Use --unified flag to use the new unified model system (recommended)

import os
import sys
import argparse

# --- PATH SETUP ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import configuration system and utilities
try:
    from config import ConfigManager, AppConfig, TradingConfig, APIConfig
    from utils.data import get_gold_data
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Configuration system not available: {e}")
    print("Falling back to legacy mode...")
    
    try:
        from utils.data import get_gold_data
        CONFIG_AVAILABLE = False
    except ImportError as e2:
        print(f"Error: Cannot import required functions: {e2}")
        print("Please ensure all dependencies are available.")
        sys.exit(1)

def train_with_unified_system(symbol, timeframe, data):
    """Train models using the new unified model architecture."""
    print(f"🚀 Training models with unified system for {symbol} {timeframe}")
    
    try:
        from models import ModelManager
        
        # Get model configuration if available
        model_config = None
        if CONFIG_AVAILABLE:
            try:
                config_manager = ConfigManager()
                app_config = config_manager.get_config()
                model_config = app_config.model if app_config else None
                print("✅ Using model configuration from config system")
            except:
                print("⚠️  Using default model configuration")
        
        # Initialize ModelManager
        manager = ModelManager(symbol, timeframe, model_config)
        print(f"✅ ModelManager initialized with {len(manager.get_available_models())} models")
        
        # Train all models
        print("🧠 Training individual models...")
        training_results = manager.train_all_models(data)
        
        # Report training results
        successful_models = [name for name, success in training_results.items() if success]
        failed_models = [name for name, success in training_results.items() if not success]
        
        print(f"✅ Successfully trained {len(successful_models)} models: {successful_models}")
        if failed_models:
            print(f"⚠️  Failed to train {len(failed_models)} models: {failed_models}")
        
        # Save all trained models
        print("💾 Saving trained models...")
        save_results = manager.save_all_models()
        saved_models = [name for name, success in save_results.items() if success]
        print(f"✅ Saved {len(saved_models)} models: {saved_models}")
        
        # Model status summary
        status = manager.get_model_status()
        print(f"\n📊 Training Summary:")
        print(f"   Total models available: {status['summary']['total_models']}")
        print(f"   Successfully trained: {status['summary']['trained_models']}")
        print(f"   Saved to disk: {status['summary']['available_models']}")
        
        return len(successful_models) > 0
        
    except Exception as e:
        print(f"❌ Error in unified training system: {e}")
        return False

def train_with_legacy_system(symbol, timeframe, data):
    """Train models using the original legacy system."""
    print(f"📋 Training models with legacy system for {symbol} {timeframe}")
    
    try:
        # Import training function from streamlit app
        sys.path.append(os.path.join(project_root, "streamlit_app"))
        from app import train_and_save_all_models
        
        # Use legacy training
        train_and_save_all_models(data, symbol, timeframe)
        print("✅ Legacy training completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error in legacy training system: {e}")
        return False

def main(args):
    """Main function to run training process with unified configuration."""
    
    print(f"🚀 Starting model training for {args.symbol} @ {args.timeframe}")
    print(f"🧠 Training system: {'Unified Architecture' if args.unified else 'Legacy System'}")
    
    # Initialize configuration if available
    training_data = None
    
    if CONFIG_AVAILABLE and not args.no_config:
        try:
            print("📊 Initializing configuration system...")
            
            # Convert timeframe display to API format for configuration
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
            print("📈 Downloading historical data using configuration system...")
            training_data = get_gold_data(outputsize=args.data_size)
            
        except Exception as e:
            print(f"⚠️  Configuration system error: {e}")
            print("📋 Falling back to legacy mode...")
    
    # Fallback to direct data fetching if needed
    if training_data is None:
        print("📈 Downloading historical data directly...")
        
        # Convert timeframe display to API format
        tf_map = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1day'}
        api_timeframe = tf_map.get(args.timeframe, args.timeframe)
        
        if not api_timeframe:
            print(f"Error: Invalid timeframe '{args.timeframe}'. Valid options: {list(tf_map.keys())}")
            return

        training_data = get_gold_data(args.apikey, interval=api_timeframe, symbol=args.symbol, outputsize=args.data_size)
    
    # Validate training data
    if training_data is not None and len(training_data) > 60:
        print(f"✅ Data successfully downloaded: {len(training_data)} rows")
        print("🧠 Starting model training...")
        
        # Choose training system
        success = False
        
        if args.unified:
            # Use new unified system
            success = train_with_unified_system(args.symbol, args.timeframe, training_data)
            
            # Fallback to legacy if unified fails and fallback is enabled
            if not success and not args.no_fallback:
                print("\n🔄 Unified system failed, falling back to legacy...")
                success = train_with_legacy_system(args.symbol, args.timeframe, training_data)
        else:
            # Use legacy system
            success = train_with_legacy_system(args.symbol, args.timeframe, training_data)
        
        # Final results
        if success:
            print("\n" + "="*60)
            print("🎉 ALL MODELS SUCCESSFULLY TRAINED!")
            print("="*60)
            
            if CONFIG_AVAILABLE:
                print(f"📊 Configuration: {args.symbol} @ {args.timeframe}")
                print(f"🔑 API Source: Twelve Data")
                print(f"📈 Data Points: {len(training_data)}")
                print(f"🧠 Training System: {'Unified Architecture' if args.unified else 'Legacy System'}")
        else:
            print("\n" + "="*60)
            print("❌ MODEL TRAINING FAILED")
            print("="*60)
            print("💡 Try using --unified flag for the new training system")
            print("💡 Or use --no-fallback to prevent fallback attempts")
    else:
        print("❌ Error: Failed to download sufficient data for training.")
        print("💡 Tips:")
        print("   - Check internet connection")
        print("   - Verify Twelve Data API key")
        print("   - Ensure trading symbol is valid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced model training script with unified architecture support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use new unified architecture (recommended)
  python train_models.py --symbol "XAU/USD" --timeframe "5m" --apikey "your_key" --unified
  
  # Use legacy training system (default)
  python train_models.py --symbol "XAU/USD" --timeframe "5m" --apikey "your_key"
  
  # Unified system with no fallback
  python train_models.py --symbol "XAU/USD" --timeframe "5m" --apikey "your_key" --unified --no-fallback
  
  # Disable configuration system
  python train_models.py --symbol "XAU/USD" --timeframe "5m" --apikey "your_key" --no-config

Notes:
  - API key can be set via TWELVE_DATA_API_KEY environment variable
  - --unified flag enables new unified model architecture (Phase 2)
  - Configuration system provides centralized parameter management (Phase 1)
  - Legacy fallback ensures compatibility with existing workflows
        """)
    
    parser.add_argument("--symbol", type=str, required=True, 
                       help="Trading symbol (e.g., 'XAU/USD', 'BTC/USD')")
    parser.add_argument("--timeframe", type=str, required=True, 
                       help="Timeframe (e.g., '5m', '1h', '15m')")
    parser.add_argument("--apikey", type=str, required=False,
                       help="Twelve Data API Key (or use TWELVE_DATA_API_KEY env var)")
    parser.add_argument("--data-size", type=int, default=5000,
                       help="Number of data points for training (default: 5000)")
    parser.add_argument("--no-config", action="store_true",
                       help="Disable unified configuration system")
    parser.add_argument("--unified", action="store_true",
                       help="Use new unified model architecture (recommended)")
    parser.add_argument("--no-fallback", action="store_true",
                       help="Disable fallback to legacy system if unified fails")
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.apikey and not os.getenv("TWELVE_DATA_API_KEY"):
        print("❌ Error: Twelve Data API Key required!")
        print("💡 Solutions:")
        print("   1. Use --apikey 'your_api_key' parameter")
        print("   2. Set environment variable: export TWELVE_DATA_API_KEY='your_api_key'")
        print("   3. Create .env file and add: TWELVE_DATA_API_KEY=your_api_key")
        sys.exit(1)
    
    # Use environment variable if apikey not provided
    if not args.apikey:
        args.apikey = os.getenv("TWELVE_DATA_API_KEY")
    
    main(args)
