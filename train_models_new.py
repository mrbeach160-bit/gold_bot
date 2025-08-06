#!/usr/bin/env python3
"""
Enhanced train_models.py using the new unified model architecture.
Maintains backward compatibility while leveraging the new ModelManager.
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def train_models_new_system(symbol: str, timeframe: str, data, use_config: bool = True):
    """Train models using the new unified system."""
    print(f"🚀 Training models using new unified system for {symbol} {timeframe}")
    
    try:
        # Import new model system
        from models import ModelManager
        
        # Initialize ModelManager
        if use_config:
            # Try to use configuration system
            try:
                from config import ConfigManager
                config_manager = ConfigManager()
                app_config = config_manager.get_config()
                model_config = app_config.model if app_config else None
                print("✅ Using configuration system")
            except:
                model_config = None
                print("⚠️  Configuration system unavailable, using defaults")
        else:
            model_config = None
            print("📋 Using default model configuration")
        
        # Create ModelManager
        manager = ModelManager(symbol, timeframe, model_config)
        print(f"✅ ModelManager initialized with {len(manager.get_available_models())} models")
        
        # Train all models
        print("🧠 Training individual models...")
        training_results = manager.train_all_models(data)
        
        # Report results
        successful_models = [name for name, success in training_results.items() if success]
        failed_models = [name for name, success in training_results.items() if not success]
        
        print(f"\n✅ Successfully trained {len(successful_models)} models: {successful_models}")
        if failed_models:
            print(f"❌ Failed to train {len(failed_models)} models: {failed_models}")
        
        # Save all trained models
        print("💾 Saving trained models...")
        save_results = manager.save_all_models()
        saved_models = [name for name, success in save_results.items() if success]
        print(f"✅ Saved {len(saved_models)} models: {saved_models}")
        
        # Display model status
        status = manager.get_model_status()
        print(f"\n📊 Model Status Summary:")
        print(f"   Total models: {status['summary']['total_models']}")
        print(f"   Trained models: {status['summary']['trained_models']}")
        print(f"   Available on disk: {status['summary']['available_models']}")
        
        return len(successful_models) > 0
        
    except Exception as e:
        print(f"❌ Error in new model system: {e}")
        return False

def train_models_legacy_system(symbol: str, timeframe: str, data):
    """Fallback to legacy training system."""
    print(f"📋 Training models using legacy system for {symbol} {timeframe}")
    
    try:
        # Import legacy training function
        sys.path.append(os.path.join(project_root, "streamlit_app"))
        from app import train_and_save_all_models
        
        # Use legacy training
        train_and_save_all_models(data, symbol, timeframe)
        print("✅ Legacy training completed")
        return True
        
    except Exception as e:
        print(f"❌ Error in legacy training: {e}")
        return False

def main(args):
    """Main training function with fallback options."""
    print(f"🚀 Starting training process for {args.symbol} @ {args.timeframe}")
    print(f"📊 Data size requested: {args.data_size}")
    
    # Import configuration system if available
    try:
        from config import ConfigManager, AppConfig, TradingConfig, APIConfig
        from utils.data import get_gold_data
        CONFIG_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️  Configuration system unavailable: {e}")
        from utils.data import get_gold_data
        CONFIG_AVAILABLE = False
    
    # Get training data
    training_data = None
    
    if CONFIG_AVAILABLE and not args.no_config:
        try:
            print("📊 Using configuration system for data fetching...")
            
            # Timeframe conversion
            tf_map = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1day'}
            api_timeframe = tf_map.get(args.timeframe, args.timeframe)
            
            # Create configuration
            trading_config = TradingConfig(symbol=args.symbol, timeframe=api_timeframe)
            api_config = APIConfig(twelve_data_key=args.apikey)
            app_config = AppConfig(trading=trading_config, api=api_config)
            
            # Load configuration and fetch data
            config_manager = ConfigManager()
            config_manager.load_config(app_config)
            training_data = get_gold_data(outputsize=args.data_size)
            
            print(f"✅ Configuration loaded: {app_config.trading.symbol} @ {app_config.trading.timeframe}")
            
        except Exception as e:
            print(f"⚠️  Configuration system error: {e}")
            print("📋 Falling back to direct data fetching...")
            
    if training_data is None:
        print("📈 Fetching data directly...")
        tf_map = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1day'}
        api_timeframe = tf_map.get(args.timeframe, args.timeframe)
        
        if not api_timeframe:
            print(f"❌ Invalid timeframe '{args.timeframe}'. Valid options: {list(tf_map.keys())}")
            return
        
        training_data = get_gold_data(args.apikey, interval=api_timeframe, symbol=args.symbol, outputsize=args.data_size)
    
    # Validate data
    if training_data is None or len(training_data) < 60:
        print("❌ Insufficient data for training. Need at least 60 data points.")
        print("💡 Tips:")
        print("   - Check internet connection")
        print("   - Verify Twelve Data API key")
        print("   - Ensure trading symbol is valid")
        return
    
    print(f"✅ Data successfully downloaded: {len(training_data)} rows")
    
    # Try training with new system first
    success = False
    
    if not args.legacy:
        print("\n🔄 Attempting training with new unified model system...")
        success = train_models_new_system(args.symbol, args.timeframe, training_data, 
                                         use_config=CONFIG_AVAILABLE and not args.no_config)
    
    # Fallback to legacy system if needed
    if not success and not args.new_only:
        print("\n🔄 Falling back to legacy training system...")
        success = train_models_legacy_system(args.symbol, args.timeframe, training_data)
    
    # Final results
    if success:
        print("\n" + "="*60)
        print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Symbol: {args.symbol}")
        print(f"⏰ Timeframe: {args.timeframe}")
        print(f"📈 Data Points: {len(training_data)}")
        print(f"🧠 Training System: {'New Unified' if not args.legacy else 'Legacy'}")
    else:
        print("\n" + "="*60)
        print("❌ MODEL TRAINING FAILED")
        print("="*60)
        print("💡 Try using --legacy flag for legacy training system")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced model training script with unified architecture support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use new unified system (recommended)
  python train_models_new.py --symbol "XAU/USD" --timeframe "5m" --apikey "your_key"
  
  # Use legacy system
  python train_models_new.py --symbol "XAU/USD" --timeframe "5m" --apikey "your_key" --legacy
  
  # New system only (no fallback)
  python train_models_new.py --symbol "XAU/USD" --timeframe "5m" --apikey "your_key" --new-only
  
  # Disable configuration system
  python train_models_new.py --symbol "XAU/USD" --timeframe "5m" --apikey "your_key" --no-config

Notes:
  - API key can be set via TWELVE_DATA_API_KEY environment variable
  - New system provides unified model management and better error handling
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
                       help="Disable configuration system")
    parser.add_argument("--legacy", action="store_true",
                       help="Use legacy training system only")
    parser.add_argument("--new-only", action="store_true",
                       help="Use new system only (no legacy fallback)")
    
    args = parser.parse_args()
    main(args)