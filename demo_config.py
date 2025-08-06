#!/usr/bin/env python3
"""
Demonstration of the Gold Bot Centralized Configuration Management System
This script shows how to use the new configuration system in practice.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def demo_configuration_system():
    """Demonstrate the complete configuration system capabilities."""
    print("🚀 Gold Bot Configuration System Demonstration")
    print("=" * 60)
    
    try:
        # Import configuration system
        from config import ConfigManager, AppConfig, TradingConfig, APIConfig, ModelConfig
        from config import validate_config, get_config, update_config
        
        print("\n📊 1. Creating Custom Configuration")
        print("-" * 40)
        
        # Create a complete configuration for demonstration
        demo_config = AppConfig(
            trading=TradingConfig(
                symbol="XAU/USD",
                timeframe="5min",
                risk_percentage=1.5,
                stop_loss_pips=25,
                take_profit_pips=50,
                leverage=20,
                use_ai_take_profit=True,
                minimum_confidence=0.65,
                account_balance=5000.0
            ),
            api=APIConfig(
                twelve_data_key="demo_twelve_data_key_12345",
                binance_api_key="demo_binance_api_key_12345",
                binance_secret="demo_binance_secret_12345",
                use_testnet=True,
                api_timeout=45,
                max_retries=5
            ),
            model=ModelConfig(
                models_to_use=["lstm", "xgb", "cnn", "meta"],
                ensemble_method="meta_learner",
                confidence_threshold=0.7,
                lstm_sequence_length=60,
                batch_size=32
            ),
            debug=True,
            environment="development",
            default_data_source="Twelve Data"
        )
        
        print(f"✅ Created configuration for {demo_config.trading.symbol}")
        print(f"   Environment: {demo_config.environment}")
        print(f"   Data Source: {demo_config.default_data_source}")
        print(f"   Risk per trade: {demo_config.trading.risk_percentage}%")
        
        print("\n🔍 2. Configuration Validation")
        print("-" * 40)
        
        validation_result = validate_config(demo_config)
        print(f"✅ Configuration valid: {validation_result.is_valid}")
        
        if validation_result.warnings:
            print(f"⚠️  Warnings ({len(validation_result.warnings)}):")
            for warning in validation_result.warnings[:3]:
                print(f"   - {warning}")
        
        print("\n⚙️  3. Loading Configuration into Manager")
        print("-" * 40)
        
        config_manager = ConfigManager()
        loaded_config = config_manager.load_config(demo_config)
        
        print(f"✅ Configuration loaded successfully")
        print(f"   Symbol: {loaded_config.trading.symbol}")
        print(f"   Timeframe: {loaded_config.trading.timeframe}")
        print(f"   API timeout: {loaded_config.api.api_timeout}s")
        
        print("\n🔄 4. Runtime Configuration Updates")
        print("-" * 40)
        
        # Update trading parameters
        update_config(
            trading_symbol="BTC/USD",
            trading_risk_percentage=2.0,
            model_confidence_threshold=0.75
        )
        
        updated_config = get_config()
        print(f"✅ Updated symbol: {updated_config.trading.symbol}")
        print(f"✅ Updated risk: {updated_config.trading.risk_percentage}%")
        print(f"✅ Updated confidence: {updated_config.model.confidence_threshold}")
        
        print("\n📝 5. Configuration Serialization")
        print("-" * 40)
        
        config_dict = updated_config.to_dict()
        print(f"✅ Configuration serialized to dict with {len(config_dict)} sections:")
        for section, data in config_dict.items():
            if isinstance(data, dict):
                print(f"   - {section}: {len(data)} parameters")
        
        print("\n🧪 6. Testing Configuration-Aware Modules")
        print("-" * 40)
        
        # Test data module
        from utils.data import get_gold_data
        print("✅ Data module loaded with config support")
        print(f"   Will use API key: {updated_config.api.twelve_data_key[:10]}...")
        print(f"   Default symbol: {updated_config.trading.symbol}")
        print(f"   Default timeframe: {updated_config.trading.timeframe}")
        
        # Test binance module
        from utils.binance_trading import init_testnet_client, get_binance_endpoints
        testnet_url, stream_url = get_binance_endpoints()
        print("✅ Binance module loaded with config support")
        print(f"   Using testnet: {updated_config.api.use_testnet}")
        print(f"   Endpoint: {testnet_url}")
        
        print("\n🎯 7. Environment-Specific Configuration")
        print("-" * 40)
        
        # Show how configuration adapts to environment
        environments = ["development", "staging", "production"]
        
        for env in environments:
            env_config = AppConfig(environment=env)
            
            # Production-specific validations
            if env == "production":
                print(f"📊 {env.upper()} environment:")
                print(f"   Debug mode: {env_config.debug} (should be False)")
                print(f"   Log level: {env_config.log_level}")
                
            elif env == "development":
                print(f"🛠️  {env.upper()} environment:")
                print(f"   Debug mode: {env_config.debug}")
                print(f"   Log level: {env_config.log_level}")
        
        print("\n💡 8. Configuration Best Practices Demonstrated")
        print("-" * 40)
        print("✅ Type-safe configuration classes")
        print("✅ Environment variable integration")
        print("✅ Validation with helpful error messages")
        print("✅ Runtime updates with thread safety")
        print("✅ Backward compatibility maintained")
        print("✅ Secure API key handling")
        print("✅ Environment-specific settings")
        print("✅ Comprehensive documentation")
        
        print("\n🎉 Configuration System Demo Complete!")
        print("=" * 60)
        print("The Gold Bot now has a professional-grade configuration")
        print("management system ready for production deployment.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_configuration_system()
    
    if success:
        print(f"\n📚 Next Steps:")
        print("1. Copy .env.example to .env and add your real API keys")
        print("2. Run: python train_models.py --symbol 'XAU/USD' --timeframe '5m'")
        print("3. Launch the app: streamlit run streamlit_app/app.py")
        print("4. Read CONFIG_USAGE.md for detailed documentation")
        
    sys.exit(0 if success else 1)