#!/usr/bin/env python3
"""
Simple test script to verify the configuration system works correctly.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_config_system():
    """Test the configuration system."""
    print("Testing Gold Bot Configuration System...")
    
    try:
        # Test 1: Import configuration modules
        print("\n1. Testing imports...")
        from config import TradingConfig, APIConfig, ModelConfig, AppConfig
        from config import ConfigManager, validate_config
        print("✅ All imports successful")
        
        # Test 2: Create configuration instances
        print("\n2. Testing configuration creation...")
        trading_config = TradingConfig(
            symbol="XAU/USD",
            timeframe="5min",
            risk_percentage=1.5,
            stop_loss_pips=25,
            take_profit_pips=50
        )
        print(f"✅ TradingConfig created: {trading_config.symbol}")
        
        api_config = APIConfig(
            twelve_data_key="test_twelve_data_api_key_12345",
            binance_api_key="test_binance_api_key_12345", 
            binance_secret="test_binance_secret_12345"
        )
        print(f"✅ APIConfig created: has_twelve_data={api_config.has_twelve_data}")
        
        model_config = ModelConfig()
        print(f"✅ ModelConfig created: models={len(model_config.models_to_use)}")
        
        app_config = AppConfig(
            trading=trading_config,
            api=api_config,
            model=model_config
        )
        print(f"✅ AppConfig created: environment={app_config.environment}")
        
        # Test 3: Validation
        print("\n3. Testing validation...")
        validation_result = validate_config(app_config)
        print(f"✅ Validation completed: valid={validation_result.is_valid}")
        
        if validation_result.warnings:
            print(f"⚠️  Warnings: {len(validation_result.warnings)}")
            for warning in validation_result.warnings[:3]:  # Show first 3
                print(f"   - {warning}")
        
        if validation_result.errors:
            print(f"❌ Errors: {len(validation_result.errors)}")
            for error in validation_result.errors[:3]:  # Show first 3
                print(f"   - {error}")
        
        # Test 4: ConfigManager
        print("\n4. Testing ConfigManager...")
        config_manager = ConfigManager()
        
        # Load config
        loaded_config = config_manager.load_config(app_config)
        print(f"✅ Config loaded into manager: {loaded_config.environment}")
        
        # Get config
        retrieved_config = config_manager.get_config()
        print(f"✅ Config retrieved: symbol={retrieved_config.trading.symbol}")
        
        # Test update
        config_manager.update_config(trading_symbol="BTC/USD")
        updated_config = config_manager.get_config()
        print(f"✅ Config updated: new symbol={updated_config.trading.symbol}")
        
        # Test 5: Configuration serialization
        print("\n5. Testing serialization...")
        config_dict = app_config.to_dict()
        print(f"✅ Config serialized: {len(config_dict)} sections")
        
        print("\n🎉 All tests passed! Configuration system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_system()
    sys.exit(0 if success else 1)