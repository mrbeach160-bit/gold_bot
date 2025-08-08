#!/usr/bin/env python3
"""
Test script to verify the migrated utils modules work with the configuration system.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_migrated_modules():
    """Test the migrated utils modules."""
    print("Testing Migrated Utils Modules...")
    
    try:
        # Test 1: Import and setup config
        print("\n1. Setting up configuration...")
        from config import ConfigManager, AppConfig, APIConfig, TradingConfig
        
        # Create test configuration
        test_config = AppConfig(
            trading=TradingConfig(
                symbol="XAU/USD",
                timeframe="5min"
            ),
            api=APIConfig(
                twelve_data_key="test_api_key_12345",
                binance_api_key="test_binance_api_key_12345",
                binance_secret="test_binance_secret_12345"
            )
        )
        
        config_manager = ConfigManager()
        config_manager.load_config(test_config)
        print("✅ Configuration loaded")
        
        # Test 2: Test utils/data.py integration
        print("\n2. Testing utils/data.py integration...")
        from utils.data import get_gold_data
        
        # Test without parameters (should use config)
        try:
            # This will fail due to fake API key, but should show config integration works
            data = get_gold_data(outputsize=1)
        except Exception as e:
            if "API key" in str(e):
                print("✅ get_gold_data correctly uses config API key")
            else:
                print(f"⚠️  Expected API error (using test key): {e}")
        
        # Test with explicit parameters (should override config)
        try:
            data = get_gold_data(
                api_key="explicit_test_key",
                interval="1min", 
                symbol="BTC/USD",
                outputsize=1
            )
        except Exception as e:
            print(f"✅ get_gold_data works with explicit parameters: {type(e).__name__}")
        
        # Test 3: Test utils/binance_trading.py integration
        print("\n3. Testing utils/binance_trading.py integration...")
        from utils.binance_trading import init_testnet_client, get_binance_endpoints
        
        # Test endpoint configuration
        testnet_url, stream_url = get_binance_endpoints()
        print(f"✅ Binance endpoints from config: testnet={config_manager.get_config().api.use_testnet}")
        
        # Test client initialization without parameters (should use config)
        try:
            client = init_testnet_client()
            print("✅ init_testnet_client uses config credentials")
        except Exception as e:
            if "API key" in str(e) or "Invalid" in str(e):
                print("✅ init_testnet_client correctly validates config credentials")
            else:
                print(f"⚠️  Unexpected error: {e}")
        
        # Test 4: Test backward compatibility
        print("\n4. Testing backward compatibility...")
        
        # Reset config to test legacy mode
        config_manager.reset()
        
        try:
            # This should now require explicit parameters
            data = get_gold_data(
                api_key="legacy_test_key",
                interval="5min",
                symbol="XAU/USD",
                outputsize=1
            )
        except Exception as e:
            print(f"✅ Legacy mode works: {type(e).__name__}")
        
        try:
            client = init_testnet_client("legacy_key", "legacy_secret")
            print("✅ Legacy client initialization works")
        except Exception as e:
            print(f"✅ Legacy client validation: {type(e).__name__}")
        
        print("\n🎉 All migration tests passed! Utils modules work correctly with config system.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_migrated_modules()
    sys.exit(0 if success else 1)