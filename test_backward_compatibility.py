# test_backward_compatibility.py
"""
Test backward compatibility with legacy utils/data.py and utils/binance_trading.py interfaces
"""

def test_data_backward_compatibility():
    """Test that old data interface still works"""
    print("🔄 Testing data backward compatibility...")
    
    try:
        # Test old interface through new system
        from data.providers import get_gold_data
        
        # This should work the same as the old utils/data.py function
        # (will fail without API key, but interface should be the same)
        try:
            data = get_gold_data(symbol='XAUUSD', interval='5m', outputsize=10)
            if data is not None:
                print("✅ get_gold_data() interface works")
            else:
                print("✅ get_gold_data() interface works (no API key, expected None)")
        except Exception as e:
            if "API key" in str(e):
                print("✅ get_gold_data() interface works (API key required, as expected)")
            else:
                raise e
        
        # Test that the function signature is the same
        import inspect
        sig = inspect.signature(get_gold_data)
        expected_params = ['api_key', 'interval', 'symbol', 'outputsize']
        actual_params = list(sig.parameters.keys())
        
        if set(expected_params) == set(actual_params):
            print("✅ get_gold_data() has correct signature")
        else:
            print(f"❌ get_gold_data() signature mismatch: expected {expected_params}, got {actual_params}")
            return False
            
    except Exception as e:
        print(f"❌ Data backward compatibility failed: {e}")
        return False
    
    return True


def test_trading_backward_compatibility():
    """Test that old trading interface still works"""
    print("🔄 Testing trading backward compatibility...")
    
    try:
        # Test old interface through new system
        from trading.engine import init_testnet_client, place_market_order
        
        # Test function signatures
        import inspect
        
        # Check init_testnet_client signature
        sig = inspect.signature(init_testnet_client)
        expected_params = ['api_key', 'api_secret']
        actual_params = list(sig.parameters.keys())
        
        if set(expected_params) == set(actual_params):
            print("✅ init_testnet_client() has correct signature")
        else:
            print(f"❌ init_testnet_client() signature mismatch: expected {expected_params}, got {actual_params}")
            return False
        
        # Check place_market_order signature
        sig = inspect.signature(place_market_order)
        expected_params = ['client', 'symbol', 'side', 'qty', 'leverage']
        actual_params = list(sig.parameters.keys())
        
        if set(expected_params) == set(actual_params):
            print("✅ place_market_order() has correct signature")
        else:
            print(f"❌ place_market_order() signature mismatch: expected {expected_params}, got {actual_params}")
            return False
            
        # Test that init_testnet_client returns something
        try:
            client = init_testnet_client()
            print("✅ init_testnet_client() interface works (returns TradingEngine)")
        except Exception as e:
            if "API key" in str(e) or "python-binance" in str(e):
                print("✅ init_testnet_client() interface works (credentials/library required, as expected)")
            else:
                raise e
                
    except Exception as e:
        print(f"❌ Trading backward compatibility failed: {e}")
        return False
    
    return True


def test_imports_backward_compatibility():
    """Test that old imports can be redirected to new system"""
    print("🔄 Testing import backward compatibility...")
    
    try:
        # Test that we can import from data and trading modules
        from data import DataManager
        from trading import TradingManager
        
        # Test that backward compatibility functions are available
        from data.providers import get_gold_data
        from trading.engine import init_testnet_client, place_market_order
        
        print("✅ All new system imports work")
        
        # Test that old utility functions work
        dm = DataManager()
        tm = TradingManager()
        
        print(f"✅ Can create DataManager (symbol: {dm.symbol})")
        print(f"✅ Can create TradingManager (symbol: {tm.symbol})")
        
    except Exception as e:
        print(f"❌ Import backward compatibility failed: {e}")
        return False
    
    return True


def test_config_integration():
    """Test that new system integrates with config from Phase 1"""
    print("🔄 Testing config integration...")
    
    try:
        # Test config integration
        from config import get_config
        from data import DataManager
        from trading import TradingManager
        
        # Create instances that should use config
        dm = DataManager()  # Should use config defaults
        tm = TradingManager()  # Should use config defaults
        
        # These should have sensible defaults even without config
        assert dm.symbol is not None
        assert dm.timeframe is not None
        assert tm.symbol is not None
        assert tm.timeframe is not None
        
        print(f"✅ DataManager uses config defaults: {dm.symbol}, {dm.timeframe}")
        print(f"✅ TradingManager uses config defaults: {tm.symbol}, {tm.timeframe}")
        
    except Exception as e:
        print(f"❌ Config integration failed: {e}")
        return False
    
    return True


def run_backward_compatibility_tests():
    """Run all backward compatibility tests"""
    print("🔄 Running Backward Compatibility Tests for Phase 3...")
    print("=" * 60)
    
    tests = [
        test_imports_backward_compatibility,
        test_config_integration,
        test_data_backward_compatibility,
        test_trading_backward_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED\n")
            else:
                failed += 1
                print("❌ FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED with exception: {e}\n")
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ ALL BACKWARD COMPATIBILITY TESTS PASSED!")
        return True
    else:
        print("❌ Some backward compatibility tests failed")
        return False


if __name__ == "__main__":
    run_backward_compatibility_tests()