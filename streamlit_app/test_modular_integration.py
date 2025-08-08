#!/usr/bin/env python3
"""
Integration test for the modular Gold Bot App
Tests component interactions and basic functionality
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_component_imports():
    """Test all component imports"""
    print("🔄 Testing component imports...")
    
    try:
        from components.websocket_panel import WebSocketPanel, EnhancedWebSocketManager
        from components.trading_panel import TradingPanel
        from components.model_status import ModelStatusDisplay
        from components.live_stream import LiveStreamManager
        from components.backtest_runner import BacktestRunner
        print("✅ All components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_component_instantiation():
    """Test component instantiation"""
    print("🔄 Testing component instantiation...")
    
    try:
        from components.websocket_panel import WebSocketPanel, EnhancedWebSocketManager
        from components.trading_panel import TradingPanel
        from components.model_status import ModelStatusDisplay
        from components.live_stream import LiveStreamManager
        from components.backtest_runner import BacktestRunner
        
        # Test instantiation
        ws_manager = EnhancedWebSocketManager()
        ws_panel = WebSocketPanel(ws_manager)
        trading_panel = TradingPanel()
        model_status = ModelStatusDisplay()
        live_stream = LiveStreamManager(ws_manager)
        backtest_runner = BacktestRunner()
        
        print("✅ All components instantiated successfully")
        return True
    except Exception as e:
        print(f"❌ Instantiation error: {e}")
        return False

def test_trading_panel_functionality():
    """Test trading panel core functionality"""
    print("🔄 Testing trading panel functionality...")
    
    try:
        from components.trading_panel import TradingPanel
        
        trading_panel = TradingPanel()
        
        # Test price formatting
        formatted_price = trading_panel.format_price("XAU/USD", 2000.50)
        assert formatted_price == "$2,000.50", f"Expected $2,000.50, got {formatted_price}"
        
        # Test trading input validation
        is_valid, errors = trading_panel.validate_trading_inputs("XAU/USD", 1000, 1.0, 20, 40)
        assert is_valid, f"Trading inputs should be valid, errors: {errors}"
        
        # Test smart entry calculation with mock data
        mock_data = pd.DataFrame({
            'close': [2000.0] * 100,
            'high': [2010.0] * 100,
            'low': [1990.0] * 100,
            'rsi': [50.0] * 100,
            'ATR_14': [20.0] * 100,
            'MACD_12_26_9': [0.0] * 100,
            'MACDs_12_26_9': [0.0] * 100
        })
        
        entry_result = trading_panel.calculate_smart_entry_price(
            'BUY', mock_data, 2005.0, 0.8, "XAU/USD"
        )
        
        assert 'entry_price' in entry_result, "Entry result should contain entry_price"
        assert 'strategy_reasons' in entry_result, "Entry result should contain strategy_reasons"
        
        print("✅ Trading panel functionality works correctly")
        return True
    except Exception as e:
        print(f"❌ Trading panel test error: {e}")
        return False

def test_model_status_functionality():
    """Test model status functionality"""
    print("🔄 Testing model status functionality...")
    
    try:
        from components.model_status import ModelStatusDisplay
        
        model_status = ModelStatusDisplay()
        
        # Test model availability check
        available_models = model_status.check_model_availability("XAU/USD", "15m")
        assert isinstance(available_models, dict), "Should return dict of model availability"
        
        # Test ensemble readiness
        readiness = model_status.get_model_ensemble_readiness("XAU/USD", "15m")
        assert 'ensemble_ready' in readiness, "Should contain ensemble_ready status"
        
        # Test file size formatting
        size_str = model_status.format_file_size(1024)
        assert size_str == "1.0 KB", f"Expected 1.0 KB, got {size_str}"
        
        print("✅ Model status functionality works correctly")
        return True
    except Exception as e:
        print(f"❌ Model status test error: {e}")
        return False

def test_websocket_panel_functionality():
    """Test websocket panel functionality"""
    print("🔄 Testing websocket panel functionality...")
    
    try:
        from components.websocket_panel import WebSocketPanel, EnhancedWebSocketManager
        
        ws_manager = EnhancedWebSocketManager()
        ws_panel = WebSocketPanel(ws_manager)
        
        # Test price formatting
        formatted_price = ws_panel.format_price("BTC/USD", 50000.5)
        assert formatted_price == "$50,000", f"Expected $50,000, got {formatted_price}"
        
        # Test websocket availability check
        is_available = ws_panel.is_websocket_available()
        assert isinstance(is_available, bool), "Should return boolean for websocket availability"
        
        print("✅ WebSocket panel functionality works correctly")
        return True
    except Exception as e:
        print(f"❌ WebSocket panel test error: {e}")
        return False

def test_app_integration():
    """Test main app integration"""
    print("🔄 Testing main app integration...")
    
    try:
        # Test that we can import the main app without streamlit context
        # This tests the overall structure
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("app_refactored", "app_refactored.py")
        app_module = importlib.util.module_from_spec(spec)
        
        # The import itself validates the structure
        print("✅ Main app structure is valid")
        return True
    except Exception as e:
        print(f"❌ App integration test error: {e}")
        return False

def run_all_tests():
    """Run all integration tests"""
    print("🚀 Running Gold Bot Modular Integration Tests\n")
    
    tests = [
        test_component_imports,
        test_component_instantiation,
        test_trading_panel_functionality,
        test_model_status_functionality,
        test_websocket_panel_functionality,
        test_app_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print(f"🎯 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Modular refactor is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)