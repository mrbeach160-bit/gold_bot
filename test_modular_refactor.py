#!/usr/bin/env python3
"""
Smoke test for modular refactoring.

This script tests that all modules can be imported and basic functions work
without breaking the existing functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime

def test_module_imports():
    """Test that all modules can be imported without errors."""
    print("Testing module imports...")
    
    try:
        from streamlit_app.modules.config import (
            initialize_feature_flags, is_feature_enabled, 
            get_model_directory, LABEL_MAP
        )
        print("✅ config module imported successfully")
        
        from streamlit_app.modules.websocket_manager import EnhancedWebSocketManager
        print("✅ websocket_manager module imported successfully")
        
        from streamlit_app.modules.smart_entry import calculate_smart_entry_price
        print("✅ smart_entry module imported successfully")
        
        from streamlit_app.modules.trading_utils import (
            validate_trading_inputs, get_pip_value, format_currency
        )
        print("✅ trading_utils module imported successfully")
        
        from streamlit_app.modules.data_utils import load_and_process_data_enhanced
        print("✅ data_utils module imported successfully")
        
        from streamlit_app.modules.models import sanitize_filename, predict_with_models
        print("✅ models module imported successfully")
        
        from streamlit_app.modules.backtest import run_backtest, format_backtest_results
        print("✅ backtest module imported successfully")
        
        from streamlit_app.modules.ui import (
            format_price, format_percentage, display_metrics_grid
        )
        print("✅ ui module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Module import failed: {e}")
        return False


def test_config_functionality():
    """Test config module functionality."""
    print("\nTesting config functionality...")
    
    try:
        from streamlit_app.modules.config import (
            initialize_feature_flags, is_feature_enabled, 
            get_model_directory, LABEL_MAP, TIMEFRAME_MAPPING
        )
        
        # Initialize feature flags
        flags = initialize_feature_flags()
        assert isinstance(flags, dict), "Feature flags should be a dictionary"
        print("✅ Feature flags initialized")
        
        # Test feature flag checking
        websocket_available = is_feature_enabled('WEBSOCKET_AVAILABLE')
        assert isinstance(websocket_available, bool), "Feature flag should be boolean"
        print("✅ Feature flag checking works")
        
        # Test constants
        assert isinstance(LABEL_MAP, dict), "LABEL_MAP should be dictionary"
        assert isinstance(TIMEFRAME_MAPPING, dict), "TIMEFRAME_MAPPING should be dictionary"
        print("✅ Constants are properly defined")
        
        # Test model directory
        model_dir = get_model_directory()
        assert isinstance(model_dir, str), "Model directory should be string"
        print("✅ Model directory path works")
        
        return True
        
    except Exception as e:
        print(f"❌ Config functionality test failed: {e}")
        return False


def test_utility_functions():
    """Test utility functions from various modules."""
    print("\nTesting utility functions...")
    
    try:
        from streamlit_app.modules.ui import format_price, format_percentage
        from streamlit_app.modules.trading_utils import get_pip_value, validate_trading_inputs
        from streamlit_app.modules.models import sanitize_filename
        
        # Test price formatting
        price_str = format_price("XAUUSD", 2000.50)
        assert isinstance(price_str, str), "Price should be formatted as string"
        assert "$" in price_str, "Gold price should include $ symbol"
        print("✅ Price formatting works")
        
        # Test percentage formatting
        pct_str = format_percentage(75.5)
        assert isinstance(pct_str, str), "Percentage should be formatted as string"
        assert "%" in pct_str, "Percentage should include % symbol"
        print("✅ Percentage formatting works")
        
        # Test pip value calculation
        pip_val = get_pip_value("XAUUSD", 2000.0)
        assert isinstance(pip_val, float), "Pip value should be float"
        assert pip_val > 0, "Pip value should be positive"
        print("✅ Pip value calculation works")
        
        # Test filename sanitization
        clean_name = sanitize_filename("XAU/USD")
        assert "/" not in clean_name, "Sanitized filename should not contain /"
        print("✅ Filename sanitization works")
        
        # Test trading input validation
        validation = validate_trading_inputs("XAUUSD", 1000, 2.0, 50, 100)
        assert isinstance(validation, dict), "Validation should return dictionary"
        assert 'valid' in validation, "Validation should have 'valid' key"
        print("✅ Trading input validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
        return False


def test_smart_entry_with_mock_data():
    """Test smart entry calculation with mock data."""
    print("\nTesting smart entry with mock data...")
    
    try:
        from streamlit_app.modules.smart_entry import calculate_smart_entry_price
        
        # Create mock data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        mock_data = pd.DataFrame({
            'open': np.random.uniform(1900, 2100, 100),
            'high': np.random.uniform(1950, 2150, 100),
            'low': np.random.uniform(1850, 2050, 100),
            'close': np.random.uniform(1900, 2100, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        # Ensure OHLC logic
        for i in range(len(mock_data)):
            mock_data.iloc[i, mock_data.columns.get_loc('high')] = max(
                mock_data.iloc[i, mock_data.columns.get_loc('open')],
                mock_data.iloc[i, mock_data.columns.get_loc('high')],
                mock_data.iloc[i, mock_data.columns.get_loc('low')],
                mock_data.iloc[i, mock_data.columns.get_loc('close')]
            )
            mock_data.iloc[i, mock_data.columns.get_loc('low')] = min(
                mock_data.iloc[i, mock_data.columns.get_loc('open')],
                mock_data.iloc[i, mock_data.columns.get_loc('high')],
                mock_data.iloc[i, mock_data.columns.get_loc('low')],
                mock_data.iloc[i, mock_data.columns.get_loc('close')]
            )
        
        # Test smart entry calculation
        result = calculate_smart_entry_price(
            signal=1,  # BUY signal
            recent_data=mock_data,
            predicted_price=2000.0,
            confidence=0.8,
            symbol="XAUUSD"
        )
        
        assert isinstance(result, dict), "Smart entry should return dictionary"
        assert 'smart_entry_price' in result, "Result should have smart_entry_price"
        assert 'risk_level' in result, "Result should have risk_level"
        assert 'fill_probability' in result, "Result should have fill_probability"
        
        entry_price = result['smart_entry_price']
        assert isinstance(entry_price, (int, float)), "Entry price should be numeric"
        assert entry_price > 0, "Entry price should be positive"
        
        print("✅ Smart entry calculation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Smart entry test failed: {e}")
        return False


def test_websocket_manager():
    """Test WebSocket manager initialization."""
    print("\nTesting WebSocket manager...")
    
    try:
        from streamlit_app.modules.websocket_manager import EnhancedWebSocketManager
        
        # Initialize manager
        manager = EnhancedWebSocketManager()
        assert hasattr(manager, 'latest_price'), "Manager should have latest_price attribute"
        assert hasattr(manager, 'connection_status'), "Manager should have connection_status attribute"
        
        # Test methods exist
        assert callable(manager.get_latest_price), "get_latest_price should be callable"
        assert callable(manager.get_connection_status), "get_connection_status should be callable"
        
        # Test initial state
        price = manager.get_latest_price("XAUUSD")
        assert price is None, "Initial price should be None"
        
        status = manager.get_connection_status("XAUUSD")
        assert isinstance(status, str), "Status should be string"
        
        print("✅ WebSocket manager initialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket manager test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("🧪 Running modular refactoring smoke tests...\n")
    
    tests = [
        test_module_imports,
        test_config_functionality,
        test_utility_functions,
        test_smart_entry_with_mock_data,
        test_websocket_manager
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Modular refactoring is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)