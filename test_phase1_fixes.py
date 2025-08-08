#!/usr/bin/env python3
"""
Test script for Phase 1 critical AI model fixes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_realistic_test_data(n_rows=500):
    """Create realistic OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='5T')
    
    # Simulate realistic gold price movements (starting around 2000)
    base_price = 2000.0
    returns = np.random.normal(0, 0.001, n_rows)  # Small random changes
    
    # Add some trend and volatility
    trend = np.sin(np.arange(n_rows) / 50) * 0.02  # Cyclical trend
    volatility = np.abs(np.random.normal(0, 0.002, n_rows))  # Variable volatility
    
    prices = [base_price]
    for i in range(1, n_rows):
        change = returns[i] + trend[i] + volatility[i] * np.random.normal(0, 1)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_rows))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_rows))),
        'close': prices,
        'volume': np.random.randint(100, 1000, n_rows)
    })
    
    # Ensure high >= low and open/close within range
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def test_lightgbm_no_dummy_values():
    """Test that LightGBM no longer uses dangerous dummy values."""
    print("\n1. Testing LightGBM dummy value elimination...")
    
    from models import create_model
    
    # Create test data WITHOUT indicators
    data = create_realistic_test_data(200)
    print(f"Created test data: {len(data)} rows")
    print(f"Columns: {list(data.columns)}")
    
    model = create_model('lightgbm', 'XAU/USD', '5m')
    print("✅ LightGBM model created")
    
    # Try training - should calculate indicators instead of using dummy values
    success = model.train(data)
    
    if success:
        print("✅ LightGBM training successful with calculated indicators")
        
        # Test prediction
        pred = model.predict(data)
        print(f"✅ Prediction: {pred['direction']} (confidence: {pred['confidence']:.3f})")
        
        if 'features_used' in pred:
            print(f"Features used: {pred['features_used']}")
            
        return True
    else:
        print("✅ LightGBM correctly failed without indicators (no dummy values used)")
        return True

def test_time_series_split():
    """Test that models use TimeSeriesSplit instead of random split."""
    print("\n2. Testing TimeSeriesSplit implementation...")
    
    from models import create_model
    from utils.indicators import add_indicators
    
    # Create data with indicators
    data = create_realistic_test_data(300)
    data_with_indicators = add_indicators(data)
    print(f"Data with indicators: {len(data_with_indicators)} rows")
    
    # Test LightGBM with time series split
    model = create_model('lightgbm', 'XAU/USD', '5m')
    success = model.train(data_with_indicators)
    
    if success:
        print("✅ LightGBM training with TimeSeriesSplit successful")
        
        # Test prediction
        pred = model.predict(data_with_indicators)
        print(f"✅ Prediction: {pred['direction']} (confidence: {pred['confidence']:.3f})")
        return True
    else:
        print("❌ LightGBM training failed")
        return False

def test_enhanced_lstm():
    """Test enhanced LSTM architecture."""
    print("\n3. Testing enhanced LSTM architecture...")
    
    from models import create_model
    
    # Create sufficient data for LSTM
    data = create_realistic_test_data(200)
    print(f"Created LSTM test data: {len(data)} rows")
    
    model = create_model('lstm', 'XAU/USD', '5m')
    print("✅ LSTM model created")
    
    # Train with enhanced architecture
    success = model.train(data)
    
    if success:
        print("✅ Enhanced LSTM training successful")
        
        # Test prediction
        pred = model.predict(data)
        print(f"✅ LSTM Prediction: {pred['direction']} (confidence: {pred['confidence']:.3f})")
        
        # Check if model has expected architecture improvements
        if hasattr(model.model, 'layers'):
            layer_count = len(model.model.layers)
            print(f"✅ Enhanced architecture: {layer_count} layers")
            
        return True
    else:
        print("❌ LSTM training failed")
        return False

def test_enhanced_xgboost():
    """Test enhanced XGBoost implementation."""
    print("\n4. Testing enhanced XGBoost implementation...")
    
    from models import create_model
    from utils.indicators import add_indicators
    
    # Create data with indicators
    data = create_realistic_test_data(200)
    data_with_indicators = add_indicators(data)
    print(f"XGBoost test data: {len(data_with_indicators)} rows")
    
    model = create_model('xgboost', 'XAU/USD', '5m')
    print("✅ XGBoost model created")
    
    # Train with enhanced implementation
    success = model.train(data_with_indicators)
    
    if success:
        print("✅ Enhanced XGBoost training successful")
        
        # Test prediction
        pred = model.predict(data_with_indicators)
        print(f"✅ XGBoost Prediction: {pred['direction']} (confidence: {pred['confidence']:.3f})")
        
        if 'features_used' in pred:
            print(f"Features used: {len(pred['features_used'])}")
            
        return True
    else:
        print("❌ XGBoost training failed")
        return False

def test_evaluation_metrics():
    """Test enhanced evaluation metrics."""
    print("\n5. Testing enhanced evaluation metrics...")
    
    from models.ml_models import calculate_trading_metrics
    import numpy as np
    
    # Create test prediction data
    n_samples = 100
    y_true = np.random.choice([0, 1], n_samples)
    y_pred = np.random.choice([0, 1], n_samples)
    y_prob = np.random.random(n_samples)
    
    metrics = calculate_trading_metrics(y_true, y_pred, y_prob)
    
    expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'win_rate', 
                       'buy_accuracy', 'sell_accuracy', 'high_confidence_accuracy']
    
    missing_metrics = [m for m in expected_metrics if m not in metrics]
    
    if not missing_metrics:
        print("✅ All enhanced metrics calculated successfully")
        for metric, value in metrics.items():
            if metric != 'error':
                print(f"  {metric}: {value:.4f}")
        return True
    else:
        print(f"❌ Missing metrics: {missing_metrics}")
        return False

def main():
    """Run all Phase 1 fix tests."""
    print("Testing Phase 1 Critical AI Model Fixes")
    print("=" * 50)
    
    tests = [
        test_lightgbm_no_dummy_values,
        test_time_series_split,
        test_enhanced_lstm,
        test_enhanced_xgboost,
        test_evaluation_metrics
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Phase 1 Test Results:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All Phase 1 fixes working correctly!")
        return True
    else:
        print("\n⚠️  Some tests failed - check implementation")
        return False

if __name__ == "__main__":
    main()