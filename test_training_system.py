#!/usr/bin/env python3
"""
Test the new training system with synthetic data.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def create_test_data(n_points=500):
    """Create synthetic OHLCV data for testing."""
    print(f"Creating synthetic test data with {n_points} points...")
    
    # Create dates
    start_date = datetime.now() - timedelta(days=n_points//24)
    dates = pd.date_range(start_date, periods=n_points, freq='5min')
    
    # Generate realistic OHLCV data with trends
    np.random.seed(42)  # For reproducible results
    
    base_price = 2000.0
    price_trend = np.cumsum(np.random.randn(n_points) * 0.001)  # Random walk
    prices = base_price + price_trend * 50
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        volatility = np.random.uniform(0.001, 0.01)
        
        open_price = price + np.random.normal(0, volatility * price)
        high_price = max(open_price, price) + np.random.exponential(volatility * price)
        low_price = min(open_price, price) - np.random.exponential(volatility * price)
        close_price = price + np.random.normal(0, volatility * price * 0.5)
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    # Add some basic technical indicators for testing
    df['rsi'] = np.random.uniform(30, 70, n_points)  # Simplified RSI
    df['MACDh_12_26_9'] = np.random.uniform(-10, 10, n_points)  # Simplified MACD
    df['ADX_14'] = np.random.uniform(20, 40, n_points)  # Simplified ADX
    
    print(f"✅ Test data created: {len(df)} rows")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    return df

def test_new_training_system():
    """Test the new training system."""
    print("Testing New Model Training System...")
    
    # Create test data
    test_data = create_test_data(500)
    
    try:
        # Test with new system
        from train_models_new import train_models_new_system
        
        print("\n🧠 Testing new unified training system...")
        success = train_models_new_system("XAU/USD", "5m", test_data, use_config=False)
        
        if success:
            print("✅ New training system test passed!")
            
            # Test ModelManager directly
            print("\n🔍 Testing ModelManager functionality...")
            from models import ModelManager
            
            manager = ModelManager("XAU/USD", "5m")
            
            # Try to load models
            loaded = manager.load_all_models()
            print(f"📂 Models loaded: {loaded}")
            
            # Get model status
            status = manager.get_model_status()
            print(f"📊 Model status: {status['summary']}")
            
            # Test predictions
            if status['summary']['trained_models'] > 0:
                print("\n🔮 Testing model predictions...")
                predictions = manager.get_predictions(test_data.tail(10))
                for model_name, pred in predictions.items():
                    if 'error' not in pred:
                        print(f"   {model_name}: {pred['direction']} (confidence: {pred['confidence']:.3f})")
                    else:
                        print(f"   {model_name}: {pred['error']}")
                
                # Test ensemble prediction
                ensemble_pred = manager.get_ensemble_prediction(test_data.tail(10))
                print(f"   Ensemble: {ensemble_pred['direction']} (confidence: {ensemble_pred['confidence']:.3f})")
            
            return True
        else:
            print("❌ New training system test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing new training system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_new_training_system()
    sys.exit(0 if success else 1)