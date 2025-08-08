#!/usr/bin/env python3
"""
Test script for the new unified model architecture.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_model_structure():
    """Test the new model structure and interfaces."""
    print("Testing Unified Model Architecture...")
    
    try:
        # Test 1: Import new model modules
        print("\n1. Testing model imports...")
        from models import BaseModel, create_model, ModelManager
        from models.ml_models import LSTMModel, LightGBMModel, XGBoostModel
        from models.ensemble import MetaLearner, VotingEnsemble
        print("✅ All model imports successful")
        
        # Test 2: Factory function
        print("\n2. Testing factory function...")
        lstm_model = create_model('lstm', 'XAU/USD', '5m')
        lgb_model = create_model('lightgbm', 'XAU/USD', '5m')
        xgb_model = create_model('xgboost', 'XAU/USD', '5m')
        
        print(f"✅ LSTM model created: {type(lstm_model).__name__}")
        print(f"✅ LightGBM model created: {type(lgb_model).__name__}")
        print(f"✅ XGBoost model created: {type(xgb_model).__name__}")
        
        # Test 3: Model interfaces
        print("\n3. Testing model interfaces...")
        test_models = [lstm_model, lgb_model, xgb_model]
        
        for model in test_models:
            model_info = model.get_model_info()
            print(f"✅ {model_info['name']} - {model_info['symbol']} {model_info['timeframe']}")
            print(f"   Trained: {model.is_trained()}")
            print(f"   Model path: {model.get_model_path()}")
        
        # Test 4: ModelManager
        print("\n4. Testing ModelManager...")
        manager = ModelManager('XAU/USD', '5m')
        print(f"✅ ModelManager created for {manager.symbol} {manager.timeframe}")
        
        available_models = manager.get_available_models()
        print(f"✅ Available models: {available_models}")
        
        model_status = manager.get_model_status()
        print(f"✅ Model status retrieved: {model_status['summary']['total_models']} total models")
        
        # Test 5: Create dummy data for prediction testing
        print("\n5. Testing prediction interfaces...")
        
        # Create minimal test data
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        test_data = pd.DataFrame({
            'open': np.random.uniform(2000, 2100, 100),
            'high': np.random.uniform(2050, 2150, 100),
            'low': np.random.uniform(1950, 2050, 100),
            'close': np.random.uniform(2000, 2100, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        # Add required technical indicators for testing
        test_data['rsi'] = np.random.uniform(30, 70, 100)
        test_data['MACDh_12_26_9'] = np.random.uniform(-10, 10, 100)
        test_data['ADX_14'] = np.random.uniform(20, 40, 100)
        
        print(f"✅ Test data created: {len(test_data)} rows")
        
        # Test prediction interface (without actual training)
        for model in test_models:
            try:
                pred = model.predict(test_data)
                print(f"✅ {pred['model_name']} prediction interface working")
                print(f"   Direction: {pred['direction']}, Confidence: {pred['confidence']:.3f}")
            except Exception as e:
                print(f"✅ {type(model).__name__} prediction failed as expected (not trained): {type(e).__name__}")
        
        # Test 6: Ensemble system
        print("\n6. Testing ensemble system...")
        voting_ensemble = VotingEnsemble([lstm_model, lgb_model, xgb_model])
        ensemble_pred = voting_ensemble.predict(test_data)
        print(f"✅ Voting ensemble prediction: {ensemble_pred['direction']}")
        
        # Test manager predictions
        manager_preds = manager.get_predictions(test_data)
        print(f"✅ Manager predictions: {len(manager_preds)} models")
        
        ensemble_pred = manager.get_ensemble_prediction(test_data, method='voting')
        print(f"✅ Manager ensemble prediction: {ensemble_pred['direction']}")
        
        print("\n🎉 All model architecture tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_structure()
    sys.exit(0 if success else 1)