#!/usr/bin/env python3
"""
Test model loading and backward compatibility.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_model_loading():
    """Test loading existing models."""
    print("Testing Model Loading and Backward Compatibility...")
    
    try:
        from models import ModelManager
        
        # Create ModelManager
        manager = ModelManager("XAU/USD", "5m")
        print(f"✅ ModelManager created")
        
        # Check what models are available on disk
        status = manager.get_model_status()
        print(f"📊 Available models on disk: {status['summary']['available_models']}")
        
        # Try to load all models
        loaded = manager.load_all_models()
        print(f"📂 Models loaded successfully: {loaded}")
        
        if loaded:
            # Test the loaded models
            print("\n🔮 Testing loaded model predictions...")
            
            # Create simple test data
            test_data = pd.DataFrame({
                'open': [2000.0, 2005.0, 2010.0],
                'high': [2010.0, 2015.0, 2020.0],
                'low': [1995.0, 2000.0, 2005.0],
                'close': [2005.0, 2010.0, 2015.0],
                'volume': [5000, 5500, 6000],
                'rsi': [50.0, 55.0, 60.0]
            })
            
            # Get predictions from all loaded models
            predictions = manager.get_predictions(test_data)
            
            for model_name, pred in predictions.items():
                if 'error' not in pred:
                    print(f"   ✅ {model_name}: {pred['direction']} (confidence: {pred['confidence']:.3f})")
                else:
                    print(f"   ⚠️  {model_name}: {pred['error']}")
            
            # Test ensemble prediction
            ensemble_pred = manager.get_ensemble_prediction(test_data)
            print(f"   🎯 Ensemble: {ensemble_pred['direction']} (confidence: {ensemble_pred['confidence']:.3f})")
            
            return True
        else:
            print("⚠️  No models could be loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_model_loading():
    """Test loading individual models directly."""
    print("\n🔍 Testing individual model loading...")
    
    try:
        from models import create_model
        
        # Test XGBoost model specifically
        xgb_model = create_model("xgboost", "XAU/USD", "5m")
        print(f"✅ XGBoost model created")
        
        # Try to load it
        loaded = xgb_model.load()
        print(f"📂 XGBoost model loaded: {loaded}")
        
        if loaded:
            # Get model info
            info = xgb_model.get_model_info()
            print(f"📋 Model info: {info['name']} - {info['trained']}")
            
            # Test prediction with simple data
            test_data = pd.DataFrame({
                'open': [2000.0],
                'high': [2010.0],
                'low': [1995.0],
                'close': [2005.0],
                'volume': [5000],
                'rsi': [50.0]
            })
            
            pred = xgb_model.predict(test_data)
            print(f"🔮 Prediction: {pred['direction']} (confidence: {pred['confidence']:.3f})")
            
            return True
        else:
            print("⚠️  Could not load XGBoost model")
            return False
            
    except Exception as e:
        print(f"❌ Error testing individual model loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_model_loading()
    success2 = test_individual_model_loading()
    
    if success1 and success2:
        print("\n🎉 All model loading tests passed!")
    else:
        print("\n⚠️  Some model loading tests failed")
    
    sys.exit(0 if (success1 and success2) else 1)