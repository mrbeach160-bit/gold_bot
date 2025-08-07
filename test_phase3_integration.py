# test_phase3_integration.py
"""
Integration test for Phase 3 Data & Trading Simplification
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import unittest

# Test data and trading systems
from data import DataManager
from data.providers import BaseProvider, YFinanceProvider
from data.processors import TechnicalIndicators, DataCleaning
from trading import TradingManager
from trading.strategies import TechnicalStrategy, MLStrategy
from trading.risk import RiskManager, PositionSizer


class TestDataSystem(unittest.TestCase):
    """Test data management system"""
    
    def setUp(self):
        self.data_manager = DataManager('XAUUSD', '5m')
    
    def test_data_manager_creation(self):
        """Test DataManager initialization"""
        self.assertEqual(self.data_manager.symbol, 'XAUUSD')
        self.assertEqual(self.data_manager.timeframe, '5m')
        self.assertIsInstance(self.data_manager.providers, dict)
        self.assertGreater(len(self.data_manager.providers), 0)
    
    def test_provider_standardization(self):
        """Test data provider standardization"""
        # Create sample OHLC data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        sample_data = pd.DataFrame({
            'datetime': dates,
            'open': np.random.uniform(2000, 2100, 100),
            'high': np.random.uniform(2100, 2200, 100),
            'low': np.random.uniform(1900, 2000, 100),
            'close': np.random.uniform(2000, 2100, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        
        # Test standardization using YFinanceProvider (concrete implementation)
        provider = YFinanceProvider()
        standardized = provider.standardize_dataframe(sample_data)
        
        self.assertIsInstance(standardized.index, pd.DatetimeIndex)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            self.assertIn(col, standardized.columns)
    
    def test_technical_indicators(self):
        """Test technical indicator calculations"""
        # Create sample price data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='5min')
        data = pd.DataFrame({
            'open': np.random.uniform(2000, 2100, 50),
            'high': np.random.uniform(2100, 2200, 50),
            'low': np.random.uniform(1900, 2000, 50),
            'close': np.random.uniform(2000, 2100, 50),
            'volume': np.random.uniform(1000, 5000, 50)
        }, index=dates)
        
        # Add indicators
        with_indicators = TechnicalIndicators.add_all_indicators(data)
        
        # Check that indicators were added
        expected_indicators = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'ema_10']
        for indicator in expected_indicators:
            self.assertIn(indicator, with_indicators.columns)
    
    def test_data_validation(self):
        """Test data validation"""
        # Valid data with proper length (need at least 50 data points for manager validation)
        dates = pd.date_range(start='2024-01-01', periods=60, freq='5min')
        valid_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 60),
            'high': np.random.uniform(110, 120, 60),
            'low': np.random.uniform(90, 100, 60),
            'close': np.random.uniform(100, 110, 60),
            'volume': np.random.uniform(1000, 5000, 60)
        }, index=dates)
        
        self.assertTrue(DataCleaning.validate_ohlc_data(valid_data))
        self.assertTrue(self.data_manager.validate_data(valid_data))
        
        # Invalid data (high < low)
        invalid_data = pd.DataFrame({
            'open': [100, 102, 101],
            'high': [95, 96, 94],  # High less than low
            'low': [99, 101, 100],
            'close': [102, 103, 101],
            'volume': [1000, 1200, 1100]
        })
        
        self.assertFalse(DataCleaning.validate_ohlc_data(invalid_data))


class TestTradingSystem(unittest.TestCase):
    """Test trading system"""
    
    def setUp(self):
        self.trading_manager = TradingManager('XAUUSD', '5m')
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        data = pd.DataFrame({
            'open': np.random.uniform(2000, 2100, 100),
            'high': np.random.uniform(2100, 2200, 100),
            'low': np.random.uniform(1900, 2000, 100),
            'close': np.random.uniform(2000, 2100, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        # Add technical indicators
        return TechnicalIndicators.add_all_indicators(data)
    
    def test_trading_manager_creation(self):
        """Test TradingManager initialization"""
        self.assertEqual(self.trading_manager.symbol, 'XAUUSD')
        self.assertEqual(self.trading_manager.timeframe, '5m')
        self.assertIsNotNone(self.trading_manager.strategy)
        self.assertIsNotNone(self.trading_manager.risk_manager)
    
    def test_strategy_signal_generation(self):
        """Test strategy signal generation"""
        strategy = TechnicalStrategy()
        signal = strategy.generate_signal(self.sample_data)
        
        # Check signal format
        required_keys = ['action', 'confidence', 'strategy_name', 'timestamp']
        for key in required_keys:
            self.assertIn(key, signal)
        
        # Check signal values
        self.assertIn(signal['action'], ['BUY', 'SELL', 'HOLD'])
        self.assertGreaterEqual(signal['confidence'], 0.0)
        self.assertLessEqual(signal['confidence'], 1.0)
    
    def test_risk_management(self):
        """Test risk management system"""
        risk_manager = RiskManager()
        
        # Test signal with proper format
        signal = {
            'action': 'BUY',
            'confidence': 0.8,
            'position_size': 1000,
            'current_price': 2050
        }
        
        # Mock account info
        account_info = {
            'total_wallet_balance': 10000,
            'positions': []
        }
        
        # Check risk limits
        risk_result = risk_manager.check_risk_limits(signal, account_info)
        
        self.assertIn('approved', risk_result)
        self.assertIn('reasons', risk_result)
        self.assertIn('adjusted_signal', risk_result)
    
    def test_position_sizing(self):
        """Test position sizing calculations"""
        sizer = PositionSizer(10000)
        
        # Test fixed percentage
        fixed_size = sizer.fixed_percentage(0.02)
        self.assertEqual(fixed_size, 0.02)
        
        # Test dynamic sizing
        dynamic_size = sizer.dynamic_sizing(0.8, 0.1, 0.02)
        self.assertGreater(dynamic_size, 0)
        self.assertLess(dynamic_size, 0.1)
    
    def test_strategy_execution(self):
        """Test strategy execution without actual trading"""
        # Disable live trading for test
        self.trading_manager.is_trading_enabled = False
        
        # Mock data manager
        class MockDataManager:
            def get_live_data(self, bars=100):
                return self.sample_data
            
            def preprocess_data(self, data):
                return data
        
        self.trading_manager.data_manager = MockDataManager()
        self.trading_manager.data_manager.sample_data = self.sample_data
        self.trading_manager.is_initialized = True
        
        # Execute strategy
        result = self.trading_manager.execute_strategy()
        
        self.assertTrue(result['success'])
        self.assertIn('signal', result)
        self.assertIn('action_taken', result)


def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Phase 3 Integration Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestTradingSystem))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print("\n✅ All Phase 3 integration tests passed!")
        return True
    else:
        print(f"\n❌ Some tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        return False


if __name__ == "__main__":
    run_integration_tests()