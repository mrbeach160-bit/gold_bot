"""
Test Production Infrastructure

Tests for Phase 4 & 5 production components to ensure they work correctly
and integrate with the existing gold bot system.
"""

import asyncio
import pytest
import time
from datetime import datetime
import logging

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_production_imports():
    """Test that all production components can be imported"""
    try:
        from production import (
            RealTimeDataStreamer,
            ModelServer, 
            ProductionRiskManager,
            PerformanceMonitor,
            SystemMonitor,
            CacheManager
        )
        logger.info("✅ All production components imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Production import failed: {e}")
        return False

def test_optimization_imports():
    """Test that all optimization components can be imported"""
    try:
        from optimization import (
            AdaptiveRetrainer,
            MicrostructureAnalyzer
        )
        logger.info("✅ All optimization components imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Optimization import failed: {e}")
        return False

def test_cache_manager():
    """Test cache manager functionality"""
    try:
        from production import CacheManager
        
        # Create cache manager (will fallback to memory if Redis not available)
        cache = CacheManager()
        
        # Test basic operations
        test_data = {'test': 'value', 'number': 42}
        cache.set('test_key', test_data, ttl=60)
        
        retrieved = cache.get('test_key')
        assert retrieved == test_data, "Cache get/set failed"
        
        # Test exists
        assert cache.exists('test_key'), "Cache exists failed"
        
        # Test delete
        cache.delete('test_key')
        assert not cache.exists('test_key'), "Cache delete failed"
        
        # Test stats
        stats = cache.get_stats()
        assert 'hits' in stats, "Cache stats missing"
        
        logger.info("✅ Cache manager tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache manager test failed: {e}")
        return False

def test_risk_manager():
    """Test production risk manager"""
    try:
        from production import ProductionRiskManager
        from production.risk_manager import RiskLimits
        
        # Create risk manager
        risk_limits = RiskLimits(max_risk_per_trade=0.02)
        risk_manager = ProductionRiskManager(risk_limits)
        
        # Test trade evaluation
        prediction = {
            'symbol': 'XAUUSD',
            'direction': 'BUY',
            'confidence': 0.75,
            'price': 2000.0
        }
        
        evaluation = risk_manager.evaluate_trade(prediction)
        assert 'action' in evaluation, "Risk evaluation missing action"
        assert evaluation['action'] in ['EXECUTE', 'SKIP', 'STOP_TRADING'], "Invalid action"
        
        # Test portfolio summary
        summary = risk_manager.get_portfolio_summary()
        assert 'account_balance' in summary, "Portfolio summary missing balance"
        
        logger.info("✅ Risk manager tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Risk manager test failed: {e}")
        return False

async def test_data_streamer():
    """Test real-time data streamer"""
    try:
        from production import RealTimeDataStreamer
        
        # Create streamer
        streamer = RealTimeDataStreamer("XAUUSD", update_interval=0.1)
        
        # Track received predictions
        received_predictions = []
        
        async def on_prediction(prediction):
            received_predictions.append(prediction)
        
        streamer.subscribe(on_prediction)
        
        # Start streaming for a short time
        stream_task = asyncio.create_task(streamer.stream_predictions())
        await asyncio.sleep(2)  # Stream for 2 seconds
        
        # Stop streaming
        streamer.stop()
        await asyncio.sleep(0.5)  # Give time to stop
        
        # Check results
        assert len(received_predictions) > 0, "No predictions received"
        
        prediction = received_predictions[0]
        assert 'direction' in prediction, "Prediction missing direction"
        assert 'confidence' in prediction, "Prediction missing confidence"
        
        # Test status
        status = streamer.get_status()
        assert 'performance_stats' in status, "Status missing performance stats"
        
        logger.info(f"✅ Data streamer tests passed ({len(received_predictions)} predictions)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data streamer test failed: {e}")
        return False

def test_monitoring():
    """Test monitoring system"""
    try:
        from production import PerformanceMonitor, SystemMonitor
        
        # Test performance monitor
        perf_monitor = PerformanceMonitor()
        
        # Record some predictions
        for i in range(5):
            prediction = {
                'direction': 'BUY' if i % 2 == 0 else 'SELL',
                'confidence': 0.7 + (i * 0.05),
                'symbol': 'XAUUSD'
            }
            perf_monitor.record_prediction(prediction, actual_result=True, latency_ms=50)
        
        metrics = perf_monitor.get_recent_metrics()
        assert isinstance(metrics, dict), "Metrics not returned as dict"
        
        # Test system monitor
        sys_monitor = SystemMonitor()
        
        # Record an error
        sys_monitor.record_error('TEST', 'Test error message')
        
        status = sys_monitor.get_system_status()
        assert 'monitoring_active' in status, "System status missing monitoring flag"
        
        logger.info("✅ Monitoring tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Monitoring test failed: {e}")
        return False

def test_adaptive_retrainer():
    """Test adaptive retraining system"""
    try:
        from optimization import AdaptiveRetrainer
        from optimization.adaptive_retrainer import RetrainingConfig
        
        # Create retrainer with test config
        config = RetrainingConfig(
            min_accuracy_threshold=0.50,
            min_samples_for_retrain=10
        )
        retrainer = AdaptiveRetrainer(config)
        
        # Record some predictions
        for i in range(15):
            prediction = {
                'direction': 'BUY' if i % 2 == 0 else 'SELL',
                'confidence': 0.6 + (i * 0.02),
                'timestamp': datetime.now()
            }
            
            features = {
                'rsi': 50 + i,
                'macd': i * 0.1,
                'price': 2000 + i
            }
            
            retrainer.record_prediction(prediction, actual_result=True, features=features)
        
        # Get status
        status = retrainer.get_status()
        assert 'buffer_size' in status, "Status missing buffer size"
        assert status['buffer_size'] > 0, "No samples in buffer"
        
        logger.info("✅ Adaptive retrainer tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Adaptive retrainer test failed: {e}")
        return False

def test_microstructure_analyzer():
    """Test microstructure analysis"""
    try:
        from optimization import MicrostructureAnalyzer
        from optimization.microstructure_analyzer import TickData, OrderBook, OrderBookLevel
        
        # Create analyzer
        analyzer = MicrostructureAnalyzer()
        
        # Create test tick data
        tick = TickData(
            timestamp=datetime.now(),
            bid=2000.0,
            ask=2000.5,
            last_price=2000.2,
            volume=100,
            bid_size=1500,
            ask_size=1200
        )
        
        # Analyze order flow
        analysis = analyzer.analyze_order_flow(tick)
        assert 'bid_ask_pressure' in analysis, "Analysis missing bid/ask pressure"
        
        # Create test order book
        orderbook = OrderBook(
            timestamp=datetime.now(),
            symbol="XAUUSD",
            bids=[OrderBookLevel(2000.0, 1500, 3)],
            asks=[OrderBookLevel(2000.5, 1200, 2)]
        )
        
        # Analyze liquidity
        liquidity_analysis = analyzer.analyze_liquidity(orderbook)
        assert 'liquidity_depth' in liquidity_analysis, "Analysis missing liquidity depth"
        
        # Test execution optimization
        prediction = {'direction': 'BUY', 'confidence': 0.75}
        microstructure = {'liquidity_depth': 5000, 'spread': 0.0025}
        
        execution_strategy = analyzer.optimize_entry_timing(prediction, microstructure)
        assert 'action' in execution_strategy, "Execution strategy missing action"
        
        logger.info("✅ Microstructure analyzer tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Microstructure analyzer test failed: {e}")
        return False

async def test_integration():
    """Test integration between components"""
    try:
        # Test that components can work together
        from production import CacheManager, ProductionRiskManager
        from optimization import AdaptiveRetrainer
        
        cache = CacheManager()
        risk_manager = ProductionRiskManager()
        retrainer = AdaptiveRetrainer()
        
        # Test workflow
        prediction = {
            'symbol': 'XAUUSD',
            'direction': 'BUY',
            'confidence': 0.75,
            'price': 2000.0
        }
        
        # Cache the prediction
        cache.cache_prediction('XAUUSD', 'v1.0', 'test_hash', prediction)
        
        # Evaluate with risk manager
        risk_eval = risk_manager.evaluate_trade(prediction)
        
        # Record with retrainer
        features = {'rsi': 65, 'macd': 1.2}
        retrainer.record_prediction(prediction, features=features)
        
        logger.info("✅ Integration tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

async def run_all_tests():
    """Run all production tests"""
    logger.info("🚀 Starting production infrastructure tests...")
    
    test_results = {}
    
    # Basic import tests
    test_results['production_imports'] = test_production_imports()
    test_results['optimization_imports'] = test_optimization_imports()
    
    # Component tests
    test_results['cache_manager'] = test_cache_manager()
    test_results['risk_manager'] = test_risk_manager()
    test_results['monitoring'] = test_monitoring()
    test_results['adaptive_retrainer'] = test_adaptive_retrainer()
    test_results['microstructure_analyzer'] = test_microstructure_analyzer()
    
    # Async tests
    test_results['data_streamer'] = await test_data_streamer()
    test_results['integration'] = await test_integration()
    
    # Results summary
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    logger.info(f"\n📊 Test Results: {passed}/{total} passed")
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    if passed == total:
        logger.info("🎉 All production tests passed!")
        return True
    else:
        logger.error(f"💥 {total - passed} tests failed")
        return False

if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)