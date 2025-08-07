"""
Production Infrastructure Package

This package contains production-ready components for real-time trading:
- Real-time data streaming
- Model serving infrastructure
- Risk management enhancements
- Monitoring and alerting
- Performance optimization
"""

__version__ = "1.0.0"
__author__ = "Gold Bot Team"

# Import main production components with fallback
__all__ = []

try:
    from .data_streamer import RealTimeDataStreamer
    __all__.append('RealTimeDataStreamer')
except ImportError as e:
    print(f"Warning: RealTimeDataStreamer not available: {e}")

try:
    from .model_server import ModelServer
    __all__.append('ModelServer')
except ImportError as e:
    print(f"Warning: ModelServer not available: {e}")

try:
    from .risk_manager import ProductionRiskManager
    __all__.append('ProductionRiskManager')
except ImportError as e:
    print(f"Warning: ProductionRiskManager not available: {e}")

try:
    from .monitor import PerformanceMonitor, SystemMonitor, MonitoringManager
    __all__.extend(['PerformanceMonitor', 'SystemMonitor', 'MonitoringManager'])
except ImportError as e:
    print(f"Warning: Monitoring components not available: {e}")

try:
    from .cache_manager import CacheManager
    __all__.append('CacheManager')
except ImportError as e:
    print(f"Warning: CacheManager not available: {e}")