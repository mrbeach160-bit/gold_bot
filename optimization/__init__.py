"""
Real-Time Model Optimization Package

Advanced optimization components for adaptive model retraining,
dynamic ensemble optimization, and continuous performance improvement.
"""

__version__ = "1.0.0"
__author__ = "Gold Bot Team"

# Import main optimization components with fallback
__all__ = []

try:
    from .adaptive_retrainer import AdaptiveRetrainer
    __all__.append('AdaptiveRetrainer')
except ImportError as e:
    print(f"Warning: AdaptiveRetrainer not available: {e}")

try:
    from .microstructure_analyzer import MicrostructureAnalyzer
    __all__.append('MicrostructureAnalyzer')
except ImportError as e:
    print(f"Warning: MicrostructureAnalyzer not available: {e}")

# Note: These components are not yet implemented but reserved for future
# try:
#     from .ensemble_optimizer import DynamicEnsembleOptimizer  
#     __all__.append('DynamicEnsembleOptimizer')
# except ImportError as e:
#     print(f"Warning: DynamicEnsembleOptimizer not available: {e}")

# try:
#     from .gpu_inference import GPUInferenceEngine
#     __all__.append('GPUInferenceEngine')
# except ImportError as e:
#     print(f"Warning: GPUInferenceEngine not available: {e}")

# try:
#     from .analytics_dashboard import AdvancedAnalytics
#     __all__.append('AdvancedAnalytics')
# except ImportError as e:
#     print(f"Warning: AdvancedAnalytics not available: {e}")