"""
Centralized Dependency Management System

Provides robust dependency checking with fallback implementations
and graceful degradation when optional packages are unavailable.
"""

import sys
import logging
import warnings
from typing import Dict, Any, Callable, Optional
from functools import wraps

# Setup logging
logger = logging.getLogger(__name__)

class DependencyManager:
    """Centralized dependency checker with fallback implementations."""
    
    def __init__(self):
        self.dependencies = {}
        self.fallbacks = {}
        self._check_all_dependencies()
    
    def _check_all_dependencies(self):
        """Check availability of all dependencies."""
        deps_to_check = {
            'tensorflow': 'tensorflow',
            'lightgbm': 'lightgbm', 
            'xgboost': 'xgboost',
            'scipy': 'scipy',
            'sklearn': 'sklearn',
            'ta': 'ta',
            'pandas_ta': 'pandas_ta',
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn',
            'websockets': 'websockets',
            'redis': 'redis',
            'psutil': 'psutil'
        }
        
        for name, module_name in deps_to_check.items():
            self.dependencies[name] = self._check_dependency(module_name)
            if not self.dependencies[name]:
                logger.warning(f"Optional dependency '{name}' not available")
    
    def _check_dependency(self, module_name: str) -> bool:
        """Check if a module is available."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def is_available(self, dependency: str) -> bool:
        """Check if a dependency is available."""
        return self.dependencies.get(dependency, False)
    
    def require(self, dependency: str, fallback_func: Optional[Callable] = None):
        """Decorator to require a dependency with optional fallback."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.is_available(dependency):
                    return func(*args, **kwargs)
                elif fallback_func:
                    logger.warning(f"Using fallback for {dependency} in {func.__name__}")
                    return fallback_func(*args, **kwargs)
                else:
                    logger.error(f"Required dependency '{dependency}' not available for {func.__name__}")
                    raise ImportError(f"Required dependency '{dependency}' not available")
            return wrapper
        return decorator
    
    def with_fallback(self, dependency: str, fallback_value: Any = None):
        """Decorator to provide fallback value when dependency unavailable."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.is_available(dependency):
                    return func(*args, **kwargs)
                else:
                    logger.warning(f"Dependency '{dependency}' unavailable in {func.__name__}, using fallback")
                    return fallback_value
            return wrapper
        return decorator
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive dependency status report."""
        return {
            'dependencies': self.dependencies.copy(),
            'total_dependencies': len(self.dependencies),
            'available_count': sum(self.dependencies.values()),
            'missing_dependencies': [name for name, available in self.dependencies.items() if not available],
            'critical_missing': self._get_critical_missing(),
            'python_version': sys.version,
            'platform': sys.platform
        }
    
    def _get_critical_missing(self) -> list:
        """Get list of critical missing dependencies."""
        critical_deps = ['scipy', 'sklearn']  # Core ML dependencies
        return [dep for dep in critical_deps if not self.is_available(dep)]
    
    def register_fallback(self, dependency: str, fallback_func: Callable):
        """Register a fallback function for a dependency."""
        self.fallbacks[dependency] = fallback_func
    
    def get_fallback(self, dependency: str) -> Optional[Callable]:
        """Get registered fallback function for a dependency."""
        return self.fallbacks.get(dependency)


# Global dependency manager instance
dependency_manager = DependencyManager()

# Convenience functions
def check_dependencies() -> Dict[str, bool]:
    """Check and return available dependencies."""
    return dependency_manager.dependencies.copy()

def is_available(dependency: str) -> bool:
    """Check if a dependency is available."""
    return dependency_manager.is_available(dependency)

def require_dependency(dependency: str, fallback_func: Optional[Callable] = None):
    """Decorator to require a dependency."""
    return dependency_manager.require(dependency, fallback_func)

def with_fallback(dependency: str, fallback_value: Any = None):
    """Decorator to provide fallback when dependency unavailable."""
    return dependency_manager.with_fallback(dependency, fallback_value)

def get_dependency_status() -> Dict[str, Any]:
    """Get comprehensive dependency status."""
    return dependency_manager.get_status_report()


# Fallback implementations for common operations
class FallbackImplementations:
    """Fallback implementations when dependencies are unavailable."""
    
    @staticmethod
    def simple_linear_regression(x, y):
        """Simple linear regression fallback for scipy.stats.linregress."""
        import numpy as np
        
        # Convert to numpy arrays
        x = np.array(x)
        y = np.array(y)
        
        # Calculate slope and intercept
        n = len(x)
        if n < 2:
            return 0, np.mean(y) if len(y) > 0 else 0, 0, None, None
        
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        # Calculate slope
        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sum((x - mean_x) ** 2)
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Calculate intercept
        intercept = mean_y - slope * mean_x
        
        # Calculate correlation coefficient
        if np.std(x) == 0 or np.std(y) == 0:
            r_value = 0
        else:
            r_value = np.corrcoef(x, y)[0, 1]
            if np.isnan(r_value):
                r_value = 0
        
        return slope, intercept, r_value, None, None
    
    @staticmethod
    def simple_peak_detection(data, distance=5):
        """Simple peak detection fallback for scipy.signal.find_peaks."""
        import numpy as np
        
        data = np.array(data)
        peaks = []
        
        for i in range(distance, len(data) - distance):
            is_peak = True
            for j in range(1, distance + 1):
                if data[i] <= data[i - j] or data[i] <= data[i + j]:
                    is_peak = False
                    break
            if is_peak:
                peaks.append(i)
        
        return np.array(peaks), {}
    
    @staticmethod
    def simple_kmeans(data, n_clusters=3):
        """Simple K-means clustering fallback."""
        import numpy as np
        
        data = np.array(data).reshape(-1, 1)
        
        if len(data) < n_clusters:
            # Return simple split
            return {
                'cluster_centers_': data[:n_clusters],
                'labels_': list(range(min(len(data), n_clusters)))
            }
        
        # Simple clustering based on quantiles
        centers = []
        for i in range(n_clusters):
            quantile = (i + 1) / (n_clusters + 1)
            centers.append([np.quantile(data, quantile)])
        
        # Assign labels based on nearest center
        labels = []
        for point in data:
            distances = [abs(point[0] - center[0]) for center in centers]
            labels.append(distances.index(min(distances)))
        
        return {
            'cluster_centers_': np.array(centers),
            'labels_': labels
        }
    
    @staticmethod
    def simple_scaler():
        """Simple scaler fallback for sklearn.preprocessing.StandardScaler."""
        import numpy as np
        
        class SimpleScaler:
            def __init__(self):
                self.mean_ = None
                self.std_ = None
            
            def fit(self, X):
                X = np.array(X)
                self.mean_ = np.mean(X, axis=0)
                self.std_ = np.std(X, axis=0)
                # Avoid division by zero
                self.std_[self.std_ == 0] = 1
                return self
            
            def transform(self, X):
                X = np.array(X)
                if self.mean_ is None or self.std_ is None:
                    raise ValueError("Scaler not fitted")
                return (X - self.mean_) / self.std_
            
            def fit_transform(self, X):
                return self.fit(X).transform(X)
        
        return SimpleScaler()


# Register fallback implementations
dependency_manager.register_fallback('scipy', FallbackImplementations.simple_linear_regression)
dependency_manager.register_fallback('sklearn', FallbackImplementations.simple_scaler)

# Export convenience functions
__all__ = [
    'DependencyManager',
    'dependency_manager', 
    'check_dependencies',
    'is_available',
    'require_dependency',
    'with_fallback',
    'get_dependency_status',
    'FallbackImplementations'
]