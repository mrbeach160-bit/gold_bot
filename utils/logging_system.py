"""
Comprehensive Logging System

Provides structured logging with configurable levels, log rotation,
performance metrics, and error tracking.
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json
import time
from functools import wraps

# Import path manager for consistent file paths
try:
    from utils.path_manager import path_manager
    PATH_MANAGER_AVAILABLE = True
except ImportError:
    PATH_MANAGER_AVAILABLE = False

class PerformanceLogger:
    """Logger for performance metrics and monitoring."""
    
    def __init__(self, name: str = "performance"):
        self.name = name
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.log_metric(f"{operation}_duration", duration)
            del self.start_times[operation]
            return duration
        return 0.0
    
    def log_metric(self, metric_name: str, value: float, metadata: Dict = None):
        """Log a performance metric."""
        timestamp = datetime.now().isoformat()
        self.metrics[metric_name] = {
            'value': value,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all logged metrics."""
        return self.metrics.copy()
    
    def timer(self, operation_name: str):
        """Decorator to automatically time function execution."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.start_timer(operation_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_timer(operation_name)
            return wrapper
        return decorator


class StructuredLogger:
    """Enhanced logger with structured output and context."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.context = {}
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers."""
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if PATH_MANAGER_AVAILABLE:
            log_file = path_manager.get_log_file_path(self.name, date_suffix=True)
        else:
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB files, keep 5
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def set_context(self, **kwargs):
        """Set logging context that will be included in all messages."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear logging context."""
        self.context.clear()
    
    def _format_message(self, message: str, extra: Dict = None) -> str:
        """Format message with context and extra data."""
        parts = [message]
        
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            parts.append(f"[Context: {context_str}]")
        
        if extra:
            extra_str = ", ".join([f"{k}={v}" for k, v in extra.items()])
            parts.append(f"[Extra: {extra_str}]")
        
        return " ".join(parts)
    
    def debug(self, message: str, **extra):
        """Log debug message."""
        self.logger.debug(self._format_message(message, extra))
    
    def info(self, message: str, **extra):
        """Log info message."""
        self.logger.info(self._format_message(message, extra))
    
    def warning(self, message: str, **extra):
        """Log warning message."""
        self.logger.warning(self._format_message(message, extra))
    
    def error(self, message: str, **extra):
        """Log error message."""
        self.logger.error(self._format_message(message, extra))
    
    def critical(self, message: str, **extra):
        """Log critical message."""
        self.logger.critical(self._format_message(message, extra))
    
    def log_exception(self, message: str = "Exception occurred", **extra):
        """Log exception with traceback."""
        self.logger.exception(self._format_message(message, extra))
    
    def log_structured(self, level: str, event: str, **data):
        """Log structured data as JSON."""
        structured_data = {
            'event': event,
            'timestamp': datetime.now().isoformat(),
            'context': self.context.copy(),
            'data': data
        }
        
        json_message = json.dumps(structured_data, default=str)
        getattr(self.logger, level.lower())(json_message)


class LogManager:
    """Central logging management."""
    
    def __init__(self):
        self.loggers = {}
        self.performance_loggers = {}
        self.default_level = "INFO"
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """Setup root logger configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Suppress overly verbose third-party loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    def get_logger(self, name: str, level: str = None) -> StructuredLogger:
        """Get or create a structured logger."""
        if name not in self.loggers:
            self.loggers[name] = StructuredLogger(name, level or self.default_level)
        return self.loggers[name]
    
    def get_performance_logger(self, name: str = "performance") -> PerformanceLogger:
        """Get or create a performance logger."""
        if name not in self.performance_loggers:
            self.performance_loggers[name] = PerformanceLogger(name)
        return self.performance_loggers[name]
    
    def set_global_level(self, level: str):
        """Set logging level for all existing loggers."""
        self.default_level = level
        for logger in self.loggers.values():
            logger.logger.setLevel(getattr(logging, level.upper()))
    
    def log_system_info(self):
        """Log system information."""
        logger = self.get_logger("system")
        
        import platform
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version()
        }
        
        # Try to get psutil info if available
        try:
            import psutil
            system_info.update({
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
            })
        except ImportError:
            logger.debug("psutil not available for detailed system info")
        
        logger.info("System Information", **system_info)
    
    def log_dependencies(self):
        """Log dependency status."""
        logger = self.get_logger("dependencies")
        
        try:
            from utils.dependency_manager import get_dependency_status
            status = get_dependency_status()
            logger.info("Dependency Status", **status)
        except ImportError:
            logger.warning("Dependency manager not available")
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log files older than specified days."""
        if not PATH_MANAGER_AVAILABLE:
            return
        
        log_dir = path_manager.get_log_path()
        if not log_dir.exists():
            return
        
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        cleaned_count = 0
        
        for log_file in log_dir.glob("*.log*"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    cleaned_count += 1
            except Exception as e:
                print(f"Could not clean up log file {log_file}: {e}")
        
        if cleaned_count > 0:
            logger = self.get_logger("cleanup")
            logger.info(f"Cleaned up {cleaned_count} old log files")


# Global log manager instance
log_manager = LogManager()

# Convenience functions
def get_logger(name: str, level: str = None) -> StructuredLogger:
    """Get a structured logger."""
    return log_manager.get_logger(name, level)

def get_performance_logger(name: str = "performance") -> PerformanceLogger:
    """Get a performance logger."""
    return log_manager.get_performance_logger(name)

def log_function_call(logger_name: str = None):
    """Decorator to log function calls with timing."""
    def decorator(func):
        logger = get_logger(logger_name or func.__module__)
        perf_logger = get_performance_logger()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            logger.debug(f"Calling {func_name}")
            
            perf_logger.start_timer(func_name)
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Completed {func_name}")
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {str(e)}")
                raise
            finally:
                duration = perf_logger.end_timer(func_name)
                logger.debug(f"{func_name} took {duration:.4f} seconds")
        
        return wrapper
    return decorator

# Initialize system logging
log_manager.log_system_info()
log_manager.log_dependencies()

# Export main classes and functions
__all__ = [
    'LogManager',
    'StructuredLogger',
    'PerformanceLogger',
    'log_manager',
    'get_logger',
    'get_performance_logger',
    'log_function_call'
]