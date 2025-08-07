# config/validation.py
"""
Enhanced configuration validation system with environment variable validation,
type checking, and runtime validation capabilities.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Union
import re
import logging
from pathlib import Path

# Import enhanced utilities
try:
    from utils.logging_system import get_logger
    from utils.dependency_manager import get_dependency_status
    from utils.path_manager import path_manager
    ENHANCED_UTILS_AVAILABLE = True
except ImportError:
    ENHANCED_UTILS_AVAILABLE = False

logger = get_logger("config_validation") if ENHANCED_UTILS_AVAILABLE else logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Enhanced result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: List[str] = None
    
    def __post_init__(self):
        if self.info is None:
            self.info = []
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_info(self, message: str):
        """Add an info message."""
        self.info.append(message)
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        self.is_valid = self.is_valid and other.is_valid


def validate_complete_configuration() -> ValidationResult:
    """
    Perform comprehensive configuration validation including environment variables,
    runtime dependencies, and configuration consistency.
    
    Returns:
        ValidationResult with comprehensive validation status
    """
    result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
    
    try:
        # 1. Validate basic environment variables
        logger.info("Validating environment variables...")
        
        # Check for at least one API configured
        has_api = bool(os.getenv('TWELVE_DATA_API_KEY') or 
                      (os.getenv('BINANCE_API_KEY') and os.getenv('BINANCE_SECRET')))
        
        if not has_api:
            result.add_warning("No API credentials configured - trading functionality will be limited")
        else:
            result.add_info("API credentials configured ✓")
        
        # Check environment setting
        env = os.getenv('ENVIRONMENT', 'development')
        if env not in ['development', 'staging', 'production']:
            result.add_error(f"Invalid environment: {env}. Must be development, staging, or production")
        else:
            result.add_info(f"Environment: {env} ✓")
        
        # 2. Validate runtime dependencies
        logger.info("Validating runtime dependencies...")
        
        if ENHANCED_UTILS_AVAILABLE:
            dep_status = get_dependency_status()
            
            # Check critical dependencies
            critical_deps = ['scipy', 'sklearn']
            missing_critical = [dep for dep in critical_deps 
                              if not dep_status['dependencies'].get(dep, False)]
            
            if missing_critical:
                result.add_error(f"Critical dependencies missing: {', '.join(missing_critical)}")
            else:
                result.add_info("Critical dependencies available ✓")
            
            # Check optional dependencies
            optional_deps = ['tensorflow', 'lightgbm', 'xgboost']
            available_optional = [dep for dep in optional_deps 
                                if dep_status['dependencies'].get(dep, False)]
            
            if available_optional:
                result.add_info(f"Optional ML dependencies: {', '.join(available_optional)} ✓")
            else:
                result.add_warning("No optional ML dependencies available - using fallback implementations")
        
        # 3. Validate file system permissions
        logger.info("Validating file system permissions...")
        
        if ENHANCED_UTILS_AVAILABLE:
            try:
                # Test model directory
                model_dir = path_manager.get_model_path()
                model_dir.mkdir(exist_ok=True)
                test_file = model_dir / "test_write.tmp"
                test_file.write_text("test")
                test_file.unlink()
                result.add_info(f"Model directory writable: {model_dir} ✓")
            except Exception as e:
                result.add_error(f"Cannot write to model directory: {e}")
            
            try:
                # Test log directory
                log_dir = path_manager.get_log_path()
                log_dir.mkdir(exist_ok=True)
                test_file = log_dir / "test_write.tmp"
                test_file.write_text("test")
                test_file.unlink()
                result.add_info(f"Log directory writable: {log_dir} ✓")
            except Exception as e:
                result.add_error(f"Cannot write to log directory: {e}")
        
        # 4. Validate configuration consistency
        logger.info("Validating configuration consistency...")
        
        # Production environment checks
        if env == 'production':
            debug = os.getenv('DEBUG', 'false').lower() in ['true', '1', 'yes']
            if debug:
                result.add_warning("Debug mode enabled in production")
            
            log_level = os.getenv('LOG_LEVEL', 'INFO')
            if log_level == 'DEBUG':
                result.add_warning("Debug logging enabled in production")
        
        # Risk management validation
        try:
            risk_pct = float(os.getenv('RISK_PERCENTAGE', '2.0'))
            if risk_pct > 5:
                result.add_warning(f"Risk percentage {risk_pct}% is high")
            elif risk_pct > 10:
                result.add_error(f"Risk percentage {risk_pct}% is extremely dangerous")
        except ValueError:
            result.add_error("Invalid RISK_PERCENTAGE format")
        
        # Symbol validation
        symbol = os.getenv('TRADING_SYMBOL', 'XAUUSD')
        if not symbol:
            result.add_error("TRADING_SYMBOL cannot be empty")
        else:
            result.add_info(f"Trading symbol: {symbol} ✓")
        
        # Log summary
        if result.errors:
            logger.error(f"Configuration validation failed with {len(result.errors)} errors")
            for error in result.errors:
                logger.error(f"  ❌ {error}")
        
        if result.warnings:
            logger.warning(f"Configuration validation has {len(result.warnings)} warnings")
            for warning in result.warnings:
                logger.warning(f"  ⚠️  {warning}")
        
        if result.info:
            logger.info("Configuration validation info:")
            for info in result.info:
                logger.info(f"  ℹ️  {info}")
        
        if result.is_valid:
            logger.info("✅ Configuration validation passed")
        else:
            logger.error("❌ Configuration validation failed")
        
        return result
        
    except Exception as e:
        logger.error(f"Configuration validation exception: {e}")
        result.add_error(f"Validation exception: {str(e)}")
        return result


def check_environment_setup() -> bool:
    """
    Quick check to see if environment is properly set up.
    
    Returns:
        True if environment is ready, False otherwise
    """
    result = validate_complete_configuration()
    return result.is_valid


def get_configuration_report() -> Dict[str, Any]:
    """
    Get comprehensive configuration report for diagnostics.
    
    Returns:
        Dictionary with configuration status and details
    """
    result = validate_complete_configuration()
    
    report = {
        'validation_status': 'PASS' if result.is_valid else 'FAIL',
        'error_count': len(result.errors),
        'warning_count': len(result.warnings),
        'info_count': len(result.info),
        'errors': result.errors,
        'warnings': result.warnings,
        'info': result.info,
        'environment_variables': {},
        'dependencies': {},
        'recommendations': []
    }
    
    # Add key environment variables (mask sensitive ones)
    env_vars = {
        'ENVIRONMENT': os.getenv('ENVIRONMENT', 'development'),
        'DEBUG': os.getenv('DEBUG', 'false'),
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
        'TRADING_SYMBOL': os.getenv('TRADING_SYMBOL', 'XAUUSD'),
        'TIMEFRAME': os.getenv('TIMEFRAME', '5m'),
        'RISK_PERCENTAGE': os.getenv('RISK_PERCENTAGE', '2.0'),
        'CONFIDENCE_THRESHOLD': os.getenv('CONFIDENCE_THRESHOLD', '0.65'),
        'TWELVE_DATA_API_KEY': '***MASKED***' if os.getenv('TWELVE_DATA_API_KEY') else 'Not set',
        'BINANCE_API_KEY': '***MASKED***' if os.getenv('BINANCE_API_KEY') else 'Not set',
        'BINANCE_SECRET': '***MASKED***' if os.getenv('BINANCE_SECRET') else 'Not set'
    }
    
    report['environment_variables'] = env_vars
    
    # Add dependency status
    if ENHANCED_UTILS_AVAILABLE:
        dep_status = get_dependency_status()
        report['dependencies'] = dep_status
    
    # Add recommendations
    if result.warnings:
        report['recommendations'].extend([f"Address warning: {w}" for w in result.warnings])
    
    if not result.is_valid:
        report['recommendations'].append("Fix configuration errors before running in production")
    
    return report


# Legacy functions for backward compatibility
def validate_config(config=None) -> ValidationResult:
    """
    Validate application configuration (legacy function for backward compatibility).
    
    Args:
        config: AppConfig instance to validate (ignored, uses environment)
        
    Returns:
        ValidationResult with validation status and messages
    """
    # Use the enhanced validator
    return validate_complete_configuration()


def validate_api_key_format(api_key: str, source: str) -> ValidationResult:
    """
    Validate a single API key format (legacy function).
    
    Args:
        api_key: API key to validate
        source: Source name for error messages
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
    
    if not api_key or len(api_key) < 10:
        result.add_error(f"{source} API key too short (minimum 10 characters)")
    elif len(api_key) > 200:
        result.add_error(f"{source} API key too long (maximum 200 characters)")
    elif any(test in api_key.lower() for test in ["test", "demo", "sandbox", "example"]):
        result.add_warning(f"{source} API key appears to be a placeholder/test key")
    else:
        result.add_info(f"{source} API key format valid ✓")
    
    return result