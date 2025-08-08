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


@dataclass
class EnvironmentVariable:
    """Environment variable specification."""
    name: str
    required: bool = True
    default: Optional[str] = None
    description: str = ""
    validator: Optional[callable] = None
    sensitive: bool = False  # Don't log value if True


class EnhancedConfigValidator:
    """Enhanced configuration validator with comprehensive validation rules."""
    
    def __init__(self):
        self.result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
        self._define_environment_variables()
    
    def _define_environment_variables(self):
        """Define expected environment variables."""
        self.env_vars = [
            # API Configuration
            EnvironmentVariable(
                "TWELVE_DATA_API_KEY", 
                required=False, 
                description="Twelve Data API key for market data",
                validator=self._validate_api_key,
                sensitive=True
            ),
            EnvironmentVariable(
                "BINANCE_API_KEY", 
                required=False, 
                description="Binance API key for trading",
                validator=self._validate_api_key,
                sensitive=True
            ),
            EnvironmentVariable(
                "BINANCE_SECRET", 
                required=False, 
                description="Binance API secret",
                validator=self._validate_api_key,
                sensitive=True
            ),
            
            # Trading Configuration
            EnvironmentVariable(
                "TRADING_SYMBOL", 
                required=False, 
                default="XAUUSD",
                description="Trading symbol",
                validator=self.validate_trading_symbol
            ),
            EnvironmentVariable(
                "TIMEFRAME", 
                required=False, 
                default="5m",
                description="Trading timeframe",
                validator=self.validate_timeframe
            ),
            EnvironmentVariable(
                "RISK_PERCENTAGE", 
                required=False, 
                default="2.0",
                description="Risk percentage per trade",
                validator=self._validate_risk_percentage
            ),
            
            # Application Configuration
            EnvironmentVariable(
                "ENVIRONMENT", 
                required=False, 
                default="development",
                description="Application environment",
                validator=self._validate_environment
            ),
            EnvironmentVariable(
                "DEBUG", 
                required=False, 
                default="true",
                description="Debug mode flag",
                validator=self._validate_boolean
            ),
            EnvironmentVariable(
                "LOG_LEVEL", 
                required=False, 
                default="INFO",
                description="Logging level",
                validator=self._validate_log_level
            ),
            
            # Model Configuration
            EnvironmentVariable(
                "CONFIDENCE_THRESHOLD", 
                required=False, 
                default="0.65",
                description="Model confidence threshold",
                validator=self._validate_confidence_threshold
            ),
            EnvironmentVariable(
                "MODELS_TO_USE", 
                required=False, 
                default="lstm,xgb,meta",
                description="Comma-separated list of models to use",
                validator=self._validate_models_list
            ),
            
            # Infrastructure Configuration
            EnvironmentVariable(
                "SERVER_HOST", 
                required=False, 
                default="0.0.0.0",
                description="Server host address",
                validator=self._validate_host
            ),
            EnvironmentVariable(
                "SERVER_PORT", 
                required=False, 
                default="8000",
                description="Server port",
                validator=self._validate_port
            ),
            EnvironmentVariable(
                "MAX_WORKERS", 
                required=False, 
                default="1",
                description="Maximum worker processes",
                validator=self._validate_positive_int
            ),
        ]
    
    def validate_environment_variables(self) -> ValidationResult:
        """Validate all environment variables."""
        env_result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
        
        for env_var in self.env_vars:
            value = os.getenv(env_var.name, env_var.default)
            
            # Check required variables
            if env_var.required and not value:
                env_result.add_error(f"Required environment variable {env_var.name} is not set")
                continue
            
            # Skip validation if not set and not required
            if not value:
                continue
            
            # Validate value format
            if env_var.validator:
                try:
                    var_valid = env_var.validator(value, env_var.name)
                    if not var_valid:
                        env_result.add_error(f"Invalid value for {env_var.name}")
                except Exception as e:
                    env_result.add_error(f"Validation error for {env_var.name}: {str(e)}")
            
            # Log successful validation (mask sensitive values)
            display_value = "***MASKED***" if env_var.sensitive else value
            env_result.add_info(f"{env_var.name}={display_value} ✓")
        
        return env_result
    
    def _validate_api_key(self, value: str, name: str) -> bool:
        """Validate API key format."""
        if not value or len(value) < 10:
            self.result.add_error(f"{name} too short (minimum 10 characters)")
            return False
        
        if len(value) > 200:
            self.result.add_error(f"{name} too long (maximum 200 characters)")
            return False
        
        # Check for test/placeholder keys
        test_patterns = ["test", "demo", "sandbox", "example", "your_", "replace_me"]
        if any(pattern in value.lower() for pattern in test_patterns):
            self.result.add_warning(f"{name} appears to be a placeholder/test key")
        
        return True
    
    def _validate_risk_percentage(self, value: str, name: str) -> bool:
        """Validate risk percentage."""
        try:
            risk = float(value)
            if risk <= 0:
                self.result.add_error(f"{name} must be positive")
                return False
            elif risk > 10:
                self.result.add_error(f"{name} cannot exceed 10% (extremely dangerous)")
                return False
            elif risk > 5:
                self.result.add_warning(f"{name} > 5% is considered high risk")
            return True
        except ValueError:
            self.result.add_error(f"{name} must be a valid number")
            return False
    
    def _validate_environment(self, value: str, name: str) -> bool:
        """Validate environment setting."""
        valid_envs = ["development", "staging", "production"]
        if value not in valid_envs:
            self.result.add_error(f"{name} must be one of: {', '.join(valid_envs)}")
            return False
        return True
    
    def _validate_boolean(self, value: str, name: str) -> bool:
        """Validate boolean values."""
        valid_bools = ["true", "false", "1", "0", "yes", "no"]
        if value.lower() not in valid_bools:
            self.result.add_error(f"{name} must be boolean (true/false, 1/0, yes/no)")
            return False
        return True
    
    def _validate_log_level(self, value: str, name: str) -> bool:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if value.upper() not in valid_levels:
            self.result.add_error(f"{name} must be one of: {', '.join(valid_levels)}")
            return False
        return True
    
    def _validate_confidence_threshold(self, value: str, name: str) -> bool:
        """Validate confidence threshold."""
        try:
            threshold = float(value)
            if not 0.5 <= threshold <= 1.0:
                self.result.add_error(f"{name} must be between 0.5 and 1.0")
                return False
            elif threshold < 0.6:
                self.result.add_warning(f"{name} < 0.6 may generate too many false signals")
            return True
        except ValueError:
            self.result.add_error(f"{name} must be a valid number")
            return False
    
    def _validate_models_list(self, value: str, name: str) -> bool:
        """Validate models list."""
        valid_models = ["lstm", "xgb", "cnn", "svc", "nb", "meta", "rf"]
        models = [model.strip() for model in value.split(",")]
        
        for model in models:
            if model not in valid_models:
                self.result.add_error(f"Invalid model '{model}' in {name}. Valid: {', '.join(valid_models)}")
                return False
        
        return True
    
    def _validate_host(self, value: str, name: str) -> bool:
        """Validate host address."""
        # Simple validation - could be enhanced
        if not value or not value.replace(".", "").replace(":", "").replace("localhost", "").isalnum():
            if value not in ["0.0.0.0", "127.0.0.1", "localhost"]:
                self.result.add_warning(f"{name} format may be invalid")
        return True
    
    def _validate_port(self, value: str, name: str) -> bool:
        """Validate port number."""
        try:
            port = int(value)
            if not 1 <= port <= 65535:
                self.result.add_error(f"{name} must be between 1 and 65535")
                return False
            elif port < 1024:
                self.result.add_warning(f"{name} < 1024 may require root privileges")
            return True
        except ValueError:
            self.result.add_error(f"{name} must be a valid port number")
            return False
    
    def _validate_positive_int(self, value: str, name: str) -> bool:
        """Validate positive integer."""
        try:
            num = int(value)
            if num <= 0:
                self.result.add_error(f"{name} must be positive")
                return False
            return True
        except ValueError:
            self.result.add_error(f"{name} must be a valid integer")
            return False
    
    def validate_runtime_dependencies(self) -> ValidationResult:
        """Validate runtime dependencies and system requirements."""
        runtime_result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
        
        if ENHANCED_UTILS_AVAILABLE:
            # Get dependency status
            dep_status = get_dependency_status()
            
            # Check critical dependencies
            critical_deps = ['scipy', 'sklearn']
            for dep in critical_deps:
                if not dep_status['dependencies'].get(dep, False):
                    runtime_result.add_error(f"Critical dependency '{dep}' not available")
            
            # Check optional dependencies
            optional_deps = ['tensorflow', 'lightgbm', 'xgboost']
            missing_optional = []
            for dep in optional_deps:
                if not dep_status['dependencies'].get(dep, False):
                    missing_optional.append(dep)
            
            if missing_optional:
                runtime_result.add_warning(f"Optional dependencies missing: {', '.join(missing_optional)}")
            
            runtime_result.add_info(f"Python version: {dep_status['python_version']}")
            runtime_result.add_info(f"Platform: {dep_status['platform']}")
            runtime_result.add_info(f"Available dependencies: {dep_status['available_count']}/{dep_status['total_dependencies']}")
        
        # Check file system permissions
        try:
            if ENHANCED_UTILS_AVAILABLE:
                model_dir = path_manager.get_model_path()
                log_dir = path_manager.get_log_path()
                
                for directory in [model_dir, log_dir]:
                    if not directory.exists():
                        try:
                            directory.mkdir(parents=True, exist_ok=True)
                            runtime_result.add_info(f"Created directory: {directory}")
                        except PermissionError:
                            runtime_result.add_error(f"No write permission for: {directory}")
                    else:
                        # Test write permissions
                        test_file = directory / "permission_test.tmp"
                        try:
                            test_file.write_text("test")
                            test_file.unlink()
                            runtime_result.add_info(f"Write permission OK: {directory}")
                        except Exception:
                            runtime_result.add_error(f"No write permission for: {directory}")
        except Exception as e:
            runtime_result.add_warning(f"Could not validate file permissions: {e}")
        
        return runtime_result
    
    def validate_configuration_consistency(self, config) -> ValidationResult:
        """Validate configuration consistency and business rules."""
        consistency_result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
        
        try:
            # Environment-specific validations
            env = os.getenv("ENVIRONMENT", "development")
            debug = os.getenv("DEBUG", "true").lower() in ["true", "1", "yes"]
            log_level = os.getenv("LOG_LEVEL", "INFO")
            
            if env == "production":
                if debug:
                    consistency_result.add_warning("Debug mode enabled in production")
                if log_level == "DEBUG":
                    consistency_result.add_warning("Debug logging enabled in production")
                
                # Check for test API keys in production
                if os.getenv("BINANCE_API_KEY"):
                    api_key = os.getenv("BINANCE_API_KEY")
                    if any(test in api_key.lower() for test in ["test", "demo", "sandbox"]):
                        consistency_result.add_error("Test API keys detected in production environment")
            
            # API configuration consistency
            has_twelve_data = bool(os.getenv("TWELVE_DATA_API_KEY"))
            has_binance = bool(os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_SECRET"))
            
            if not has_twelve_data and not has_binance:
                consistency_result.add_error("At least one API (Twelve Data or Binance) must be configured")
            
            # Model configuration validation
            models_str = os.getenv("MODELS_TO_USE", "lstm,xgb,meta")
            models = [m.strip() for m in models_str.split(",")]
            
            if "meta" in models and len(models) < 3:
                consistency_result.add_warning("Meta learner requires multiple base models for effective ensemble")
            
            # Resource allocation checks
            max_workers = int(os.getenv("MAX_WORKERS", "1"))
            if max_workers > 4:
                consistency_result.add_warning("High worker count may cause resource contention")
            
            consistency_result.add_info("Configuration consistency check completed")
            
        except Exception as e:
            consistency_result.add_error(f"Configuration consistency check failed: {str(e)}")
        
        return consistency_result
    
    # Keep original validation methods for backward compatibility
    def validate_api_key_format(self, api_key: Optional[str], source: str) -> bool:
        """Validate API key format (backward compatibility)."""
        return self._validate_api_key(api_key or "", source)
    
    def validate_trading_symbol(self, symbol: str, name: str = None) -> bool:
        """Validate trading symbol format."""
        if not symbol or not symbol.strip():
            self.result.add_error("Trading symbol cannot be empty")
            return False
        
        symbol = symbol.strip().upper()
        
        # Common symbol patterns
        patterns = [
            r'^[A-Z]{3,6}/[A-Z]{3,4}$',  # Forex/Crypto format (e.g., XAU/USD, BTC/USD)
            r'^[A-Z]{3,10}USDT?$',       # Crypto format (e.g., BTCUSDT, ETHUSDT)
            r'^[A-Z]{2,6}$',             # Stock format (e.g., AAPL, MSFT)
        ]
        
        valid_format = any(re.match(pattern, symbol) for pattern in patterns)
        
        if not valid_format:
            self.result.add_error(f"Invalid symbol format: {symbol}. Expected: XAU/USD, BTCUSDT, or AAPL")
            return False
        
        return True
    
    def validate_timeframe(self, timeframe: str, name: str = None) -> bool:
        """Validate timeframe format."""
        valid_timeframes = [
            "1min", "5min", "15min", "30min", "1h", "4h", "1day",
            "1m", "5m", "15m", "30m", "1d"
        ]
        
        if timeframe not in valid_timeframes:
            self.result.add_error(f"Invalid timeframe: {timeframe}. Valid: {', '.join(valid_timeframes)}")
            return False
        
        return True


def validate_complete_configuration() -> ValidationResult:
    """
    Perform comprehensive configuration validation including environment variables,
    runtime dependencies, and configuration consistency.
    
    Returns:
        ValidationResult with comprehensive validation status
    """
    validator = EnhancedConfigValidator()
    final_result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
    
    try:
        # 1. Validate environment variables
        logger.info("Validating environment variables...")
        env_result = validator.validate_environment_variables()
        final_result.merge(env_result)
        
        # 2. Validate runtime dependencies
        logger.info("Validating runtime dependencies...")
        runtime_result = validator.validate_runtime_dependencies()
        final_result.merge(runtime_result)
        
        # 3. Validate configuration consistency
        logger.info("Validating configuration consistency...")
        consistency_result = validator.validate_configuration_consistency(None)
        final_result.merge(consistency_result)
        
        # Log summary
        if final_result.errors:
            logger.error(f"Configuration validation failed with {len(final_result.errors)} errors")
            for error in final_result.errors:
                logger.error(f"  ❌ {error}")
        
        if final_result.warnings:
            logger.warning(f"Configuration validation has {len(final_result.warnings)} warnings")
            for warning in final_result.warnings:
                logger.warning(f"  ⚠️  {warning}")
        
        if final_result.info:
            logger.info("Configuration validation info:")
            for info in final_result.info:
                logger.info(f"  ℹ️  {info}")
        
        if final_result.is_valid:
            logger.info("✅ Configuration validation passed")
        else:
            logger.error("❌ Configuration validation failed")
        
        return final_result
        
    except Exception as e:
        logger.error(f"Configuration validation exception: {e}")
        final_result.add_error(f"Validation exception: {str(e)}")
        return final_result


def validate_config(config) -> ValidationResult:
    """
    Validate application configuration (legacy function for backward compatibility).
    
    Args:
        config: AppConfig instance to validate
        
    Returns:
        ValidationResult with validation status and messages
    """
    # For backward compatibility, use the enhanced validator
    return validate_complete_configuration()


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
    
    # Add environment variables (mask sensitive ones)
    validator = EnhancedConfigValidator()
    for env_var in validator.env_vars:
        value = os.getenv(env_var.name, env_var.default)
        if value:
            report['environment_variables'][env_var.name] = {
                'set': True,
                'value': '***MASKED***' if env_var.sensitive else value,
                'description': env_var.description
            }
        else:
            report['environment_variables'][env_var.name] = {
                'set': False,
                'required': env_var.required,
                'description': env_var.description
            }
    
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


# Additional validation methods that were orphaned but should be in a validator class
class ConfigValidator:
    """Basic configuration validator for backward compatibility."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_api_key_format(self, api_key: str, source: str) -> bool:
        """
        Validate API key format.
        
        Args:
            api_key: API key to validate
            source: Source name (for error messages)
            
        Returns:
            True if valid, False otherwise
        """
        if not api_key or not api_key.strip():
            self.warnings.append(f"{source} API key is empty or missing")
            return False
        
        # Remove whitespace
        api_key = api_key.strip()
        
        # Basic format validation
        if len(api_key) < 10:
            self.errors.append(f"{source} API key too short (minimum 10 characters)")
            return False
        
        if len(api_key) > 200:
            self.errors.append(f"{source} API key too long (maximum 200 characters)")
            return False
        
        # Check for obvious test/demo keys
        test_patterns = [
            "test", "demo", "sandbox", "example", "your_api_key",
            "replace_me", "insert_here", "api_key_here"
        ]
        
        for pattern in test_patterns:
            if pattern.lower() in api_key.lower():
                self.warnings.append(f"{source} API key appears to be a placeholder/test key")
                break
        
        return True
    
    def validate_trading_symbol(self, symbol: str) -> bool:
        """
        Validate trading symbol format.
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not symbol or not symbol.strip():
            self.errors.append("Trading symbol cannot be empty")
            return False
        
        symbol = symbol.strip().upper()
        
        # Common symbol patterns
        patterns = [
            r'^[A-Z]{3,6}/[A-Z]{3,4}$',  # Forex/Crypto format (e.g., XAU/USD, BTC/USD)
            r'^[A-Z]{3,10}USDT?$',       # Crypto format (e.g., BTCUSDT, ETHUSDT)
            r'^[A-Z]{2,6}$',             # Stock format (e.g., AAPL, MSFT)
        ]
        
        valid_format = any(re.match(pattern, symbol) for pattern in patterns)
        
        if not valid_format:
            self.errors.append(f"Invalid symbol format: {symbol}. Expected formats: XAU/USD, BTCUSDT, or AAPL")
            return False
        
        return True
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """
        Validate timeframe format.
        
        Args:
            timeframe: Timeframe to validate
            
        Returns:
            True if valid, False otherwise
        """
        valid_timeframes = [
            "1min", "5min", "15min", "30min", "1h", "4h", "1day",
            "1m", "5m", "15m", "30m", "1d"
        ]
        
        if timeframe not in valid_timeframes:
            self.errors.append(f"Invalid timeframe: {timeframe}. Valid options: {', '.join(valid_timeframes)}")
            return False
        
        return True
    
    def validate_risk_parameters(self, risk_percentage: float, stop_loss_pips: int, take_profit_pips: int) -> bool:
        """
        Validate risk management parameters.
        
        Args:
            risk_percentage: Risk percentage per trade
            stop_loss_pips: Stop loss in pips
            take_profit_pips: Take profit in pips
            
        Returns:
            True if valid, False otherwise
        """
        valid = True
        
        # Risk percentage validation
        if risk_percentage <= 0:
            self.errors.append("Risk percentage must be positive")
            valid = False
        elif risk_percentage > 10:
            self.errors.append("Risk percentage cannot exceed 10% (extremely dangerous)")
            valid = False
        elif risk_percentage > 5:
            self.warnings.append("Risk percentage > 5% is considered high risk")
        
        # Stop loss validation
        if stop_loss_pips <= 0:
            self.errors.append("Stop loss pips must be positive")
            valid = False
        elif stop_loss_pips < 5:
            self.warnings.append("Stop loss < 5 pips may be too tight (frequent stop-outs)")
        elif stop_loss_pips > 200:
            self.warnings.append("Stop loss > 200 pips is very wide")
        
        # Take profit validation
        if take_profit_pips <= 0:
            self.errors.append("Take profit pips must be positive")
            valid = False
        elif take_profit_pips < stop_loss_pips:
            self.warnings.append("Take profit is smaller than stop loss (negative risk:reward ratio)")
        elif take_profit_pips > stop_loss_pips * 10:
            self.warnings.append("Take profit is very large compared to stop loss (may rarely be hit)")
        
        return valid
    
    def validate_model_parameters(self, models_to_use: List[str], confidence_threshold: float) -> bool:
        """
        Validate model configuration parameters.
        
        Args:
            models_to_use: List of models to use
            confidence_threshold: Confidence threshold for trading signals
            
        Returns:
            True if valid, False otherwise
        """
        valid = True
        
        # Validate models list
        valid_models = ["lstm", "xgb", "cnn", "svc", "nb", "meta"]
        
        if not models_to_use:
            self.errors.append("At least one model must be specified")
            valid = False
        else:
            for model in models_to_use:
                if model not in valid_models:
                    self.errors.append(f"Invalid model: {model}. Valid models: {', '.join(valid_models)}")
                    valid = False
        
        # Validate confidence threshold
        if not 0.5 <= confidence_threshold <= 1.0:
            self.errors.append("Confidence threshold must be between 0.5 and 1.0")
            valid = False
        elif confidence_threshold < 0.6:
            self.warnings.append("Confidence threshold < 0.6 may generate too many false signals")
        
        return valid
    
    def validate_api_consistency(self, api_config) -> bool:
        """
        Validate API configuration consistency.
        
        Args:
            api_config: APIConfig instance
            
        Returns:
            True if valid, False otherwise
        """
        valid = True
        
        # Check if at least one API is configured
        if not api_config.has_twelve_data and not api_config.has_binance:
            self.errors.append("At least one API (Twelve Data or Binance) must be configured")
            valid = False
        
        # Validate Twelve Data API key format
        if api_config.twelve_data_key:
            self.validate_api_key_format(api_config.twelve_data_key, "Twelve Data")
        
        # Validate Binance API credentials
        if api_config.binance_api_key and api_config.binance_secret:
            self.validate_api_key_format(api_config.binance_api_key, "Binance API Key")
            self.validate_api_key_format(api_config.binance_secret, "Binance Secret")
            
            # Warn about testnet usage in production
            if not api_config.use_testnet:
                self.warnings.append("Using Binance mainnet - ensure this is intended for production trading")
        
        return valid
    
    def validate_environment_consistency(self, app_config) -> bool:
        """
        Validate environment-specific configuration consistency.
        
        Args:
            app_config: AppConfig instance
            
        Returns:
            True if valid, False otherwise
        """
        valid = True
        
        # Production environment checks
        if app_config.environment == "production":
            if app_config.debug:
                self.warnings.append("Debug mode enabled in production environment")
            
            if app_config.log_level == "DEBUG":
                self.warnings.append("Debug logging enabled in production environment")
            
            if app_config.api.use_testnet:
                self.errors.append("Cannot use testnet in production environment")
                valid = False
        
        # Development environment recommendations
        elif app_config.environment == "development":
            if not app_config.api.use_testnet and app_config.api.has_binance:
                self.warnings.append("Consider using testnet in development environment")
        
        return valid


def validate_config(config) -> ValidationResult:
    """
    Validate complete application configuration.
    
    Args:
        config: AppConfig instance to validate
        
    Returns:
        ValidationResult with validation status and messages
    """
    validator = ConfigValidator()
    
    try:
        # Validate trading configuration
        validator.validate_trading_symbol(config.trading.symbol)
        validator.validate_timeframe(config.trading.timeframe)
        validator.validate_risk_parameters(
            config.trading.risk_percentage,
            config.trading.stop_loss_pips,
            config.trading.take_profit_pips
        )
        
        # Validate API configuration
        validator.validate_api_consistency(config.api)
        
        # Validate model configuration
        validator.validate_model_parameters(
            config.model.models_to_use,
            config.model.confidence_threshold
        )
        
        # Validate environment consistency
        validator.validate_environment_consistency(config)
        
        # Additional business rule validations
        if config.trading.account_balance < 100:
            validator.warnings.append("Account balance < $100 may not be sufficient for meaningful trading")
        
        if config.trading.max_positions > 10:
            validator.warnings.append("Max positions > 10 may be difficult to manage")
        
        # Log validation results
        if validator.errors:
            logger.error(f"Configuration validation errors: {', '.join(validator.errors)}")
        
        if validator.warnings:
            logger.warning(f"Configuration validation warnings: {', '.join(validator.warnings)}")
        
        is_valid = len(validator.errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=validator.errors,
            warnings=validator.warnings
        )
    
    except Exception as e:
        logger.error(f"Configuration validation failed with exception: {e}")
        return ValidationResult(
            is_valid=False,
            errors=[f"Validation exception: {str(e)}"],
            warnings=[]
        )


def validate_api_key_format(api_key: str, source: str) -> ValidationResult:
    """
    Validate a single API key format.
    
    Args:
        api_key: API key to validate
        source: Source name for error messages
        
    Returns:
        ValidationResult
    """
    validator = ConfigValidator()
    is_valid = validator.validate_api_key_format(api_key, source)
    
    return ValidationResult(
        is_valid=is_valid,
        errors=validator.errors,
        warnings=validator.warnings
    )