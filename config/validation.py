# config/validation.py
"""
Configuration validation system with type checking and business rule validation.
"""

from dataclasses import dataclass
from typing import List, Optional, Any
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ConfigValidator:
    """Configuration validator with comprehensive validation rules."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_api_key_format(self, api_key: Optional[str], source: str) -> bool:
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