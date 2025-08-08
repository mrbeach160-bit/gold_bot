# config/manager.py
"""
ConfigManager - Singleton pattern for global configuration access.
Provides thread-safe configuration management with runtime updates.
"""

import threading
from typing import Optional, Dict, Any, Callable
from .settings import AppConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Singleton ConfigManager for global configuration access.
    Thread-safe implementation with configuration validation and update capabilities.
    """
    
    _instance: Optional['ConfigManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ConfigManager':
        """Ensure singleton instance creation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize ConfigManager if not already initialized."""
        if not hasattr(self, '_initialized'):
            self._config: Optional[AppConfig] = None
            self._update_callbacks: Dict[str, Callable] = {}
            self._config_lock = threading.RLock()
            self._initialized = True
            logger.info("ConfigManager initialized")
    
    def load_config(self, config: Optional[AppConfig] = None) -> AppConfig:
        """
        Load configuration into the manager.
        
        Args:
            config: Optional AppConfig instance. If None, loads from environment.
            
        Returns:
            The loaded AppConfig instance.
            
        Raises:
            ValueError: If configuration validation fails.
        """
        with self._config_lock:
            if config is None:
                config = AppConfig.from_env()
            
            # Validate configuration
            from .validation import validate_config
            validation_result = validate_config(config)
            
            if not validation_result.is_valid:
                error_msg = f"Configuration validation failed: {', '.join(validation_result.errors)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    logger.warning(f"Configuration warning: {warning}")
            
            self._config = config
            logger.info(f"Configuration loaded successfully for environment: {config.environment}")
            
            # Notify callbacks
            self._notify_update_callbacks()
            
            return self._config
    
    def get_config(self) -> AppConfig:
        """
        Get the current configuration.
        
        Returns:
            Current AppConfig instance.
            
        Raises:
            RuntimeError: If configuration has not been loaded.
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration values at runtime.
        
        Args:
            **kwargs: Configuration values to update using dot notation.
                     Example: trading_symbol="BTC/USD", api_timeout=60
        """
        with self._config_lock:
            if self._config is None:
                raise RuntimeError("Configuration not loaded. Call load_config() first.")
            
            for key, value in kwargs.items():
                self._update_nested_value(self._config, key, value)
            
            # Re-validate after updates
            from .validation import validate_config
            validation_result = validate_config(self._config)
            
            if not validation_result.is_valid:
                logger.error(f"Configuration update validation failed: {', '.join(validation_result.errors)}")
                raise ValueError(f"Configuration update validation failed: {', '.join(validation_result.errors)}")
            
            logger.info(f"Configuration updated: {kwargs}")
            self._notify_update_callbacks()
    
    def _update_nested_value(self, config: AppConfig, key: str, value: Any) -> None:
        """Update nested configuration value using dot notation."""
        if '.' in key:
            section, sub_key = key.split('.', 1)
            section_obj = getattr(config, section)
            self._update_nested_value(section_obj, sub_key, value)
        else:
            if '_' in key:
                # Handle underscore notation (e.g., trading_symbol -> trading.symbol)
                parts = key.split('_', 1)
                if hasattr(config, parts[0]):
                    section_obj = getattr(config, parts[0])
                    if hasattr(section_obj, parts[1]):
                        setattr(section_obj, parts[1], value)
                        return
            
            # Direct attribute update
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
    
    def register_update_callback(self, name: str, callback: Callable[[AppConfig], None]) -> None:
        """
        Register a callback to be called when configuration is updated.
        
        Args:
            name: Unique name for the callback.
            callback: Function to call with updated config.
        """
        self._update_callbacks[name] = callback
        logger.debug(f"Registered update callback: {name}")
    
    def unregister_update_callback(self, name: str) -> None:
        """
        Unregister an update callback.
        
        Args:
            name: Name of the callback to remove.
        """
        if name in self._update_callbacks:
            del self._update_callbacks[name]
            logger.debug(f"Unregistered update callback: {name}")
    
    def _notify_update_callbacks(self) -> None:
        """Notify all registered callbacks of configuration updates."""
        for name, callback in self._update_callbacks.items():
            try:
                callback(self._config)
            except Exception as e:
                logger.error(f"Error calling update callback '{name}': {e}")
    
    def get_trading_config(self) -> 'TradingConfig':
        """Get trading configuration section."""
        return self.get_config().trading
    
    def get_api_config(self) -> 'APIConfig':
        """Get API configuration section."""
        return self.get_config().api
    
    def get_model_config(self) -> 'ModelConfig':
        """Get model configuration section."""
        return self.get_config().model
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get_config().debug
    
    def get_environment(self) -> str:
        """Get current environment."""
        return self.get_config().environment
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.get_environment() == "production"
    
    def get_default_data_source(self) -> str:
        """Get the default data source."""
        return self.get_config().default_data_source
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert current configuration to dictionary."""
        if self._config is None:
            return {}
        return self._config.to_dict()
    
    def reset(self) -> None:
        """Reset configuration (mainly for testing)."""
        with self._config_lock:
            self._config = None
            self._update_callbacks.clear()
            logger.info("Configuration reset")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """
    Convenience function to get the current configuration.
    
    Returns:
        Current AppConfig instance.
    """
    return config_manager.get_config()


def update_config(**kwargs) -> None:
    """
    Convenience function to update configuration.
    
    Args:
        **kwargs: Configuration values to update.
    """
    config_manager.update_config(**kwargs)


def load_config(config: Optional[AppConfig] = None) -> AppConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config: Optional AppConfig instance.
        
    Returns:
        Loaded AppConfig instance.
    """
    return config_manager.load_config(config)