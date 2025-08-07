# config/settings.py
"""
Type-safe configuration classes using dataclasses for the Gold Bot application.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class TradingConfig:
    """Trading-related configuration parameters."""
    
    symbol: str = field(default="XAU/USD")
    timeframe: str = field(default="5min")
    risk_percentage: float = field(default=1.0)
    max_positions: int = field(default=3)
    stop_loss_pips: int = field(default=20)
    take_profit_pips: int = field(default=40)
    leverage: int = field(default=20)
    use_ai_take_profit: bool = field(default=True)
    minimum_confidence: float = field(default=0.55)
    account_balance: float = field(default=1000.0)
    
    def __post_init__(self):
        """Validate trading configuration after initialization."""
        if not 0.1 <= self.risk_percentage <= 10.0:
            raise ValueError("Risk percentage must be between 0.1 and 10.0")
        if not 5 <= self.stop_loss_pips <= 200:
            raise ValueError("Stop loss pips must be between 5 and 200")
        if not 5 <= self.take_profit_pips <= 500:
            raise ValueError("Take profit pips must be between 5 and 500")
        if not 0.5 <= self.minimum_confidence <= 1.0:
            raise ValueError("Minimum confidence must be between 0.5 and 1.0")


@dataclass
class APIConfig:
    """API configuration for external services."""
    
    twelve_data_key: Optional[str] = field(default=None)
    binance_api_key: Optional[str] = field(default=None)
    binance_secret: Optional[str] = field(default=None)
    use_testnet: bool = field(default=True)
    api_timeout: int = field(default=30)
    max_retries: int = field(default=3)
    
    def __post_init__(self):
        """Load API keys from environment variables if not provided."""
        if self.twelve_data_key is None:
            self.twelve_data_key = os.getenv("TWELVE_DATA_API_KEY")
        
        if self.binance_api_key is None:
            self.binance_api_key = os.getenv("BINANCE_API_KEY")
        
        if self.binance_secret is None:
            self.binance_secret = os.getenv("BINANCE_API_SECRET")
        
        if self.use_testnet is None:
            self.use_testnet = os.getenv("USE_TESTNET", "true").lower() == "true"
    
    @property
    def has_twelve_data(self) -> bool:
        """Check if Twelve Data API key is available."""
        return self.twelve_data_key is not None and len(self.twelve_data_key.strip()) > 0
    
    @property
    def has_binance(self) -> bool:
        """Check if Binance API credentials are available."""
        return (self.binance_api_key is not None and 
                self.binance_secret is not None and
                len(self.binance_api_key.strip()) > 0 and
                len(self.binance_secret.strip()) > 0)


@dataclass
class ModelConfig:
    """Enhanced machine learning model configuration with tuning parameters."""
    
    models_to_use: List[str] = field(default_factory=lambda: [
        "lstm", "xgb", "cnn", "svc", "nb", "meta"
    ])
    ensemble_method: str = field(default="meta_learner")
    retrain_interval: int = field(default=7)  # days
    confidence_threshold: float = field(default=0.6)
    
    # Enhanced tuning parameters
    enable_hyperparameter_tuning: bool = field(default=True)
    enable_meta_learner_tuning: bool = field(default=True)
    
    # LSTM enhanced parameters
    lstm_sequence_length: int = field(default=60)
    lstm_epochs: int = field(default=50)  # Enhanced from 10
    lstm_dropout: float = field(default=0.3)
    lstm_use_batch_norm: bool = field(default=True)
    lstm_use_early_stopping: bool = field(default=True)
    lstm_patience: int = field(default=10)
    lstm_units: List[int] = field(default_factory=lambda: [50, 50])
    
    # CNN enhanced parameters  
    cnn_sequence_length: int = field(default=30)  # Optimized for CNN
    cnn_epochs: int = field(default=30)
    cnn_dropout: float = field(default=0.4)
    cnn_filter_sizes: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    cnn_use_global_pooling: bool = field(default=True)
    cnn_use_batch_norm: bool = field(default=True)
    
    # XGBoost enhanced parameters
    xgb_max_depth: int = field(default=6)  # Enhanced from 3
    xgb_n_estimators: int = field(default=300)  # Enhanced from 100
    xgb_learning_rate: float = field(default=0.05)  # Enhanced from 0.1
    xgb_subsample: float = field(default=0.8)
    xgb_colsample_bytree: float = field(default=0.8)
    xgb_reg_alpha: float = field(default=0.1)
    xgb_reg_lambda: float = field(default=1.0)
    
    # SVC enhanced parameters
    svc_c_range: List[float] = field(default_factory=lambda: [0.1, 1, 10, 100])
    svc_gamma_range: List[str] = field(default_factory=lambda: ['scale', 'auto'])
    svc_kernels: List[str] = field(default_factory=lambda: ['rbf', 'poly', 'linear'])
    svc_enable_grid_search: bool = field(default=True)
    
    # Meta-learner enhanced parameters
    meta_n_iter: int = field(default=50)  # For RandomizedSearchCV
    meta_cv_folds: int = field(default=3)
    meta_scoring: str = field(default='f1_macro')
    
    # Training parameters
    train_test_split: float = field(default=0.8)
    validation_split: float = field(default=0.2)
    batch_size: int = field(default=32)
    epochs: int = field(default=50)  # Enhanced default
    
    def __post_init__(self):
        """Validate enhanced model configuration."""
        valid_ensemble_methods = ["meta_learner", "voting", "averaging"]
        if self.ensemble_method not in valid_ensemble_methods:
            raise ValueError(f"Ensemble method must be one of {valid_ensemble_methods}")
        
        if not 0.5 <= self.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.5 and 1.0")
        
        # Validate LSTM parameters
        if not 10 <= self.lstm_sequence_length <= 100:
            raise ValueError("LSTM sequence length must be between 10 and 100")
        if not 0.1 <= self.lstm_dropout <= 0.8:
            raise ValueError("LSTM dropout must be between 0.1 and 0.8")
        
        # Validate CNN parameters
        if not 10 <= self.cnn_sequence_length <= 50:
            raise ValueError("CNN sequence length must be between 10 and 50")
        if not 0.1 <= self.cnn_dropout <= 0.8:
            raise ValueError("CNN dropout must be between 0.1 and 0.8")
        
        # Validate XGBoost parameters
        if not 3 <= self.xgb_max_depth <= 15:
            raise ValueError("XGBoost max_depth must be between 3 and 15")
        if not 0.01 <= self.xgb_learning_rate <= 0.3:
            raise ValueError("XGBoost learning_rate must be between 0.01 and 0.3")


@dataclass
class AppConfig:
    """Main application configuration containing all sub-configurations."""
    
    # Sub-configurations
    trading: TradingConfig = field(default_factory=TradingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Application-specific settings
    debug: bool = field(default=False)
    log_level: str = field(default="INFO")
    environment: str = field(default="development")
    
    # Data settings
    default_data_source: str = field(default="Twelve Data")
    data_cache_ttl: int = field(default=300)  # seconds
    max_data_points: int = field(default=5000)
    
    # WebSocket settings
    enable_websocket: bool = field(default=True)
    websocket_timeout: int = field(default=30)
    websocket_retry_attempts: int = field(default=3)
    
    # UI settings
    page_title: str = field(default="Gold Bot Trading System")
    theme: str = field(default="dark")
    auto_refresh_interval: int = field(default=30)  # seconds
    
    def __post_init__(self):
        """Load configuration from environment variables and validate."""
        # Load environment-specific settings
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.environment = os.getenv("ENVIRONMENT", "development").lower()
        
        # Load data source preference
        self.default_data_source = os.getenv("DEFAULT_DATA_SOURCE", "Twelve Data")
        
        # Validate settings
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Log level must be one of {valid_log_levels}")
        
        valid_environments = ["development", "staging", "production"]
        if self.environment not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        
        valid_data_sources = ["Twelve Data", "Binance"]
        if self.default_data_source not in valid_data_sources:
            raise ValueError(f"Data source must be one of {valid_data_sources}")
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "trading": {
                "symbol": self.trading.symbol,
                "timeframe": self.trading.timeframe,
                "risk_percentage": self.trading.risk_percentage,
                "max_positions": self.trading.max_positions,
                "stop_loss_pips": self.trading.stop_loss_pips,
                "take_profit_pips": self.trading.take_profit_pips,
                "leverage": self.trading.leverage,
                "use_ai_take_profit": self.trading.use_ai_take_profit,
                "minimum_confidence": self.trading.minimum_confidence,
                "account_balance": self.trading.account_balance,
            },
            "api": {
                "has_twelve_data": self.api.has_twelve_data,
                "has_binance": self.api.has_binance,
                "use_testnet": self.api.use_testnet,
                "api_timeout": self.api.api_timeout,
                "max_retries": self.api.max_retries,
            },
            "model": {
                "models_to_use": self.model.models_to_use,
                "ensemble_method": self.model.ensemble_method,
                "retrain_interval": self.model.retrain_interval,
                "confidence_threshold": self.model.confidence_threshold,
            },
            "app": {
                "debug": self.debug,
                "log_level": self.log_level,
                "environment": self.environment,
                "default_data_source": self.default_data_source,
                "enable_websocket": self.enable_websocket,
            }
        }