# Configuration System Usage Examples

This document demonstrates how to use the new centralized configuration management system in the Gold Bot trading application.

## Quick Start

### 1. Basic Setup with Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit with your actual values
TWELVE_DATA_API_KEY=your_twelve_data_api_key
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_secret
ENVIRONMENT=development
DEBUG=true
DEFAULT_SYMBOL=XAU/USD
```

### 2. Using Configuration in Python Code

```python
from config import ConfigManager, AppConfig, get_config

# Initialize configuration manager
config_manager = ConfigManager()

# Load configuration from environment variables
app_config = AppConfig.from_env()
config_manager.load_config(app_config)

# Get configuration anywhere in your code
config = get_config()
print(f"Trading symbol: {config.trading.symbol}")
print(f"API timeout: {config.api.api_timeout}")
print(f"Debug mode: {config.debug}")
```

### 3. Configuration-Aware Data Fetching

```python
from utils.data import get_gold_data

# Old way (still supported for backward compatibility)
data = get_gold_data("your_api_key", "5min", "XAU/USD", 1000)

# New way - uses configuration automatically
data = get_gold_data(outputsize=1000)  # Uses config for API key, symbol, timeframe

# Override specific values
data = get_gold_data(symbol="BTC/USD", outputsize=500)  # Uses config for API key and timeframe
```

## Advanced Usage

### 1. Custom Configuration

```python
from config import AppConfig, TradingConfig, APIConfig, ModelConfig

# Create custom configuration
custom_config = AppConfig(
    trading=TradingConfig(
        symbol="BTC/USD",
        timeframe="1h",
        risk_percentage=2.0,
        stop_loss_pips=30,
        take_profit_pips=60
    ),
    api=APIConfig(
        twelve_data_key="your_key",
        use_testnet=True
    ),
    model=ModelConfig(
        confidence_threshold=0.7,
        models_to_use=["lstm", "xgb", "meta"]
    )
)

# Load custom configuration
config_manager.load_config(custom_config)
```

### 2. Runtime Configuration Updates

```python
from config import update_config

# Update trading parameters
update_config(
    trading_symbol="EUR/USD",
    trading_risk_percentage=1.5,
    model_confidence_threshold=0.65
)

# Update API settings
update_config(api_timeout=45, api_max_retries=5)
```

### 3. Configuration Validation

```python
from config import validate_config, AppConfig

config = AppConfig.from_env()
validation_result = validate_config(config)

if not validation_result.is_valid:
    print("Configuration errors:")
    for error in validation_result.errors:
        print(f"  - {error}")

if validation_result.warnings:
    print("Configuration warnings:")
    for warning in validation_result.warnings:
        print(f"  - {warning}")
```

### 4. Configuration Callbacks

```python
def on_config_update(config):
    print(f"Configuration updated! New symbol: {config.trading.symbol}")

# Register callback
config_manager.register_update_callback("my_callback", on_config_update)

# Update will trigger callback
update_config(trading_symbol="GBP/USD")
```

## Training Models with Configuration

### 1. Using train_models.py with Configuration

```bash
# Traditional way (still supported)
python train_models.py --symbol "XAU/USD" --timeframe "5m" --apikey "your_key"

# New way - uses environment variables
export TWELVE_DATA_API_KEY="your_key"
python train_models.py --symbol "XAU/USD" --timeframe "5m"

# With custom data size
python train_models.py --symbol "BTC/USD" --timeframe "1h" --data-size 10000
```

### 2. Programmatic Model Training

```python
from config import ConfigManager, AppConfig, TradingConfig, APIConfig
from utils.model_manager import train_and_save_all_models
from utils.data import get_gold_data

# Setup configuration
config = AppConfig(
    trading=TradingConfig(symbol="XAU/USD", timeframe="5min"),
    api=APIConfig(twelve_data_key="your_key")
)

config_manager = ConfigManager()
config_manager.load_config(config)

# Get data using configuration
data = get_gold_data(outputsize=5000)

# Train models
if data is not None and len(data) > 60:
    train_and_save_all_models(data, "5m")
```

## Environment-Specific Configurations

### Development Environment

```bash
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
USE_TESTNET=true
DEFAULT_RISK_PERCENTAGE=0.5
```

### Production Environment

```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
USE_TESTNET=false
DEFAULT_RISK_PERCENTAGE=1.0
API_TIMEOUT=60
```

## Migration from Legacy Code

### Before (Legacy)

```python
# Hardcoded values scattered throughout the code
api_key = "your_api_key"
symbol = "XAU/USD"
timeframe = "5min"
risk_percent = 1.0

data = get_gold_data(api_key, timeframe, symbol, 1000)
```

### After (Configuration System)

```python
# Centralized configuration
from config import get_config
from utils.data import get_gold_data

config = get_config()
data = get_gold_data(outputsize=1000)  # Uses config automatically
```

## Security Best Practices

1. **Never commit .env files** - They contain sensitive API keys
2. **Use different API keys for different environments**
3. **Set API key permissions appropriately** (read-only for market data)
4. **Enable IP restrictions** on your API keys when possible
5. **Use testnet for development** and testing
6. **Regularly rotate API keys**
7. **Monitor API usage** and set up alerts

## Troubleshooting

### Common Issues

1. **"Configuration not loaded" error**
   ```python
   # Solution: Initialize configuration first
   from config import load_config
   load_config()
   ```

2. **"API key not configured" error**
   ```bash
   # Solution: Set environment variable or use .env file
   export TWELVE_DATA_API_KEY="your_key"
   ```

3. **Configuration validation failures**
   ```python
   # Solution: Check validation messages
   from config import validate_config, AppConfig
   result = validate_config(AppConfig.from_env())
   print(result.errors)
   ```

### Testing Configuration

```python
# Test configuration loading
python test_config.py

# Test migrated modules
python test_migration.py

# Test with custom environment
ENVIRONMENT=staging python test_config.py
```

## API Reference

### Configuration Classes

- `TradingConfig`: Trading-related parameters (symbol, timeframe, risk management)
- `APIConfig`: API credentials and settings
- `ModelConfig`: Machine learning model configuration
- `AppConfig`: Main configuration container

### Manager Functions

- `ConfigManager()`: Singleton configuration manager
- `get_config()`: Get current configuration
- `load_config(config)`: Load configuration
- `update_config(**kwargs)`: Update configuration at runtime
- `validate_config(config)`: Validate configuration

### Environment Variables

See `.env.example` for a complete list of supported environment variables.