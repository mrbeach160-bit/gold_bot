import os
import sys
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET

# Add project root to path for config import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import get_config

# Configuration-based endpoints
def get_binance_endpoints():
    """Get Binance endpoints based on configuration."""
    try:
        config = get_config()
        if config.api.use_testnet:
            return "https://testnet.binancefuture.com", "wss://stream.binancefuture.com/stream"
        else:
            return "https://fapi.binance.com", "wss://fstream.binance.com/stream"
    except RuntimeError:
        # Fallback to testnet if config not available
        return "https://testnet.binancefuture.com", "wss://stream.binancefuture.com/stream"

TESTNET_URL, STREAM_URL = get_binance_endpoints()

def init_testnet_client(api_key: str = None, api_secret: str = None) -> Client:
    """
    Inisialisasi client Futures dengan integrasi configuration system.
    
    Args:
        api_key: API key (optional, will use config if not provided)
        api_secret: API secret (optional, will use config if not provided)
    """
    # Get configuration values if not provided
    try:
        config = get_config()
        
        if api_key is None:
            api_key = config.api.binance_api_key
            if not api_key:
                raise ValueError("Binance API key not configured. Please set BINANCE_API_KEY environment variable or pass api_key parameter.")
        
        if api_secret is None:
            api_secret = config.api.binance_secret
            if not api_secret:
                raise ValueError("Binance API secret not configured. Please set BINANCE_API_SECRET environment variable or pass api_secret parameter.")
        
        # Use configured testnet setting
        use_testnet = config.api.use_testnet
        
    except RuntimeError:
        # Configuration not loaded, use parameters as provided
        if api_key is None:
            raise ValueError("API key is required when configuration is not loaded")
        if api_secret is None:
            raise ValueError("API secret is required when configuration is not loaded")
        
        # Default to testnet for safety
        use_testnet = True
    
    client = Client(api_key, api_secret)
    
    # Override endpoint REST/WS based on configuration
    if use_testnet:
        client.FUTURES_URL = f"https://testnet.binancefuture.com/fapi"
        client.FUTURES_STREAM_URL = "wss://stream.binancefuture.com/stream"
    else:
        client.FUTURES_URL = f"https://fapi.binance.com/fapi"
        client.FUTURES_STREAM_URL = "wss://fstream.binance.com/stream"
    
    return client

def place_market_order(
    client: Client,
    symbol: str,
    side: str,
    qty: float,
    leverage: int = None
) -> dict:
    """
    Kirim order MARKET di akun UM Futures dengan integrasi configuration system.
    side: 'BUY' = long, 'SELL' = short
    
    Args:
        client: Binance client
        symbol: Trading symbol
        side: Order side ('BUY' or 'SELL')
        qty: Quantity
        leverage: Leverage (optional, will use config if not provided)
    """
    # Get leverage from config if not provided
    if leverage is None:
        try:
            config = get_config()
            leverage = config.trading.leverage
        except RuntimeError:
            leverage = 20  # Default leverage
    
    # Pastikan leverage sudah disetel
    client.futures_change_leverage(symbol=symbol, leverage=leverage)

    # Kirim order MARKET
    order = client.futures_create_order(
        symbol=symbol,
        side=SIDE_BUY if side == "BUY" else SIDE_SELL,
        type=ORDER_TYPE_MARKET,
        quantity=qty
    )
    return order
