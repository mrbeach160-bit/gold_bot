from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET

TESTNET_URL = "https://testnet.binancefuture.com"
STREAM_URL = "wss://stream.binancefuture.com/stream"

def init_testnet_client(api_key: str, api_secret: str) -> Client:
    """
    Inisialisasi client Futures testnet.
    """
    client = Client(api_key, api_secret)
    # Override endpoint REST/WS ke testnet Futures
    client.FUTURES_URL = f"{TESTNET_URL}/fapi"
    client.FUTURES_STREAM_URL = STREAM_URL
    return client

def place_market_order(
    client: Client,
    symbol: str,
    side: str,
    qty: float,
    leverage: int = 20
) -> dict:
    """
    Kirim order MARKET di akun UM Futures (testnet).
    side: 'BUY' = long, 'SELL' = short
    """
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
