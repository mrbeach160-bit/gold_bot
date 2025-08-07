# trading/engine.py
"""
Trading Execution Engine for Gold Bot

This module provides:
- Order execution (market, limit, stop orders)
- Position management (open, close, modify positions)
- Account information (balance, equity, margin)
- Connection management and error handling

Migrated and enhanced from utils/binance_trading.py
"""

import os
import sys
from typing import Optional, Dict, Any, List
from decimal import Decimal
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import get_config

# Import trading libraries with error handling
try:
    from binance.client import Client
    from binance.enums import (
        SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT, 
        ORDER_TYPE_STOP_MARKET, TIME_IN_FORCE_GTC
    )
    HAS_BINANCE = True
except ImportError:
    print("Warning: python-binance not installed, trading functionality limited")
    HAS_BINANCE = False


class TradingEngine:
    """Core trading execution engine"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, use_testnet: bool = None):
        """
        Initialize trading engine
        
        Args:
            api_key: Binance API key (optional, uses config if not provided)
            api_secret: Binance API secret (optional, uses config if not provided)  
            use_testnet: Whether to use testnet (optional, uses config if not provided)
        """
        self.config = None
        try:
            self.config = get_config()
        except RuntimeError:
            pass
        
        # Initialize connection parameters
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet
        
        if self.config:
            if self.api_key is None:
                self.api_key = self.config.api.binance_api_key
            if self.api_secret is None:
                self.api_secret = self.config.api.binance_secret
            if self.use_testnet is None:
                self.use_testnet = self.config.api.use_testnet
        
        # Default to testnet for safety
        if self.use_testnet is None:
            self.use_testnet = True
        
        # Initialize client
        self.client = None
        self._initialize_client()
        
        # Trading state
        self.is_connected = False
        self.last_error = None
    
    def _initialize_client(self):
        """Initialize Binance client with proper configuration"""
        if not HAS_BINANCE:
            raise RuntimeError("python-binance not installed")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret are required")
        
        try:
            self.client = Client(self.api_key, self.api_secret)
            
            # Configure endpoints based on testnet setting
            if self.use_testnet:
                self.client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
                self.client.FUTURES_STREAM_URL = "wss://stream.binancefuture.com/stream"
            else:
                self.client.FUTURES_URL = "https://fapi.binance.com/fapi"
                self.client.FUTURES_STREAM_URL = "wss://fstream.binance.com/stream"
            
            # Test connection
            self.client.futures_ping()
            self.is_connected = True
            
        except Exception as e:
            self.last_error = str(e)
            self.is_connected = False
            raise RuntimeError(f"Failed to initialize trading client: {e}")
    
    def check_connection(self) -> bool:
        """Check if connection to exchange is active"""
        if not self.client:
            return False
        
        try:
            self.client.futures_ping()
            self.is_connected = True
            return True
        except Exception as e:
            self.last_error = str(e)
            self.is_connected = False
            return False
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information including balance and positions"""
        if not self._ensure_connection():
            return None
        
        try:
            account = self.client.futures_account()
            
            # Extract key information
            info = {
                'total_wallet_balance': float(account['totalWalletBalance']),
                'total_unrealized_pnl': float(account['totalUnrealizedPnL']),
                'total_margin_balance': float(account['totalMarginBalance']),
                'total_position_initial_margin': float(account['totalPositionInitialMargin']),
                'total_open_order_initial_margin': float(account['totalOpenOrderInitialMargin']),
                'available_balance': float(account['availableBalance']),
                'max_withdraw_amount': float(account['maxWithdrawAmount']),
                'assets': [],
                'positions': []
            }
            
            # Process assets
            for asset in account['assets']:
                if float(asset['walletBalance']) > 0:
                    info['assets'].append({
                        'asset': asset['asset'],
                        'wallet_balance': float(asset['walletBalance']),
                        'unrealized_profit': float(asset['unrealizedProfit']),
                        'margin_balance': float(asset['marginBalance']),
                        'maint_margin': float(asset['maintMargin']),
                        'initial_margin': float(asset['initialMargin'])
                    })
            
            # Process positions
            for position in account['positions']:
                if float(position['positionAmt']) != 0:
                    info['positions'].append({
                        'symbol': position['symbol'],
                        'position_amt': float(position['positionAmt']),
                        'entry_price': float(position['entryPrice']),
                        'mark_price': float(position['markPrice']),
                        'pnl': float(position['unRealizedProfit']),
                        'percentage': float(position['percentage']),
                        'isolated': position['isolated'],
                        'position_side': position['positionSide']
                    })
            
            return info
            
        except Exception as e:
            self.last_error = str(e)
            print(f"Error getting account info: {e}")
            return None
    
    def place_market_order(self, symbol: str, side: str, quantity: float, leverage: int = None) -> Optional[Dict[str, Any]]:
        """
        Place market order - migrated from utils/binance_trading.py
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            leverage: Leverage setting (optional, uses config if not provided)
            
        Returns:
            Order response dict or None if failed
        """
        if not self._ensure_connection():
            return None
        
        # Get leverage from config if not provided
        if leverage is None and self.config:
            leverage = self.config.trading.leverage
        if leverage is None:
            leverage = 20  # Default leverage
        
        try:
            # Set leverage
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            
            # Place market order
            order = self.client.futures_create_order(
                symbol=symbol,
                side=SIDE_BUY if side.upper() == "BUY" else SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            return self._standardize_order_response(order)
            
        except Exception as e:
            self.last_error = str(e)
            print(f"Error placing market order: {e}")
            return None
    
    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float, 
                         time_in_force: str = 'GTC') -> Optional[Dict[str, Any]]:
        """
        Place limit order
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            price: Limit price
            time_in_force: Time in force ('GTC', 'IOC', 'FOK')
            
        Returns:
            Order response dict or None if failed
        """
        if not self._ensure_connection():
            return None
        
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=SIDE_BUY if side.upper() == "BUY" else SIDE_SELL,
                type=ORDER_TYPE_LIMIT,
                quantity=quantity,
                price=price,
                timeInForce=TIME_IN_FORCE_GTC
            )
            
            return self._standardize_order_response(order)
            
        except Exception as e:
            self.last_error = str(e)
            print(f"Error placing limit order: {e}")
            return None
    
    def place_stop_order(self, symbol: str, side: str, quantity: float, stop_price: float) -> Optional[Dict[str, Any]]:
        """
        Place stop market order
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL' 
            quantity: Order quantity
            stop_price: Stop trigger price
            
        Returns:
            Order response dict or None if failed
        """
        if not self._ensure_connection():
            return None
        
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=SIDE_BUY if side.upper() == "BUY" else SIDE_SELL,
                type=ORDER_TYPE_STOP_MARKET,
                quantity=quantity,
                stopPrice=stop_price
            )
            
            return self._standardize_order_response(order)
            
        except Exception as e:
            self.last_error = str(e)
            print(f"Error placing stop order: {e}")
            return None
    
    def close_position(self, symbol: str, position_side: str = None) -> Optional[Dict[str, Any]]:
        """
        Close position for given symbol
        
        Args:
            symbol: Trading symbol
            position_side: Position side ('LONG' or 'SHORT') for hedge mode
            
        Returns:
            Order response dict or None if failed
        """
        if not self._ensure_connection():
            return None
        
        try:
            # Get current position
            positions = self.get_positions(symbol)
            if not positions:
                return None
            
            position = positions[0]  # Take first position
            position_amt = position['position_amt']
            
            if position_amt == 0:
                return {'status': 'NO_POSITION', 'message': 'No position to close'}
            
            # Determine side to close position
            close_side = 'SELL' if position_amt > 0 else 'BUY'
            close_quantity = abs(position_amt)
            
            return self.place_market_order(symbol, close_side, close_quantity)
            
        except Exception as e:
            self.last_error = str(e)
            print(f"Error closing position: {e}")
            return None
    
    def get_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Get current positions
        
        Args:
            symbol: Specific symbol to get position for (optional)
            
        Returns:
            List of position dicts
        """
        if not self._ensure_connection():
            return []
        
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            
            # Filter out zero positions and standardize
            active_positions = []
            for pos in positions:
                if float(pos['positionAmt']) != 0:
                    active_positions.append({
                        'symbol': pos['symbol'],
                        'position_amt': float(pos['positionAmt']),
                        'entry_price': float(pos['entryPrice']),
                        'mark_price': float(pos['markPrice']),
                        'pnl': float(pos['unRealizedProfit']),
                        'percentage': float(pos['percentage']),
                        'position_side': pos['positionSide'],
                        'isolated': pos['isolated'],
                        'leverage': int(pos['leverage'])
                    })
            
            return active_positions
            
        except Exception as e:
            self.last_error = str(e)
            print(f"Error getting positions: {e}")
            return []
    
    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Get open orders
        
        Args:
            symbol: Specific symbol to get orders for (optional)
            
        Returns:
            List of order dicts
        """
        if not self._ensure_connection():
            return []
        
        try:
            orders = self.client.futures_get_open_orders(symbol=symbol)
            
            standardized_orders = []
            for order in orders:
                standardized_orders.append(self._standardize_order_response(order))
            
            return standardized_orders
            
        except Exception as e:
            self.last_error = str(e)
            print(f"Error getting open orders: {e}")
            return []
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an open order
        
        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_connection():
            return False
        
        try:
            self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
            return True
            
        except Exception as e:
            self.last_error = str(e)
            print(f"Error canceling order: {e}")
            return False
    
    def _ensure_connection(self) -> bool:
        """Ensure connection is active, retry if needed"""
        if not self.is_connected:
            try:
                self._initialize_client()
            except Exception:
                return False
        
        return self.is_connected
    
    def _standardize_order_response(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize order response format"""
        return {
            'order_id': order.get('orderId'),
            'client_order_id': order.get('clientOrderId'),
            'symbol': order.get('symbol'),
            'side': order.get('side'),
            'type': order.get('type'),
            'quantity': float(order.get('origQty', 0)),
            'price': float(order.get('price', 0)),
            'stop_price': float(order.get('stopPrice', 0)),
            'status': order.get('status'),
            'time_in_force': order.get('timeInForce'),
            'timestamp': order.get('updateTime', order.get('transactTime')),
            'executed_qty': float(order.get('executedQty', 0)),
            'cumulative_quote_qty': float(order.get('cumQuoteQty', 0))
        }


# Backward compatibility functions
def init_testnet_client(api_key: str = None, api_secret: str = None) -> TradingEngine:
    """
    Backward compatibility function for initializing trading client
    
    This function maintains the exact same interface as the original utils/binance_trading.py
    but now returns the new TradingEngine instead of a raw Binance client.
    """
    return TradingEngine(api_key, api_secret, use_testnet=True)


def place_market_order(client, symbol: str, side: str, qty: float, leverage: int = None) -> dict:
    """
    Backward compatibility function for placing market orders
    
    This function maintains the exact same interface as the original utils/binance_trading.py
    but now works with both legacy Client objects and new TradingEngine instances.
    """
    if isinstance(client, TradingEngine):
        return client.place_market_order(symbol, side, qty, leverage)
    else:
        # Handle legacy Client objects
        engine = TradingEngine()
        engine.client = client
        engine.is_connected = True
        return engine.place_market_order(symbol, side, qty, leverage)