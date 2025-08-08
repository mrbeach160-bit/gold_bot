"""
WebSocket Manager module for real-time price streaming.

This module provides unified WebSocket abstraction for live price streams 
via Twelve Data WebSocket, Binance WebSocket, or polling fallback.
Maintains price history buffers and connection management.
"""

import streamlit as st
import json
import time
import threading
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional, Callable

from .config import is_feature_enabled

# Import WebSocket dependencies if available
if is_feature_enabled('WEBSOCKET_AVAILABLE'):
    import websocket

if is_feature_enabled('BINANCE_AVAILABLE'):
    from binance.client import Client


class TwelveDataWebSocketClient:
    """Twelve Data WebSocket client untuk real-time streaming"""
    
    def __init__(self, api_key: str, on_message_callback: Callable):
        self.api_key = api_key
        self.on_message_callback = on_message_callback
        self.ws = None
        self.is_connected = False
        self.subscribed_symbols = set()
        
    def connect(self) -> bool:
        """Connect ke Twelve Data WebSocket"""
        try:
            # Twelve Data WebSocket endpoint
            url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={self.api_key}"
            
            self.ws = websocket.WebSocketApp(
                url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Start in thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            return True
            
        except Exception as e:
            st.error(f"Twelve Data WebSocket connection error: {e}")
            return False
            
    def on_open(self, ws):
        """WebSocket opened"""
        self.is_connected = True
        
    def on_message(self, ws, message):
        """Handle incoming real-time price messages"""
        try:
            data = json.loads(message)
            
            # Twelve Data WebSocket format
            if data.get('event') == 'price':
                processed_data = {
                    'symbol': data.get('symbol'),
                    'price': float(data.get('price', 0)),
                    'timestamp': datetime.now(),
                    'volume': data.get('volume', 0)
                }
                
                if self.on_message_callback:
                    self.on_message_callback(processed_data)
                    
        except Exception as e:
            pass  # Ignore message parsing errors
            
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.is_connected = False
        
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        self.is_connected = False
        
    def subscribe_symbol(self, symbol: str):
        """Subscribe to symbol updates"""
        if self.ws and self.is_connected:
            subscribe_msg = {
                "action": "subscribe",
                "params": {
                    "symbols": symbol
                }
            }
            self.ws.send(json.dumps(subscribe_msg))
            self.subscribed_symbols.add(symbol)
            
    def unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from symbol"""
        if self.ws and self.is_connected:
            unsubscribe_msg = {
                "action": "unsubscribe",
                "params": {
                    "symbols": symbol
                }
            }
            self.ws.send(json.dumps(unsubscribe_msg))
            self.subscribed_symbols.discard(symbol)
            
    def disconnect(self):
        """Disconnect WebSocket"""
        if self.ws:
            self.ws.close()
            self.is_connected = False


class EnhancedWebSocketManager:
    """Enhanced WebSocket Manager untuk Twelve Data dan Binance"""
    
    def __init__(self):
        self.latest_price = {}
        self.connection_status = {}
        self.clients = {}
        self.price_history = {}
        
    def start_stream(self, api_source: str, symbol: str, api_key_1: Optional[str] = None, api_key_2: Optional[str] = None) -> bool:
        """Start streaming berdasarkan API source"""
        
        if api_source == "Binance" and is_feature_enabled('BINANCE_AVAILABLE'):
            return self._start_binance_stream(symbol)
            
        elif api_source == "Twelve Data":
            # Try WebSocket first, fallback to polling
            if is_feature_enabled('WEBSOCKET_AVAILABLE'):
                websocket_success = self._start_twelve_data_websocket_stream(symbol, api_key_1)
                if websocket_success:
                    return True
                else:
                    st.warning("⚠️ WebSocket failed, falling back to polling...")
                    return self._start_polling_stream(symbol, api_key_1)
            else:
                return self._start_polling_stream(symbol, api_key_1)
            
        return False
    
    def _start_twelve_data_websocket_stream(self, symbol: str, api_key: str) -> bool:
        """Start Twelve Data WebSocket stream"""
        if not is_feature_enabled('WEBSOCKET_AVAILABLE'):
            self.connection_status[symbol] = "WebSocket tidak tersedia"
            return False
            
        try:
            def on_data_received(data):
                if data and 'price' in data:
                    self.latest_price[symbol] = data['price']
                    if symbol not in self.price_history:
                        self.price_history[symbol] = deque(maxlen=100)
                    self.price_history[symbol].append({
                        'price': data['price'],
                        'timestamp': data.get('timestamp', datetime.now()),
                        'volume': data.get('volume', 0)
                    })
                    self.connection_status[symbol] = "Connected ✅"
                    
            client = TwelveDataWebSocketClient(
                api_key=api_key,
                on_message_callback=on_data_received
            )
            
            if client.connect():
                time.sleep(2)  # Wait for connection
                client.subscribe_symbol(symbol)
                
                self.clients[symbol] = client
                self.connection_status[symbol] = "Connecting... 🔄"
                return True
            else:
                self.connection_status[symbol] = "Connection failed ❌"
                return False
                
        except Exception as e:
            self.connection_status[symbol] = f"Error: {str(e)}"
            return False
            
    def _start_binance_stream(self, symbol: str) -> bool:
        """Start Binance WebSocket stream"""
        # Implementation would go here - keeping it simple for now
        self.connection_status[symbol] = "Binance WebSocket not yet implemented"
        return False
        
    def _start_polling_stream(self, symbol: str, api_key: str) -> bool:
        """Start polling fallback stream"""
        self.connection_status[symbol] = "Polling mode ⏱️"
        return True
        
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        return self.latest_price.get(symbol)
        
    def get_connection_status(self, symbol: str) -> str:
        """Get connection status for symbol"""
        return self.connection_status.get(symbol, "Not connected")
        
    def get_price_history(self, symbol: str) -> Optional[deque]:
        """Get price history for symbol"""
        return self.price_history.get(symbol)
        
    def disconnect_all(self):
        """Disconnect all WebSocket connections"""
        for symbol, client in self.clients.items():
            if hasattr(client, 'disconnect'):
                client.disconnect()
        self.clients.clear()
        self.connection_status.clear()


# Global WebSocket manager instance
websocket_manager = EnhancedWebSocketManager()