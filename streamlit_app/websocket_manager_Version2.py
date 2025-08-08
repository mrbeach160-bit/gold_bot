import streamlit as st
import threading
import time
from collections import deque
from datetime import datetime

# Import clients (make sure they're in same directory)
try:
    from websocket_clients_Version2 import BinanceWebSocketClient, TwelveDataWebSocketClient
    WEBSOCKET_CLIENTS_AVAILABLE = True
except ImportError:
    print("⚠️ WebSocket clients not found")
    WEBSOCKET_CLIENTS_AVAILABLE = False

class WebSocketManager:
    def __init__(self):
        self.clients = {}
        self.latest_data = {}
        self.connection_status = {}
        self.price_history = {}
        
    def start_websocket(self, api_source, symbol, api_key_1=None, api_key_2=None):
        """Start WebSocket connection based on API source"""
        if not WEBSOCKET_CLIENTS_AVAILABLE:
            return False
            
        client_key = f"{api_source}_{symbol}"
        
        if client_key in self.clients:
            self.stop_websocket(client_key)
            
        def on_data_received(data):
            self.latest_data[client_key] = data
            self.connection_status[client_key] = "Connected ✅"
            
            # Store price history
            if client_key not in self.price_history:
                self.price_history[client_key] = deque(maxlen=100)
            self.price_history[client_key].append({
                'price': data.get('price'),
                'timestamp': data.get('timestamp', datetime.now()),
                'volume': data.get('volume', 0)
            })
            
        try:
            if api_source == "Binance":
                client = BinanceWebSocketClient(
                    symbol=symbol.replace('/', '').upper(),  # Fix symbol format
                    on_message_callback=on_data_received
                )
            elif api_source == "Twelve Data":
                if not api_key_1:
                    raise ValueError("API key required for Twelve Data")
                client = TwelveDataWebSocketClient(
                    symbol=symbol,
                    api_key=api_key_1,
                    on_message_callback=on_data_received
                )
            else:
                raise ValueError(f"Unsupported API source: {api_source}")
                
            self.clients[client_key] = client
            self.connection_status[client_key] = "Connecting... 🔄"
            client.connect()
            
            return True
            
        except Exception as e:
            st.error(f"Failed to start WebSocket for {symbol}: {e}")
            self.connection_status[client_key] = f"Error: {str(e)[:30]}"
            return False
            
    def stop_websocket(self, client_key):
        """Stop WebSocket connection"""
        if client_key in self.clients:
            try:
                self.clients[client_key].disconnect()
                del self.clients[client_key]
                self.connection_status[client_key] = "Disconnected"
            except Exception as e:
                self.connection_status[client_key] = f"Disconnect Error: {str(e)[:20]}"
            
    def get_latest_price(self, api_source, symbol):
        """Get latest price from WebSocket stream"""
        client_key = f"{api_source}_{symbol}"
        data = self.latest_data.get(client_key)
        return data.get('price') if data else None
    
    def get_price(self, symbol):
        """Get price from any available WebSocket connection for the symbol"""
        # Try different API sources
        for api_source in ["Binance", "Twelve Data"]:
            price = self.get_latest_price(api_source, symbol)
            if price and price > 0:
                return price
        return None
        
    def get_connection_status(self, api_source, symbol):
        """Get connection status"""
        client_key = f"{api_source}_{symbol}"
        return self.connection_status.get(client_key, "Not Connected")
        
    def is_connected(self, api_source, symbol):
        """Check if WebSocket is connected"""
        client_key = f"{api_source}_{symbol}"
        client = self.clients.get(client_key)
        return client.is_connected if client else False
        
    def get_price_history(self, api_source, symbol, limit=10):
        """Get recent price history"""
        client_key = f"{api_source}_{symbol}"
        history = self.price_history.get(client_key, [])
        return list(history)[-limit:] if history else []
        
    def disconnect_all(self):
        """Disconnect all WebSocket connections"""
        for client_key in list(self.clients.keys()):
            self.stop_websocket(client_key)

# Global WebSocket manager instance
if 'ws_manager' not in st.session_state:
    st.session_state.ws_manager = WebSocketManager()