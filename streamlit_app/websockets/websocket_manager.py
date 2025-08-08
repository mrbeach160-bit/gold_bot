# websocket_manager.py - Enhanced WebSocket Manager
import websocket
import json
import threading
import time
from datetime import datetime
from collections import deque
import streamlit as st

from .twelve_data_client import TwelveDataWebSocketClient

# Check for Binance availability
try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

# Check for WebSocket availability
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False


class EnhancedWebSocketManager:
    """Enhanced WebSocket Manager untuk Twelve Data dan Binance"""
    
    def __init__(self):
        self.latest_price = {}
        self.connection_status = {}
        self.clients = {}
        self.price_history = {}
        
    def start_stream(self, api_source, symbol, api_key_1=None, api_key_2=None):
        """Start streaming berdasarkan API source"""
        
        if api_source == "Binance" and BINANCE_AVAILABLE:
            return self._start_binance_stream(symbol)
            
        elif api_source == "Twelve Data":
            # Try WebSocket first, fallback to polling
            if WEBSOCKET_AVAILABLE:
                websocket_success = self._start_twelve_data_websocket_stream(symbol, api_key_1)
                if websocket_success:
                    return True
                else:
                    st.warning("⚠️ WebSocket failed, falling back to polling...")
                    return self._start_polling_stream(symbol, api_key_1)
            else:
                return self._start_polling_stream(symbol, api_key_1)
            
        return False
    
    def _start_twelve_data_websocket_stream(self, symbol, api_key):
        """Start Twelve Data WebSocket stream"""
        if not WEBSOCKET_AVAILABLE:
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
                self.connection_status[symbol] = "Connection Failed"
                return False
                
        except Exception as e:
            self.connection_status[symbol] = f"Twelve Data WS Error: {str(e)[:30]}"
            return False
        
    def _start_binance_stream(self, symbol):
        """Start Binance WebSocket stream"""
        if not WEBSOCKET_AVAILABLE:
            self.connection_status[symbol] = "WebSocket tidak tersedia"
            return False
            
        try:
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    price = float(data.get('c', 0))
                    if price > 0:
                        self.latest_price[symbol] = price
                        if symbol not in self.price_history:
                            self.price_history[symbol] = deque(maxlen=100)
                        self.price_history[symbol].append({
                            'price': price,
                            'timestamp': datetime.now(),
                            'volume': float(data.get('v', 0))
                        })
                        self.connection_status[symbol] = "Connected ✅"
                except Exception as e:
                    pass
                    
            def on_error(ws, error):
                self.connection_status[symbol] = f"Error: {str(error)[:30]}"
                
            def on_close(ws, close_status_code, close_msg):
                self.connection_status[symbol] = "Disconnected"
                
            def on_open(ws):
                self.connection_status[symbol] = "Connected ✅"
                
            # Buat WebSocket connection
            symbol_clean = symbol.replace('/', '').lower()
            url = f"wss://stream.binance.com:9443/ws/{symbol_clean}@ticker"
            
            ws = websocket.WebSocketApp(url,
                on_message=on_message,
                on_error=on_error, 
                on_close=on_close,
                on_open=on_open)
            
            # Start di thread terpisah
            def run_ws():
                try:
                    ws.run_forever()
                except Exception as e:
                    self.connection_status[symbol] = f"Thread Error: {str(e)[:30]}"
                    
            thread = threading.Thread(target=run_ws, daemon=True)
            thread.start()
            
            self.clients[symbol] = ws
            self.connection_status[symbol] = "Connecting... 🔄"
            return True
            
        except Exception as e:
            self.connection_status[symbol] = f"Start Error: {str(e)[:30]}"
            return False
    
    def _start_polling_stream(self, symbol, api_key):
        """Start polling untuk Twelve Data (fallback)"""
        try:
            def poll_data():
                while symbol in self.clients:
                    try:
                        # Import here to avoid circular imports
                        from ..data.data_loader import get_gold_data
                        
                        # Polling setiap 30 detik
                        data = get_gold_data(api_key, interval='1min', symbol=symbol, outputsize=1)
                        if data is not None and not data.empty:
                            price = data['close'].iloc[-1]
                            self.latest_price[symbol] = price
                            if symbol not in self.price_history:
                                self.price_history[symbol] = deque(maxlen=100)
                            self.price_history[symbol].append({
                                'price': price,
                                'timestamp': datetime.now(),
                                'volume': data['volume'].iloc[-1] if 'volume' in data.columns else 0
                            })
                            self.connection_status[symbol] = "Polling ✅"
                        time.sleep(30)  # 30 second interval
                    except Exception as e:
                        self.connection_status[symbol] = f"Polling Error: {str(e)[:30]}"
                        time.sleep(60)  # Wait longer on error
            
            # Start polling thread
            thread = threading.Thread(target=poll_data, daemon=True)
            thread.start()
            
            self.clients[symbol] = thread
            self.connection_status[symbol] = "Starting Polling... 🔄"
            return True
            
        except Exception as e:
            self.connection_status[symbol] = f"Polling Start Error: {str(e)[:30]}"
            return False
    
    def get_price(self, symbol):
        """Get latest price"""
        return self.latest_price.get(symbol)
        
    def get_status(self, symbol):
        """Get connection status"""
        return self.connection_status.get(symbol, "Not Connected")
        
    def get_price_history(self, symbol, limit=10):
        """Get recent price history"""
        history = self.price_history.get(symbol, [])
        return list(history)[-limit:] if history else []
        
    def is_connected(self, symbol):
        """Check if connected"""
        status = self.get_status(symbol)
        return "✅" in status
        
    def disconnect(self, symbol):
        """Disconnect stream"""
        try:
            if symbol in self.clients:
                client = self.clients[symbol]
                if hasattr(client, 'disconnect'):
                    client.disconnect()
                del self.clients[symbol]
            self.connection_status[symbol] = "Disconnected"
            if symbol in self.latest_price:
                del self.latest_price[symbol]
        except Exception as e:
            pass