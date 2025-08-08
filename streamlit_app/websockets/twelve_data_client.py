# twelve_data_client.py - Twelve Data WebSocket Client
import websocket
import json
import threading
from datetime import datetime
import streamlit as st


class TwelveDataWebSocketClient:
    """Twelve Data WebSocket client untuk real-time streaming"""
    
    def __init__(self, api_key, on_message_callback):
        self.api_key = api_key
        self.on_message_callback = on_message_callback
        self.ws = None
        self.is_connected = False
        self.subscribed_symbols = set()
        
    def connect(self):
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
        
    def subscribe_symbol(self, symbol):
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
            
    def unsubscribe_symbol(self, symbol):
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