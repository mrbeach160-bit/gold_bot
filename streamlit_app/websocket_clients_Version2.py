import websocket
import json
import threading
import time
import pandas as pd
from datetime import datetime
import logging
from collections import deque

class BinanceWebSocketClient:
    def __init__(self, symbol, on_message_callback):
        self.symbol = symbol.lower()
        self.on_message_callback = on_message_callback
        self.ws = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3  # Reduce attempts
        self.data_buffer = deque(maxlen=100)  # Smaller buffer
        
    def connect(self):
        """Connect to Binance WebSocket"""
        try:
            websocket.enableTrace(False)
            url = f"wss://stream.binance.com:9443/ws/{self.symbol}@ticker"
            self.ws = websocket.WebSocketApp(
                url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Start WebSocket in a separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
        except Exception as e:
            print(f"Binance WebSocket connection error: {e}")  # Use print instead of logging
            
    def on_open(self, ws):
        """WebSocket connection opened"""
        self.is_connected = True
        self.reconnect_attempts = 0
        print(f"✅ Binance WebSocket connected for {self.symbol}")
        
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Process ticker data
            processed_data = {
                'symbol': data['s'],
                'price': float(data['c']),
                'volume': float(data['v']),
                'high': float(data['h']),
                'low': float(data['l']),
                'open': float(data['o']),
                'timestamp': datetime.fromtimestamp(int(data['E']) / 1000)
            }
            
            self.data_buffer.append(processed_data)
            
            # Call the callback function
            if self.on_message_callback:
                self.on_message_callback(processed_data)
                
        except Exception as e:
            pass  # Silent error handling
            
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.is_connected = False
        
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        self.is_connected = False
        
        # Auto-reconnect disabled for stability
        # if self.reconnect_attempts < self.max_reconnect_attempts:
        #     self.reconnect_attempts += 1
        #     time.sleep(5)
        #     self.connect()
            
    def get_latest_data(self):
        """Get the latest data from buffer"""
        return list(self.data_buffer)[-1] if self.data_buffer else None
        
    def disconnect(self):
        """Disconnect WebSocket"""
        if self.ws:
            self.ws.close()
            self.is_connected = False

class TwelveDataWebSocketClient:
    def __init__(self, symbol, api_key, on_message_callback):
        self.symbol = symbol
        self.api_key = api_key
        self.on_message_callback = on_message_callback
        self.ws = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.data_buffer = deque(maxlen=100)
        
    def connect(self):
        """Connect to TwelveData WebSocket"""
        try:
            websocket.enableTrace(False)
            url = "wss://ws.twelvedata.com/v1/quotes/price"
            self.ws = websocket.WebSocketApp(
                url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Start WebSocket in a separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
        except Exception as e:
            print(f"TwelveData WebSocket connection error: {e}")
            
    def on_open(self, ws):
        """WebSocket connection opened"""
        try:
            # Subscribe to symbol
            subscribe_message = {
                "action": "subscribe",
                "params": {
                    "symbols": self.symbol,
                    "apikey": self.api_key
                }
            }
            ws.send(json.dumps(subscribe_message))
            self.is_connected = True
            self.reconnect_attempts = 0
            print(f"✅ TwelveData WebSocket connected for {self.symbol}")
        except Exception as e:
            print(f"❌ Failed to subscribe: {e}")
        
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if data.get('event') == 'price':
                processed_data = {
                    'symbol': data.get('symbol'),
                    'price': float(data.get('price', 0)),
                    'timestamp': datetime.now()
                }
                
                self.data_buffer.append(processed_data)
                
                # Call the callback function
                if self.on_message_callback:
                    self.on_message_callback(processed_data)
                    
        except Exception as e:
            pass  # Silent error handling
            
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.is_connected = False
        
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        self.is_connected = False
            
    def get_latest_data(self):
        """Get the latest data from buffer"""
        return list(self.data_buffer)[-1] if self.data_buffer else None
        
    def disconnect(self):
        """Disconnect WebSocket"""
        if self.ws:
            self.ws.close()
            self.is_connected = False