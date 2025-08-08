"""
WebSocket Panel Component
Handles real-time data streaming and WebSocket connections
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import json
import threading
from datetime import datetime
from collections import deque

# Safe imports with error handling
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False


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


class EnhancedWebSocketManager:
    """Enhanced WebSocket Manager untuk Twelve Data dan Binance"""
    
    def __init__(self):
        self.latest_price = {}
        self.connection_status = {}
        self.clients = {}
        self.price_history = {}
        
    def start_stream(self, api_source, symbol, api_key_1=None, api_key_2=None):
        """Start streaming berdasarkan API source"""
        
        if api_source == "Binance":
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
                        # Placeholder for polling logic
                        # This would need the get_gold_data function from utils
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


class WebSocketPanel:
    """WebSocket Panel Component for real-time data display"""
    
    def __init__(self, ws_manager=None):
        if ws_manager is None:
            self.ws_manager = EnhancedWebSocketManager()
        else:
            self.ws_manager = ws_manager
    
    @staticmethod
    def format_price(symbol, price):
        """Format price berdasarkan symbol"""
        try:
            price = float(price)
            if 'XAU' in symbol.upper() or 'GOLD' in symbol.upper():
                return f"${price:,.2f}"
            elif 'BTC' in symbol.upper():
                return f"${price:,.0f}"  # Round to nearest dollar for BTC
            elif 'ETH' in symbol.upper():
                return f"${price:,.2f}"
            elif 'JPY' in symbol.upper():
                return f"{price:.3f}"
            else:
                return f"{price:.5f}"
        except (ValueError, TypeError):
            return str(price)
    
    def render_websocket_panel(self, api_source, symbol, api_key_1=None, api_key_2=None):
        """Render the WebSocket connection panel"""
        if not WEBSOCKET_AVAILABLE or api_source not in ["Binance", "Twelve Data"]:
            return None
            
        with st.expander("🌐 Real-time Connection", expanded=True):
            ws_cols = st.columns([2, 1, 1])
            
            # Status WebSocket
            ws_status = self.ws_manager.get_status(symbol)
            ws_price = self.ws_manager.get_price(symbol)
            
            if "✅" in ws_status:
                if api_source == "Twelve Data":
                    ws_cols[0].success(f"🟢 Twelve Data WebSocket: {ws_status}")
                else:
                    ws_cols[0].success(f"🟢 Binance WebSocket: {ws_status}")
            elif "Error" in ws_status:
                ws_cols[0].error(f"🔴 Status: {ws_status}")
            else:
                ws_cols[0].warning(f"🟡 Status: {ws_status}")
            
            # Live Price Display
            if ws_price and ws_price > 0:
                price_history = self.ws_manager.get_price_history(symbol, 5)
                if len(price_history) >= 2:
                    prev_price = price_history[-2]['price']
                    price_change = ws_price - prev_price
                    ws_cols[1].metric(
                        "🔴 Live Price", 
                        self.format_price(symbol, ws_price),
                        delta=f"{self.format_price(symbol, price_change)}"
                    )
                else:
                    ws_cols[1].metric("🔴 Live Price", self.format_price(symbol, ws_price))
            else:
                ws_cols[1].info("⏳ Waiting for data...")
            
            # Connect/Disconnect Button
            if ws_cols[2].button("🔗 Connect" if not self.ws_manager.is_connected(symbol) else "🔌 Disconnect"):
                if self.ws_manager.is_connected(symbol):
                    self.ws_manager.disconnect(symbol)
                    st.info("🔌 Stream disconnected")
                else:
                    success = self.ws_manager.start_stream(api_source, symbol, api_key_1, api_key_2)
                    if success:
                        if api_source == "Twelve Data":
                            st.success(f"🔗 Twelve Data WebSocket stream initiated")
                        else:
                            st.success(f"🔗 {api_source} stream initiated")
                    else:
                        st.error("❌ Failed to start stream")
                time.sleep(1)
                st.rerun()
                
            # Price History Chart (Mini)
            if ws_price and len(self.ws_manager.get_price_history(symbol, 20)) > 5:
                self.render_price_chart(symbol, api_source)
    
    def render_price_chart(self, symbol, api_source):
        """Render mini price chart"""
        price_history = self.ws_manager.get_price_history(symbol, 20)
        prices = [p['price'] for p in price_history]
        timestamps = [p['timestamp'] for p in price_history]
        
        price_df = pd.DataFrame({
            'Time': timestamps,
            'Price': prices
        })
        fig_mini = px.line(
            price_df, 
            x='Time', 
            y='Price', 
            title=f"📊 Live Price Feed - {api_source} (Last 20 updates)",
            height=300
        )
        fig_mini.update_layout(template="plotly_dark")
        st.plotly_chart(fig_mini, use_container_width=True)
    
    def get_realtime_price(self, symbol):
        """Get current real-time price"""
        return self.ws_manager.get_price(symbol)
    
    def get_connection_status(self, symbol):
        """Get connection status"""
        return self.ws_manager.get_status(symbol)
    
    def is_websocket_available(self):
        """Check if WebSocket is available"""
        return WEBSOCKET_AVAILABLE