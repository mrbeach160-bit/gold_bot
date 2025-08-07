# data/providers/binance.py
"""
Binance Data Provider

Provides market data through Binance API.
Supports spot and futures markets for cryptocurrencies.
"""

import requests
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import time
from .base import BaseProvider


class BinanceProvider(BaseProvider):
    """Binance API provider implementation"""
    
    def __init__(self):
        super().__init__("Binance")
        self.base_url = "https://api.binance.com"
        self.rate_limit_delay = 0.5  # seconds between requests
        
    def connect(self, **kwargs) -> bool:
        """Connect to Binance API (no authentication required for market data)"""
        try:
            # Test connection with exchange info
            response = requests.get(
                f"{self.base_url}/api/v3/exchangeInfo",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'symbols' in data:
                    self.is_connected = True
                    return True
                    
            print(f"❌ Binance connection failed: {response.status_code}")
            self.is_connected = False
            return False
            
        except Exception as e:
            print(f"❌ Binance connection error: {e}")
            self.is_connected = False
            return False
            
    def get_historical_data(self, symbol: str, timeframe: str,
                           start_date: datetime = None,
                           end_date: datetime = None,
                           bars: int = None) -> pd.DataFrame:
        """Get historical OHLCV data from Binance"""
        if not self.is_connected:
            print("❌ Binance not connected")
            return pd.DataFrame()
            
        try:
            # Convert symbol and timeframe
            binance_symbol = self._convert_symbol(symbol)
            binance_interval = self._convert_timeframe(timeframe)
            
            # Set limit (max 1000 for Binance)
            limit = min(bars or 100, 1000)
            
            # Build request parameters
            params = {
                'symbol': binance_symbol,
                'interval': binance_interval,
                'limit': limit
            }
            
            # Add time range if specified
            if start_date:
                params['startTime'] = int(start_date.timestamp() * 1000)
            if end_date:
                params['endTime'] = int(end_date.timestamp() * 1000)
                
            # Make API request
            response = requests.get(
                f"{self.base_url}/api/v3/klines",
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ Binance API error: {response.status_code}")
                return pd.DataFrame()
                
            data = response.json()
            
            if not data:
                print(f"⚠️  No data received for {symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            if df.empty:
                return pd.DataFrame()
                
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
            
            # Select and rename required columns
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            # Standardize the dataframe
            standardized = self.standardize_dataframe(df)
            
            print(f"✅ Binance: Retrieved {len(standardized)} bars for {symbol}")
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return standardized
            
        except Exception as e:
            print(f"❌ Binance error for {symbol}: {e}")
            return pd.DataFrame()
            
    def get_live_data(self, symbol: str, bars: int = 100) -> pd.DataFrame:
        """Get latest live data"""
        return self.get_historical_data(
            symbol=symbol,
            timeframe="5m",
            bars=bars
        )
        
    def _convert_symbol(self, symbol: str) -> str:
        """Convert standard symbol to Binance format"""
        symbol_map = {
            'BTCUSD': 'BTCUSDT',
            'ETHUSD': 'ETHUSDT',
            'BNBUSD': 'BNBUSDT',
            'ADAUSD': 'ADAUSDT',
            'XRPUSD': 'XRPUSDT',
            'SOLUSD': 'SOLUSDT',
            'DOTUSD': 'DOTUSDT',
            'AVAXUSD': 'AVAXUSDT'
        }
        
        # If symbol already ends with USDT, use as-is
        if symbol.endswith('USDT'):
            return symbol
            
        return symbol_map.get(symbol, symbol + 'USDT')
        
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to Binance format"""
        timeframe_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            'D': '1d',
            'W': '1w',
            'M': '1M'
        }
        
        return timeframe_map.get(timeframe, '5m')