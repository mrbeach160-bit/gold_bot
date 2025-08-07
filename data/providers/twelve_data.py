# data/providers/twelve_data.py
"""
Twelve Data API Provider

Provides market data through Twelve Data API.
Supports real-time and historical data for multiple asset classes.
"""

import requests
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import time
from .base import BaseProvider


class TwelveDataProvider(BaseProvider):
    """Twelve Data API provider implementation"""
    
    def __init__(self, api_key: str = None):
        super().__init__("Twelve Data")
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.rate_limit_delay = 1  # seconds between requests
        
    def connect(self, **kwargs) -> bool:
        """Connect to Twelve Data API"""
        if not self.api_key:
            print("⚠️  Twelve Data API key not provided")
            self.is_connected = False
            return False
            
        try:
            # Test connection with API status
            response = requests.get(
                f"{self.base_url}/time_series",
                params={
                    'symbol': 'AAPL',
                    'interval': '1day',
                    'outputsize': 1,
                    'apikey': self.api_key
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'status' not in data or data.get('status') != 'error':
                    self.is_connected = True
                    return True
                    
            print(f"❌ Twelve Data connection failed: {response.status_code}")
            self.is_connected = False
            return False
            
        except Exception as e:
            print(f"❌ Twelve Data connection error: {e}")
            self.is_connected = False
            return False
            
    def get_historical_data(self, symbol: str, timeframe: str,
                           start_date: datetime = None,
                           end_date: datetime = None,
                           bars: int = None) -> pd.DataFrame:
        """Get historical OHLCV data from Twelve Data"""
        if not self.is_connected:
            print("❌ Twelve Data not connected")
            return pd.DataFrame()
            
        try:
            # Convert symbol and timeframe
            td_symbol = self._convert_symbol(symbol)
            td_interval = self._convert_timeframe(timeframe)
            
            # Set output size
            outputsize = min(bars or 100, 5000)  # Twelve Data limit
            
            # Build request parameters
            params = {
                'symbol': td_symbol,
                'interval': td_interval,
                'outputsize': outputsize,
                'apikey': self.api_key,
                'format': 'JSON'
            }
            
            # Add date range if specified
            if start_date:
                params['start_date'] = start_date.strftime('%Y-%m-%d')
            if end_date:
                params['end_date'] = end_date.strftime('%Y-%m-%d')
                
            # Make API request
            response = requests.get(
                f"{self.base_url}/time_series",
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ Twelve Data API error: {response.status_code}")
                return pd.DataFrame()
                
            data = response.json()
            
            if 'status' in data and data['status'] == 'error':
                print(f"❌ Twelve Data API error: {data.get('message', 'Unknown error')}")
                return pd.DataFrame()
                
            if 'values' not in data:
                print(f"⚠️  No data received for {symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            
            if df.empty:
                return pd.DataFrame()
                
            # Rename columns to match our standard
            df.rename(columns={
                'datetime': 'datetime',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }, inplace=True)
            
            # Standardize the dataframe
            standardized = self.standardize_dataframe(df)
            
            print(f"✅ Twelve Data: Retrieved {len(standardized)} bars for {symbol}")
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return standardized
            
        except Exception as e:
            print(f"❌ Twelve Data error for {symbol}: {e}")
            return pd.DataFrame()
            
    def get_live_data(self, symbol: str, bars: int = 100) -> pd.DataFrame:
        """Get latest live data"""
        return self.get_historical_data(
            symbol=symbol,
            timeframe="5m",
            bars=bars
        )
        
    def _convert_symbol(self, symbol: str) -> str:
        """Convert standard symbol to Twelve Data format"""
        symbol_map = {
            'XAUUSD': 'XAU/USD',
            'XAGUSD': 'XAG/USD',
            'EURUSD': 'EUR/USD',
            'GBPUSD': 'GBP/USD',
            'USDJPY': 'USD/JPY',
            'BTCUSD': 'BTC/USD',
            'ETHUSD': 'ETH/USD'
        }
        
        return symbol_map.get(symbol, symbol)
        
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to Twelve Data format"""
        timeframe_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1day',
            'D': '1day',
            'W': '1week',
            'M': '1month'
        }
        
        return timeframe_map.get(timeframe, '5min')