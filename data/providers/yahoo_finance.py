# data/providers/yahoo_finance.py
"""
Yahoo Finance Data Provider

Provides market data through yfinance library.
Supports stocks, commodities, forex and cryptocurrencies.
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from .base import BaseProvider


class YFinanceProvider(BaseProvider):
    """Yahoo Finance data provider implementation"""
    
    def __init__(self):
        super().__init__("Yahoo Finance")
        self.rate_limit_delay = 1  # seconds between requests
        
    def connect(self, **kwargs) -> bool:
        """Connect to Yahoo Finance (no authentication required)"""
        try:
            # Test connection with a simple request
            test_ticker = yf.Ticker("GC=F")  # Gold futures
            info = test_ticker.info
            self.is_connected = True
            return True
        except Exception as e:
            print(f"❌ Yahoo Finance connection failed: {e}")
            self.is_connected = False
            return False
            
    def get_historical_data(self, symbol: str, timeframe: str,
                           start_date: datetime = None,
                           end_date: datetime = None,
                           bars: int = None) -> pd.DataFrame:
        """Get historical OHLCV data from Yahoo Finance"""
        try:
            # Convert symbol to Yahoo Finance format
            yf_symbol = self._convert_symbol(symbol)
            
            # Convert timeframe to yfinance format
            yf_interval = self._convert_timeframe(timeframe)
            
            # Set date range
            if bars and not start_date:
                # Calculate start date based on bars and timeframe
                days_back = self._calculate_days_back(bars, timeframe)
                start_date = datetime.now() - timedelta(days=days_back)
                
            if not end_date:
                end_date = datetime.now()
                
            # Create ticker object
            ticker = yf.Ticker(yf_symbol)
            
            # Fetch data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                print(f"⚠️  No data received for {symbol}")
                return pd.DataFrame()
                
            # Reset index to get datetime as column
            data.reset_index(inplace=True)
            
            # Rename columns to match our standard
            data.rename(columns={
                'Datetime': 'datetime',
                'Date': 'datetime', 
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            # Standardize the dataframe
            standardized = self.standardize_dataframe(data)
            
            print(f"✅ Yahoo Finance: Retrieved {len(standardized)} bars for {symbol}")
            return standardized
            
        except Exception as e:
            print(f"❌ Yahoo Finance error for {symbol}: {e}")
            return pd.DataFrame()
            
    def get_live_data(self, symbol: str, bars: int = 100) -> pd.DataFrame:
        """Get latest live data (same as historical for Yahoo Finance)"""
        return self.get_historical_data(
            symbol=symbol,
            timeframe="5m",
            bars=bars
        )
        
    def _convert_symbol(self, symbol: str) -> str:
        """Convert standard symbol to Yahoo Finance format"""
        symbol_map = {
            'XAUUSD': 'GC=F',  # Gold futures
            'XAGUSD': 'SI=F',  # Silver futures
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'JPY=X',
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD'
        }
        
        return symbol_map.get(symbol, symbol)
        
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to Yahoo Finance format"""
        timeframe_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '1h',  # Yahoo doesn't have 4h, use 1h
            '1d': '1d',
            'D': '1d',
            'W': '1wk',
            'M': '1mo'
        }
        
        return timeframe_map.get(timeframe, '5m')
        
    def _calculate_days_back(self, bars: int, timeframe: str) -> int:
        """Calculate how many days back to fetch data"""
        timeframe_days = {
            '1m': bars / (24 * 60),
            '5m': bars / (24 * 12),
            '15m': bars / (24 * 4),
            '30m': bars / (24 * 2),
            '1h': bars / 24,
            '4h': bars / 6,
            '1d': bars,
            'D': bars,
            'W': bars * 7,
            'M': bars * 30
        }
        
        days = timeframe_days.get(timeframe, bars / 24)
        return max(int(days) + 1, 7)  # Minimum 7 days