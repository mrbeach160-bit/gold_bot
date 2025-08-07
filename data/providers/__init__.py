# data/providers/__init__.py
"""
Data Provider Abstraction Layer

This module provides standardized interfaces for multiple data sources:
- BaseProvider: Abstract base class for all data providers
- YFinanceProvider: Yahoo Finance data integration
- TwelveDataProvider: Twelve Data API integration  
- BinanceProvider: Binance API integration

All providers standardize data to OHLCV format with consistent column names.
"""

from .base import BaseProvider
from .yahoo_finance import YFinanceProvider
from .twelve_data import TwelveDataProvider
from .binance import BinanceProvider

__all__ = [
    'BaseProvider',
    'YFinanceProvider', 
    'TwelveDataProvider',
    'BinanceProvider'
]