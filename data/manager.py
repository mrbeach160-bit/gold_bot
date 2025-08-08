# data/manager.py
"""
Data Manager - Centralized Data Management System

The DataManager provides a unified interface for:
- Multiple data provider management (Yahoo Finance, Twelve Data, Binance)
- Data standardization and validation
- Technical indicator calculation
- Data caching and preprocessing
- Configuration-driven provider selection
"""

import pandas as pd
import os
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import providers and processors
from .providers import BaseProvider, YFinanceProvider, TwelveDataProvider, BinanceProvider
from .processors import TechnicalIndicators, DataCleaning

# Import configuration system
try:
    from config import get_config
except ImportError:
    get_config = None


class DataManager:
    """Centralized data management system"""
    
    def __init__(self, symbol: str = None, timeframe: str = None, 
                 preferred_providers: List[str] = None):
        """
        Initialize Data Manager
        
        Args:
            symbol: Trading symbol (uses config default if None)
            timeframe: Data timeframe (uses config default if None)  
            preferred_providers: List of preferred provider names
        """
        # Load configuration
        self.config = None
        try:
            if get_config:
                self.config = get_config()
        except Exception:
            pass
            
        # Set symbol and timeframe
        self.symbol = symbol or (self.config.trading.symbol if self.config else 'XAUUSD')
        self.timeframe = timeframe or (self.config.trading.timeframe if self.config else '5m')
        
        # Initialize providers
        self.providers = {}
        self.active_provider = None
        self._initialize_providers(preferred_providers)
        
        # Data cache
        self.cached_data = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes
        
        print(f"📊 DataManager initialized - Symbol: {self.symbol}, Timeframe: {self.timeframe}")
        
    def _initialize_providers(self, preferred_providers: List[str] = None):
        """Initialize available data providers"""
        provider_priority = preferred_providers or ['yahoo_finance', 'twelve_data', 'binance']
        
        # Initialize Yahoo Finance provider (always available)
        try:
            yf_provider = YFinanceProvider()
            if yf_provider.connect():
                self.providers['yahoo_finance'] = yf_provider
                if not self.active_provider:
                    self.active_provider = 'yahoo_finance'
                print("✅ Yahoo Finance provider connected")
            else:
                print("❌ Yahoo Finance provider failed to connect")
        except Exception as e:
            print(f"❌ Yahoo Finance provider error: {e}")
            
        # Initialize Twelve Data provider (if API key available)
        try:
            api_key = None
            if self.config and hasattr(self.config, 'data_providers') and hasattr(self.config.data_providers, 'twelve_data_api_key'):
                api_key = self.config.data_providers.twelve_data_api_key
            elif 'TWELVE_DATA_API_KEY' in os.environ:
                api_key = os.environ['TWELVE_DATA_API_KEY']
                
            if api_key:
                td_provider = TwelveDataProvider(api_key)
                if td_provider.connect():
                    self.providers['twelve_data'] = td_provider
                    if 'twelve_data' in provider_priority and (not self.active_provider or provider_priority.index('twelve_data') < provider_priority.index(self.active_provider)):
                        self.active_provider = 'twelve_data'
                    print("✅ Twelve Data provider connected")
                else:
                    print("❌ Twelve Data provider failed to connect")
            else:
                print("⚠️  Twelve Data API key not found")
        except Exception as e:
            print(f"❌ Twelve Data provider error: {e}")
            
        # Initialize Binance provider (for crypto symbols)
        try:
            if self.symbol.upper() in ['BTCUSD', 'ETHUSD', 'BNBUSD'] or 'BTC' in self.symbol.upper():
                binance_provider = BinanceProvider()
                if binance_provider.connect():
                    self.providers['binance'] = binance_provider
                    if 'binance' in provider_priority and (not self.active_provider or provider_priority.index('binance') < provider_priority.index(self.active_provider)):
                        self.active_provider = 'binance'
                    print("✅ Binance provider connected")
                else:
                    print("❌ Binance provider failed to connect")
        except Exception as e:
            print(f"❌ Binance provider error: {e}")
            
        if not self.providers:
            print("⚠️  No external providers available, using mock data for testing")
            self._setup_mock_provider()
            
        print(f"📊 Active provider: {self.active_provider}")
        
    def _setup_mock_provider(self):
        """Setup mock provider for testing when no external providers work"""
        class MockProvider(BaseProvider):
            def __init__(self):
                super().__init__("Mock Provider")
                self.is_connected = True
                
            def connect(self, **kwargs) -> bool:
                return True
                
            def get_historical_data(self, symbol: str, timeframe: str, 
                                   start_date: datetime = None, 
                                   end_date: datetime = None,
                                   bars: int = None) -> pd.DataFrame:
                """Generate mock OHLCV data"""
                import numpy as np
                
                bars = bars or 100
                # Create sample dates
                dates = pd.date_range(start='2024-01-01', periods=bars, freq='5min')
                
                # Generate realistic price data with some trend and volatility
                base_price = 2050 if 'XAU' in symbol.upper() else 100
                price_changes = np.random.normal(0, 0.01, bars).cumsum()
                prices = base_price * (1 + price_changes)
                
                # Generate OHLCV data
                opens = prices
                highs = opens * (1 + np.random.uniform(0, 0.02, bars))
                lows = opens * (1 - np.random.uniform(0, 0.02, bars))
                closes = opens + np.random.uniform(-0.01, 0.01, bars) * opens
                volumes = np.random.uniform(1000, 5000, bars)
                
                df = pd.DataFrame({
                    'datetime': dates,
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': volumes
                })
                
                return self.standardize_dataframe(df)
                
            def get_live_data(self, symbol: str, bars: int = 100) -> pd.DataFrame:
                return self.get_historical_data(symbol, '5m', bars=bars)
                
        mock_provider = MockProvider()
        self.providers['mock'] = mock_provider
        self.active_provider = 'mock'
        
    def get_historical_data(self, symbol: str = None, timeframe: str = None,
                           start_date: datetime = None, end_date: datetime = None,
                           bars: int = None, force_refresh: bool = False) -> pd.DataFrame:
        """Get historical market data"""
        symbol = symbol or self.symbol
        timeframe = timeframe or self.timeframe
        bars = bars or 100
        
        # Check cache first
        cache_key = f"{symbol}_{timeframe}_{bars}"
        if not force_refresh and self._is_cache_valid(cache_key):
            print(f"📋 Using cached data for {symbol}")
            return self.cached_data[cache_key]
            
        # Try providers in order of preference
        data = pd.DataFrame()
        providers_tried = []
        
        # Try active provider first
        if self.active_provider and self.active_provider in self.providers:
            providers_tried.append(self.active_provider)
            try:
                provider = self.providers[self.active_provider]
                data = provider.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    bars=bars
                )
                if not data.empty:
                    print(f"✅ Data retrieved from {self.active_provider}")
                else:
                    print(f"⚠️  No data from {self.active_provider}")
            except Exception as e:
                print(f"❌ Error with {self.active_provider}: {e}")
                
        # Try fallback providers if needed
        if data.empty:
            for provider_name, provider in self.providers.items():
                if provider_name not in providers_tried:
                    providers_tried.append(provider_name)
                    try:
                        data = provider.get_historical_data(
                            symbol=symbol,
                            timeframe=timeframe,
                            start_date=start_date,
                            end_date=end_date,
                            bars=bars
                        )
                        if not data.empty:
                            print(f"✅ Fallback data retrieved from {provider_name}")
                            self.active_provider = provider_name  # Switch to working provider
                            break
                        else:
                            print(f"⚠️  No data from fallback {provider_name}")
                    except Exception as e:
                        print(f"❌ Error with fallback {provider_name}: {e}")
                        
        if data.empty:
            print(f"❌ Failed to retrieve data from all providers: {providers_tried}")
            return pd.DataFrame()
            
        # Validate and clean data
        if not self.validate_data(data):
            print("⚠️  Data validation failed, attempting to clean")
            data = DataCleaning.clean_ohlc_data(data)
            
        # Add technical indicators
        if len(data) >= 50:  # Need sufficient data for indicators
            data = TechnicalIndicators.add_all_indicators(data)
            
        # Cache the data
        self.cached_data[cache_key] = data
        self.cache_expiry[cache_key] = time.time() + self.cache_duration
        
        return data
        
    def get_live_data(self, symbol: str = None, bars: int = 100) -> pd.DataFrame:
        """Get latest live market data"""
        return self.get_historical_data(
            symbol=symbol,
            bars=bars,
            force_refresh=True  # Always get fresh data for live feed
        )
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for ML models"""
        if data.empty:
            return data
            
        processed = data.copy()
        
        # Add additional features
        if len(processed) > 1:
            processed['price_change_pct'] = processed['close'].pct_change()
            processed['volatility'] = processed['price_change_pct'].rolling(20).std()
            processed['volume_sma'] = processed['volume'].rolling(20).mean() if 'volume' in processed.columns else 0
            
        # Fill any remaining NaN values
        processed = processed.fillna(method='ffill').fillna(0)
        
        return processed
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data quality"""
        if data.empty:
            return False
            
        # Check minimum length
        if len(data) < 10:
            return False
            
        # Use data cleaning validation
        validation_result = DataCleaning.validate_data_quality(data)
        return validation_result['valid']
        
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cached_data:
            return False
            
        if cache_key not in self.cache_expiry:
            return False
            
        return time.time() < self.cache_expiry[cache_key]
        
    def clear_cache(self):
        """Clear data cache"""
        self.cached_data.clear()
        self.cache_expiry.clear()
        print("🗑️  Data cache cleared")
        
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        status = {}
        for name, provider in self.providers.items():
            status[name] = provider.get_provider_info()
            
        return {
            'providers': status,
            'active_provider': self.active_provider,
            'symbol': self.symbol,
            'timeframe': self.timeframe
        }
        
    def switch_provider(self, provider_name: str) -> bool:
        """Switch to a different provider"""
        if provider_name in self.providers:
            if self.providers[provider_name].validate_connection():
                self.active_provider = provider_name
                print(f"🔄 Switched to provider: {provider_name}")
                return True
            else:
                print(f"❌ Provider {provider_name} is not connected")
                return False
        else:
            print(f"❌ Provider {provider_name} not available")
            return False