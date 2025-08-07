# data/providers/base.py
"""
Base Provider Interface

Abstract base class that defines the standard interface for all data providers.
Ensures consistent data format and method signatures across all providers.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime


class BaseProvider(ABC):
    """Abstract base class for data providers"""
    
    def __init__(self, name: str):
        """Initialize provider with name"""
        self.name = name
        self.is_connected = False
        
    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """Connect to data source"""
        pass
        
    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: str, 
                           start_date: datetime = None, 
                           end_date: datetime = None,
                           bars: int = None) -> pd.DataFrame:
        """Get historical OHLCV data"""
        pass
        
    @abstractmethod
    def get_live_data(self, symbol: str, bars: int = 100) -> pd.DataFrame:
        """Get latest live data"""
        pass
        
    def standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize dataframe to consistent OHLCV format
        
        Expected columns: datetime, open, high, low, close, volume
        Returns: DataFrame with datetime index and OHLCV columns
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            
        # Make a copy to avoid modifying original
        standardized = df.copy()
        
        # Ensure datetime column exists and convert to index
        datetime_cols = ['datetime', 'timestamp', 'time', 'date']
        datetime_col = None
        
        for col in datetime_cols:
            if col in standardized.columns:
                datetime_col = col
                break
                
        if datetime_col:
            if not isinstance(standardized.index, pd.DatetimeIndex):
                standardized['datetime'] = pd.to_datetime(standardized[datetime_col])
                standardized.set_index('datetime', inplace=True)
        elif not isinstance(standardized.index, pd.DatetimeIndex):
            # If no datetime column found, try to convert index
            try:
                standardized.index = pd.to_datetime(standardized.index)
            except:
                # Create datetime index if all else fails
                standardized.index = pd.date_range(
                    start='2024-01-01', periods=len(standardized), freq='5min'
                )
        
        # Standardize column names to lowercase
        column_mapping = {}
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        for col in standardized.columns:
            col_lower = col.lower()
            if col_lower in required_cols:
                column_mapping[col] = col_lower
                
        standardized.rename(columns=column_mapping, inplace=True)
        
        # Ensure all required columns exist
        for col in required_cols:
            if col not in standardized.columns:
                if col == 'volume':
                    standardized[col] = 1000  # Default volume
                else:
                    standardized[col] = 0  # Default price
                    
        # Select only required columns
        standardized = standardized[required_cols]
        
        # Ensure data types
        for col in required_cols:
            standardized[col] = pd.to_numeric(standardized[col], errors='coerce')
            
        # Remove any rows with NaN values
        standardized.dropna(inplace=True)
        
        return standardized
        
    def validate_connection(self) -> bool:
        """Validate if connection is active"""
        return self.is_connected
        
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            'name': self.name,
            'connected': self.is_connected,
            'type': self.__class__.__name__
        }