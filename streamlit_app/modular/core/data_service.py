"""
Data Service for the modular application.
Handles data fetching and preprocessing.
"""

import pandas as pd
import streamlit as st
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data import get_gold_data


class DataService:
    """Service for handling data operations."""
    
    def __init__(self):
        self.timeframe_mapping = {
            '1m': '1min',
            '5m': '5min', 
            '15m': '15min',
            '30m': '30min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1day'
        }
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_data(_self, symbol: str, timeframe: str, size: int = 500, api_key: str = None) -> pd.DataFrame:
        """
        Fetch historical data for the given symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            timeframe: Timeframe (e.g., '15m', '1h')
            size: Number of data points to fetch
            api_key: API key for data source
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Map display timeframe to API parameter
            api_timeframe = _self.timeframe_mapping.get(timeframe, timeframe)
            
            # Fetch data using existing utility
            data = get_gold_data(
                api_key=api_key,
                interval=api_timeframe, 
                symbol=symbol,
                outputsize=size
            )
            
            if data is None or data.empty:
                st.error(f"No data received for {symbol} {timeframe}")
                return pd.DataFrame()
                
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()
                
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    def validate_data(self, data: pd.DataFrame, min_rows: int = 100) -> bool:
        """
        Validate that data meets minimum requirements.
        
        Args:
            data: DataFrame to validate
            min_rows: Minimum number of rows required
            
        Returns:
            True if valid, False otherwise
        """
        if data is None or data.empty:
            return False
            
        if len(data) < min_rows:
            st.warning(f"Insufficient data: {len(data)} rows (minimum {min_rows} required)")
            return False
            
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return False
            
        return True
    
    def prepare_latest_window(self, data: pd.DataFrame, window_size: int = 60) -> pd.DataFrame:
        """
        Prepare the latest window of data for prediction.
        
        Args:
            data: Full dataset
            window_size: Size of window to extract
            
        Returns:
            Latest window of data
        """
        if len(data) < window_size:
            st.warning(f"Not enough data for window size {window_size}. Using all available {len(data)} rows.")
            return data
            
        return data.tail(window_size).copy()