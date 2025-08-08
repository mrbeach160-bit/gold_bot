"""
Data utilities module for historical data fetching, preprocessing, and feature engineering.

This module handles downloading historical OHLCV data, merging indicators,
cleaning, scaling, and caching operations for trading analysis.
"""

import pandas as pd
import numpy as np
import streamlit as st
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

from .config import is_feature_enabled

# Import data utilities if available
if is_feature_enabled('UTILS_AVAILABLE'):
    from utils.data import get_gold_data
    from utils.indicators import add_indicators, get_support_resistance, compute_rsi

if is_feature_enabled('BINANCE_AVAILABLE'):
    from binance.client import Client


def get_binance_data(api_key: str, api_secret: str, interval: str, symbol: str, 
                    outputsize: int = 500) -> Optional[pd.DataFrame]:
    """
    Fetch historical data from Binance API.
    
    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        interval: Time interval (1m, 5m, 1h, etc.)
        symbol: Trading symbol
        outputsize: Number of candles to fetch
        
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        if not is_feature_enabled('BINANCE_AVAILABLE'):
            st.error("Binance module not available")
            return None
            
        client = Client(api_key, api_secret)
        
        # Map interval format
        interval_map = {
            '1min': Client.KLINE_INTERVAL_1MINUTE,
            '5min': Client.KLINE_INTERVAL_5MINUTE,
            '15min': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY
        }
        
        binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_1HOUR)
        
        # Get historical klines
        klines = client.get_historical_klines(
            symbol, 
            binance_interval, 
            limit=outputsize
        )
        
        if not klines:
            st.error("No data received from Binance")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp and select relevant columns
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching Binance data: {e}")
        return None


def load_and_process_data_enhanced(api_source: str, symbol: str, interval: str, 
                                 api_key_1: str, api_key_2: Optional[str] = None, 
                                 outputsize: int = 500) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Enhanced data loading with comprehensive processing and validation.
    
    Args:
        api_source: Data source ("Twelve Data" or "Binance")
        symbol: Trading symbol
        interval: Time interval
        api_key_1: Primary API key
        api_key_2: Secondary API key
        outputsize: Number of data points
        
    Returns:
        Tuple of (processed_dataframe, success_flag)
    """
    try:
        st.info(f"📡 Loading {outputsize} candles of {symbol} ({interval}) from {api_source}...")
        
        # Fetch raw data based on source
        if api_source == "Binance":
            if not api_key_1 or not api_key_2:
                st.error("Binance requires both API key and secret")
                return None, False
            df = get_binance_data(api_key_1, api_key_2, interval, symbol, outputsize)
        
        elif api_source == "Twelve Data":
            if not api_key_1:
                st.error("Twelve Data requires API key")
                return None, False
            
            if is_feature_enabled('UTILS_AVAILABLE'):
                df = get_gold_data(api_key_1, interval=interval, symbol=symbol, outputsize=outputsize)
            else:
                df = _fetch_twelve_data_fallback(api_key_1, interval, symbol, outputsize)
        
        else:
            st.error(f"Unsupported API source: {api_source}")
            return None, False
        
        if df is None or df.empty:
            st.error(f"❌ No data received from {api_source}")
            return None, False
        
        # Data validation
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            return None, False
        
        # Clean data
        df = _clean_ohlcv_data(df)
        
        if len(df) < 50:
            st.error(f"❌ Insufficient data after cleaning: {len(df)} rows (minimum 50 required)")
            return None, False
        
        # Add technical indicators
        df = _add_technical_indicators(df)
        
        # Final validation
        if df.isnull().any().any():
            st.warning("⚠️ Data contains null values, applying forward fill...")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Data quality metrics
        data_quality = _assess_data_quality(df)
        if data_quality['score'] < 0.7:
            st.warning(f"⚠️ Data quality score: {data_quality['score']:.2f}")
            for issue in data_quality['issues']:
                st.warning(f"  - {issue}")
        
        st.success(f"✅ Successfully loaded {len(df)} candles with {len(df.columns)} features")
        
        return df, True
        
    except Exception as e:
        st.error(f"❌ Error in data loading: {e}")
        return None, False


def _fetch_twelve_data_fallback(api_key: str, interval: str, symbol: str, outputsize: int) -> Optional[pd.DataFrame]:
    """Fallback method to fetch Twelve Data without utils module."""
    try:
        # Convert symbol format for Twelve Data
        symbol_formatted = symbol.replace('/', '')
        
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': symbol_formatted,
            'interval': interval,
            'outputsize': outputsize,
            'apikey': api_key,
            'format': 'JSON'
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'values' not in data:
            st.error(f"Invalid response from Twelve Data: {data.get('message', 'Unknown error')}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data['values'])
        
        # Rename columns to standard format
        df.rename(columns={
            'datetime': 'timestamp',
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }, inplace=True)
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching Twelve Data: {e}")
        return None


def _clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate OHLCV data."""
    
    # Remove rows with invalid prices
    df = df[df['open'] > 0]
    df = df[df['high'] > 0] 
    df = df[df['low'] > 0]
    df = df[df['close'] > 0]
    
    # Fix OHLC logic violations
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    # Remove outliers (prices beyond 3 standard deviations)
    for col in ['open', 'high', 'low', 'close']:
        mean_price = df[col].mean()
        std_price = df[col].std()
        lower_bound = mean_price - (3 * std_price)
        upper_bound = mean_price + (3 * std_price)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]
    
    # Sort by timestamp
    df.sort_index(inplace=True)
    
    return df


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe."""
    
    try:
        # If utils available, use them
        if is_feature_enabled('UTILS_AVAILABLE'):
            df = add_indicators(df)
        else:
            # Fallback indicator calculations
            df = _calculate_basic_indicators(df)
        
        return df
        
    except Exception as e:
        st.warning(f"Error adding indicators: {e}")
        return _calculate_basic_indicators(df)


def _calculate_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic technical indicators as fallback."""
    
    try:
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['RSI_14'] = df['rsi']  # Alias
        
        # Moving Averages
        df['EMA_10'] = df['close'].ewm(span=10).mean()
        df['EMA_20'] = df['close'].ewm(span=20).mean()
        df['SMA_50'] = df['close'].rolling(50).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['MACD_12_26_9'] = exp1 - exp2
        df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9).mean()
        df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR_14'] = true_range.rolling(14).mean()
        
        # Stochastic
        lowest_low = df['low'].rolling(14).min()
        highest_high = df['high'].rolling(14).max()
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        df['STOCHk_14_3_3'] = k_percent.rolling(3).mean()
        df['STOCHd_14_3_3'] = df['STOCHk_14_3_3'].rolling(3).mean()
        
        # Bollinger Bands
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['BBL_20_2.0'] = bb_middle - (bb_std * 2)
        df['BBM_20_2.0'] = bb_middle
        df['BBU_20_2.0'] = bb_middle + (bb_std * 2)
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating basic indicators: {e}")
        return df


def _assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Assess data quality and return metrics."""
    
    quality_score = 1.0
    issues = []
    
    # Check for missing data
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_ratio > 0.1:
        quality_score -= 0.3
        issues.append(f"High missing data ratio: {missing_ratio:.2%}")
    elif missing_ratio > 0.05:
        quality_score -= 0.1
        issues.append(f"Moderate missing data: {missing_ratio:.2%}")
    
    # Check for data gaps (timestamp consistency)
    time_diffs = df.index.to_series().diff().dropna()
    if len(time_diffs) > 0:
        median_diff = time_diffs.median()
        outlier_diffs = time_diffs[time_diffs > median_diff * 2]
        if len(outlier_diffs) > len(time_diffs) * 0.05:
            quality_score -= 0.2
            issues.append(f"Irregular time intervals detected: {len(outlier_diffs)} outliers")
    
    # Check for price consistency
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            volatility = df[col].pct_change().std()
            if volatility > 0.1:  # More than 10% volatility
                quality_score -= 0.1
                issues.append(f"High volatility in {col}: {volatility:.2%}")
    
    # Check data recency
    if len(df) > 0:
        latest_time = df.index[-1]
        time_since_latest = datetime.now() - latest_time.to_pydatetime()
        if time_since_latest.days > 7:
            quality_score -= 0.2
            issues.append(f"Data is {time_since_latest.days} days old")
    
    return {
        'score': max(0, quality_score),
        'issues': issues,
        'missing_ratio': missing_ratio,
        'data_points': len(df),
        'features': len(df.columns)
    }


def get_historical_data_cached(api_source: str, symbol: str, interval: str, 
                             api_key_1: str, api_key_2: Optional[str] = None,
                             cache_minutes: int = 5) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Get historical data with caching to reduce API calls.
    
    Args:
        api_source: Data source
        symbol: Trading symbol
        interval: Time interval
        api_key_1: Primary API key
        api_key_2: Secondary API key
        cache_minutes: Cache duration in minutes
        
    Returns:
        Tuple of (dataframe, from_cache)
    """
    cache_key = f"{api_source}_{symbol}_{interval}"
    
    # Check cache
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}
    
    if cache_key in st.session_state.data_cache:
        cached_data, cache_time = st.session_state.data_cache[cache_key]
        age_minutes = (datetime.now() - cache_time).total_seconds() / 60
        
        if age_minutes < cache_minutes:
            st.info(f"📋 Using cached data (age: {age_minutes:.1f} minutes)")
            return cached_data, True
    
    # Fetch fresh data
    df, success = load_and_process_data_enhanced(
        api_source, symbol, interval, api_key_1, api_key_2
    )
    
    if success and df is not None:
        # Cache the data
        st.session_state.data_cache[cache_key] = (df, datetime.now())
        st.info("💾 Data cached for future use")
    
    return df, False