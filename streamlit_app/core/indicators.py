"""
Technical Indicators Module

Contains technical analysis indicator calculations.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import pandas_ta with error handling
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("Warning: pandas_ta tidak terinstall. Menggunakan implementasi manual.")


def compute_rsi(series, period=14):
    """
    Compute RSI manually without pandas_ta dependency
    
    Args:
        series: Price series (typically close prices)
        period: RSI calculation period
        
    Returns:
        pandas.Series: RSI values
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use exponential weighted moving average for more responsive RSI
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    # Handle edge cases
    rsi = rsi.fillna(50)  # Neutral RSI for NaN values
    rsi = rsi.clip(0, 100)  # Ensure RSI is between 0 and 100
    
    return rsi


def compute_macd(close, fast=12, slow=26, signal=9):
    """
    Compute MACD manually
    
    Args:
        close: Close price series
        fast: Fast EMA period
        slow: Slow EMA period  
        signal: Signal line period
        
    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def compute_bollinger_bands(close, length=20, std=2):
    """
    Compute Bollinger Bands manually
    
    Args:
        close: Close price series
        length: Moving average length
        std: Standard deviation multiplier
        
    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    sma = close.rolling(window=length).mean()
    rolling_std = close.rolling(window=length).std()
    
    upper_band = sma + (rolling_std * std)
    lower_band = sma - (rolling_std * std)
    
    return upper_band, sma, lower_band


def compute_atr(high, low, close, period=14):
    """
    Compute Average True Range
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR calculation period
        
    Returns:
        pandas.Series: ATR values
    """
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    return atr


def compute_adx(high, low, close, length=14):
    """
    Compute ADX manually (simplified version)
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        length: ADX calculation period
        
    Returns:
        pandas.Series: ADX values
    """
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                       np.maximum(high - high.shift(1), 0), 0)
    dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), 
                        np.maximum(low.shift(1) - low, 0), 0)
    
    dm_plus = pd.Series(dm_plus, index=close.index)
    dm_minus = pd.Series(dm_minus, index=close.index)
    
    # Smoothed values
    atr = tr.ewm(alpha=1/length).mean()
    di_plus = 100 * (dm_plus.ewm(alpha=1/length).mean() / atr)
    di_minus = 100 * (dm_minus.ewm(alpha=1/length).mean() / atr)
    
    # ADX calculation
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = dx.ewm(alpha=1/length).mean()
    
    return adx.fillna(0)


def add_indicators(df):
    """
    Add technical indicators to dataframe
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame: Original DataFrame with added indicators
    """
    if df is None or df.empty:
        raise ValueError("DataFrame kosong atau None")
    
    if 'close' not in df.columns:
        raise ValueError("Column 'close' tidak ditemukan dalam DataFrame")
    
    df_copy = df.copy()
    
    try:
        # Add RSI
        df_copy['rsi'] = compute_rsi(df_copy['close'])
        
        # Add MACD
        macd_line, signal_line, histogram = compute_macd(df_copy['close'])
        df_copy['MACD_12_26_9'] = macd_line
        df_copy['MACDs_12_26_9'] = signal_line
        df_copy['MACDh_12_26_9'] = histogram
        
        # Add Bollinger Bands
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(df_copy['close'])
        df_copy['BB_upper'] = bb_upper
        df_copy['BB_middle'] = bb_middle
        df_copy['BB_lower'] = bb_lower
        
        # Add ATR
        if all(col in df_copy.columns for col in ['high', 'low']):
            df_copy['ATR_14'] = compute_atr(df_copy['high'], df_copy['low'], df_copy['close'])
        
        # Add ADX if OHLC data available
        if all(col in df_copy.columns for col in ['high', 'low']):
            df_copy['ADX_14'] = compute_adx(df_copy['high'], df_copy['low'], df_copy['close'])
        
        return df_copy
        
    except Exception as e:
        print(f"Error adding indicators: {e}")
        return df_copy