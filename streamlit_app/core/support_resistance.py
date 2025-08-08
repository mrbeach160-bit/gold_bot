"""
Support and Resistance Level Calculation

Contains functions for calculating support and resistance levels from price data.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def get_support_resistance(data, n=20):
    """
    Find support and resistance levels
    
    Args:
        data: DataFrame with OHLC data
        n: lookback period for finding local extremes
        
    Returns:
        tuple: (support_levels, resistance_levels) as pandas Series
    """
    if len(data) < n*2+1:
        return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    
    # Find local minima and maxima
    lows = data['low'].rolling(window=n*2+1, center=True).min()
    highs = data['high'].rolling(window=n*2+1, center=True).max()
    
    support_levels = lows[lows == data['low']].dropna()
    resistance_levels = highs[highs == data['high']].dropna()
    
    return support_levels, resistance_levels


def get_fibonacci_levels(data, lookback_period=252):
    """
    Calculate Fibonacci retracement levels
    
    Args:
        data: DataFrame with OHLC data
        lookback_period: number of periods to look back for high/low
        
    Returns:
        dict: Fibonacci levels with names and values
    """
    if len(data) < lookback_period:
        lookback_data = data
    else:
        lookback_data = data.tail(lookback_period)
    
    high_price = lookback_data['high'].max()
    low_price = lookback_data['low'].min()
    price_range = high_price - low_price
    
    if price_range == 0:
        return {f'Level {i}': high_price for i in range(7)}
    
    levels = {
        'Fib 0% (High)': high_price,
        'Fib 23.6%': high_price - (price_range * 0.236),
        'Fib 38.2%': high_price - (price_range * 0.382),
        'Fib 50%': high_price - (price_range * 0.5),
        'Fib 61.8%': high_price - (price_range * 0.618),
        'Fib 100% (Low)': low_price,
    }
    return levels