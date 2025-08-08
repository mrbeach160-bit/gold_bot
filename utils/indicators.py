# utils/indicators.py
# Versi yang sudah diperbaiki - menghapus OBV dan MFI yang bermasalah

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import pandas_ta dengan error handling
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("Warning: pandas_ta tidak terinstall. Menggunakan implementasi manual.")

def compute_rsi(series, period=14):
    """Compute RSI manually without pandas_ta dependency"""
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
    """Compute MACD manually"""
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def compute_bollinger_bands(close, length=20, std=2):
    """Compute Bollinger Bands manually"""
    sma = close.rolling(window=length).mean()
    rolling_std = close.rolling(window=length).std()
    
    upper_band = sma + (rolling_std * std)
    lower_band = sma - (rolling_std * std)
    
    return upper_band, sma, lower_band

def compute_adx(high, low, close, length=14):
    """Compute ADX manually (simplified version)"""
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

def compute_parabolic_sar(high, low, close, af_start=0.02, af_max=0.2):
    """Compute Parabolic SAR manually (simplified)"""
    length = len(close)
    psar = pd.Series(index=close.index, dtype=float)
    
    if length < 2:
        return psar.fillna(close.iloc[0] if len(close) > 0 else 0)
    
    # Initialize
    psar.iloc[0] = low.iloc[0]
    trend = 1  # 1 for uptrend, -1 for downtrend
    af = af_start
    ep = high.iloc[0] if trend == 1 else low.iloc[0]
    
    for i in range(1, length):
        # Calculate PSAR
        psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
        
        # Check for trend reversal
        if trend == 1 and low.iloc[i] <= psar.iloc[i]:
            # Reversal to downtrend
            trend = -1
            psar.iloc[i] = ep
            af = af_start
            ep = low.iloc[i]
        elif trend == -1 and high.iloc[i] >= psar.iloc[i]:
            # Reversal to uptrend
            trend = 1
            psar.iloc[i] = ep
            af = af_start
            ep = high.iloc[i]
        else:
            # Continue current trend
            if trend == 1:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_start, af_max)
                # Ensure PSAR doesn't go above previous two lows
                psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1])
                if i > 1:
                    psar.iloc[i] = min(psar.iloc[i], low.iloc[i-2])
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_start, af_max)
                # Ensure PSAR doesn't go below previous two highs
                psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1])
                if i > 1:
                    psar.iloc[i] = max(psar.iloc[i], high.iloc[i-2])
    
    return psar.fillna(method='ffill').fillna(close)

def get_support_resistance(data, n=20):
    """Find support and resistance levels"""
    if len(data) < n*2+1:
        return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    
    # Find local minima and maxima
    lows = data['low'].rolling(window=n*2+1, center=True).min()
    highs = data['high'].rolling(window=n*2+1, center=True).max()
    
    support_levels = lows[lows == data['low']].dropna()
    resistance_levels = highs[highs == data['high']].dropna()
    
    return support_levels, resistance_levels

def get_fibonacci_levels(data, lookback_period=252):
    """Calculate Fibonacci retracement levels"""
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

def add_indicators(df):
    """Add technical indicators to dataframe"""
    if df is None or df.empty:
        raise ValueError("DataFrame kosong atau None")
    
    if 'close' not in df.columns:
        raise ValueError("DataFrame harus memiliki kolom 'close'")
    
    # Ensure we have minimum required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame harus memiliki kolom: {missing_cols}")
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    try:
        # 1. Exponential Moving Averages
        df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_fast"] = df["EMA_10"]  # Alias for backward compatibility
        df["ema_slow"] = df["EMA_20"]  # Alias for backward compatibility
        
        # EMA Signal
        df["ema_signal"] = df.apply(
            lambda row: "BUY" if row["ema_fast"] > row["ema_slow"] 
            else "SELL" if row["ema_fast"] < row["ema_slow"] 
            else "HOLD", axis=1
        )
        
        # 2. RSI
        df["rsi"] = compute_rsi(df["close"])
        
        # 3. MACD
        macd_line, signal_line, histogram = compute_macd(df["close"])
        df["MACD_12_26_9"] = macd_line
        df["MACDs_12_26_9"] = signal_line
        df["MACDh_12_26_9"] = histogram
        
        # 4. Bollinger Bands
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(df["close"])
        df["BBU_20_2.0"] = bb_upper
        df["BBM_20_2.0"] = bb_middle
        df["BBL_20_2.0"] = bb_lower
        
        # 5. ADX
        df["ADX_14"] = compute_adx(df["high"], df["low"], df["close"])
        
        # 6. Parabolic SAR
        df["PSAR_0.02_0.2"] = compute_parabolic_sar(df["high"], df["low"], df["close"])
        
        # 7. Simple Moving Averages
        df["SMA_20"] = df["close"].rolling(window=20).mean()
        df["SMA_50"] = df["close"].rolling(window=50).mean()
        
        # 8. Stochastic Oscillator
        lowest_low = df["low"].rolling(window=14).min()
        highest_high = df["high"].rolling(window=14).max()
        k_percent = 100 * ((df["close"] - lowest_low) / (highest_high - lowest_low))
        df["STOCH_k"] = k_percent.rolling(window=3).mean()
        df["STOCH_d"] = df["STOCH_k"].rolling(window=3).mean()
        
        # 9. Commodity Channel Index (CCI)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mean_deviation = typical_price.rolling(window=20).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        df["CCI_20"] = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        # 10. Williams %R
        highest_high_14 = df["high"].rolling(window=14).max()
        lowest_low_14 = df["low"].rolling(window=14).min()
        df["WILLR_14"] = -100 * ((highest_high_14 - df["close"]) / (highest_high_14 - lowest_low_14))
        
        # 11. Average True Range (ATR)
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift(1))
        tr3 = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR_14"] = tr.ewm(alpha=1/14).mean()
        
        # 12. Price Rate of Change (ROC)
        df["ROC_10"] = ((df["close"] - df["close"].shift(10)) / df["close"].shift(10)) * 100
        
        # Fill any remaining NaN values with forward fill, then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Drop rows that still have NaN values (usually first few rows)
        df = df.dropna()
        
        print(f"Successfully added indicators. DataFrame shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error adding indicators: {str(e)}")
        # Return original dataframe with basic indicators if advanced ones fail
        try:
            df["EMA_10"] = df["close"].ewm(span=10).mean()
            df["EMA_20"] = df["close"].ewm(span=20).mean()
            df["rsi"] = compute_rsi(df["close"])
            df = df.fillna(method='ffill').fillna(method='bfill').dropna()
            print("Fallback to basic indicators successful")
            return df
        except Exception as e2:
            print(f"Even basic indicators failed: {str(e2)}")
            raise e2