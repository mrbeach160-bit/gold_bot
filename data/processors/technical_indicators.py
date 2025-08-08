# data/processors/technical_indicators.py
"""
Technical Indicators Module

Provides comprehensive technical analysis indicators including:
- Trend indicators (EMA, SMA, MACD)
- Momentum indicators (RSI, Stochastic)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class TechnicalIndicators:
    """Technical analysis indicators calculator"""
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average (EMA)"""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA)"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return williams_r
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume (OBV)"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        return obv
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to a dataframe"""
        if df.empty or len(df) < 50:
            return df.copy()
            
        result = df.copy()
        
        try:
            # RSI
            result['rsi'] = TechnicalIndicators.rsi(result['close'])
            
            # Moving Averages
            result['ema_10'] = TechnicalIndicators.ema(result['close'], 10)
            result['ema_20'] = TechnicalIndicators.ema(result['close'], 20)
            result['sma_50'] = TechnicalIndicators.sma(result['close'], 50)
            
            # MACD
            macd_data = TechnicalIndicators.macd(result['close'])
            result['macd'] = macd_data['macd']
            result['macd_signal'] = macd_data['signal']
            result['macd_histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            bb_data = TechnicalIndicators.bollinger_bands(result['close'])
            result['bb_upper'] = bb_data['bb_upper']
            result['bb_middle'] = bb_data['bb_middle']
            result['bb_lower'] = bb_data['bb_lower']
            
            # ATR
            result['atr'] = TechnicalIndicators.atr(
                result['high'], result['low'], result['close']
            )
            
            # Stochastic
            stoch_data = TechnicalIndicators.stochastic(
                result['high'], result['low'], result['close']
            )
            result['stoch_k'] = stoch_data['stoch_k']
            result['stoch_d'] = stoch_data['stoch_d']
            
            # Williams %R
            result['williams_r'] = TechnicalIndicators.williams_r(
                result['high'], result['low'], result['close']
            )
            
            # OBV
            result['obv'] = TechnicalIndicators.obv(result['close'], result['volume'])
            
            # Price-based indicators
            result['price_change'] = result['close'].pct_change()
            result['price_range'] = (result['high'] - result['low']) / result['close']
            result['body_size'] = abs(result['close'] - result['open']) / result['close']
            
        except Exception as e:
            print(f"⚠️  Error adding technical indicators: {e}")
            
        return result
    
    @staticmethod
    def get_signals(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on technical indicators"""
        if df.empty or 'rsi' not in df.columns:
            return {'signal': 'HOLD', 'strength': 0.0, 'reasons': []}
            
        latest = df.iloc[-1]
        signals = []
        strength = 0.0
        
        # RSI signals
        if latest['rsi'] < 30:
            signals.append("RSI oversold")
            strength += 0.3
        elif latest['rsi'] > 70:
            signals.append("RSI overbought")
            strength -= 0.3
            
        # MACD signals
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            if latest['macd'] > latest['macd_signal']:
                signals.append("MACD bullish")
                strength += 0.2
            else:
                signals.append("MACD bearish")
                strength -= 0.2
                
        # Moving average signals
        if 'ema_10' in df.columns and 'ema_20' in df.columns:
            if latest['ema_10'] > latest['ema_20']:
                signals.append("EMA bullish cross")
                strength += 0.2
            else:
                signals.append("EMA bearish cross")
                strength -= 0.2
        
        # Determine overall signal
        if strength > 0.3:
            signal = 'BUY'
        elif strength < -0.3:
            signal = 'SELL'
        else:
            signal = 'HOLD'
            
        return {
            'signal': signal,
            'strength': abs(strength),
            'reasons': signals
        }