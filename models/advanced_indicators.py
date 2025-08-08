# models/advanced_indicators.py
"""
Enhanced technical indicators for Phase 2 implementation.
Advanced RSI, MACD, Bollinger Bands, and additional indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Try to import technical analysis libraries
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AdvancedRSI:
    """Multi-timeframe RSI with divergence detection."""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate_rsi(self, series: pd.Series, period: int = None) -> pd.Series:
        """Calculate RSI with improved algorithm."""
        if period is None:
            period = self.period
        
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use Wilder's smoothing (exponential moving average)
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        
        # Calculate RSI
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        rsi = rsi.fillna(50)
        rsi = rsi.clip(0, 100)
        
        return rsi
    
    def detect_divergence(self, price_series: pd.Series, rsi_series: pd.Series, 
                         window: int = 20) -> pd.Series:
        """Detect RSI-price divergences."""
        if len(price_series) < window * 2:
            return pd.Series(index=price_series.index, data=0)
        
        divergence_signal = pd.Series(index=price_series.index, data=0)
        
        for i in range(window, len(price_series) - window):
            # Get windows for analysis
            price_window = price_series.iloc[i-window:i+window]
            rsi_window = rsi_series.iloc[i-window:i+window]
            
            # Find local maxima and minima
            price_max_idx = price_window.idxmax()
            price_min_idx = price_window.idxmin()
            rsi_max_idx = rsi_window.idxmax()
            rsi_min_idx = rsi_window.idxmin()
            
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (price_min_idx == price_window.index[window] and 
                rsi_min_idx != rsi_window.index[window]):
                if (price_window.iloc[window] < price_window.iloc[0] and
                    rsi_window.iloc[window] > rsi_window.iloc[0]):
                    divergence_signal.iloc[i] = 1  # Bullish divergence
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            if (price_max_idx == price_window.index[window] and 
                rsi_max_idx != rsi_window.index[window]):
                if (price_window.iloc[window] > price_window.iloc[0] and
                    rsi_window.iloc[window] < rsi_window.iloc[0]):
                    divergence_signal.iloc[i] = -1  # Bearish divergence
        
        return divergence_signal
    
    def add_advanced_rsi_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced RSI features."""
        result_data = data.copy()
        
        # Standard RSI
        result_data['rsi'] = self.calculate_rsi(data['close'])
        
        # Multi-period RSI
        result_data['rsi_fast'] = self.calculate_rsi(data['close'], period=7)
        result_data['rsi_slow'] = self.calculate_rsi(data['close'], period=21)
        
        # RSI divergence
        result_data['rsi_divergence'] = self.detect_divergence(
            data['close'], result_data['rsi']
        )
        
        # RSI momentum and acceleration
        result_data['rsi_momentum'] = result_data['rsi'].diff()
        result_data['rsi_acceleration'] = result_data['rsi_momentum'].diff()
        
        # RSI levels (overbought/oversold zones)
        result_data['rsi_overbought'] = (result_data['rsi'] > 70).astype(int)
        result_data['rsi_oversold'] = (result_data['rsi'] < 30).astype(int)
        result_data['rsi_neutral'] = (
            (result_data['rsi'] >= 40) & (result_data['rsi'] <= 60)
        ).astype(int)
        
        return result_data


class EnhancedMACD:
    """Enhanced MACD with histogram slope and acceleration."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def calculate_macd(self, series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD components."""
        ema_fast = series.ewm(span=self.fast).mean()
        ema_slow = series.ewm(span=self.slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def add_enhanced_macd_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced MACD features."""
        result_data = data.copy()
        
        # Standard MACD
        macd_line, signal_line, histogram = self.calculate_macd(data['close'])
        
        result_data['macd'] = macd_line
        result_data['macd_signal'] = signal_line
        result_data['macd_histogram'] = histogram
        
        # MACD histogram slope and acceleration
        result_data['macd_hist_slope'] = histogram.diff()
        result_data['macd_hist_acceleration'] = result_data['macd_hist_slope'].diff()
        
        # MACD crossover signals
        result_data['macd_bullish_cross'] = (
            (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        ).astype(int)
        result_data['macd_bearish_cross'] = (
            (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        ).astype(int)
        
        # MACD zero line crossovers
        result_data['macd_above_zero'] = (macd_line > 0).astype(int)
        result_data['macd_zero_cross_up'] = (
            (macd_line > 0) & (macd_line.shift(1) <= 0)
        ).astype(int)
        result_data['macd_zero_cross_down'] = (
            (macd_line < 0) & (macd_line.shift(1) >= 0)
        ).astype(int)
        
        # MACD strength indicators
        result_data['macd_strength'] = abs(macd_line - signal_line)
        result_data['macd_momentum'] = macd_line.diff()
        
        return result_data


class AdvancedBollingerBands:
    """Bollinger Bands with position and width dynamics."""
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std
    
    def calculate_bollinger_bands(self, series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = series.rolling(window=self.window).mean()
        std = series.rolling(window=self.window).std()
        
        upper_band = sma + (std * self.num_std)
        lower_band = sma - (std * self.num_std)
        
        return upper_band, sma, lower_band
    
    def add_bollinger_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced Bollinger Band features."""
        result_data = data.copy()
        
        # Standard Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['close'])
        
        result_data['bb_upper'] = bb_upper
        result_data['bb_middle'] = bb_middle
        result_data['bb_lower'] = bb_lower
        
        # Position within bands (0 = lower band, 1 = upper band)
        result_data['bb_percent'] = (
            (data['close'] - bb_lower) / (bb_upper - bb_lower)
        ).clip(0, 1)
        
        # Band width dynamics
        result_data['bb_width'] = (bb_upper - bb_lower) / bb_middle
        result_data['bb_width_ratio'] = (
            result_data['bb_width'] / result_data['bb_width'].rolling(window=self.window).mean()
        )
        
        # Bollinger Band squeeze detection
        bb_width_sma = result_data['bb_width'].rolling(window=20).mean()
        bb_width_std = result_data['bb_width'].rolling(window=20).std()
        result_data['bb_squeeze'] = (
            result_data['bb_width'] < (bb_width_sma - bb_width_std)
        ).astype(int)
        
        # Band touch signals
        result_data['bb_touch_upper'] = (data['high'] >= bb_upper).astype(int)
        result_data['bb_touch_lower'] = (data['low'] <= bb_lower).astype(int)
        
        # Band breakout signals
        result_data['bb_breakout_up'] = (
            (data['close'] > bb_upper) & (data['close'].shift(1) <= bb_upper.shift(1))
        ).astype(int)
        result_data['bb_breakout_down'] = (
            (data['close'] < bb_lower) & (data['close'].shift(1) >= bb_lower.shift(1))
        ).astype(int)
        
        return result_data


class StochasticOscillator:
    """Stochastic with %K/%D crossovers and divergences."""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
    
    def calculate_stochastic(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic %K and %D."""
        # Calculate raw %K
        lowest_low = data['low'].rolling(window=self.k_period).min()
        highest_high = data['high'].rolling(window=self.k_period).max()
        
        raw_k = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
        
        # Smooth %K
        percent_k = raw_k.rolling(window=self.smooth_k).mean()
        
        # Calculate %D (moving average of %K)
        percent_d = percent_k.rolling(window=self.d_period).mean()
        
        return percent_k, percent_d
    
    def add_stochastic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Stochastic oscillator features."""
        result_data = data.copy()
        
        # Standard Stochastic
        stoch_k, stoch_d = self.calculate_stochastic(data)
        
        result_data['stoch_k'] = stoch_k
        result_data['stoch_d'] = stoch_d
        
        # Stochastic crossovers
        result_data['stoch_bullish_cross'] = (
            (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))
        ).astype(int)
        result_data['stoch_bearish_cross'] = (
            (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))
        ).astype(int)
        
        # Stochastic levels
        result_data['stoch_overbought'] = (stoch_k > 80).astype(int)
        result_data['stoch_oversold'] = (stoch_k < 20).astype(int)
        
        # Stochastic momentum
        result_data['stoch_momentum'] = stoch_k.diff()
        result_data['stoch_acceleration'] = result_data['stoch_momentum'].diff()
        
        return result_data


class ADXIndicator:
    """ADX/DMI for trend strength and directional movement."""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate_adx(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX, +DI, and -DI."""
        # True Range
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift(1))
        low_close = abs(data['low'] - data['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = data['high'].diff()
        minus_dm = -data['low'].diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Smoothed values using Wilder's smoothing
        atr = tr.ewm(alpha=1/self.period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/self.period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/self.period, adjust=False).mean() / atr)
        
        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/self.period, adjust=False).mean()
        
        return adx, plus_di, minus_di
    
    def add_adx_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add ADX and directional movement features."""
        result_data = data.copy()
        
        # Calculate ADX components
        adx, plus_di, minus_di = self.calculate_adx(data)
        
        result_data['adx'] = adx
        result_data['plus_di'] = plus_di
        result_data['minus_di'] = minus_di
        
        # Trend strength classification
        result_data['trend_weak'] = (adx < 25).astype(int)
        result_data['trend_strong'] = (adx > 50).astype(int)
        result_data['trend_very_strong'] = (adx > 75).astype(int)
        
        # Directional bias
        result_data['bullish_bias'] = (plus_di > minus_di).astype(int)
        result_data['bearish_bias'] = (minus_di > plus_di).astype(int)
        
        # ADX momentum
        result_data['adx_momentum'] = adx.diff()
        result_data['adx_rising'] = (result_data['adx_momentum'] > 0).astype(int)
        
        return result_data


class VolumeIndicators:
    """Volume-based indicators: OBV, Volume ROC, A/D Line."""
    
    def __init__(self):
        pass
    
    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=data.index, data=0.0)
        obv.iloc[0] = data['volume'].iloc[0]
        
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def calculate_ad_line(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        # Money Flow Multiplier
        mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        mfm = mfm.fillna(0)  # Handle division by zero
        
        # Money Flow Volume
        mfv = mfm * data['volume']
        
        # Accumulation/Distribution Line
        ad_line = mfv.cumsum()
        
        return ad_line
    
    def add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        if 'volume' not in data.columns:
            print("Warning: Volume data not available")
            return data
        
        result_data = data.copy()
        
        # On-Balance Volume
        result_data['obv'] = self.calculate_obv(data)
        
        # OBV moving averages
        result_data['obv_sma'] = result_data['obv'].rolling(window=20).mean()
        result_data['obv_trend'] = (result_data['obv'] > result_data['obv_sma']).astype(int)
        
        # Volume Rate of Change
        result_data['volume_roc'] = data['volume'].pct_change(periods=10) * 100
        
        # Accumulation/Distribution Line
        result_data['ad_line'] = self.calculate_ad_line(data)
        
        # Volume-Price Trend (VPT)
        price_change_pct = data['close'].pct_change()
        result_data['vpt'] = (price_change_pct * data['volume']).cumsum()
        
        # Volume moving averages and ratios
        result_data['volume_sma'] = data['volume'].rolling(window=20).mean()
        result_data['volume_ratio'] = data['volume'] / result_data['volume_sma']
        
        # Volume spikes
        vol_std = data['volume'].rolling(window=20).std()
        result_data['volume_spike'] = (
            data['volume'] > (result_data['volume_sma'] + 2 * vol_std)
        ).astype(int)
        
        return result_data


class EnhancedIndicators:
    """Complete enhanced technical indicators package."""
    
    def __init__(self):
        self.rsi_calculator = AdvancedRSI()
        self.macd_calculator = EnhancedMACD()
        self.bb_calculator = AdvancedBollingerBands()
        self.stoch_calculator = StochasticOscillator()
        self.adx_calculator = ADXIndicator()
        self.volume_calculator = VolumeIndicators()
    
    def add_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all enhanced technical indicators."""
        print("Adding enhanced technical indicators...")
        
        enhanced_data = data.copy()
        
        try:
            enhanced_data = self.rsi_calculator.add_advanced_rsi_features(enhanced_data)
            print("✅ Advanced RSI indicators added")
        except Exception as e:
            print(f"❌ Error adding RSI indicators: {e}")
        
        try:
            enhanced_data = self.macd_calculator.add_enhanced_macd_features(enhanced_data)
            print("✅ Enhanced MACD indicators added")
        except Exception as e:
            print(f"❌ Error adding MACD indicators: {e}")
        
        try:
            enhanced_data = self.bb_calculator.add_bollinger_features(enhanced_data)
            print("✅ Advanced Bollinger Bands added")
        except Exception as e:
            print(f"❌ Error adding Bollinger Bands: {e}")
        
        try:
            enhanced_data = self.stoch_calculator.add_stochastic_features(enhanced_data)
            print("✅ Stochastic oscillator added")
        except Exception as e:
            print(f"❌ Error adding Stochastic: {e}")
        
        try:
            enhanced_data = self.adx_calculator.add_adx_features(enhanced_data)
            print("✅ ADX/DMI indicators added")
        except Exception as e:
            print(f"❌ Error adding ADX: {e}")
        
        try:
            enhanced_data = self.volume_calculator.add_volume_features(enhanced_data)
            print("✅ Volume indicators added")
        except Exception as e:
            print(f"❌ Error adding volume indicators: {e}")
        
        print(f"Enhanced indicators data shape: {enhanced_data.shape}")
        return enhanced_data