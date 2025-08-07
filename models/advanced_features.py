# models/advanced_features.py
"""
Advanced feature engineering for Phase 2 implementation.
Multi-timeframe features, volatility clustering, support/resistance, and more.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import dependency manager for robust dependency handling
try:
    from utils.dependency_manager import dependency_manager, FallbackImplementations
    from utils.dependency_manager import is_available, require_dependency, with_fallback
    DEPENDENCY_MANAGER_AVAILABLE = True
except ImportError:
    # Fallback for systems without dependency manager
    DEPENDENCY_MANAGER_AVAILABLE = False
    print("Warning: Dependency manager not available, using legacy imports")

# Import dependencies with fallbacks
if DEPENDENCY_MANAGER_AVAILABLE:
    HAS_SCIPY = is_available('scipy')
    HAS_SKLEARN = is_available('sklearn')
    
    if HAS_SCIPY:
        from scipy import stats
        from scipy.signal import find_peaks
    
    if HAS_SKLEARN:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
else:
    # Legacy import handling
    try:
        from scipy import stats
        from scipy.signal import find_peaks
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        print("Warning: scipy not available, some advanced features will be disabled")

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        HAS_SKLEARN = True
    except ImportError:
        HAS_SKLEARN = False
        print("Warning: sklearn not available, some features will be disabled")


class MultiTimeframeFeatures:
    """Generate multi-timeframe features for enhanced model input."""
    
    def __init__(self, base_timeframe: str = '5m'):
        self.base_timeframe = base_timeframe
        self.higher_timeframes = {
            '5m': ['15m', '1h', '4h', '1d'],
            '15m': ['1h', '4h', '1d'],
            '1h': ['4h', '1d'],
            '4h': ['1d']
        }
    
    def resample_to_timeframe(self, data: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Resample data to higher timeframe."""
        # Map timeframe strings to pandas frequency strings
        tf_map = {
            '15m': '15T', '1h': '1H', '4h': '4H', '1d': '1D'
        }
        
        if target_tf not in tf_map:
            return data
        
        freq = tf_map[target_tf]
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.set_index('datetime' if 'datetime' in data.columns else data.index)
        
        # Resample OHLCV data
        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def calculate_trend_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate trend features for given timeframe."""
        if len(data) < 20:
            return {'trend_strength': 0.0, 'trend_direction': 0.0}
        
        # Simple trend calculation using linear regression
        close_prices = data['close'].values
        x = np.arange(len(close_prices))
        
        try:
            if HAS_SCIPY:
                slope, intercept, r_value, _, _ = stats.linregress(x, close_prices)
            elif DEPENDENCY_MANAGER_AVAILABLE:
                # Use fallback implementation
                slope, intercept, r_value, _, _ = FallbackImplementations.simple_linear_regression(x, close_prices)
            else:
                # Manual calculation fallback
                slope, intercept, r_value = self._manual_linear_regression(x, close_prices)
            
            trend_strength = abs(r_value)  # Correlation coefficient magnitude
            trend_direction = 1 if slope > 0 else -1 if slope < 0 else 0
            
        except Exception as e:
            print(f"Warning: Error calculating trend features: {e}")
            trend_strength = 0.0
            trend_direction = 0.0
        
        return {
            'trend_strength': trend_strength,
            'trend_direction': trend_direction
        }
    
    def _manual_linear_regression(self, x, y):
        """Manual linear regression calculation as ultimate fallback."""
        x = np.array(x)
        y = np.array(y)
        
        n = len(x)
        if n < 2:
            return 0, np.mean(y) if len(y) > 0 else 0, 0
        
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        # Calculate slope
        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sum((x - mean_x) ** 2)
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Calculate intercept
        intercept = mean_y - slope * mean_x
        
        # Calculate correlation coefficient
        if np.std(x) == 0 or np.std(y) == 0:
            r_value = 0
        else:
            r_value = np.corrcoef(x, y)[0, 1]
            if np.isnan(r_value):
                r_value = 0
        
        return slope, intercept, r_value
    
    def add_multi_timeframe_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add multi-timeframe trend features to the base data."""
        if self.base_timeframe not in self.higher_timeframes:
            return data
        
        result_data = data.copy()
        
        for tf in self.higher_timeframes[self.base_timeframe]:
            try:
                # Resample to higher timeframe
                higher_tf_data = self.resample_to_timeframe(data, tf)
                
                if len(higher_tf_data) < 10:
                    continue
                
                # Calculate trend features
                trend_features = self.calculate_trend_features(higher_tf_data)
                
                # Add features with timeframe suffix
                result_data[f'trend_strength_{tf}'] = trend_features['trend_strength']
                result_data[f'trend_direction_{tf}'] = trend_features['trend_direction']
                
                # Add higher timeframe price ratios
                if len(higher_tf_data) > 0:
                    latest_htf_close = higher_tf_data['close'].iloc[-1]
                    result_data[f'price_ratio_{tf}'] = data['close'] / latest_htf_close
                
            except Exception as e:
                print(f"Warning: Could not calculate {tf} features: {e}")
                continue
        
        return result_data


class VolatilityFeatures:
    """GARCH-based and other volatility features."""
    
    def __init__(self, window: int = 20):
        self.window = window
    
    def calculate_realized_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate realized volatility using high-low estimators."""
        # Parkinson volatility estimator (uses high-low)
        high_low_ratio = np.log(data['high'] / data['low'])
        parkinson_vol = np.sqrt(high_low_ratio ** 2 / (4 * np.log(2)))
        
        return parkinson_vol.rolling(window=self.window).mean()
    
    def calculate_garch_proxy(self, returns: pd.Series) -> pd.Series:
        """Calculate GARCH-like volatility clustering proxy."""
        # Simple GARCH(1,1) proxy using exponential smoothing
        alpha = 0.1  # Weight for recent squared returns
        beta = 0.8   # Weight for previous volatility
        
        vol_series = returns.copy() * 0  # Initialize
        vol_series.iloc[0] = returns.std()  # Initial volatility
        
        for i in range(1, len(returns)):
            if not np.isnan(returns.iloc[i-1]):
                vol_series.iloc[i] = np.sqrt(
                    alpha * returns.iloc[i-1]**2 + 
                    beta * vol_series.iloc[i-1]**2
                )
            else:
                vol_series.iloc[i] = vol_series.iloc[i-1]
        
        return vol_series
    
    def add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add various volatility features."""
        result_data = data.copy()
        
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Realized volatility
        result_data['realized_vol'] = self.calculate_realized_volatility(data)
        
        # GARCH proxy volatility
        result_data['garch_vol'] = self.calculate_garch_proxy(returns)
        
        # Volatility ratios and trends
        result_data['vol_ratio'] = (
            result_data['realized_vol'] / 
            result_data['realized_vol'].rolling(window=self.window*2).mean()
        )
        
        # Volume-price volatility
        if 'volume' in data.columns:
            volume_norm = data['volume'] / data['volume'].rolling(window=self.window).mean()
            price_vol = returns.abs()
            result_data['volume_vol_corr'] = (
                volume_norm.rolling(window=self.window).corr(price_vol)
            )
        
        return result_data


class SupportResistanceLevels:
    """Automated support and resistance level detection."""
    
    def __init__(self, lookback: int = 50, min_touches: int = 2):
        self.lookback = lookback
        self.min_touches = min_touches
    
    def find_swing_points(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Find swing highs and lows using peak detection."""
        if not HAS_SCIPY:
            if DEPENDENCY_MANAGER_AVAILABLE:
                # Use fallback implementation
                high_peaks, _ = FallbackImplementations.simple_peak_detection(data['high'].values, distance=5)
                low_peaks, _ = FallbackImplementations.simple_peak_detection(-data['low'].values, distance=5)
            else:
                # Simple local maxima/minima fallback
                highs = data['high'].rolling(window=5, center=True).max() == data['high']
                lows = data['low'].rolling(window=5, center=True).min() == data['low']
                return highs.values, lows.values
        else:
            # Use scipy for more sophisticated peak detection
            high_peaks, _ = find_peaks(data['high'].values, distance=5)
            low_peaks, _ = find_peaks(-data['low'].values, distance=5)
        
        # Convert to boolean arrays
        highs = np.zeros(len(data), dtype=bool)
        lows = np.zeros(len(data), dtype=bool)
        highs[high_peaks] = True
        lows[low_peaks] = True
        
        return highs, lows
    
    def detect_levels(self, data: pd.DataFrame) -> List[Dict]:
        """Detect support and resistance levels."""
        if len(data) < self.lookback:
            return []
        
        # Get recent data for analysis
        recent_data = data.tail(self.lookback).copy()
        highs, lows = self.find_swing_points(recent_data)
        
        # Extract swing high and low prices
        swing_highs = recent_data.loc[highs, 'high'].values
        swing_lows = recent_data.loc[lows, 'low'].values
        
        levels = []
        
        # Cluster similar price levels
        if HAS_SKLEARN and len(swing_highs) > 0:
            # Resistance levels from swing highs
            if len(swing_highs) >= self.min_touches:
                try:
                    kmeans = KMeans(n_clusters=min(3, len(swing_highs)), random_state=42)
                    clusters = kmeans.fit_predict(swing_highs.reshape(-1, 1))
                    
                    for i, center in enumerate(kmeans.cluster_centers_):
                        level_price = center[0]
                        touches = np.sum(clusters == i)
                        if touches >= self.min_touches:
                            levels.append({
                                'type': 'resistance',
                                'price': level_price,
                                'touches': touches,
                                'strength': min(touches / 5.0, 1.0)
                            })
                except Exception as e:
                    print(f"Warning: K-means clustering failed: {e}")
            
            # Support levels from swing lows
            if len(swing_lows) >= self.min_touches:
                try:
                    kmeans = KMeans(n_clusters=min(3, len(swing_lows)), random_state=42)
                    clusters = kmeans.fit_predict(swing_lows.reshape(-1, 1))
                    
                    for i, center in enumerate(kmeans.cluster_centers_):
                        level_price = center[0]
                        touches = np.sum(clusters == i)
                        if touches >= self.min_touches:
                            levels.append({
                                'type': 'support',
                                'price': level_price,
                                'touches': touches,
                                'strength': min(touches / 5.0, 1.0)
                            })
                except Exception as e:
                    print(f"Warning: K-means clustering failed: {e}")
        
        elif DEPENDENCY_MANAGER_AVAILABLE and len(swing_highs) > 0:
            # Use fallback clustering implementation
            try:
                # Resistance levels from swing highs
                if len(swing_highs) >= self.min_touches:
                    cluster_result = FallbackImplementations.simple_kmeans(swing_highs, n_clusters=min(3, len(swing_highs)))
                    for i, center in enumerate(cluster_result['cluster_centers_']):
                        level_price = center[0]
                        touches = cluster_result['labels_'].count(i)
                        if touches >= self.min_touches:
                            levels.append({
                                'type': 'resistance',
                                'price': level_price,
                                'touches': touches,
                                'strength': min(touches / 5.0, 1.0)
                            })
                
                # Support levels from swing lows
                if len(swing_lows) >= self.min_touches:
                    cluster_result = FallbackImplementations.simple_kmeans(swing_lows, n_clusters=min(3, len(swing_lows)))
                    for i, center in enumerate(cluster_result['cluster_centers_']):
                        level_price = center[0]
                        touches = cluster_result['labels_'].count(i)
                        if touches >= self.min_touches:
                            levels.append({
                                'type': 'support',
                                'price': level_price,
                                'touches': touches,
                                'strength': min(touches / 5.0, 1.0)
                            })
            except Exception as e:
                print(f"Warning: Fallback clustering failed: {e}")
        
        return levels
    
    def add_sr_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add support/resistance distance features."""
        result_data = data.copy()
        
        # Initialize default values
        result_data['dist_to_support'] = 0.0
        result_data['dist_to_resistance'] = 0.0
        result_data['sr_strength'] = 0.0
        
        # Detect levels
        levels = self.detect_levels(data)
        
        if levels:
            current_price = data['close'].iloc[-1]
            
            # Find nearest support and resistance
            support_levels = [l for l in levels if l['type'] == 'support']
            resistance_levels = [l for l in levels if l['type'] == 'resistance']
            
            if support_levels:
                nearest_support = max(
                    [l for l in support_levels if l['price'] < current_price], 
                    key=lambda x: x['price'], 
                    default=None
                )
                if nearest_support:
                    support_dist = (current_price - nearest_support['price']) / current_price
                    result_data['dist_to_support'] = support_dist
                    result_data['sr_strength'] += nearest_support['strength']
            
            if resistance_levels:
                nearest_resistance = min(
                    [l for l in resistance_levels if l['price'] > current_price], 
                    key=lambda x: x['price'], 
                    default=None
                )
                if nearest_resistance:
                    resistance_dist = (nearest_resistance['price'] - current_price) / current_price
                    result_data['dist_to_resistance'] = resistance_dist
                    result_data['sr_strength'] += nearest_resistance['strength']
        
        return result_data


class SeasonalityFeatures:
    """Time-based patterns and seasonality features."""
    
    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add hour/day/week seasonality features."""
        result_data = data.copy()
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'datetime' in data.columns:
                datetime_col = pd.to_datetime(data['datetime'])
            else:
                print("Warning: No datetime information available for seasonality features")
                return result_data
        else:
            datetime_col = data.index
        
        # Hour of day features (cyclical encoding)
        hour = datetime_col.hour
        result_data['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        result_data['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week features (cyclical encoding)
        day_of_week = datetime_col.dayofweek
        result_data['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        result_data['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # Market session indicators
        result_data['asian_session'] = ((hour >= 23) | (hour <= 8)).astype(int)
        result_data['london_session'] = ((hour >= 8) & (hour <= 16)).astype(int)
        result_data['new_york_session'] = ((hour >= 13) & (hour <= 22)).astype(int)
        result_data['overlap_session'] = (
            (result_data['london_session'] & result_data['new_york_session'])
        ).astype(int)
        
        return result_data


class RegimeDetection:
    """Bull/bear/sideways market classification."""
    
    def __init__(self, trend_window: int = 50, volatility_window: int = 20):
        self.trend_window = trend_window
        self.volatility_window = volatility_window
    
    def detect_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime."""
        if len(data) < max(self.trend_window, self.volatility_window):
            return 'sideways'
        
        # Trend detection using moving averages
        short_ma = data['close'].rolling(window=self.trend_window//2).mean()
        long_ma = data['close'].rolling(window=self.trend_window).mean()
        
        # Volatility measurement
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=self.volatility_window).std()
        
        # Current values
        current_price = data['close'].iloc[-1]
        current_short_ma = short_ma.iloc[-1]
        current_long_ma = long_ma.iloc[-1]
        current_vol = volatility.iloc[-1]
        
        # Regime classification logic
        trend_strength = abs(current_short_ma - current_long_ma) / current_long_ma
        
        if trend_strength > 0.02:  # Strong trend threshold
            if current_short_ma > current_long_ma:
                if current_vol < volatility.quantile(0.7):
                    return 'trending_bullish'
                else:
                    return 'volatile_bullish'
            else:
                if current_vol < volatility.quantile(0.7):
                    return 'trending_bearish'
                else:
                    return 'volatile_bearish'
        else:
            if current_vol > volatility.quantile(0.8):
                return 'volatile_sideways'
            else:
                return 'calm_sideways'
    
    def add_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add regime detection features."""
        result_data = data.copy()
        
        # Detect current regime
        current_regime = self.detect_regime(data)
        
        # One-hot encode regime
        regimes = ['trending_bullish', 'trending_bearish', 'volatile_bullish', 
                  'volatile_bearish', 'volatile_sideways', 'calm_sideways']
        
        for regime in regimes:
            result_data[f'regime_{regime}'] = 1 if current_regime == regime else 0
        
        # Add regime strength indicators
        if len(data) >= self.trend_window:
            short_ma = data['close'].rolling(window=self.trend_window//2).mean()
            long_ma = data['close'].rolling(window=self.trend_window).mean()
            result_data['trend_strength'] = abs(short_ma - long_ma) / long_ma
            
            returns = data['close'].pct_change()
            vol_ratio = (
                returns.rolling(window=self.volatility_window).std() / 
                returns.rolling(window=self.volatility_window*2).std()
            )
            result_data['volatility_regime'] = vol_ratio
        
        return result_data


class AdvancedDataPipeline:
    """Complete advanced feature engineering pipeline."""
    
    def __init__(self, base_timeframe: str = '5m'):
        self.feature_generators = [
            MultiTimeframeFeatures(base_timeframe),
            VolatilityFeatures(),
            SupportResistanceLevels(),
            SeasonalityFeatures(),
            RegimeDetection()
        ]
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature generators and return enhanced dataset."""
        enhanced_data = data.copy()
        
        print("Applying advanced feature engineering...")
        
        for generator in self.feature_generators:
            try:
                if isinstance(generator, MultiTimeframeFeatures):
                    enhanced_data = generator.add_multi_timeframe_features(enhanced_data)
                elif isinstance(generator, VolatilityFeatures):
                    enhanced_data = generator.add_volatility_features(enhanced_data)
                elif isinstance(generator, SupportResistanceLevels):
                    enhanced_data = generator.add_sr_features(enhanced_data)
                elif isinstance(generator, SeasonalityFeatures):
                    enhanced_data = generator.add_time_features(enhanced_data)
                elif isinstance(generator, RegimeDetection):
                    enhanced_data = generator.add_regime_features(enhanced_data)
                    
                print(f"✅ Applied {generator.__class__.__name__}")
                
            except Exception as e:
                print(f"❌ Error applying {generator.__class__.__name__}: {e}")
                continue
        
        # Handle missing data intelligently
        enhanced_data = self._handle_missing_data(enhanced_data)
        
        print(f"Enhanced data shape: {enhanced_data.shape}")
        return enhanced_data
    
    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Intelligent missing data handling."""
        # Forward fill recent missing values
        data = data.fillna(method='ffill', limit=5)
        
        # Fill remaining NaN with sensible defaults
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if data[col].isna().any():
                if 'ratio' in col or 'strength' in col:
                    data[col] = data[col].fillna(1.0)
                elif 'dist_to' in col:
                    data[col] = data[col].fillna(0.0)
                elif 'vol' in col:
                    data[col] = data[col].fillna(data[col].median())
                else:
                    data[col] = data[col].fillna(0.0)
        
        return data