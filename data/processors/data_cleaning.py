# data/processors/data_cleaning.py
"""
Data Cleaning and Validation Module

Provides data quality assurance including:
- OHLCV data validation
- Outlier detection and removal
- Missing data handling
- Data consistency checks
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple


class DataCleaning:
    """Data cleaning and validation utilities"""
    
    @staticmethod
    def validate_ohlc_data(df: pd.DataFrame) -> bool:
        """
        Validate OHLCV data for consistency
        
        Checks:
        - High >= Open, Close, Low
        - Low <= Open, Close, High
        - All prices > 0
        - Volume >= 0
        """
        if df.empty:
            return False
            
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return False
            
        try:
            # Check price relationships
            high_valid = (df['high'] >= df['open']) & (df['high'] >= df['close']) & (df['high'] >= df['low'])
            low_valid = (df['low'] <= df['open']) & (df['low'] <= df['close']) & (df['low'] <= df['high'])
            
            # Check positive prices
            positive_prices = (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)
            
            # Check volume if present
            volume_valid = True
            if 'volume' in df.columns:
                volume_valid = df['volume'] >= 0
                
            return high_valid.all() and low_valid.all() and positive_prices.all() and volume_valid.all()
            
        except Exception:
            return False
    
    @staticmethod
    def clean_ohlc_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean OHLCV data by fixing common issues"""
        if df.empty:
            return df
            
        cleaned = df.copy()
        
        # Remove rows with any NaN values in OHLC
        ohlc_cols = ['open', 'high', 'low', 'close']
        available_cols = [col for col in ohlc_cols if col in cleaned.columns]
        cleaned = cleaned.dropna(subset=available_cols)
        
        if cleaned.empty:
            return cleaned
            
        # Fix price relationships
        for idx in cleaned.index:
            try:
                # Ensure high is the maximum
                max_price = max(cleaned.loc[idx, 'open'], cleaned.loc[idx, 'close'])
                if cleaned.loc[idx, 'high'] < max_price:
                    cleaned.loc[idx, 'high'] = max_price
                    
                # Ensure low is the minimum  
                min_price = min(cleaned.loc[idx, 'open'], cleaned.loc[idx, 'close'])
                if cleaned.loc[idx, 'low'] > min_price:
                    cleaned.loc[idx, 'low'] = min_price
                    
                # Ensure all prices are positive
                for col in ohlc_cols:
                    if col in cleaned.columns and cleaned.loc[idx, col] <= 0:
                        if idx > 0:
                            cleaned.loc[idx, col] = cleaned.iloc[idx-1][col]
                        else:
                            cleaned.loc[idx, col] = 100  # Default price
                            
            except Exception:
                continue
                
        # Fix volume
        if 'volume' in cleaned.columns:
            cleaned['volume'] = cleaned['volume'].fillna(0)
            cleaned['volume'] = cleaned['volume'].clip(lower=0)
            
        return cleaned
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str = 'close', 
                       method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers from data"""
        if df.empty or column not in df.columns:
            return df
            
        cleaned = df.copy()
        
        if method == 'iqr':
            Q1 = cleaned[column].quantile(0.25)
            Q3 = cleaned[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (cleaned[column] >= lower_bound) & (cleaned[column] <= upper_bound)
            cleaned = cleaned[mask]
            
        elif method == 'zscore':
            z_scores = np.abs((cleaned[column] - cleaned[column].mean()) / cleaned[column].std())
            mask = z_scores < threshold
            cleaned = cleaned[mask]
            
        return cleaned
    
    @staticmethod
    def fill_missing_data(df: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
        """Fill missing data in OHLCV dataframe"""
        if df.empty:
            return df
            
        filled = df.copy()
        
        if method == 'forward':
            filled = filled.fillna(method='ffill')
        elif method == 'backward':
            filled = filled.fillna(method='bfill')
        elif method == 'interpolate':
            filled = filled.interpolate(method='linear')
        elif method == 'mean':
            filled = filled.fillna(filled.mean())
            
        return filled
    
    @staticmethod
    def detect_gaps(df: pd.DataFrame, expected_frequency: str = '5min') -> pd.DataFrame:
        """Detect time gaps in data"""
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return pd.DataFrame()
            
        # Calculate expected time differences
        freq_minutes = {
            '1min': 1,
            '5min': 5,
            '15min': 15,
            '30min': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        
        expected_diff = pd.Timedelta(minutes=freq_minutes.get(expected_frequency, 5))
        
        # Find actual gaps
        time_diffs = df.index.to_series().diff()
        gaps = time_diffs > expected_diff * 1.5  # Allow 50% tolerance
        
        gap_info = []
        for idx, is_gap in gaps.items():
            if is_gap:
                gap_info.append({
                    'start_time': df.index[df.index.get_loc(idx) - 1],
                    'end_time': idx,
                    'duration': time_diffs[idx],
                    'expected': expected_diff
                })
                
        return pd.DataFrame(gap_info)
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        if df.empty:
            return {'valid': False, 'errors': ['Empty dataset']}
            
        errors = []
        warnings = []
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            
        # Check data types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column {col} is not numeric")
                    
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            warnings.append(f"Found {missing_count} missing values")
            
        # Check OHLC relationships
        if not DataCleaning.validate_ohlc_data(df):
            errors.append("OHLC data validation failed")
            
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate rows")
            
        # Check data length
        if len(df) < 50:
            warnings.append("Dataset is very small (< 50 rows)")
            
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'row_count': len(df),
            'missing_values': missing_count,
            'duplicate_rows': duplicate_count
        }