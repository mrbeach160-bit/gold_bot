"""
Feature Service for the modular application.
Handles feature engineering and technical indicators.
"""

import pandas as pd
import numpy as np
import streamlit as st
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.indicators import add_indicators


class FeatureService:
    """Service for feature engineering and technical indicators."""
    
    def __init__(self):
        # Define feature sets used by different models
        self.base_features = ['open', 'high', 'low', 'close']
        
        self.technical_features = [
            'rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20', 'ATR_14', 'STOCHk_14_3_3'
        ]
        
        self.engineered_features = [
            'price_change', 'high_low_ratio', 'open_close_ratio'
        ]
        
        # Feature sets for different models
        self.xgb_features = self.base_features + self.technical_features + self.engineered_features
        self.svc_features = self.base_features + self.technical_features
        self.nb_features = self.base_features + self.technical_features
        self.cnn_features = self.base_features + ['rsi', 'MACD_12_26_9', 'EMA_10', 'EMA_20']
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        try:
            # Use existing utility to add indicators
            data_with_indicators = add_indicators(data.copy())
            return data_with_indicators
            
        except Exception as e:
            st.warning(f"Error adding technical indicators: {str(e)}")
            # Return original data if indicators fail
            return data.copy()
    
    def add_engineered_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to the data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with engineered features added
        """
        try:
            df = data.copy()
            
            # Price change percentage
            df['price_change'] = df['close'].pct_change()
            
            # High/Low ratio
            df['high_low_ratio'] = df['high'] / df['low']
            
            # Open/Close ratio  
            df['open_close_ratio'] = df['open'] / df['close']
            
            # Fill NaN values
            df['price_change'] = df['price_change'].fillna(0)
            df['high_low_ratio'] = df['high_low_ratio'].fillna(1)
            df['open_close_ratio'] = df['open_close_ratio'].fillna(1)
            
            return df
            
        except Exception as e:
            st.warning(f"Error adding engineered features: {str(e)}")
            return data.copy()
    
    def prepare_feature_set(self, data: pd.DataFrame, model_type: str) -> pd.DataFrame:
        """
        Prepare feature set for a specific model type.
        
        Args:
            data: DataFrame with full feature set
            model_type: Type of model ('xgb', 'svc', 'nb', 'cnn')
            
        Returns:
            DataFrame with features for the specified model
        """
        try:
            if model_type == 'xgb':
                features = self.xgb_features
            elif model_type == 'svc':
                features = self.svc_features
            elif model_type == 'nb':
                features = self.nb_features
            elif model_type == 'cnn':
                features = self.cnn_features
            else:
                st.warning(f"Unknown model type: {model_type}")
                features = self.base_features
            
            # Select available features
            available_features = [f for f in features if f in data.columns]
            missing_features = [f for f in features if f not in data.columns]
            
            if missing_features:
                st.warning(f"Missing features for {model_type}: {missing_features}")
            
            # Return feature subset with missing values filled
            feature_data = data[available_features].copy()
            
            # Fill missing values with appropriate defaults
            for col in feature_data.columns:
                if col in ['rsi']:
                    feature_data[col] = feature_data[col].fillna(50)  # Neutral RSI
                elif col in ['MACD_12_26_9', 'price_change']:
                    feature_data[col] = feature_data[col].fillna(0)  # No change
                elif col in ['high_low_ratio', 'open_close_ratio']:
                    feature_data[col] = feature_data[col].fillna(1)  # Neutral ratio
                elif col in ['EMA_10', 'EMA_20']:
                    feature_data[col] = feature_data[col].fillna(data['close'])  # Use close price
                else:
                    feature_data[col] = feature_data[col].fillna(0)  # Default to 0
            
            return feature_data
            
        except Exception as e:
            st.error(f"Error preparing features for {model_type}: {str(e)}")
            return pd.DataFrame()
    
    def prepare_lstm_sequence(self, data: pd.DataFrame, sequence_length: int = 60) -> np.ndarray:
        """
        Prepare sequence data for LSTM model.
        
        Args:
            data: DataFrame with close prices
            sequence_length: Length of sequence to use
            
        Returns:
            Array of close prices for LSTM input
        """
        try:
            if len(data) < sequence_length:
                st.warning(f"Not enough data for LSTM sequence. Need {sequence_length}, got {len(data)}")
                # Pad with the first available value
                close_prices = data['close'].values
                if len(close_prices) > 0:
                    padded = np.pad(close_prices, (sequence_length - len(close_prices), 0), 
                                  mode='constant', constant_values=close_prices[0])
                    return padded
                else:
                    return np.zeros(sequence_length)
            
            return data['close'].tail(sequence_length).values
            
        except Exception as e:
            st.error(f"Error preparing LSTM sequence: {str(e)}")
            return np.zeros(sequence_length)
    
    def prepare_cnn_window(self, data: pd.DataFrame, window_size: int = 20) -> np.ndarray:
        """
        Prepare windowed data for CNN model.
        
        Args:
            data: DataFrame with features
            window_size: Size of the window
            
        Returns:
            Array with windowed features for CNN
        """
        try:
            feature_data = self.prepare_feature_set(data, 'cnn')
            
            if len(feature_data) < window_size:
                st.warning(f"Not enough data for CNN window. Need {window_size}, got {len(feature_data)}")
                # Pad with the last available values
                if len(feature_data) > 0:
                    last_row = feature_data.iloc[-1:].values
                    padding_needed = window_size - len(feature_data)
                    padding = np.tile(last_row, (padding_needed, 1))
                    windowed_data = np.vstack([padding, feature_data.values])
                else:
                    windowed_data = np.zeros((window_size, len(self.cnn_features)))
            else:
                windowed_data = feature_data.tail(window_size).values
            
            return windowed_data
            
        except Exception as e:
            st.error(f"Error preparing CNN window: {str(e)}")
            return np.zeros((window_size, len(self.cnn_features)))
    
    def get_latest_row_features(self, data: pd.DataFrame, model_type: str) -> np.ndarray:
        """
        Get the latest row of features for single-row prediction models.
        
        Args:
            data: DataFrame with features
            model_type: Type of model ('xgb', 'svc', 'nb')
            
        Returns:
            Array with latest row features
        """
        try:
            feature_data = self.prepare_feature_set(data, model_type)
            
            if feature_data.empty:
                st.error(f"No feature data available for {model_type}")
                return np.array([])
            
            return feature_data.iloc[-1:].values
            
        except Exception as e:
            st.error(f"Error getting latest row features for {model_type}: {str(e)}")
            return np.array([])