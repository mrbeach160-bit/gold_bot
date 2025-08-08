# trading/strategies.py
"""
Trading Strategies Framework for Gold Bot

This module provides:
- Base strategy interface
- ML-based trading strategy
- Technical analysis strategy
- Hybrid strategies combining ML and technical analysis

All strategies return standardized signal format for consistent execution.
"""

import pandas as pd
import numpy as np
import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import get_config


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str = None):
        """
        Initialize strategy
        
        Args:
            name: Strategy name (optional)
        """
        self.name = name or self.__class__.__name__
        self.config = None
        try:
            self.config = get_config()
        except RuntimeError:
            pass
        
        # Strategy parameters
        self.parameters = {}
        self.backtesting_mode = False
        self.last_signal = None
        self.signal_history = []
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal from market data
        
        Args:
            data: Market data with OHLC and indicators
            
        Returns:
            Signal dict with standardized format:
            {
                'action': 'BUY'|'SELL'|'HOLD',
                'confidence': float,  # 0.0 to 1.0
                'stop_loss': float,   # Price level
                'take_profit': float, # Price level
                'position_size': float, # Lot size
                'strategy_name': str,
                'timestamp': datetime,
                'risk_reward': float
            }
        """
        pass
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set strategy parameters"""
        self.parameters.update(parameters)
    
    def set_backtesting_mode(self, mode: bool):
        """Set backtesting mode"""
        self.backtesting_mode = mode
    
    def get_signal_history(self) -> List[Dict[str, Any]]:
        """Get history of generated signals"""
        return self.signal_history.copy()
    
    def _create_signal(self, action: str, confidence: float, current_price: float,
                      stop_loss: float = None, take_profit: float = None,
                      position_size: float = None, risk_reward: float = None) -> Dict[str, Any]:
        """
        Create standardized signal dict
        
        Args:
            action: Trading action ('BUY', 'SELL', 'HOLD')
            confidence: Signal confidence (0.0 to 1.0)
            current_price: Current market price
            stop_loss: Stop loss price level
            take_profit: Take profit price level
            position_size: Position size in lots
            risk_reward: Risk/reward ratio
            
        Returns:
            Standardized signal dict
        """
        # Default position size from config
        if position_size is None and self.config:
            position_size = self.config.trading.position_size
        if position_size is None:
            position_size = 0.01  # Default
        
        # Calculate stop loss and take profit if not provided
        if action in ['BUY', 'SELL'] and (stop_loss is None or take_profit is None):
            atr_multiplier = 2.0  # Default ATR multiplier
            
            if action == 'BUY':
                if stop_loss is None:
                    stop_loss = current_price * 0.99  # 1% stop loss
                if take_profit is None:
                    take_profit = current_price * 1.02  # 2% take profit
            else:  # SELL
                if stop_loss is None:
                    stop_loss = current_price * 1.01  # 1% stop loss
                if take_profit is None:
                    take_profit = current_price * 0.98  # 2% take profit
        
        # Calculate risk/reward ratio
        if risk_reward is None and action != 'HOLD':
            if action == 'BUY':
                risk = current_price - stop_loss
                reward = take_profit - current_price
            else:  # SELL
                risk = stop_loss - current_price
                reward = current_price - take_profit
            
            risk_reward = reward / risk if risk > 0 else 0
        
        signal = {
            'action': action,
            'confidence': max(0.0, min(1.0, confidence)),  # Clamp between 0 and 1
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'strategy_name': self.name,
            'timestamp': datetime.now(),
            'risk_reward': risk_reward or 0,
            'current_price': current_price
        }
        
        # Store signal
        self.last_signal = signal
        self.signal_history.append(signal)
        
        # Keep history limited to last 1000 signals
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        return signal


class MLStrategy(BaseStrategy):
    """Machine Learning based trading strategy"""
    
    def __init__(self, model_manager=None):
        """
        Initialize ML strategy
        
        Args:
            model_manager: ML model manager from Phase 2
        """
        super().__init__("MLStrategy")
        self.model_manager = model_manager
        
        # Default parameters
        self.parameters = {
            'confidence_threshold': 0.6,
            'prediction_threshold': 0.55,
            'lookback_periods': 60,
            'model_type': 'ensemble'  # 'lstm', 'xgb', 'ensemble'
        }
    
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate ML-based trading signal"""
        if data is None or data.empty or len(data) < self.parameters['lookback_periods']:
            return self._create_signal('HOLD', 0.0, data['close'].iloc[-1] if not data.empty else 0)
        
        current_price = data['close'].iloc[-1]
        
        try:
            # Get model prediction if available
            if self.model_manager:
                prediction = self._get_model_prediction(data)
                if prediction is not None:
                    return self._convert_prediction_to_signal(prediction, current_price)
            
            # Fallback to simple ML-like signal based on indicators
            return self._generate_indicator_based_signal(data)
            
        except Exception as e:
            print(f"Error in MLStrategy: {e}")
            return self._create_signal('HOLD', 0.0, current_price)
    
    def _get_model_prediction(self, data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Get prediction from ML model"""
        try:
            # This would integrate with the model manager from Phase 2
            # For now, return None to use fallback
            return None
            
        except Exception as e:
            print(f"Error getting model prediction: {e}")
            return None
    
    def _convert_prediction_to_signal(self, prediction: Dict[str, float], current_price: float) -> Dict[str, Any]:
        """Convert model prediction to trading signal"""
        # Assuming prediction contains probability scores
        buy_prob = prediction.get('buy_probability', 0.5)
        sell_prob = prediction.get('sell_probability', 0.5)
        
        threshold = self.parameters['prediction_threshold']
        
        if buy_prob > threshold and buy_prob > sell_prob:
            action = 'BUY'
            confidence = buy_prob
        elif sell_prob > threshold and sell_prob > buy_prob:
            action = 'SELL'
            confidence = sell_prob
        else:
            action = 'HOLD'
            confidence = max(buy_prob, sell_prob)
        
        return self._create_signal(action, confidence, current_price)
    
    def _generate_indicator_based_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signal based on technical indicators as ML proxy"""
        current_price = data['close'].iloc[-1]
        
        # Use technical indicators as features for ML-like decision
        features = {}
        
        # RSI feature
        if 'rsi' in data.columns:
            rsi = data['rsi'].iloc[-1]
            features['rsi_oversold'] = 1 if rsi < 30 else 0
            features['rsi_overbought'] = 1 if rsi > 70 else 0
        
        # MACD feature
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            macd = data['macd'].iloc[-1]
            macd_signal = data['macd_signal'].iloc[-1]
            features['macd_bullish'] = 1 if macd > macd_signal else 0
        
        # Bollinger Bands feature
        if all(col in data.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            bb_upper = data['bb_upper'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            bb_middle = data['bb_middle'].iloc[-1]
            
            features['bb_oversold'] = 1 if current_price < bb_lower else 0
            features['bb_overbought'] = 1 if current_price > bb_upper else 0
        
        # Simple scoring system
        bullish_score = (
            features.get('rsi_oversold', 0) +
            features.get('macd_bullish', 0) +
            features.get('bb_oversold', 0)
        )
        
        bearish_score = (
            features.get('rsi_overbought', 0) +
            (1 - features.get('macd_bullish', 0)) +
            features.get('bb_overbought', 0)
        )
        
        # Generate signal
        if bullish_score >= 2:
            action = 'BUY'
            confidence = min(0.8, 0.5 + bullish_score * 0.1)
        elif bearish_score >= 2:
            action = 'SELL'
            confidence = min(0.8, 0.5 + bearish_score * 0.1)
        else:
            action = 'HOLD'
            confidence = 0.3
        
        return self._create_signal(action, confidence, current_price)


class TechnicalStrategy(BaseStrategy):
    """Pure technical analysis trading strategy"""
    
    def __init__(self):
        super().__init__("TechnicalStrategy")
        
        # Default parameters
        self.parameters = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'use_macd': True,
            'use_bollinger_bands': True,
            'use_ema_crossover': True,
            'min_confidence': 0.5
        }
    
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate technical analysis based signal"""
        if data is None or data.empty or len(data) < 20:
            return self._create_signal('HOLD', 0.0, data['close'].iloc[-1] if not data.empty else 0)
        
        current_price = data['close'].iloc[-1]
        signals = []
        
        # RSI Signal
        rsi_signal = self._get_rsi_signal(data)
        if rsi_signal:
            signals.append(rsi_signal)
        
        # MACD Signal
        if self.parameters['use_macd']:
            macd_signal = self._get_macd_signal(data)
            if macd_signal:
                signals.append(macd_signal)
        
        # Bollinger Bands Signal
        if self.parameters['use_bollinger_bands']:
            bb_signal = self._get_bollinger_signal(data)
            if bb_signal:
                signals.append(bb_signal)
        
        # EMA Crossover Signal
        if self.parameters['use_ema_crossover']:
            ema_signal = self._get_ema_crossover_signal(data)
            if ema_signal:
                signals.append(ema_signal)
        
        # Combine signals
        return self._combine_signals(signals, current_price)
    
    def _get_rsi_signal(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get RSI-based signal"""
        if 'rsi' not in data.columns or len(data) < 2:
            return None
        
        rsi_current = data['rsi'].iloc[-1]
        rsi_prev = data['rsi'].iloc[-2]
        
        if rsi_current < self.parameters['rsi_oversold'] and rsi_prev >= rsi_current:
            return {'action': 'BUY', 'strength': 0.8, 'indicator': 'RSI'}
        elif rsi_current > self.parameters['rsi_overbought'] and rsi_prev <= rsi_current:
            return {'action': 'SELL', 'strength': 0.8, 'indicator': 'RSI'}
        
        return None
    
    def _get_macd_signal(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get MACD-based signal"""
        if not all(col in data.columns for col in ['macd', 'macd_signal']) or len(data) < 2:
            return None
        
        macd_current = data['macd'].iloc[-1]
        macd_signal_current = data['macd_signal'].iloc[-1]
        macd_prev = data['macd'].iloc[-2]
        macd_signal_prev = data['macd_signal'].iloc[-2]
        
        # MACD bullish crossover
        if (macd_current > macd_signal_current and macd_prev <= macd_signal_prev):
            return {'action': 'BUY', 'strength': 0.7, 'indicator': 'MACD'}
        # MACD bearish crossover
        elif (macd_current < macd_signal_current and macd_prev >= macd_signal_prev):
            return {'action': 'SELL', 'strength': 0.7, 'indicator': 'MACD'}
        
        return None
    
    def _get_bollinger_signal(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get Bollinger Bands signal"""
        if not all(col in data.columns for col in ['bb_upper', 'bb_lower']) or len(data) < 2:
            return None
        
        close_current = data['close'].iloc[-1]
        close_prev = data['close'].iloc[-2]
        bb_upper = data['bb_upper'].iloc[-1]
        bb_lower = data['bb_lower'].iloc[-1]
        bb_upper_prev = data['bb_upper'].iloc[-2]
        bb_lower_prev = data['bb_lower'].iloc[-2]
        
        # Price bouncing off lower band (oversold)
        if close_prev <= bb_lower_prev and close_current > bb_lower:
            return {'action': 'BUY', 'strength': 0.6, 'indicator': 'Bollinger'}
        # Price bouncing off upper band (overbought)
        elif close_prev >= bb_upper_prev and close_current < bb_upper:
            return {'action': 'SELL', 'strength': 0.6, 'indicator': 'Bollinger'}
        
        return None
    
    def _get_ema_crossover_signal(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get EMA crossover signal"""
        if not all(col in data.columns for col in ['ema_10', 'ema_20']) or len(data) < 2:
            return None
        
        ema_fast_current = data['ema_10'].iloc[-1]
        ema_slow_current = data['ema_20'].iloc[-1]
        ema_fast_prev = data['ema_10'].iloc[-2]
        ema_slow_prev = data['ema_20'].iloc[-2]
        
        # Golden cross (bullish)
        if ema_fast_current > ema_slow_current and ema_fast_prev <= ema_slow_prev:
            return {'action': 'BUY', 'strength': 0.7, 'indicator': 'EMA_Cross'}
        # Death cross (bearish)
        elif ema_fast_current < ema_slow_current and ema_fast_prev >= ema_slow_prev:
            return {'action': 'SELL', 'strength': 0.7, 'indicator': 'EMA_Cross'}
        
        return None
    
    def _combine_signals(self, signals: List[Dict[str, Any]], current_price: float) -> Dict[str, Any]:
        """Combine multiple technical signals"""
        if not signals:
            return self._create_signal('HOLD', 0.0, current_price)
        
        # Count signals by action
        buy_signals = [s for s in signals if s['action'] == 'BUY']
        sell_signals = [s for s in signals if s['action'] == 'SELL']
        
        # Calculate weighted scores
        buy_score = sum(s['strength'] for s in buy_signals)
        sell_score = sum(s['strength'] for s in sell_signals)
        
        # Determine final action
        if buy_score > sell_score and buy_score >= self.parameters['min_confidence']:
            action = 'BUY'
            confidence = min(0.9, buy_score / len(signals))
        elif sell_score > buy_score and sell_score >= self.parameters['min_confidence']:
            action = 'SELL'
            confidence = min(0.9, sell_score / len(signals))
        else:
            action = 'HOLD'
            confidence = max(buy_score, sell_score) / len(signals) if signals else 0.0
        
        return self._create_signal(action, confidence, current_price)


class HybridStrategy(BaseStrategy):
    """Hybrid strategy combining ML and technical analysis"""
    
    def __init__(self, model_manager=None):
        super().__init__("HybridStrategy")
        self.ml_strategy = MLStrategy(model_manager)
        self.technical_strategy = TechnicalStrategy()
        
        # Default parameters
        self.parameters = {
            'ml_weight': 0.6,
            'technical_weight': 0.4,
            'min_agreement_threshold': 0.5,
            'confidence_boost': 0.1
        }
    
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate hybrid signal combining ML and technical analysis"""
        if data is None or data.empty:
            return self._create_signal('HOLD', 0.0, data['close'].iloc[-1] if not data.empty else 0)
        
        current_price = data['close'].iloc[-1]
        
        try:
            # Get signals from both strategies
            ml_signal = self.ml_strategy.generate_signal(data)
            technical_signal = self.technical_strategy.generate_signal(data)
            
            # Combine signals
            return self._combine_hybrid_signals(ml_signal, technical_signal, current_price)
            
        except Exception as e:
            print(f"Error in HybridStrategy: {e}")
            return self._create_signal('HOLD', 0.0, current_price)
    
    def _combine_hybrid_signals(self, ml_signal: Dict[str, Any], 
                               technical_signal: Dict[str, Any], 
                               current_price: float) -> Dict[str, Any]:
        """Combine ML and technical signals"""
        ml_weight = self.parameters['ml_weight']
        tech_weight = self.parameters['technical_weight']
        
        # Calculate weighted scores
        ml_action = ml_signal['action']
        tech_action = technical_signal['action']
        
        ml_confidence = ml_signal['confidence']
        tech_confidence = technical_signal['confidence']
        
        # Check for agreement
        actions_agree = ml_action == tech_action
        
        if actions_agree and ml_action != 'HOLD':
            # Both strategies agree on direction
            action = ml_action
            confidence = (ml_confidence * ml_weight + tech_confidence * tech_weight)
            
            # Boost confidence when strategies agree
            confidence = min(0.95, confidence + self.parameters['confidence_boost'])
            
        elif ml_action != 'HOLD' and tech_action == 'HOLD':
            # Only ML strategy has signal
            action = ml_action
            confidence = ml_confidence * ml_weight
            
        elif tech_action != 'HOLD' and ml_action == 'HOLD':
            # Only technical strategy has signal
            action = tech_action
            confidence = tech_confidence * tech_weight
            
        else:
            # Strategies disagree or both are HOLD
            if ml_confidence * ml_weight > tech_confidence * tech_weight:
                action = ml_action
                confidence = ml_confidence * ml_weight
            else:
                action = tech_action
                confidence = tech_confidence * tech_weight
            
            # Reduce confidence when strategies disagree
            if not actions_agree and ml_action != 'HOLD' and tech_action != 'HOLD':
                confidence *= 0.7  # Reduce confidence by 30%
        
        # Apply minimum agreement threshold
        if confidence < self.parameters['min_agreement_threshold']:
            action = 'HOLD'
        
        return self._create_signal(action, confidence, current_price)


# Strategy factory function
def create_strategy(strategy_type: str, **kwargs) -> BaseStrategy:
    """
    Create strategy instance based on type
    
    Args:
        strategy_type: Type of strategy ('ml', 'technical', 'hybrid')
        **kwargs: Additional arguments for strategy initialization
        
    Returns:
        Strategy instance
    """
    strategy_type = strategy_type.lower()
    
    if strategy_type == 'ml':
        return MLStrategy(kwargs.get('model_manager'))
    elif strategy_type == 'technical':
        return TechnicalStrategy()
    elif strategy_type == 'hybrid':
        return HybridStrategy(kwargs.get('model_manager'))
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


# Default strategy selection based on config
def get_default_strategy(**kwargs) -> BaseStrategy:
    """Get default strategy based on configuration"""
    try:
        config = get_config()
        strategy_type = config.strategy.default_strategy if hasattr(config, 'strategy') else 'technical'
    except RuntimeError:
        strategy_type = 'technical'  # Safe default
    
    return create_strategy(strategy_type, **kwargs)