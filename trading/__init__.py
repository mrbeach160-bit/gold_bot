# trading/__init__.py
"""
Unified Trading System for Gold Bot

This module provides consolidated trading functionality including:
- Trading execution engine
- Strategy framework
- Risk management system
- Centralized trading management through TradingManager

Key Components:
- engine: Trading execution and order management
- strategies: Trading strategy implementations  
- risk: Risk management and position sizing
- manager: Centralized trading management interface
"""

from .manager import TradingManager
from .engine import TradingEngine
from .strategies import BaseStrategy, MLStrategy, TechnicalStrategy
from .risk import RiskManager, PositionSizer

__all__ = [
    'TradingManager',
    'TradingEngine',
    'BaseStrategy',
    'MLStrategy', 
    'TechnicalStrategy',
    'RiskManager',
    'PositionSizer'
]