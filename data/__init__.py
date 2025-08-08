# data/__init__.py
"""
Data Management System for Gold Bot Phase 3

This module provides comprehensive data management functionality including:
- Centralized data management through DataManager
- Multiple data provider support (Twelve Data, Yahoo Finance, Binance)
- Data processing and technical indicators
- Data validation and cleaning
- Configuration-driven API management

Key Components:
- manager: DataManager class for centralized data operations
- providers: Data provider abstraction layer
- processors: Data processing and technical indicators
"""

from .manager import DataManager

__all__ = [
    'DataManager'
]