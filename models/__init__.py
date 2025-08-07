# models/__init__.py
"""
Unified Model Architecture for Gold Bot
Provides standardized model interfaces, factory patterns, and centralized management.
"""

from .base import BaseModel, create_model
from .ml_models import LSTMModel, LightGBMModel, XGBoostModel, CNNModel, SVCModel, NaiveBayesModel
from .ensemble import MetaLearner, VotingEnsemble, WeightedVotingEnsemble
from .manager import ModelManager

__all__ = [
    'BaseModel',
    'create_model',
    'LSTMModel', 
    'LightGBMModel',
    'XGBoostModel',
    'CNNModel',
    'SVCModel',
    'NaiveBayesModel',
    'MetaLearner',
    'VotingEnsemble',
    'WeightedVotingEnsemble',
    'ModelManager'
]