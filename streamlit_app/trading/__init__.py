# Trading module for position management and backtesting

from .position_manager import (
    calculate_position_info,
    calculate_ai_take_profit,
    execute_trade_exit_realistic,
    calculate_realistic_pnl
)

__all__ = [
    'calculate_position_info',
    'calculate_ai_take_profit',
    'execute_trade_exit_realistic', 
    'calculate_realistic_pnl'
]