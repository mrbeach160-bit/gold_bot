# trading/risk.py
"""
Risk Management System for Gold Bot

This module provides:
- Position sizing (Kelly criterion, fixed percentage, dynamic sizing)
- Risk limits (max drawdown, daily loss limits)
- Exposure control (max positions, correlation limits)
- Stop loss management (trailing stops, break-even moves)
- Risk metrics calculation
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import get_config


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_daily_loss: float = 0.02  # 2% max daily loss
    max_drawdown: float = 0.10    # 10% max drawdown
    max_position_size: float = 0.05  # 5% max position size
    max_open_positions: int = 3   # Maximum number of open positions
    max_correlation: float = 0.8  # Maximum correlation between positions
    stop_loss_percent: float = 0.01  # 1% stop loss
    take_profit_percent: float = 0.02  # 2% take profit


class PositionSizer:
    """Position sizing calculations"""
    
    def __init__(self, account_balance: float = 10000.0):
        """
        Initialize position sizer
        
        Args:
            account_balance: Account balance for calculations
        """
        self.account_balance = account_balance
        
        # Load config if available
        self.config = None
        try:
            self.config = get_config()
        except RuntimeError:
            pass
    
    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            win_rate: Historical win rate (0.0 to 1.0)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive number)
            
        Returns:
            Optimal position size as fraction of capital
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.01  # Default small size
        
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply safety margin (Kelly/4 rule)
        kelly_fraction = max(0, min(kelly_fraction / 4, 0.25))  # Cap at 25%
        
        return kelly_fraction
    
    def fixed_percentage(self, risk_percent: float = 0.02) -> float:
        """
        Calculate position size based on fixed percentage of account
        
        Args:
            risk_percent: Percentage of account to risk (default 2%)
            
        Returns:
            Position size as fraction of capital
        """
        return max(0.001, min(risk_percent, 0.1))  # Min 0.1%, max 10%
    
    def dynamic_sizing(self, signal_confidence: float, volatility: float, 
                      base_size: float = 0.02) -> float:
        """
        Calculate dynamic position size based on signal confidence and volatility
        
        Args:
            signal_confidence: Signal confidence (0.0 to 1.0)
            volatility: Market volatility measure
            base_size: Base position size
            
        Returns:
            Adjusted position size
        """
        # Adjust for confidence
        confidence_multiplier = signal_confidence
        
        # Adjust for volatility (inverse relationship)
        volatility_adjustment = 1.0 / (1.0 + volatility * 10)
        
        adjusted_size = base_size * confidence_multiplier * volatility_adjustment
        
        return max(0.001, min(adjusted_size, 0.1))  # Bounds check
    
    def calculate_position_size(self, signal: Dict[str, Any], method: str = 'fixed',
                              account_balance: float = None, **kwargs) -> float:
        """
        Calculate position size using specified method
        
        Args:
            signal: Trading signal with confidence and risk data
            method: Sizing method ('kelly', 'fixed', 'dynamic')
            account_balance: Account balance (uses instance balance if None)
            **kwargs: Additional parameters for sizing methods
            
        Returns:
            Position size in account currency
        """
        if account_balance is not None:
            self.account_balance = account_balance
        
        if method == 'kelly':
            win_rate = kwargs.get('win_rate', 0.5)
            avg_win = kwargs.get('avg_win', 1.0)
            avg_loss = kwargs.get('avg_loss', 1.0)
            fraction = self.kelly_criterion(win_rate, avg_win, avg_loss)
            
        elif method == 'dynamic':
            confidence = signal.get('confidence', 0.5)
            volatility = kwargs.get('volatility', 0.1)
            base_size = kwargs.get('base_size', 0.02)
            fraction = self.dynamic_sizing(confidence, volatility, base_size)
            
        else:  # fixed
            risk_percent = kwargs.get('risk_percent', 0.02)
            fraction = self.fixed_percentage(risk_percent)
        
        # Convert to position size in currency
        position_size = self.account_balance * fraction
        
        return position_size


class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, limits: RiskLimits = None):
        """
        Initialize risk manager
        
        Args:
            limits: Risk limits configuration
        """
        self.limits = limits or RiskLimits()
        self.position_sizer = PositionSizer()
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.peak_balance = 0.0
        self.current_drawdown = 0.0
        self.open_positions = []
        self.trade_history = []
        
        # Load config if available
        self.config = None
        try:
            self.config = get_config()
            if self.config and hasattr(self.config, 'risk'):
                self._update_limits_from_config()
        except RuntimeError:
            pass
    
    def _update_limits_from_config(self):
        """Update risk limits from configuration"""
        risk_config = self.config.risk
        
        if hasattr(risk_config, 'max_daily_loss'):
            self.limits.max_daily_loss = risk_config.max_daily_loss
        if hasattr(risk_config, 'max_drawdown'):
            self.limits.max_drawdown = risk_config.max_drawdown
        if hasattr(risk_config, 'max_position_size'):
            self.limits.max_position_size = risk_config.max_position_size
        if hasattr(risk_config, 'max_open_positions'):
            self.limits.max_open_positions = risk_config.max_open_positions
    
    def check_risk_limits(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if trade signal passes risk limits
        
        Args:
            signal: Trading signal to evaluate
            account_info: Current account information
            
        Returns:
            Risk check result with approval status and reasons
        """
        result = {
            'approved': True,
            'reasons': [],
            'warnings': [],
            'adjusted_signal': signal.copy()
        }
        
        account_balance = account_info.get('total_wallet_balance', 10000.0)
        
        # Update tracking variables
        self._update_account_state(account_info)
        
        # Check daily loss limit
        if self.daily_pnl <= -self.limits.max_daily_loss * account_balance:
            result['approved'] = False
            result['reasons'].append('Daily loss limit exceeded')
        
        # Check drawdown limit
        if self.current_drawdown >= self.limits.max_drawdown:
            result['approved'] = False
            result['reasons'].append('Maximum drawdown limit exceeded')
        
        # Check maximum open positions
        if len(self.open_positions) >= self.limits.max_open_positions:
            result['approved'] = False
            result['reasons'].append('Maximum open positions limit reached')
        
        # Check position size limit
        position_size = signal.get('position_size', 0)
        max_position_value = account_balance * self.limits.max_position_size
        
        if position_size > max_position_value:
            result['warnings'].append('Position size exceeds limit, reducing')
            result['adjusted_signal']['position_size'] = max_position_value
        
        # Check correlation with existing positions
        correlation_check = self._check_position_correlation(signal)
        if not correlation_check['approved']:
            result['approved'] = False
            result['reasons'].extend(correlation_check['reasons'])
        
        # Risk/reward check
        risk_reward = signal.get('risk_reward', 0)
        if risk_reward < 1.0:
            result['warnings'].append('Poor risk/reward ratio')
        
        return result
    
    def calculate_stop_loss(self, entry_price: float, side: str, atr: float = None,
                           method: str = 'percentage') -> float:
        """
        Calculate stop loss level
        
        Args:
            entry_price: Entry price for position
            side: Position side ('BUY' or 'SELL')
            atr: Average True Range for ATR-based stops
            method: Stop loss method ('percentage', 'atr', 'support_resistance')
            
        Returns:
            Stop loss price level
        """
        if method == 'atr' and atr is not None:
            # ATR-based stop loss
            multiplier = 2.0  # Default ATR multiplier
            if side == 'BUY':
                return entry_price - (atr * multiplier)
            else:
                return entry_price + (atr * multiplier)
        
        elif method == 'percentage':
            # Percentage-based stop loss
            stop_percent = self.limits.stop_loss_percent
            if side == 'BUY':
                return entry_price * (1 - stop_percent)
            else:
                return entry_price * (1 + stop_percent)
        
        else:
            # Default percentage method
            return self.calculate_stop_loss(entry_price, side, method='percentage')
    
    def calculate_take_profit(self, entry_price: float, side: str, 
                             risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit level
        
        Args:
            entry_price: Entry price for position
            side: Position side ('BUY' or 'SELL')
            risk_reward_ratio: Target risk/reward ratio
            
        Returns:
            Take profit price level
        """
        stop_loss = self.calculate_stop_loss(entry_price, side)
        
        if side == 'BUY':
            risk = entry_price - stop_loss
            return entry_price + (risk * risk_reward_ratio)
        else:
            risk = stop_loss - entry_price
            return entry_price - (risk * risk_reward_ratio)
    
    def update_trailing_stop(self, position: Dict[str, Any], current_price: float) -> Optional[float]:
        """
        Update trailing stop loss
        
        Args:
            position: Position information
            current_price: Current market price
            
        Returns:
            New stop loss level or None if no update
        """
        entry_price = position.get('entry_price', 0)
        current_stop = position.get('stop_loss', 0)
        side = position.get('side', 'BUY')
        
        trailing_percent = 0.005  # 0.5% trailing stop
        
        if side == 'BUY':
            # For long positions, only move stop loss up
            new_stop = current_price * (1 - trailing_percent)
            if new_stop > current_stop:
                return new_stop
        else:
            # For short positions, only move stop loss down
            new_stop = current_price * (1 + trailing_percent)
            if new_stop < current_stop:
                return new_stop
        
        return None
    
    def _update_account_state(self, account_info: Dict[str, Any]):
        """Update internal risk tracking state"""
        current_balance = account_info.get('total_wallet_balance', 0)
        
        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Calculate current drawdown
        if self.peak_balance > 0:
            self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
        
        # Update open positions from account info
        self.open_positions = account_info.get('positions', [])
    
    def _check_position_correlation(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Check correlation with existing positions"""
        # Simple correlation check - in production, this would be more sophisticated
        symbol = signal.get('symbol', 'UNKNOWN')
        action = signal.get('action', 'HOLD')
        
        # Count positions in same direction
        same_direction_count = 0
        for pos in self.open_positions:
            pos_side = 'BUY' if pos.get('position_amt', 0) > 0 else 'SELL'
            if pos_side == action:
                same_direction_count += 1
        
        # Limit positions in same direction
        if same_direction_count >= 2:
            return {
                'approved': False,
                'reasons': ['Too many positions in same direction']
            }
        
        return {'approved': True, 'reasons': []}
    
    def record_trade(self, trade: Dict[str, Any]):
        """Record completed trade for risk analysis"""
        trade['timestamp'] = datetime.now()
        self.trade_history.append(trade)
        
        # Update PnL tracking
        pnl = trade.get('pnl', 0)
        self.total_pnl += pnl
        
        # Update daily PnL (reset daily if needed)
        self.daily_pnl += pnl
        
        # Keep history limited
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Calculate current risk metrics"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_consecutive_losses': 0,
                'current_drawdown': self.current_drawdown
            }
        
        trades = pd.DataFrame(self.trade_history)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        
        # Profit factor
        total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for _, trade in trades.iterrows():
            if trade['pnl'] < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_consecutive_losses': max_consecutive_losses,
            'current_drawdown': self.current_drawdown,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl
        }
    
    def emergency_stop_check(self) -> bool:
        """Check if emergency stop conditions are met"""
        # Emergency stop conditions
        emergency_conditions = [
            self.current_drawdown >= self.limits.max_drawdown * 0.8,  # 80% of max drawdown
            self.daily_pnl <= -self.limits.max_daily_loss * 0.8,     # 80% of daily limit
        ]
        
        return any(emergency_conditions)
    
    def reset_daily_tracking(self):
        """Reset daily tracking variables (call at start of new trading day)"""
        self.daily_pnl = 0.0


# Factory function
def create_risk_manager(config_limits: bool = True) -> RiskManager:
    """
    Create risk manager instance
    
    Args:
        config_limits: Whether to load limits from configuration
        
    Returns:
        RiskManager instance
    """
    if config_limits:
        return RiskManager()
    else:
        return RiskManager(RiskLimits())