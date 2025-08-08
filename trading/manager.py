# trading/manager.py
"""
Centralized Trading Management for Gold Bot

The TradingManager provides a unified interface for:
- Strategy execution and signal generation
- Risk management and position sizing  
- Order execution and position management
- Trading status monitoring and control
- Integration with data and configuration systems
"""

import pandas as pd
import os
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import get_config
from .engine import TradingEngine
from .strategies import BaseStrategy, create_strategy, get_default_strategy
from .risk import RiskManager, PositionSizer, RiskLimits


class TradingManager:
    """Centralized trading management system"""
    
    def __init__(self, symbol: str = None, timeframe: str = None, 
                 strategy: BaseStrategy = None, data_manager=None):
        """
        Initialize Trading Manager
        
        Args:
            symbol: Trading symbol (uses config default if None)
            timeframe: Trading timeframe (uses config default if None)
            strategy: Trading strategy instance (uses default if None)
            data_manager: Data manager instance for market data
        """
        # Load configuration
        self.config = None
        try:
            self.config = get_config()
        except RuntimeError:
            pass
        
        # Set symbol and timeframe
        self.symbol = symbol
        self.timeframe = timeframe
        
        if self.config:
            if self.symbol is None:
                self.symbol = self.config.trading.symbol
            if self.timeframe is None:
                self.timeframe = self.config.trading.timeframe
        
        # Set defaults if still None
        if self.symbol is None:
            self.symbol = 'XAUUSD'
        if self.timeframe is None:
            self.timeframe = '5m'
        
        # Initialize components
        self.data_manager = data_manager
        self.trading_engine = None
        self.strategy = strategy or get_default_strategy()
        self.risk_manager = RiskManager()
        self.position_sizer = PositionSizer()
        
        # Trading state
        self.is_trading_enabled = False
        self.is_initialized = False
        self.last_signal = None
        self.last_execution_time = None
        self.execution_interval = 60  # 1 minute between executions
        
        # Performance tracking
        self.trades_today = 0
        self.signals_generated = 0
        self.successful_executions = 0
        self.failed_executions = 0
        
        # Initialize trading engine if credentials available
        self._initialize_trading_engine()
    
    def _initialize_trading_engine(self):
        """Initialize trading engine with proper error handling"""
        try:
            self.trading_engine = TradingEngine()
            if self.trading_engine.is_connected:
                print(f"✅ Trading engine initialized successfully")
            else:
                print(f"⚠️  Trading engine initialized but not connected")
        except Exception as e:
            print(f"⚠️  Could not initialize trading engine: {e}")
            self.trading_engine = None
    
    def initialize_trading(self, enable_trading: bool = False) -> bool:
        """
        Initialize trading system
        
        Args:
            enable_trading: Whether to enable live trading
            
        Returns:
            True if initialization successful
        """
        try:
            # Check data manager
            if self.data_manager is None:
                print("⚠️  Warning: No data manager provided")
                return False
            
            # Test data connection
            test_data = self.data_manager.get_live_data(bars=10)
            if test_data is None or test_data.empty:
                print("❌ Failed to get market data")
                return False
            
            print(f"✅ Market data connection verified")
            
            # Check trading engine if live trading enabled
            if enable_trading:
                if self.trading_engine is None or not self.trading_engine.is_connected:
                    print("❌ Trading engine not available for live trading")
                    return False
                
                # Test account access
                account_info = self.trading_engine.get_account_info()
                if account_info is None:
                    print("❌ Could not access trading account")
                    return False
                
                print(f"✅ Trading account connected - Balance: {account_info.get('total_wallet_balance', 0)}")
            
            # Initialize risk manager with account info
            if enable_trading and self.trading_engine:
                account_info = self.trading_engine.get_account_info()
                if account_info:
                    balance = account_info.get('total_wallet_balance', 10000)
                    self.position_sizer.account_balance = balance
            
            self.is_trading_enabled = enable_trading
            self.is_initialized = True
            
            print(f"✅ Trading manager initialized successfully")
            print(f"📊 Symbol: {self.symbol}, Timeframe: {self.timeframe}")
            print(f"🎯 Strategy: {self.strategy.name}")
            print(f"💰 Live Trading: {'Enabled' if enable_trading else 'Disabled (Paper Trading)'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Trading initialization failed: {e}")
            return False
    
    def execute_strategy(self, strategy_name: str = None, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Execute trading strategy and generate signal
        
        Args:
            strategy_name: Strategy to use (optional, uses current strategy if None)
            data: Market data (optional, fetches if None)
            
        Returns:
            Execution result with signal and status
        """
        if not self.is_initialized:
            return {'success': False, 'error': 'Trading manager not initialized'}
        
        try:
            # Switch strategy if requested
            if strategy_name and strategy_name != self.strategy.name:
                self.strategy = create_strategy(strategy_name)
            
            # Get market data if not provided
            if data is None:
                if self.data_manager is None:
                    return {'success': False, 'error': 'No data manager available'}
                
                data = self.data_manager.get_live_data(bars=100)
                if data is None or data.empty:
                    return {'success': False, 'error': 'Could not fetch market data'}
                
                # Preprocess data with indicators
                data = self.data_manager.preprocess_data(data)
            
            # Generate trading signal
            signal = self.strategy.generate_signal(data)
            self.last_signal = signal
            self.signals_generated += 1
            
            # Check if it's time to execute (rate limiting)
            if not self._should_execute_now():
                return {
                    'success': True,
                    'signal': signal,
                    'action_taken': 'WAIT',
                    'message': 'Rate limited - waiting for next execution window'
                }
            
            # Execute signal if trading enabled
            execution_result = None
            if self.is_trading_enabled and signal['action'] != 'HOLD':
                execution_result = self._execute_signal(signal, data)
            
            # Update execution time
            self.last_execution_time = datetime.now()
            
            return {
                'success': True,
                'signal': signal,
                'execution_result': execution_result,
                'action_taken': signal['action'],
                'timestamp': datetime.now(),
                'strategy_used': self.strategy.name
            }
            
        except Exception as e:
            self.failed_executions += 1
            return {'success': False, 'error': str(e)}
    
    def _execute_signal(self, signal: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute trading signal with risk management
        
        Args:
            signal: Trading signal to execute
            data: Market data for context
            
        Returns:
            Execution result
        """
        try:
            # Get account info for risk checks
            account_info = self.trading_engine.get_account_info()
            if account_info is None:
                return {'success': False, 'error': 'Could not get account info'}
            
            # Risk management check
            risk_check = self.risk_manager.check_risk_limits(signal, account_info)
            
            if not risk_check['approved']:
                return {
                    'success': False,
                    'error': 'Trade rejected by risk management',
                    'reasons': risk_check['reasons']
                }
            
            # Use adjusted signal from risk manager
            adjusted_signal = risk_check['adjusted_signal']
            
            # Calculate position size
            position_size = self._calculate_position_size(adjusted_signal, account_info)
            
            # Place order
            order_result = None
            if adjusted_signal['action'] == 'BUY':
                order_result = self.trading_engine.place_market_order(
                    self.symbol, 'BUY', position_size
                )
            elif adjusted_signal['action'] == 'SELL':
                order_result = self.trading_engine.place_market_order(
                    self.symbol, 'SELL', position_size
                )
            
            if order_result:
                self.successful_executions += 1
                self.trades_today += 1
                
                # Set stop loss and take profit if order successful
                if adjusted_signal.get('stop_loss') and adjusted_signal.get('take_profit'):
                    self._set_stop_loss_take_profit(order_result, adjusted_signal)
                
                return {
                    'success': True,
                    'order_result': order_result,
                    'position_size': position_size,
                    'risk_warnings': risk_check.get('warnings', [])
                }
            else:
                self.failed_executions += 1
                return {
                    'success': False,
                    'error': 'Order execution failed',
                    'last_error': self.trading_engine.last_error
                }
            
        except Exception as e:
            self.failed_executions += 1
            return {'success': False, 'error': str(e)}
    
    def _calculate_position_size(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> float:
        """Calculate appropriate position size"""
        try:
            # Get position sizing method from config
            sizing_method = 'fixed'
            if self.config and hasattr(self.config.trading, 'position_sizing_method'):
                sizing_method = self.config.trading.position_sizing_method
            
            # Calculate position size
            account_balance = account_info.get('total_wallet_balance', 10000)
            self.position_sizer.account_balance = account_balance
            
            if sizing_method == 'dynamic':
                # Get volatility from data
                volatility = 0.1  # Default
                if self.data_manager:
                    recent_data = self.data_manager.get_live_data(bars=20)
                    if recent_data is not None and len(recent_data) > 1:
                        returns = recent_data['close'].pct_change().dropna()
                        volatility = returns.std() if len(returns) > 0 else 0.1
                
                position_size = self.position_sizer.calculate_position_size(
                    signal, 'dynamic', account_balance, volatility=volatility
                )
            else:
                # Use fixed percentage method
                risk_percent = 0.02  # Default 2%
                if self.config and hasattr(self.config.trading, 'risk_per_trade'):
                    risk_percent = self.config.trading.risk_per_trade
                
                position_size = self.position_sizer.calculate_position_size(
                    signal, 'fixed', account_balance, risk_percent=risk_percent
                )
            
            # Convert to lot size for trading
            # For simplicity, assume direct position size
            # In production, would convert based on instrument specifications
            return max(0.01, position_size / account_balance)  # Minimum 0.01 lots
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0.01  # Default small position
    
    def _set_stop_loss_take_profit(self, order_result: Dict[str, Any], signal: Dict[str, Any]):
        """Set stop loss and take profit orders"""
        try:
            # This would place stop loss and take profit orders
            # Implementation depends on trading platform capabilities
            pass
        except Exception as e:
            print(f"Error setting stop loss/take profit: {e}")
    
    def _should_execute_now(self) -> bool:
        """Check if enough time has passed since last execution"""
        if self.last_execution_time is None:
            return True
        
        time_since_last = datetime.now() - self.last_execution_time
        return time_since_last.total_seconds() >= self.execution_interval
    
    def manage_positions(self) -> Dict[str, Any]:
        """
        Manage open positions (trailing stops, risk monitoring)
        
        Returns:
            Position management result
        """
        if not self.is_trading_enabled or not self.trading_engine:
            return {'success': False, 'error': 'Trading not enabled'}
        
        try:
            # Get current positions
            positions = self.trading_engine.get_positions(self.symbol)
            
            if not positions:
                return {'success': True, 'message': 'No positions to manage'}
            
            management_actions = []
            
            for position in positions:
                # Update trailing stops
                current_price = self._get_current_price()
                if current_price:
                    new_stop = self.risk_manager.update_trailing_stop(position, current_price)
                    if new_stop:
                        # Would update stop loss order here
                        management_actions.append({
                            'action': 'update_stop_loss',
                            'position': position['symbol'],
                            'new_stop': new_stop
                        })
                
                # Check for emergency conditions
                if self.risk_manager.emergency_stop_check():
                    # Close all positions in emergency
                    close_result = self.trading_engine.close_position(position['symbol'])
                    management_actions.append({
                        'action': 'emergency_close',
                        'position': position['symbol'],
                        'result': close_result
                    })
            
            return {
                'success': True,
                'positions_managed': len(positions),
                'actions_taken': management_actions
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _get_current_price(self) -> Optional[float]:
        """Get current market price"""
        try:
            if self.data_manager:
                return self.data_manager.get_latest_price()
        except Exception:
            pass
        return None
    
    def check_risk_limits(self) -> Dict[str, Any]:
        """
        Check current risk limits and status
        
        Returns:
            Risk status information
        """
        try:
            # Get account info
            account_info = {}
            if self.trading_engine:
                account_info = self.trading_engine.get_account_info() or {}
            
            # Get risk metrics
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            # Check emergency conditions
            emergency_stop = self.risk_manager.emergency_stop_check()
            
            return {
                'success': True,
                'risk_metrics': risk_metrics,
                'emergency_stop_required': emergency_stop,
                'account_balance': account_info.get('total_wallet_balance', 0),
                'open_positions': len(account_info.get('positions', [])),
                'daily_pnl': risk_metrics.get('daily_pnl', 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_trading_status(self) -> Dict[str, Any]:
        """
        Get comprehensive trading status
        
        Returns:
            Trading status information
        """
        try:
            status = {
                'is_initialized': self.is_initialized,
                'is_trading_enabled': self.is_trading_enabled,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'strategy': self.strategy.name,
                'last_signal': self.last_signal,
                'last_execution_time': self.last_execution_time,
                'performance': {
                    'signals_generated': self.signals_generated,
                    'successful_executions': self.successful_executions,
                    'failed_executions': self.failed_executions,
                    'trades_today': self.trades_today,
                    'success_rate': (
                        self.successful_executions / max(1, self.successful_executions + self.failed_executions)
                    )
                }
            }
            
            # Add engine status
            if self.trading_engine:
                status['engine_connected'] = self.trading_engine.is_connected
                status['last_engine_error'] = self.trading_engine.last_error
            else:
                status['engine_connected'] = False
                status['last_engine_error'] = 'Trading engine not initialized'
            
            # Add risk status
            risk_status = self.check_risk_limits()
            if risk_status['success']:
                status['risk_status'] = risk_status
            
            return status
            
        except Exception as e:
            return {'error': str(e)}
    
    def emergency_stop(self) -> Dict[str, Any]:
        """
        Emergency stop all trading activity
        
        Returns:
            Emergency stop result
        """
        try:
            results = []
            
            # Disable trading
            self.is_trading_enabled = False
            results.append('Trading disabled')
            
            # Close all positions if trading engine available
            if self.trading_engine:
                positions = self.trading_engine.get_positions()
                for position in positions:
                    close_result = self.trading_engine.close_position(position['symbol'])
                    results.append(f"Closed position {position['symbol']}: {close_result}")
            
            # Cancel all open orders
            if self.trading_engine:
                orders = self.trading_engine.get_open_orders()
                for order in orders:
                    cancel_result = self.trading_engine.cancel_order(
                        order['symbol'], order['order_id']
                    )
                    results.append(f"Cancelled order {order['order_id']}: {cancel_result}")
            
            return {
                'success': True,
                'message': 'Emergency stop executed',
                'actions_taken': results
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def switch_strategy(self, strategy_name: str, **kwargs) -> bool:
        """Switch to different trading strategy"""
        try:
            new_strategy = create_strategy(strategy_name, **kwargs)
            self.strategy = new_strategy
            print(f"✅ Switched to strategy: {strategy_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to switch strategy: {e}")
            return False
    
    def set_symbol(self, symbol: str):
        """Change trading symbol"""
        self.symbol = symbol
        if self.data_manager:
            self.data_manager.set_symbol(symbol)
    
    def set_timeframe(self, timeframe: str):
        """Change trading timeframe"""
        self.timeframe = timeframe
        if self.data_manager:
            self.data_manager.set_timeframe(timeframe)
    
    def enable_trading(self):
        """Enable live trading"""
        if self.trading_engine and self.trading_engine.is_connected:
            self.is_trading_enabled = True
            print("✅ Live trading enabled")
        else:
            print("❌ Cannot enable trading - engine not connected")
    
    def disable_trading(self):
        """Disable live trading (paper trading mode)"""
        self.is_trading_enabled = False
        print("⚠️  Live trading disabled - paper trading mode")


# Factory function
def create_trading_manager(symbol: str = None, timeframe: str = None, 
                          strategy_name: str = None, data_manager=None) -> TradingManager:
    """
    Create trading manager instance
    
    Args:
        symbol: Trading symbol
        timeframe: Trading timeframe
        strategy_name: Strategy to use
        data_manager: Data manager instance
        
    Returns:
        TradingManager instance
    """
    strategy = None
    if strategy_name:
        strategy = create_strategy(strategy_name)
    
    return TradingManager(symbol, timeframe, strategy, data_manager)