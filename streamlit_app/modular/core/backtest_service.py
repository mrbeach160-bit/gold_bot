"""
Backtest Service for the modular application.
Handles backtesting with real predictions.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from .prediction_service import PredictionService


class BacktestService:
    """Service for running backtests with real model predictions."""
    
    def __init__(self, prediction_service: PredictionService):
        self.prediction_service = prediction_service
        self.evaluation_frequency = 10  # Evaluate every N bars
    
    def run_backtest(self, data: pd.DataFrame, symbol: str, timeframe: str,
                    initial_balance: float = 10000.0, risk_per_trade: float = 2.0,
                    sl_percentage: float = 0.5, tp_percentage: float = 1.0,
                    use_atr_sl_tp: bool = False, evaluation_freq: int = 10) -> Dict[str, Any]:
        """
        Run backtest using real model predictions.
        
        Args:
            data: Historical data for backtesting
            symbol: Trading symbol
            timeframe: Timeframe
            initial_balance: Starting balance
            risk_per_trade: Risk percentage per trade
            sl_percentage: Stop loss percentage (if not using ATR)
            tp_percentage: Take profit percentage (if not using ATR)
            use_atr_sl_tp: Whether to use ATR-based SL/TP
            evaluation_freq: How often to make predictions (every N bars)
            
        Returns:
            Dictionary with backtest results
        """
        try:
            if data is None or data.empty:
                return self._empty_backtest_result("No data provided")
            
            if len(data) < 100:
                return self._empty_backtest_result(f"Insufficient data: {len(data)} rows")
            
            # Initialize backtest state
            balance = initial_balance
            trades = []
            equity_curve = []
            current_trade = None
            
            # Prepare data with features
            from .feature_service import FeatureService
            feature_service = FeatureService()
            data_with_features = feature_service.add_technical_indicators(data.copy())
            data_with_features = feature_service.add_engineered_features(data_with_features)
            
            # Run backtest
            with st.progress(0) as progress_bar:
                for i in range(100, len(data_with_features)):  # Start after enough data for features
                    # Update progress
                    progress = (i - 100) / (len(data_with_features) - 100)
                    progress_bar.progress(progress)
                    
                    current_bar = data_with_features.iloc[i]
                    current_price = current_bar['close']
                    current_time = current_bar.name if hasattr(current_bar, 'name') else i
                    
                    # Check for trade exits first
                    if current_trade:
                        exit_result = self._check_trade_exit(current_trade, current_bar, use_atr_sl_tp)
                        if exit_result:
                            # Close trade
                            trade_pnl = self._calculate_trade_pnl(current_trade, exit_result['exit_price'])
                            balance += trade_pnl
                            
                            # Record completed trade
                            completed_trade = {
                                'entry_time': current_trade['entry_time'],
                                'exit_time': current_time,
                                'side': current_trade['side'],
                                'entry_price': current_trade['entry_price'],
                                'exit_price': exit_result['exit_price'],
                                'pnl': trade_pnl,
                                'exit_reason': exit_result['reason'],
                                'duration_bars': i - current_trade['entry_bar']
                            }
                            trades.append(completed_trade)
                            current_trade = None
                    
                    # Record equity curve
                    current_equity = balance
                    if current_trade:
                        # Add unrealized PnL
                        unrealized_pnl = self._calculate_trade_pnl(current_trade, current_price)
                        current_equity += unrealized_pnl
                    
                    equity_curve.append({
                        'time': current_time,
                        'equity': current_equity,
                        'bar_index': i
                    })
                    
                    # Check for new trade entries (only if no current trade and at evaluation frequency)
                    if not current_trade and i % evaluation_freq == 0:
                        # Get historical window for prediction
                        window_data = data_with_features.iloc[max(0, i-200):i+1]  # Use up to 200 bars
                        
                        try:
                            # Make prediction
                            prediction_result = self.prediction_service.make_predictions(
                                window_data, symbol, timeframe
                            )
                            
                            ensemble = prediction_result.get('ensemble', {})
                            signal_direction = ensemble.get('direction', 'HOLD')
                            signal_confidence = ensemble.get('confidence', 0.0)
                            
                            # Check entry criteria
                            if signal_direction in ['BUY', 'SELL'] and signal_confidence >= 0.55:
                                # Calculate position size
                                position_size = self._calculate_position_size(
                                    balance, risk_per_trade, current_price, sl_percentage
                                )
                                
                                if position_size > 0:
                                    # Calculate SL/TP levels
                                    sl_price, tp_price = self._compute_sl_tp(
                                        current_price, signal_direction, 
                                        sl_percentage, tp_percentage,
                                        current_bar.get('ATR_14') if use_atr_sl_tp else None
                                    )
                                    
                                    # Enter trade
                                    current_trade = {
                                        'entry_time': current_time,
                                        'entry_bar': i,
                                        'side': signal_direction,
                                        'entry_price': current_price,
                                        'sl_price': sl_price,
                                        'tp_price': tp_price,
                                        'position_size': position_size,
                                        'signal_confidence': signal_confidence
                                    }
                        
                        except Exception as e:
                            # Log prediction error but continue backtest
                            if len(trades) < 5:  # Only show first few errors
                                st.warning(f"Prediction error at bar {i}: {str(e)}")
            
            # Close any remaining open trade
            if current_trade:
                final_price = data_with_features.iloc[-1]['close']
                trade_pnl = self._calculate_trade_pnl(current_trade, final_price)
                balance += trade_pnl
                
                completed_trade = {
                    'entry_time': current_trade['entry_time'],
                    'exit_time': data_with_features.index[-1],
                    'side': current_trade['side'],
                    'entry_price': current_trade['entry_price'],
                    'exit_price': final_price,
                    'pnl': trade_pnl,
                    'exit_reason': 'end_of_data',
                    'duration_bars': len(data_with_features) - 1 - current_trade['entry_bar']
                }
                trades.append(completed_trade)
            
            # Calculate results
            results = self._calculate_backtest_results(
                initial_balance, balance, trades, equity_curve
            )
            
            return results
            
        except Exception as e:
            st.error(f"Backtest error: {str(e)}")
            return self._empty_backtest_result(f"Backtest failed: {str(e)}")
    
    def _compute_sl_tp(self, entry_price: float, side: str, 
                      sl_pct: float = 0.5, tp_pct: float = 1.0, 
                      atr: Optional[float] = None) -> Tuple[float, float]:
        """
        Compute stop loss and take profit levels.
        
        Args:
            entry_price: Entry price
            side: 'BUY' or 'SELL'
            sl_pct: Stop loss percentage
            tp_pct: Take profit percentage
            atr: ATR value (if using ATR-based levels)
            
        Returns:
            Tuple of (sl_price, tp_price)
        """
        if atr and atr > 0:
            # ATR-based SL/TP
            sl_distance = atr * 1.5  # 1.5x ATR for SL
            tp_distance = atr * 2.5  # 2.5x ATR for TP
        else:
            # Percentage-based SL/TP
            sl_distance = entry_price * (sl_pct / 100)
            tp_distance = entry_price * (tp_pct / 100)
        
        if side == 'BUY':
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:  # SELL
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        
        return sl_price, tp_price
    
    def _check_trade_exit(self, trade: Dict[str, Any], current_bar: pd.Series, 
                         use_atr: bool) -> Optional[Dict[str, Any]]:
        """
        Check if trade should be exited.
        
        Args:
            trade: Current trade dictionary
            current_bar: Current price bar
            use_atr: Whether using ATR-based levels
            
        Returns:
            Dictionary with exit info or None if no exit
        """
        current_high = current_bar['high']
        current_low = current_bar['low']
        current_close = current_bar['close']
        
        if trade['side'] == 'BUY':
            # Check stop loss
            if current_low <= trade['sl_price']:
                return {'exit_price': trade['sl_price'], 'reason': 'stop_loss'}
            # Check take profit
            if current_high >= trade['tp_price']:
                return {'exit_price': trade['tp_price'], 'reason': 'take_profit'}
        else:  # SELL
            # Check stop loss
            if current_high >= trade['sl_price']:
                return {'exit_price': trade['sl_price'], 'reason': 'stop_loss'}
            # Check take profit
            if current_low <= trade['tp_price']:
                return {'exit_price': trade['tp_price'], 'reason': 'take_profit'}
        
        return None
    
    def _calculate_position_size(self, balance: float, risk_pct: float, 
                               entry_price: float, sl_pct: float) -> float:
        """Calculate position size based on risk management."""
        risk_amount = balance * (risk_pct / 100)
        sl_distance = entry_price * (sl_pct / 100)
        
        if sl_distance > 0:
            position_size = risk_amount / sl_distance
            return min(position_size, balance * 0.95)  # Max 95% of balance
        
        return 0
    
    def _calculate_trade_pnl(self, trade: Dict[str, Any], exit_price: float) -> float:
        """Calculate P&L for a trade."""
        if trade['side'] == 'BUY':
            pnl = (exit_price - trade['entry_price']) * trade['position_size']
        else:  # SELL
            pnl = (trade['entry_price'] - exit_price) * trade['position_size']
        
        return pnl
    
    def _calculate_backtest_results(self, initial_balance: float, final_balance: float,
                                  trades: List[Dict], equity_curve: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive backtest results."""
        total_return_pct = ((final_balance - initial_balance) / initial_balance) * 100
        
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(trades) * 100
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            profit_factor = (sum(t['pnl'] for t in winning_trades) / 
                           abs(sum(t['pnl'] for t in losing_trades))) if losing_trades else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'success': True,
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return_pct': total_return_pct,
            'total_trades': len(trades),
            'winning_trades': len([t for t in trades if t['pnl'] > 0]),
            'losing_trades': len([t for t in trades if t['pnl'] <= 0]),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trades': trades[:10],  # First 10 trades for display
            'equity_curve': equity_curve,
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'evaluation_frequency': self.evaluation_frequency
        }
    
    def _calculate_max_drawdown(self, equity_curve: List[Dict]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not equity_curve:
            return 0
        
        equity_values = [point['equity'] for point in equity_curve]
        peak = equity_values[0]
        max_dd = 0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _empty_backtest_result(self, message: str) -> Dict[str, Any]:
        """Return empty backtest result with error message."""
        return {
            'success': False,
            'message': message,
            'initial_balance': 0,
            'final_balance': 0,
            'total_return_pct': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'trades': [],
            'equity_curve': [],
            'max_drawdown': 0
        }