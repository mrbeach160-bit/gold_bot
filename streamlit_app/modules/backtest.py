"""
Backtesting module for historical strategy simulation.

This module provides comprehensive backtesting functionality with realistic
trade execution, performance metrics, and drawdown analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from .smart_entry import calculate_smart_entry_price
from .trading_utils import (
    calculate_position_info, execute_trade_exit_realistic, 
    calculate_realistic_pnl, calculate_ai_take_profit,
    get_conversion_rate, get_pip_value
)
from .config import is_feature_enabled

# Import utils if available
if is_feature_enabled('UTILS_AVAILABLE'):
    from utils.indicators import get_support_resistance


def run_backtest(symbol: str, data: pd.DataFrame, initial_balance: float, 
                risk_percent: float, sl_pips: int, tp_pips: int, 
                predict_func: callable, all_models: Dict[str, Any], 
                api_source: str, api_key_1: str, api_key_2: Optional[str] = None, 
                use_ai_tp: bool = False) -> Dict[str, Any]:
    """
    Run comprehensive backtesting with smart entry logic and realistic execution.
    
    Args:
        symbol: Trading symbol
        data: Historical OHLCV data
        initial_balance: Starting balance
        risk_percent: Risk percentage per trade
        sl_pips: Stop loss in pips
        tp_pips: Take profit in pips
        predict_func: Model prediction function
        all_models: Dictionary of trained models
        api_source: Data source
        api_key_1: Primary API key
        api_key_2: Secondary API key
        use_ai_tp: Whether to use AI take profit
        
    Returns:
        Dictionary with backtest results and metrics
    """
    try:
        if len(data) < 100:
            return {
                'error': 'Insufficient data for backtesting',
                'total_trades': 0,
                'final_balance': initial_balance,
                'total_pnl': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Initialize backtest state
        balance = initial_balance
        trades = []
        active_trade = None
        balance_history = [initial_balance]
        trade_signals = []
        
        # Get conversion rate for position sizing
        quote_currency = _extract_quote_currency(symbol)
        conversion_rate = get_conversion_rate(quote_currency, api_source, api_key_1, api_key_2)
        
        # Progress tracking
        progress_bar = st.progress(0, text="Starting backtest...")
        total_candles = len(data)
        
        # Main backtest loop
        for i in range(60, len(data)):  # Start from index 60 to have enough history
            current_candle = data.iloc[i]
            recent_data = data.iloc[max(0, i-60):i+1]
            
            # Update progress
            if i % max(1, total_candles // 20) == 0:
                progress = i / total_candles
                progress_bar.progress(progress, text=f"Backtesting... {progress:.0%}")
            
            # Check for trade exit first
            if active_trade:
                exit_price, exit_reason = execute_trade_exit_realistic(
                    current_candle.to_dict(), active_trade, slippage=0.001
                )
                
                if exit_reason in ['STOP_LOSS', 'TAKE_PROFIT']:
                    # Close the trade
                    pnl_info = calculate_realistic_pnl(
                        active_trade['entry_price'], exit_price,
                        active_trade['position_size'], active_trade['type'], symbol
                    )
                    
                    trade_result = {
                        'entry_time': active_trade['entry_time'],
                        'exit_time': current_candle.name,
                        'type': active_trade['type'],
                        'entry_price': active_trade['entry_price'],
                        'exit_price': exit_price,
                        'position_size': active_trade['position_size'],
                        'pnl': pnl_info['net_pnl'],
                        'exit_reason': exit_reason,
                        'hold_time': (current_candle.name - active_trade['entry_time']).total_seconds() / 3600,  # hours
                        'pips_gained': pnl_info.get('pips_gained', 0)
                    }
                    
                    trades.append(trade_result)
                    balance += pnl_info['net_pnl']
                    balance_history.append(balance)
                    active_trade = None
            
            # Look for new trade entry (only if no active trade)
            if not active_trade and i < len(data) - 10:  # Leave buffer at end
                # Get prediction from models
                try:
                    signal, confidence, predicted_price = predict_func(all_models, recent_data)
                    
                    # Convert signal to numeric if needed
                    if isinstance(signal, str):
                        signal_numeric = 1 if signal == "BUY" else -1 if signal == "SELL" else 0
                    else:
                        signal_numeric = signal
                    
                    trade_signals.append({
                        'time': current_candle.name,
                        'signal': signal_numeric,
                        'confidence': confidence,
                        'predicted_price': predicted_price
                    })
                    
                    # Only trade if signal is strong enough
                    if abs(signal_numeric) == 1 and confidence >= 0.6:
                        # Calculate smart entry price
                        smart_entry_result = calculate_smart_entry_price(
                            signal_numeric, recent_data, predicted_price, confidence, symbol
                        )
                        
                        # Check if entry is acceptable
                        if (smart_entry_result.get('risk_level') != 'REJECTED' and 
                            smart_entry_result.get('fill_probability', 0) >= 0.7):
                            
                            entry_price = smart_entry_result['smart_entry_price']
                            
                            # Calculate take profit
                            if use_ai_tp:
                                try:
                                    if is_feature_enabled('UTILS_AVAILABLE'):
                                        supports, resistances = get_support_resistance(recent_data)
                                    else:
                                        supports = np.array([entry_price * 0.98, entry_price * 0.96])
                                        resistances = np.array([entry_price * 1.02, entry_price * 1.04])
                                    
                                    atr_value = recent_data['ATR_14'].iloc[-1] if 'ATR_14' in recent_data.columns else entry_price * 0.001
                                    ai_tp = calculate_ai_take_profit(
                                        signal_numeric, entry_price, supports, resistances, atr_value
                                    )
                                    if ai_tp:
                                        tp_price = ai_tp
                                    else:
                                        # Fallback to standard TP calculation
                                        pip_value = get_pip_value(symbol, entry_price)
                                        if signal_numeric == 1:
                                            tp_price = entry_price + (tp_pips * pip_value / 100000)
                                        else:
                                            tp_price = entry_price - (tp_pips * pip_value / 100000)
                                except:
                                    # Fallback calculation
                                    pip_value = get_pip_value(symbol, entry_price)
                                    if signal_numeric == 1:
                                        tp_price = entry_price + (tp_pips * pip_value / 100000)
                                    else:
                                        tp_price = entry_price - (tp_pips * pip_value / 100000)
                            else:
                                tp_price = None
                            
                            # Calculate position info
                            position_info = calculate_position_info(
                                signal_numeric, symbol, entry_price, sl_pips, tp_pips,
                                balance, risk_percent, conversion_rate, tp_price
                            )
                            
                            if position_info and position_info.get('position_size', 0) > 0:
                                # Create active trade
                                active_trade = {
                                    'entry_time': current_candle.name,
                                    'type': 'BUY' if signal_numeric == 1 else 'SELL',
                                    'entry_price': entry_price,
                                    'sl': position_info['stop_loss_price'],
                                    'tp': position_info['take_profit_price'],
                                    'position_size': position_info['position_size'],
                                    'signal_confidence': confidence,
                                    'smart_entry_info': smart_entry_result
                                }
                
                except Exception as e:
                    # Continue backtest even if prediction fails
                    trade_signals.append({
                        'time': current_candle.name,
                        'signal': 0,
                        'confidence': 0,
                        'predicted_price': current_candle['close'],
                        'error': str(e)
                    })
                    continue
        
        # Close any remaining active trade at the end
        if active_trade:
            final_price = data['close'].iloc[-1]
            pnl_info = calculate_realistic_pnl(
                active_trade['entry_price'], final_price,
                active_trade['position_size'], active_trade['type'], symbol
            )
            
            trade_result = {
                'entry_time': active_trade['entry_time'],
                'exit_time': data.index[-1],
                'type': active_trade['type'],
                'entry_price': active_trade['entry_price'],
                'exit_price': final_price,
                'position_size': active_trade['position_size'],
                'pnl': pnl_info['net_pnl'],
                'exit_reason': 'END_OF_DATA',
                'hold_time': (data.index[-1] - active_trade['entry_time']).total_seconds() / 3600,
                'pips_gained': pnl_info.get('pips_gained', 0)
            }
            
            trades.append(trade_result)
            balance += pnl_info['net_pnl']
            balance_history.append(balance)
        
        progress_bar.progress(1.0, text="Backtest complete!")
        
        # Calculate performance metrics
        metrics = _calculate_backtest_metrics(
            trades, balance_history, initial_balance, data
        )
        
        return {
            'trades': trades,
            'trade_signals': trade_signals,
            'balance_history': balance_history,
            'final_balance': balance,
            'total_pnl': balance - initial_balance,
            'total_pnl_percent': ((balance - initial_balance) / initial_balance) * 100,
            **metrics
        }
        
    except Exception as e:
        st.error(f"Backtest error: {e}")
        return {
            'error': str(e),
            'total_trades': 0,
            'final_balance': initial_balance,
            'total_pnl': 0,
            'win_rate': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }


def _calculate_backtest_metrics(trades: List[Dict], balance_history: List[float], 
                               initial_balance: float, data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive backtest performance metrics."""
    
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'max_drawdown_percent': 0,
            'sharpe_ratio': 0,
            'calmar_ratio': 0,
            'expectancy': 0,
            'recovery_factor': 0,
            'avg_hold_time': 0
        }
    
    # Basic trade statistics
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] < 0]
    
    total_trades = len(trades)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
    
    total_wins = sum(t['pnl'] for t in winning_trades)
    total_losses = abs(sum(t['pnl'] for t in losing_trades))
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # Drawdown calculation
    peak = initial_balance
    max_drawdown = 0
    max_drawdown_percent = 0
    
    for balance in balance_history:
        if balance > peak:
            peak = balance
        drawdown = peak - balance
        drawdown_percent = (drawdown / peak) * 100 if peak > 0 else 0
        
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        if drawdown_percent > max_drawdown_percent:
            max_drawdown_percent = drawdown_percent
    
    # Return-based metrics
    returns = []
    for i in range(1, len(balance_history)):
        ret = (balance_history[i] - balance_history[i-1]) / balance_history[i-1]
        returns.append(ret)
    
    if returns:
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        # Annualize Sharpe (assuming daily returns)
        sharpe_ratio = sharpe_ratio * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Calmar ratio (annual return / max drawdown)
    final_balance = balance_history[-1]
    total_return = (final_balance - initial_balance) / initial_balance
    # Approximate annualized return
    days = len(data)
    annual_return = total_return * (365 / days) if days > 0 else 0
    calmar_ratio = annual_return / (max_drawdown_percent / 100) if max_drawdown_percent > 0 else 0
    
    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Recovery factor
    recovery_factor = total_return / (max_drawdown_percent / 100) if max_drawdown_percent > 0 else 0
    
    # Average hold time
    avg_hold_time = np.mean([t['hold_time'] for t in trades]) if trades else 0
    
    return {
        'total_trades': total_trades,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'max_drawdown_percent': max_drawdown_percent,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'expectancy': expectancy,
        'recovery_factor': recovery_factor,
        'avg_hold_time': avg_hold_time,
        'total_wins_value': total_wins,
        'total_losses_value': total_losses
    }


def _extract_quote_currency(symbol: str) -> str:
    """Extract quote currency from symbol."""
    symbol_upper = symbol.upper().replace('/', '')
    
    if 'USD' in symbol_upper:
        return 'USD'
    elif 'EUR' in symbol_upper:
        return 'EUR'
    elif 'GBP' in symbol_upper:
        return 'GBP'
    elif 'JPY' in symbol_upper:
        return 'JPY'
    elif 'CHF' in symbol_upper:
        return 'CHF'
    elif 'CAD' in symbol_upper:
        return 'CAD'
    elif 'AUD' in symbol_upper:
        return 'AUD'
    else:
        return 'USD'  # Default


def format_backtest_results(results: Dict[str, Any]) -> Dict[str, str]:
    """Format backtest results for display."""
    
    if results.get('error'):
        return {'Error': results['error']}
    
    formatted = {
        'Total Trades': str(results.get('total_trades', 0)),
        'Win Rate': f"{results.get('win_rate', 0):.1%}",
        'Final Balance': f"${results.get('final_balance', 0):,.2f}",
        'Total P&L': f"${results.get('total_pnl', 0):,.2f}",
        'Total Return': f"{results.get('total_pnl_percent', 0):,.1f}%",
        'Max Drawdown': f"{results.get('max_drawdown_percent', 0):.1f}%",
        'Profit Factor': f"{results.get('profit_factor', 0):.2f}",
        'Sharpe Ratio': f"{results.get('sharpe_ratio', 0):.2f}",
        'Expectancy': f"${results.get('expectancy', 0):.2f}",
        'Avg Hold Time': f"{results.get('avg_hold_time', 0):.1f} hours"
    }
    
    if results.get('avg_win', 0) > 0:
        formatted['Avg Win'] = f"${results['avg_win']:.2f}"
    if results.get('avg_loss', 0) < 0:
        formatted['Avg Loss'] = f"${results['avg_loss']:.2f}"
    
    return formatted


def generate_backtest_summary(results: Dict[str, Any]) -> str:
    """Generate a text summary of backtest results."""
    
    if results.get('error'):
        return f"Backtest failed: {results['error']}"
    
    total_trades = results.get('total_trades', 0)
    win_rate = results.get('win_rate', 0)
    total_pnl = results.get('total_pnl', 0)
    max_dd = results.get('max_drawdown_percent', 0)
    sharpe = results.get('sharpe_ratio', 0)
    
    if total_trades == 0:
        return "No trades were executed during the backtest period."
    
    summary = f"""
    **Backtest Summary:**
    
    • **{total_trades}** total trades executed
    • **{win_rate:.1%}** win rate
    • **${total_pnl:,.2f}** total profit/loss
    • **{max_dd:.1f}%** maximum drawdown
    • **{sharpe:.2f}** Sharpe ratio
    
    """
    
    if total_pnl > 0:
        summary += "✅ **Strategy was profitable** during the backtest period."
    else:
        summary += "❌ **Strategy was unprofitable** during the backtest period."
    
    if win_rate >= 0.6:
        summary += "\n✅ **High win rate** indicates good signal quality."
    elif win_rate >= 0.4:
        summary += "\n⚠️ **Moderate win rate** - consider optimizing entry conditions."
    else:
        summary += "\n❌ **Low win rate** - strategy needs significant improvement."
    
    if max_dd <= 10:
        summary += "\n✅ **Low drawdown** indicates good risk management."
    elif max_dd <= 20:
        summary += "\n⚠️ **Moderate drawdown** - monitor risk closely."
    else:
        summary += "\n❌ **High drawdown** - reduce risk or improve strategy."
    
    return summary