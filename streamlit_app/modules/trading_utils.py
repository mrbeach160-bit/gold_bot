"""
Trading utilities module.

This module contains utilities for position sizing, take profit calculation,
trade validation, risk management, PnL calculation, and trade execution simulation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union

from .config import is_feature_enabled


def validate_trading_inputs(symbol: str, balance: float, risk_percent: float, 
                          sl_pips: int, tp_pips: int) -> Dict[str, Any]:
    """
    Validate trading inputs with comprehensive checks.
    
    Args:
        symbol: Trading symbol
        balance: Account balance
        risk_percent: Risk percentage per trade
        sl_pips: Stop loss in pips
        tp_pips: Take profit in pips
        
    Returns:
        Dict with validation results and errors
    """
    errors = []
    warnings = []
    
    # Balance validation
    if balance < 100:
        errors.append("Account balance must be at least $100")
    elif balance < 500:
        warnings.append("Low account balance may limit position sizes")
    
    # Risk validation
    if risk_percent < 0.1:
        errors.append("Risk percentage must be at least 0.1%")
    elif risk_percent > 10:
        errors.append("Risk percentage cannot exceed 10% per trade")
    elif risk_percent > 5:
        warnings.append("High risk percentage (>5%) - consider reducing")
    
    # Stop loss validation
    if sl_pips < 5:
        errors.append("Stop loss must be at least 5 pips")
    elif sl_pips > 200:
        warnings.append("Large stop loss (>200 pips) may increase risk")
    
    # Take profit validation
    if tp_pips < 5:
        errors.append("Take profit must be at least 5 pips")
    elif tp_pips > 500:
        warnings.append("Large take profit (>500 pips) may reduce fill probability")
    
    # Risk/reward ratio
    rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 0
    if rr_ratio < 1:
        warnings.append(f"Poor risk/reward ratio: {rr_ratio:.2f} (consider ≥1.5)")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "risk_reward_ratio": rr_ratio
    }


def calculate_position_info(signal: int, symbol: str, entry_price: float, sl_pips: int, 
                          tp_pips: int, balance: float, risk_percent: float, 
                          conversion_rate_to_usd: float, take_profit_price: Optional[float] = None, 
                          leverage: int = 20) -> Dict[str, Any]:
    """
    Calculate comprehensive position information including size, risk, and profit targets.
    
    Args:
        signal: 1 for BUY, -1 for SELL
        symbol: Trading symbol
        entry_price: Entry price
        sl_pips: Stop loss in pips
        tp_pips: Take profit in pips
        balance: Account balance
        risk_percent: Risk percentage
        conversion_rate_to_usd: USD conversion rate
        take_profit_price: Optional specific TP price
        leverage: Trading leverage
        
    Returns:
        Dict with position calculations
    """
    try:
        # Calculate pip value
        pip_value = get_pip_value(symbol, entry_price)
        
        # Calculate position size based on risk
        risk_amount = balance * (risk_percent / 100)
        sl_amount = sl_pips * pip_value
        
        if sl_amount <= 0:
            raise ValueError("Invalid stop loss amount")
        
        position_size = (risk_amount / sl_amount) * leverage
        
        # Calculate stop loss and take profit prices
        if signal == 1:  # BUY
            stop_loss_price = entry_price - (sl_pips * pip_value / 100000)
            if take_profit_price is None:
                take_profit_price = entry_price + (tp_pips * pip_value / 100000)
        else:  # SELL
            stop_loss_price = entry_price + (sl_pips * pip_value / 100000)
            if take_profit_price is None:
                take_profit_price = entry_price - (tp_pips * pip_value / 100000)
        
        # Calculate potential profit/loss
        potential_loss = sl_pips * pip_value * position_size / leverage
        potential_profit = tp_pips * pip_value * position_size / leverage
        
        # Required margin
        required_margin = (position_size * entry_price) / leverage
        
        # Margin level
        margin_level = (balance / required_margin) * 100 if required_margin > 0 else 0
        
        # Risk validation
        if required_margin > balance * 0.8:
            risk_warning = "HIGH MARGIN USAGE"
        elif required_margin > balance * 0.5:
            risk_warning = "MEDIUM MARGIN USAGE"
        else:
            risk_warning = "SAFE MARGIN USAGE"
        
        return {
            "position_size": round(position_size, 2),
            "entry_price": round(entry_price, 5),
            "stop_loss_price": round(stop_loss_price, 5),
            "take_profit_price": round(take_profit_price, 5),
            "potential_loss": round(potential_loss, 2),
            "potential_profit": round(potential_profit, 2),
            "required_margin": round(required_margin, 2),
            "margin_level": round(margin_level, 2),
            "pip_value": pip_value,
            "risk_amount": round(risk_amount, 2),
            "risk_warning": risk_warning,
            "leverage": leverage,
            "risk_reward_ratio": round(potential_profit / potential_loss, 2) if potential_loss > 0 else 0
        }
        
    except Exception as e:
        st.error(f"Error calculating position info: {e}")
        return {
            "position_size": 0,
            "entry_price": entry_price,
            "stop_loss_price": entry_price,
            "take_profit_price": entry_price,
            "potential_loss": 0,
            "potential_profit": 0,
            "required_margin": 0,
            "margin_level": 0,
            "pip_value": 0,
            "risk_amount": 0,
            "risk_warning": "CALCULATION ERROR",
            "leverage": leverage,
            "risk_reward_ratio": 0
        }


def calculate_ai_take_profit(signal: int, entry_price: float, supports: np.ndarray, 
                           resistances: np.ndarray, atr_value: float, 
                           current_price: Optional[float] = None) -> float:
    """
    Calculate AI-optimized take profit level based on technical analysis.
    
    Args:
        signal: 1 for BUY, -1 for SELL
        entry_price: Entry price
        supports: Array of support levels
        resistances: Array of resistance levels
        atr_value: Average True Range value
        current_price: Current market price
        
    Returns:
        Optimized take profit price
    """
    try:
        if signal == 1:  # BUY - target nearest resistance
            valid_resistances = resistances[resistances > entry_price]
            if len(valid_resistances) > 0:
                target_resistance = valid_resistances.min()
                # Take profit slightly below resistance with ATR buffer
                tp_price = target_resistance - (atr_value * 0.2)
                # Ensure minimum profit target
                min_tp = entry_price + (atr_value * 1.5)
                tp_price = max(tp_price, min_tp)
            else:
                # No resistance found, use ATR-based target
                tp_price = entry_price + (atr_value * 2.0)
                
        else:  # SELL - target nearest support
            valid_supports = supports[supports < entry_price]
            if len(valid_supports) > 0:
                target_support = valid_supports.max()
                # Take profit slightly above support with ATR buffer
                tp_price = target_support + (atr_value * 0.2)
                # Ensure minimum profit target
                max_tp = entry_price - (atr_value * 1.5)
                tp_price = min(tp_price, max_tp)
            else:
                # No support found, use ATR-based target
                tp_price = entry_price - (atr_value * 2.0)
        
        return round(tp_price, 5)
        
    except Exception as e:
        # Fallback to simple ATR-based calculation
        if signal == 1:
            return round(entry_price + (atr_value * 1.5), 5)
        else:
            return round(entry_price - (atr_value * 1.5), 5)


def execute_trade_exit_realistic(current_candle: Dict[str, Any], active_trade: Dict[str, Any], 
                               slippage: float = 0.001) -> Tuple[bool, float, str]:
    """
    Execute realistic trade exit simulation with slippage and market conditions.
    
    Args:
        current_candle: Current OHLC data
        active_trade: Active trade information
        slippage: Slippage factor for execution
        
    Returns:
        Tuple of (exited, exit_price, exit_reason)
    """
    try:
        trade_type = active_trade["type"]
        entry_price = active_trade["entry_price"]
        stop_loss = active_trade["stop_loss"]
        take_profit = active_trade["take_profit"]
        
        high = current_candle.get("high", 0)
        low = current_candle.get("low", 0)
        close = current_candle.get("close", 0)
        
        if trade_type == "BUY":
            # Check stop loss hit
            if low <= stop_loss:
                exit_price = stop_loss * (1 - slippage)  # Slippage against trader
                return True, exit_price, "STOP_LOSS"
            
            # Check take profit hit
            if high >= take_profit:
                exit_price = take_profit * (1 - slippage)  # Slippage against trader
                return True, exit_price, "TAKE_PROFIT"
                
        else:  # SELL
            # Check stop loss hit
            if high >= stop_loss:
                exit_price = stop_loss * (1 + slippage)  # Slippage against trader
                return True, exit_price, "STOP_LOSS"
            
            # Check take profit hit
            if low <= take_profit:
                exit_price = take_profit * (1 + slippage)  # Slippage against trader
                return True, exit_price, "TAKE_PROFIT"
        
        return False, close, "ACTIVE"
        
    except Exception as e:
        return False, active_trade.get("entry_price", 0), f"ERROR: {str(e)}"


def calculate_realistic_pnl(entry_price: float, exit_price: float, position_size: float, 
                          trade_type: str, symbol: str) -> Dict[str, Any]:
    """
    Calculate realistic PnL including spreads, commissions, and swap costs.
    
    Args:
        entry_price: Trade entry price
        exit_price: Trade exit price
        position_size: Position size in lots
        trade_type: "BUY" or "SELL"
        symbol: Trading symbol
        
    Returns:
        Dict with PnL breakdown
    """
    try:
        # Get pip value
        pip_value = get_pip_value(symbol, entry_price)
        
        # Calculate raw PnL
        if trade_type == "BUY":
            price_diff = exit_price - entry_price
        else:  # SELL
            price_diff = entry_price - exit_price
        
        # Convert to pips
        pips_gained = price_diff / (pip_value / 100000) if pip_value > 0 else 0
        
        # Calculate gross PnL
        gross_pnl = pips_gained * pip_value * position_size
        
        # Realistic trading costs
        spread_cost = _get_spread_cost(symbol) * position_size
        commission = _get_commission_cost(symbol, position_size)
        swap_cost = _get_swap_cost(symbol, position_size, trade_type)
        
        # Net PnL
        total_costs = spread_cost + commission + swap_cost
        net_pnl = gross_pnl - total_costs
        
        return {
            "gross_pnl": round(gross_pnl, 2),
            "spread_cost": round(spread_cost, 2),
            "commission": round(commission, 2),
            "swap_cost": round(swap_cost, 2),
            "total_costs": round(total_costs, 2),
            "net_pnl": round(net_pnl, 2),
            "pips_gained": round(pips_gained, 1),
            "pip_value": pip_value
        }
        
    except Exception as e:
        return {
            "gross_pnl": 0,
            "spread_cost": 0,
            "commission": 0,
            "swap_cost": 0,
            "total_costs": 0,
            "net_pnl": 0,
            "pips_gained": 0,
            "pip_value": 0,
            "error": str(e)
        }


def get_conversion_rate(quote_currency: str, api_source: str, api_key_1: str, 
                       api_key_2: Optional[str] = None) -> float:
    """
    Get USD conversion rate for position sizing calculations.
    
    Args:
        quote_currency: Quote currency (USD, EUR, etc.)
        api_source: Data source
        api_key_1: Primary API key
        api_key_2: Secondary API key
        
    Returns:
        USD conversion rate
    """
    try:
        if quote_currency.upper() == "USD":
            return 1.0
        
        # For now, return simplified rates
        # In production, this would fetch real-time rates
        conversion_rates = {
            "EUR": 1.10,
            "GBP": 1.25,
            "JPY": 0.007,
            "CHF": 1.05,
            "CAD": 0.75,
            "AUD": 0.65
        }
        
        return conversion_rates.get(quote_currency.upper(), 1.0)
        
    except Exception:
        return 1.0  # Default to 1:1 if conversion fails


def get_pip_value(symbol: str, price: float) -> float:
    """
    Calculate pip value for position sizing.
    
    Args:
        symbol: Trading symbol
        price: Current price
        
    Returns:
        Pip value in account currency
    """
    try:
        symbol_upper = symbol.upper().replace('/', '')
        
        # Forex pairs
        if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
            return 0.1  # $0.10 per pip for gold
        elif 'JPY' in symbol_upper:
            return 0.001  # For JPY pairs
        elif any(pair in symbol_upper for pair in ['EUR', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF']):
            return 0.0001  # Standard forex pip
        
        # Crypto
        elif 'BTC' in symbol_upper:
            return price * 0.0001  # 0.01% of price
        elif any(crypto in symbol_upper for crypto in ['ETH', 'LTC', 'XRP']):
            return price * 0.0001
        
        # Default
        else:
            return 0.0001
            
    except Exception:
        return 0.0001  # Safe default


def _get_spread_cost(symbol: str) -> float:
    """Get typical spread cost for symbol."""
    spreads = {
        'XAUUSD': 0.5,
        'EURUSD': 0.1,
        'GBPUSD': 0.2,
        'USDJPY': 0.1,
        'BTCUSD': 10.0,
        'ETHUSD': 1.0
    }
    return spreads.get(symbol.upper().replace('/', ''), 0.3)


def _get_commission_cost(symbol: str, position_size: float) -> float:
    """Get commission cost for symbol and position size."""
    # Simplified commission structure
    if 'XAU' in symbol.upper():
        return position_size * 0.5  # $0.50 per lot
    elif any(crypto in symbol.upper() for crypto in ['BTC', 'ETH']):
        return position_size * 0.25  # 0.25% commission
    else:
        return position_size * 0.1  # $0.10 per lot for forex


def _get_swap_cost(symbol: str, position_size: float, trade_type: str) -> float:
    """Get overnight swap cost."""
    # Simplified swap calculation - typically very small for short-term trades
    if 'XAU' in symbol.upper():
        return position_size * (-0.2 if trade_type == 'BUY' else 0.1)
    else:
        return 0  # Ignore for simplicity