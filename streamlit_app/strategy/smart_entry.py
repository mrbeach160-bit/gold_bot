"""
Smart Entry Price Calculation Module

Contains the main smart entry logic for both momentum and pullback modes.
"""

import pandas as pd
import numpy as np
from .modes import ENTRY_MODE_MOMENTUM, ENTRY_MODE_PULLBACK, DEFAULT_ENTRY_MODE
from ..core.support_resistance import get_support_resistance

# Import streamlit for error handling (temporary - should be refactored to use proper logging)
try:
    import streamlit as st
except ImportError:
    # Fallback for when streamlit is not available (e.g., in tests)
    class MockStreamlit:
        def error(self, msg): print(f"ERROR: {msg}")
    st = MockStreamlit()


def calculate_smart_entry_price(signal, recent_data, predicted_price, confidence, symbol="XAUUSD", mode=DEFAULT_ENTRY_MODE):
    """
    🧠 SMART AI ENTRY PRICE CALCULATION
    
    Menghitung entry price berdasarkan multiple factors:
    - Support/Resistance levels
    - RSI conditions (oversold/overbought)
    - MACD momentum
    - ATR volatility buffer
    - AI confidence adjustment
    - Entry mode (momentum vs pullback)
    
    Args:
        signal: 'BUY' atau 'SELL'
        recent_data: DataFrame dengan OHLC dan indicators
        predicted_price: LSTM prediction price
        confidence: AI confidence level (0-1)
        symbol: Trading symbol untuk pip calculation
        mode: Entry mode ('momentum' or 'pullback')
        
    Returns:
        dict: {
            'entry_price': float,
            'strategy_reasons': list,
            'risk_level': str,
            'expected_fill_probability': float
        }
    """
    try:
        current_price = recent_data['close'].iloc[-1]
        
        # Get technical indicators
        rsi = recent_data['rsi'].iloc[-1] if 'rsi' in recent_data.columns else 50
        atr = recent_data['ATR_14'].iloc[-1] if 'ATR_14' in recent_data.columns else current_price * 0.01
        macd = recent_data['MACD_12_26_9'].iloc[-1] if 'MACD_12_26_9' in recent_data.columns else 0
        macd_signal = recent_data['MACDs_12_26_9'].iloc[-1] if 'MACDs_12_26_9' in recent_data.columns else 0
        
        # Get Support/Resistance levels
        supports, resistances = get_support_resistance(recent_data)
        
        # Initialize strategy reasons
        strategy_reasons = []
        risk_level = "MEDIUM"
        
        if signal == "BUY":
            if mode == ENTRY_MODE_PULLBACK:
                entry_price, reasons, risk = _calculate_buy_entry_pullback(
                    current_price, predicted_price, supports, resistances,
                    rsi, atr, macd, macd_signal, confidence
                )
            else:  # Default to momentum mode
                entry_price, reasons, risk = _calculate_buy_entry_momentum(
                    current_price, predicted_price, supports, resistances,
                    rsi, atr, macd, macd_signal, confidence
                )
        elif signal == "SELL":
            if mode == ENTRY_MODE_PULLBACK:
                entry_price, reasons, risk = _calculate_sell_entry_pullback(
                    current_price, predicted_price, supports, resistances,
                    rsi, atr, macd, macd_signal, confidence
                )
            else:  # Default to momentum mode
                entry_price, reasons, risk = _calculate_sell_entry_momentum(
                    current_price, predicted_price, supports, resistances,
                    rsi, atr, macd, macd_signal, confidence
                )
        else:
            return {
                'entry_price': current_price,
                'strategy_reasons': ['HOLD signal - no entry'],
                'risk_level': 'LOW',
                'expected_fill_probability': 0.0
            }
        
        strategy_reasons.extend(reasons)
        risk_level = risk
        
        # Validate entry_price is a valid number
        if not isinstance(entry_price, (int, float)) or entry_price <= 0:
            return {
                'entry_price': current_price,
                'strategy_reasons': [f'Invalid entry price calculated: {entry_price}, using current price'],
                'risk_level': 'HIGH',
                'expected_fill_probability': 0.5
            }
        
        # Calculate price distance for fill probability assessment
        price_distance = abs(entry_price - current_price) / current_price if current_price > 0 else 0
        
        # Calculate expected fill probability with realistic assessment
        if mode == ENTRY_MODE_PULLBACK:
            # Pullback mode has lower fill probability due to waiting for retracement
            fill_probability = _calculate_pullback_fill_probability(price_distance)
        else:
            # Momentum mode has higher fill probability due to immediate execution
            fill_probability = _calculate_momentum_fill_probability(price_distance)
        
        # Minimum fill probability threshold (70%)
        if fill_probability < 0.7:
            return {
                'entry_price': current_price,
                'strategy_reasons': [f'REJECTED: Fill probability {fill_probability:.1%} below 70% minimum'],
                'risk_level': 'REJECTED',
                'expected_fill_probability': fill_probability
            }
        
        # Validate entry price reasonableness
        max_deviation = 0.005  # 0.5% max deviation from current price
        if price_distance > max_deviation:
            return {
                'entry_price': current_price,
                'strategy_reasons': [f'REJECTED: Entry price deviation {price_distance:.2%} exceeds maximum {max_deviation:.1%}'],
                'risk_level': 'REJECTED',
                'expected_fill_probability': 0.0
            }
        
        return {
            'entry_price': entry_price,
            'strategy_reasons': strategy_reasons,
            'risk_level': risk_level,
            'expected_fill_probability': fill_probability
        }
        
    except Exception as e:
        st.error(f"❌ Error in smart entry calculation: {e}")
        return {
            'entry_price': current_price if 'current_price' in locals() else 0,
            'strategy_reasons': [f'Fallback to current price due to error: {str(e)}'],
            'risk_level': 'HIGH',
            'expected_fill_probability': 0.5
        }


def _calculate_buy_entry_momentum(current_price, predicted_price, supports, resistances, 
                                  rsi, atr, macd, macd_signal, confidence):
    """
    Calculate optimal BUY entry price using MOMENTUM strategy (existing logic)
    """
    strategy_reasons = []
    risk_level = "MEDIUM"
    
    # Base entry calculation
    base_entry = current_price
    
    # 1. Support level analysis
    valid_supports = supports[supports <= current_price * 1.005]  # Within 0.5%
    if not valid_supports.empty:
        nearest_support = valid_supports.max()
        support_buffer = atr * 0.1  # Small buffer above support
        base_entry = nearest_support + support_buffer
        strategy_reasons.append(f"Entry above support: ${nearest_support:.2f}")
        risk_level = "LOW"
    else:
        # No nearby support, use current price with small premium
        base_entry = current_price + (atr * 0.05)
        strategy_reasons.append("Entry at market with small premium")
        risk_level = "MEDIUM"
    
    # 2. RSI condition adjustment
    rsi_adjustment = 0
    if rsi < 30:  # Oversold - great for BUY
        rsi_adjustment = -atr * 0.2  # Better entry price
        strategy_reasons.append(f"RSI oversold advantage ({rsi:.1f})")
        risk_level = "LOW"
    elif rsi < 40:  # Mild oversold
        rsi_adjustment = -atr * 0.1
        strategy_reasons.append(f"RSI favorable ({rsi:.1f})")
    elif rsi > 70:  # Overbought - risky for BUY
        rsi_adjustment = atr * 0.1  # Premium for risky entry
        strategy_reasons.append(f"RSI overbought risk ({rsi:.1f})")
        risk_level = "HIGH"
    
    # 3. MACD momentum check
    macd_adjustment = 0
    if macd > macd_signal:  # Bullish momentum
        macd_adjustment = atr * 0.05  # Small premium for momentum
        strategy_reasons.append("MACD bullish momentum")
    else:
        strategy_reasons.append("MACD neutral/bearish")
    
    # 4. Confidence factor
    confidence_adjustment = (confidence - 0.5) * atr * 0.1  # -0.05 to +0.05 ATR
    if confidence > 0.8:
        strategy_reasons.append(f"High confidence ({confidence:.1%})")
    elif confidence < 0.6:
        strategy_reasons.append(f"Low confidence ({confidence:.1%})")
        risk_level = "HIGH"
    
    # 5. Predicted price consideration
    pred_adjustment = 0
    if predicted_price > current_price:
        pred_factor = min(0.3, (predicted_price - current_price) / current_price)
        pred_adjustment = pred_factor * atr * 0.5
        strategy_reasons.append("AI predicts price increase")
    
    # Final entry price calculation
    final_entry = base_entry + rsi_adjustment + macd_adjustment + confidence_adjustment + pred_adjustment
    
    return final_entry, strategy_reasons, risk_level


def _calculate_buy_entry_pullback(current_price, predicted_price, supports, resistances, 
                                  rsi, atr, macd, macd_signal, confidence):
    """
    Calculate optimal BUY entry price using PULLBACK strategy (NEW)
    
    Targets slight retrace toward nearest support or volatility zone.
    """
    strategy_reasons = []
    risk_level = "MEDIUM"
    
    # Calculate volatility anchor
    volatility_anchor = max(atr, current_price * 0.0015)
    
    # Base pullback entry - target retrace zone
    retrace_factor = np.random.uniform(0.15, 0.30)  # Random between 15-30% of volatility
    base_pullback = current_price - (retrace_factor * volatility_anchor)
    
    strategy_reasons.append(f"Pullback target: {retrace_factor:.1%} retrace")
    
    # 1. Support level confluence
    valid_supports = supports[supports <= current_price * 1.01]  # Within 1%
    if not valid_supports.empty:
        nearest_support = valid_supports.max()
        support_distance = abs(base_pullback - nearest_support) / current_price
        
        if support_distance < 0.002:  # Within 0.2% of support
            base_pullback = nearest_support + (atr * 0.05)  # Slightly above support
            strategy_reasons.append(f"Support confluence at ${nearest_support:.2f}")
            risk_level = "LOW"
        else:
            strategy_reasons.append(f"Pullback zone alignment")
    else:
        strategy_reasons.append("Volatility-based pullback zone")
    
    # 2. RSI oversold fine-tuning
    rsi_adjustment = 0
    if rsi < 35:  # Very oversold - wait for more pullback
        rsi_adjustment = -volatility_anchor * 0.1
        strategy_reasons.append("RSI oversold fine-tune - deeper pullback")
        risk_level = "LOW"
    elif rsi < 45:  # Mildly oversold - good pullback zone
        rsi_adjustment = 0
        strategy_reasons.append("RSI pullback zone optimal")
    elif rsi > 60:  # Not oversold enough - higher pullback target
        rsi_adjustment = volatility_anchor * 0.05
        strategy_reasons.append("RSI not oversold - conservative pullback")
        risk_level = "HIGH"
    
    # 3. Confidence adjustment
    confidence_adjustment = (confidence - 0.5) * volatility_anchor * 0.1
    if confidence > 0.8:
        strategy_reasons.append(f"High confidence pullback ({confidence:.1%})")
    elif confidence < 0.6:
        strategy_reasons.append(f"Low confidence pullback ({confidence:.1%})")
        risk_level = "HIGH"
    
    # Final pullback entry price
    final_entry = base_pullback + rsi_adjustment + confidence_adjustment
    
    # Ensure pullback entry is below current price
    if final_entry >= current_price:
        final_entry = current_price - (volatility_anchor * 0.1)
        strategy_reasons.append("Adjusted to ensure pullback below current price")
    
    return final_entry, strategy_reasons, risk_level


def _calculate_sell_entry_momentum(current_price, predicted_price, supports, resistances,
                                   rsi, atr, macd, macd_signal, confidence):
    """
    Calculate optimal SELL entry price using MOMENTUM strategy (existing logic)
    """
    strategy_reasons = []
    risk_level = "MEDIUM"
    
    # Base entry calculation
    base_entry = current_price
    
    # 1. Resistance level analysis
    valid_resistances = resistances[resistances >= current_price * 0.995]  # Within 0.5%
    if not valid_resistances.empty:
        nearest_resistance = valid_resistances.min()
        resistance_buffer = atr * 0.1  # Small buffer below resistance
        base_entry = nearest_resistance - resistance_buffer
        strategy_reasons.append(f"Entry below resistance: ${nearest_resistance:.2f}")
        risk_level = "LOW"
    else:
        # No nearby resistance, use current price with small discount
        base_entry = current_price - (atr * 0.05)
        strategy_reasons.append("Entry at market with small discount")
        risk_level = "MEDIUM"
    
    # 2. RSI condition adjustment
    rsi_adjustment = 0
    if rsi > 70:  # Overbought - great for SELL
        rsi_adjustment = atr * 0.2  # Better entry price (higher)
        strategy_reasons.append(f"RSI overbought advantage ({rsi:.1f})")
        risk_level = "LOW"
    elif rsi > 60:  # Mild overbought
        rsi_adjustment = atr * 0.1
        strategy_reasons.append(f"RSI favorable ({rsi:.1f})")
    elif rsi < 30:  # Oversold - risky for SELL
        rsi_adjustment = -atr * 0.1  # Discount for risky entry
        strategy_reasons.append(f"RSI oversold risk ({rsi:.1f})")
        risk_level = "HIGH"
    
    # 3. MACD momentum check
    macd_adjustment = 0
    if macd < macd_signal:  # Bearish momentum
        macd_adjustment = atr * 0.05  # Small premium (higher entry)
        strategy_reasons.append("MACD bearish momentum")
    else:
        strategy_reasons.append("MACD neutral/bullish")
    
    # 4. Confidence factor
    confidence_adjustment = (confidence - 0.5) * atr * 0.1
    if confidence > 0.8:
        strategy_reasons.append(f"High confidence ({confidence:.1%})")
    elif confidence < 0.6:
        strategy_reasons.append(f"Low confidence ({confidence:.1%})")
        risk_level = "HIGH"
    
    # 5. Predicted price consideration
    pred_adjustment = 0
    if predicted_price < current_price:
        pred_factor = min(0.3, (current_price - predicted_price) / current_price)
        pred_adjustment = pred_factor * atr * 0.5
        strategy_reasons.append("AI predicts price decrease")
    
    # Final entry price calculation
    final_entry = base_entry + rsi_adjustment + macd_adjustment + confidence_adjustment + pred_adjustment
    
    return final_entry, strategy_reasons, risk_level


def _calculate_sell_entry_pullback(current_price, predicted_price, supports, resistances,
                                   rsi, atr, macd, macd_signal, confidence):
    """
    Calculate optimal SELL entry price using PULLBACK strategy (NEW)
    
    Targets relief bounce toward resistance for better entry opportunities.
    """
    strategy_reasons = []
    risk_level = "MEDIUM"
    
    # Calculate volatility anchor
    volatility_anchor = max(atr, current_price * 0.0015)
    
    # Base pullback entry - target relief bounce zone
    bounce_factor = np.random.uniform(0.15, 0.30)  # Random between 15-30% of volatility
    base_pullback = current_price + (bounce_factor * volatility_anchor)
    
    strategy_reasons.append(f"Pullback target: {bounce_factor:.1%} relief bounce")
    
    # 1. Resistance level confluence
    valid_resistances = resistances[resistances >= current_price * 0.99]  # Within 1%
    if not valid_resistances.empty:
        nearest_resistance = valid_resistances.min()
        resistance_distance = abs(base_pullback - nearest_resistance) / current_price
        
        if resistance_distance < 0.002:  # Within 0.2% of resistance
            base_pullback = nearest_resistance - (atr * 0.05)  # Slightly below resistance
            strategy_reasons.append(f"Resistance confluence at ${nearest_resistance:.2f}")
            risk_level = "LOW"
        else:
            strategy_reasons.append(f"Pullback bounce zone alignment")
    else:
        strategy_reasons.append("Volatility-based bounce zone")
    
    # 2. RSI overbought fine-tuning
    rsi_adjustment = 0
    if rsi > 65:  # Very overbought - wait for more bounce
        rsi_adjustment = volatility_anchor * 0.1
        strategy_reasons.append("RSI overbought fine-tune - higher bounce")
        risk_level = "LOW"
    elif rsi > 55:  # Mildly overbought - good bounce zone
        rsi_adjustment = 0
        strategy_reasons.append("RSI bounce zone optimal")
    elif rsi < 40:  # Not overbought enough - lower bounce target
        rsi_adjustment = -volatility_anchor * 0.05
        strategy_reasons.append("RSI not overbought - conservative bounce")
        risk_level = "HIGH"
    
    # 3. Confidence adjustment
    confidence_adjustment = (confidence - 0.5) * volatility_anchor * 0.1
    if confidence > 0.8:
        strategy_reasons.append(f"High confidence bounce ({confidence:.1%})")
    elif confidence < 0.6:
        strategy_reasons.append(f"Low confidence bounce ({confidence:.1%})")
        risk_level = "HIGH"
    
    # Final pullback entry price
    final_entry = base_pullback + rsi_adjustment + confidence_adjustment
    
    # Ensure pullback entry is above current price
    if final_entry <= current_price:
        final_entry = current_price + (volatility_anchor * 0.1)
        strategy_reasons.append("Adjusted to ensure bounce above current price")
    
    return final_entry, strategy_reasons, risk_level


def _calculate_momentum_fill_probability(price_distance):
    """Calculate fill probability for momentum mode"""
    if price_distance <= 0.001:  # 0.1%
        return 0.95
    elif price_distance <= 0.002:  # 0.2%
        return 0.90
    elif price_distance <= 0.003:  # 0.3%
        return 0.80
    elif price_distance <= 0.005:  # 0.5%
        return 0.70
    else:
        return max(0.1, 0.7 * (1 - price_distance * 100))  # Realistic decay


def _calculate_pullback_fill_probability(price_distance):
    """Calculate fill probability for pullback mode (generally lower due to waiting)"""
    if price_distance <= 0.002:  # 0.2%
        return 0.85
    elif price_distance <= 0.003:  # 0.3%
        return 0.75
    elif price_distance <= 0.005:  # 0.5%
        return 0.70
    else:
        # Reduce probability more aggressively for pullback
        return max(0.1, 0.6 * (1 - price_distance * 80))


# Backward compatibility - keep original function names
_calculate_buy_entry = _calculate_buy_entry_momentum
_calculate_sell_entry = _calculate_sell_entry_momentum