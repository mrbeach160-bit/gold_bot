# smart_entry.py - Smart AI Entry Price calculations
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Check for validation utilities
try:
    from ..validation_utils import (
        validate_signal_realtime, validate_take_profit_realtime, 
        is_signal_expired, get_signal_quality_score,
        get_price_staleness_indicator, format_quality_indicator
    )
    VALIDATION_UTILS_AVAILABLE = True
except ImportError:
    VALIDATION_UTILS_AVAILABLE = False

# Import indicators
try:
    from utils.indicators import get_support_resistance
except ImportError:
    st.error("utils/indicators.py not found. Please ensure indicators module is available.")


def calculate_smart_entry_price(signal, recent_data, predicted_price, confidence, symbol="XAUUSD"):
    """
    🧠 SMART AI ENTRY PRICE CALCULATION
    
    Menghitung entry price berdasarkan multiple factors:
    - Support/Resistance levels
    - RSI conditions (oversold/overbought)
    - MACD momentum
    - ATR volatility buffer
    - AI confidence adjustment
    
    Args:
        signal: 'BUY' atau 'SELL'
        recent_data: DataFrame dengan OHLC dan indicators
        predicted_price: LSTM prediction price
        confidence: AI confidence level (0-1)
        symbol: Trading symbol untuk pip calculation
        
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
            entry_price, reasons, risk = _calculate_buy_entry(
                current_price, predicted_price, supports, resistances,
                rsi, atr, macd, macd_signal, confidence
            )
        elif signal == "SELL":
            entry_price, reasons, risk = _calculate_sell_entry(
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
        # This was the missing critical calculation that caused the undefined variable error
        price_distance = abs(entry_price - current_price) / current_price if current_price > 0 else 0
        
        # Calculate expected fill probability with realistic assessment
        if price_distance <= 0.001:  # 0.1%
            fill_probability = 0.95
        elif price_distance <= 0.002:  # 0.2%
            fill_probability = 0.90
        elif price_distance <= 0.003:  # 0.3%
            fill_probability = 0.80
        elif price_distance <= 0.005:  # 0.5%
            fill_probability = 0.70
        else:
            fill_probability = max(0.1, 0.7 * (1 - price_distance * 100))  # Realistic decay
        
        # Minimum fill probability threshold (70%)
        if fill_probability < 0.7:
            return {
                'entry_price': current_price,
                'strategy_reasons': [f'REJECTED: Fill probability {fill_probability:.1%} below 70% minimum'],
                'risk_level': 'REJECTED',
                'expected_fill_probability': fill_probability
            }
        
        # Validate entry price reasonableness - TIGHTENED to 0.5%
        max_deviation = 0.005  # 0.5% max deviation from current price (was 2%)
        if price_distance > max_deviation:
            # REJECT signal instead of adjusting when threshold exceeded
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
            'entry_price': current_price,
            'strategy_reasons': [f'Fallback to current price due to error: {str(e)}'],
            'risk_level': 'HIGH',
            'expected_fill_probability': 0.5
        }


def _calculate_buy_entry(current_price, predicted_price, supports, resistances, 
                        rsi, atr, macd, macd_signal, confidence):
    """
    Calculate optimal BUY entry price
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


def _calculate_sell_entry(current_price, predicted_price, supports, resistances,
                         rsi, atr, macd, macd_signal, confidence):
    """
    Calculate optimal SELL entry price
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


def display_smart_signal_results(signal, confidence, smart_entry_result, position_info, symbol, ws_price=None, current_price=None):
    """
    Enhanced UI display dengan Smart AI strategy reasoning and real-time validation
    """
    if signal == "HOLD":
        st.info("🔄 **HOLD** - Menunggu opportunity yang lebih baik")
        return
    
    # Check if signal is REJECTED
    if smart_entry_result.get('risk_level') == 'REJECTED':
        st.error("❌ **SIGNAL REJECTED**")
        st.warning("Signal failed validation criteria:")
        for reason in smart_entry_result['strategy_reasons']:
            st.markdown(f"• {reason}")
        return
    
    # Main signal display
    signal_color = "🟢" if signal == "BUY" else "🔴"
    confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
    
    # Real-time validation and quality scoring
    validation_result = None
    if VALIDATION_UTILS_AVAILABLE and position_info:
        real_time_price = ws_price if ws_price and ws_price > 0 else current_price
        if real_time_price:
            validation_result = validate_signal_realtime(
                signal=signal,
                entry_price=smart_entry_result['entry_price'],
                take_profit=position_info.get('take_profit'),
                stop_loss=position_info.get('stop_loss'),
                current_price=current_price or real_time_price,
                ws_price=ws_price,
                confidence=confidence,
                symbol=symbol
            )
    
    st.markdown(f"""
    ## {signal_color} **{signal} SIGNAL**
    **Confidence:** {confidence:.1%} `{confidence_bar}`
    """)
    
    # Display validation results if available
    if validation_result:
        if not validation_result['is_valid']:
            st.error(f"❌ **SIGNAL VALIDATION FAILED**: {validation_result['rejection_reason']}")
            return
        
        # Display quality score and staleness
        quality_indicator = format_quality_indicator(validation_result['quality_score']) if VALIDATION_UTILS_AVAILABLE else None
        staleness_info = get_price_staleness_indicator(
            current_price or validation_result['real_time_price'], 
            ws_price
        ) if VALIDATION_UTILS_AVAILABLE and current_price else None
        
        if quality_indicator:
            st.markdown(f"**Signal Quality:** {quality_indicator['emoji']} {quality_indicator['text']}")
        
        if staleness_info:
            if staleness_info['color'] == 'red':
                st.error(staleness_info['message'])
            elif staleness_info['color'] == 'orange':
                st.warning(staleness_info['message'])
            else:
                st.success(staleness_info['message'])
        
        # Display warnings if any
        if validation_result.get('warnings'):
            with st.expander("⚠️ Signal Warnings", expanded=False):
                for warning in validation_result['warnings']:
                    st.warning(warning)
    
    # Smart Entry Information
    entry_price = smart_entry_result['entry_price']
    fill_probability = smart_entry_result['expected_fill_probability']
    risk_level = smart_entry_result['risk_level']
    
    # Import format_price function locally to avoid circular imports
    try:
        from ..utils.formatters import format_price
    except ImportError:
        def format_price(symbol, price):
            return f"{price:.5f}"
    
    # Color coding for risk level
    risk_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
    risk_color = risk_colors.get(risk_level, "⚪")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Smart Entry Price",
            format_price(symbol, entry_price),
            help="AI-optimized entry price based on multiple factors"
        )
    
    with col2:
        fill_prob_color = "🟢" if fill_probability >= 0.8 else "🟡" if fill_probability >= 0.7 else "🔴"
        st.metric(
            f"{fill_prob_color} Fill Probability",
            f"{fill_probability:.1%}",
            help="Estimated probability of order execution"
        )
    
    with col3:
        st.metric(
            f"{risk_color} Risk Level",
            risk_level,
            help="Entry risk assessment based on market conditions"
        )
    
    # Real-time price comparison if available
    if ws_price and current_price:
        price_diff_pct = abs(ws_price - current_price) / current_price
        entry_diff_pct = abs(entry_price - ws_price) / ws_price
        
        st.markdown("### 📡 **Real-time Price Analysis**")
        rt_cols = st.columns(3)
        rt_cols[0].metric("API Price", format_price(symbol, current_price))
        rt_cols[1].metric("WebSocket Price", format_price(symbol, ws_price))
        rt_cols[2].metric("Entry Deviation", f"{entry_diff_pct:.2%}")
    
    # Strategy Reasoning
    st.markdown("### 🧠 **Smart Entry Strategy**")
    
    for i, reason in enumerate(smart_entry_result['strategy_reasons'], 1):
        st.markdown(f"**{i}.** {reason}")
    
    # Position Details
    if position_info:
        st.markdown("### 📊 **Position Details**")
        
        detail_cols = st.columns(4)
        detail_cols[0].metric("Position Size", f"{position_info['position_size']:.4f} {symbol.split('/')[0] if '/' in symbol else 'units'}")
        detail_cols[1].metric("Stop Loss", format_price(symbol, position_info['stop_loss']))
        detail_cols[2].metric("Take Profit", format_price(symbol, position_info['take_profit']))
        detail_cols[3].metric("Risk Amount", f"${position_info['risk_amount']:.2f}")
        
        # Risk-Reward Ratio
        sl_dist = abs(position_info['entry_price'] - position_info['stop_loss'])
        tp_dist = abs(position_info['take_profit'] - position_info['entry_price'])
        rr_ratio = (tp_dist / sl_dist) if sl_dist > 0 else 0
        rr_color = "🟢" if rr_ratio >= 2 else "🟡" if rr_ratio >= 1.5 else "🔴"
        st.metric(f"{rr_color} Risk:Reward Ratio", f"1:{rr_ratio:.2f}")
        
        # Signal expiry countdown if validation available
        if validation_result and VALIDATION_UTILS_AVAILABLE:
            signal_timestamp = datetime.now()  # In practice, this should be passed from signal generation
            expiry_info = is_signal_expired(signal_timestamp, 30)
            if expiry_info['remaining_seconds'] > 0:
                st.info(f"⏰ Signal expires in {expiry_info['remaining_seconds']} seconds")
            else:
                st.error("⏰ Signal has expired - generate new signal for current market conditions")