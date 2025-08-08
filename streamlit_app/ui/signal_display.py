"""
Signal Display UI Module

Contains functions for displaying trading signals and related information.
"""

from datetime import datetime
from ..utils.price_format import format_price

# Import validation utilities if available
try:
    from ..validation_utils import (
        validate_signal_realtime, is_signal_expired, get_signal_quality_score,
        get_price_staleness_indicator, format_quality_indicator
    )
    VALIDATION_UTILS_AVAILABLE = True
except ImportError:
    VALIDATION_UTILS_AVAILABLE = False

# Import streamlit for UI
try:
    import streamlit as st
except ImportError:
    # Fallback for when streamlit is not available (e.g., in tests)
    class MockStreamlit:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def success(self, msg): print(f"SUCCESS: {msg}")
        def markdown(self, msg): print(f"MD: {msg}")
        def metric(self, *args, **kwargs): print(f"METRIC: {args}")
        def columns(self, n): return [self] * n
        def expander(self, *args, **kwargs): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    st = MockStreamlit()


def display_smart_signal_results(signal, confidence, smart_entry_result, position_info, symbol, ws_price=None, current_price=None):
    """
    Enhanced UI display with Smart AI strategy reasoning and real-time validation
    
    Args:
        signal: Trading signal ('BUY', 'SELL', 'HOLD')
        confidence: AI confidence level (0-1)
        smart_entry_result: Result from smart entry calculation
        position_info: Position information dictionary
        symbol: Trading symbol
        ws_price: WebSocket real-time price (optional)
        current_price: Current API price (optional)
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
    
    # Render main signal header
    render_signal_header(signal, confidence, smart_entry_result, symbol, ws_price, current_price)
    
    # Render validation results if available
    render_validation_results(signal, smart_entry_result, position_info, symbol, ws_price, current_price)
    
    # Render entry metrics
    render_entry_metrics(smart_entry_result, symbol, ws_price, current_price)
    
    # Render strategy reasoning
    render_strategy_reasons(smart_entry_result)
    
    # Render position details
    if position_info:
        render_position_details(position_info, symbol)


def render_signal_header(signal, confidence, smart_entry_result, symbol, ws_price=None, current_price=None):
    """Render the main signal header with confidence indicator"""
    signal_color = "🟢" if signal == "BUY" else "🔴"
    confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
    
    st.markdown(f"""
    ## {signal_color} **{signal} SIGNAL**
    **Confidence:** {confidence:.1%} `{confidence_bar}`
    """)


def render_validation_results(signal, smart_entry_result, position_info, symbol, ws_price=None, current_price=None):
    """Render real-time validation results if available"""
    if not VALIDATION_UTILS_AVAILABLE or not position_info:
        return
    
    real_time_price = ws_price if ws_price and ws_price > 0 else current_price
    if not real_time_price:
        return
    
    validation_result = validate_signal_realtime(
        signal=signal,
        entry_price=smart_entry_result['entry_price'],
        take_profit=position_info.get('take_profit'),
        stop_loss=position_info.get('stop_loss'),
        current_price=current_price or real_time_price,
        ws_price=ws_price,
        confidence=smart_entry_result.get('confidence', 0.5),
        symbol=symbol
    )
    
    if not validation_result['is_valid']:
        st.error(f"❌ **SIGNAL VALIDATION FAILED**: {validation_result['rejection_reason']}")
        return
    
    # Display quality score and staleness
    quality_indicator = format_quality_indicator(validation_result['quality_score'])
    staleness_info = get_price_staleness_indicator(
        current_price or validation_result['real_time_price'], 
        ws_price
    ) if current_price else None
    
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


def render_entry_metrics(smart_entry_result, symbol, ws_price=None, current_price=None):
    """Render smart entry metrics"""
    entry_price = smart_entry_result['entry_price']
    fill_probability = smart_entry_result['expected_fill_probability']
    risk_level = smart_entry_result['risk_level']
    
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


def render_strategy_reasons(smart_entry_result):
    """Render strategy reasoning"""
    st.markdown("### 🧠 **Smart Entry Strategy**")
    
    for i, reason in enumerate(smart_entry_result['strategy_reasons'], 1):
        st.markdown(f"**{i}.** {reason}")


def render_position_details(position_info, symbol):
    """Render position details section"""
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
    if VALIDATION_UTILS_AVAILABLE:
        signal_timestamp = datetime.now()  # In practice, this should be passed from signal generation
        expiry_info = is_signal_expired(signal_timestamp, 30)
        if expiry_info['remaining_seconds'] > 0:
            st.info(f"⏰ Signal expires in {expiry_info['remaining_seconds']} seconds")
        else:
            st.error("⏰ Signal has expired - generate new signal for current market conditions")