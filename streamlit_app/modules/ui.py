"""
UI utilities module for formatting and display helpers.

This module provides reusable UI elements including metric cards,
expandable sections, formatting helpers, and trading signal displays.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from .config import is_feature_enabled

# Import validation utilities if available
if is_feature_enabled('VALIDATION_UTILS_AVAILABLE'):
    from ..validation_utils import (
        format_quality_indicator, get_price_staleness_indicator
    )


def format_price(symbol: str, price: Union[float, int]) -> str:
    """
    Format price based on symbol type with appropriate decimal places.
    
    Args:
        symbol: Trading symbol
        price: Price value to format
        
    Returns:
        Formatted price string
    """
    try:
        price = float(price)
        symbol_upper = symbol.upper()
        
        if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
            return f"${price:,.2f}"
        elif 'BTC' in symbol_upper:
            return f"${price:,.1f}"
        elif 'ETH' in symbol_upper:
            return f"${price:,.2f}"
        elif 'JPY' in symbol_upper:
            return f"{price:.3f}"
        elif any(pair in symbol_upper for pair in ['EUR', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF']):
            return f"{price:.5f}"
        else:
            return f"{price:.5f}"
    except (ValueError, TypeError):
        return str(price)


def format_percentage(value: Union[float, int], decimals: int = 1) -> str:
    """Format percentage value with specified decimal places."""
    try:
        return f"{float(value):.{decimals}f}%"
    except (ValueError, TypeError):
        return "0.0%"


def format_currency(amount: Union[float, int], currency: str = "USD") -> str:
    """Format currency amount with appropriate symbol."""
    try:
        amount = float(amount)
        if currency.upper() == "USD":
            return f"${amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"
    except (ValueError, TypeError):
        return f"0.00 {currency}"


def display_smart_signal_results(signal: int, confidence: float, smart_entry_result: Dict[str, Any], 
                                position_info: Dict[str, Any], symbol: str, 
                                ws_price: Optional[float] = None, current_price: Optional[float] = None):
    """
    Display comprehensive smart signal results with detailed analysis.
    
    Args:
        signal: Trading signal (1=BUY, -1=SELL, 0=HOLD)
        confidence: Model confidence (0-1)
        smart_entry_result: Smart entry calculation results
        position_info: Position sizing information
        symbol: Trading symbol
        ws_price: WebSocket real-time price
        current_price: Current market price
    """
    
    # Signal header with confidence
    signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
    signal_color = "🟢" if signal == 1 else "🔴" if signal == -1 else "🟡"
    
    st.header(f"{signal_color} {signal_text} Signal")
    
    # Confidence and quality indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_color = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.5 else "🔴"
        st.metric("AI Confidence", f"{confidence:.1%}", help="Model prediction confidence")
        st.write(f"{confidence_color} {'High' if confidence >= 0.7 else 'Medium' if confidence >= 0.5 else 'Low'} Confidence")
    
    with col2:
        if ws_price:
            st.metric("Live Price", format_price(symbol, ws_price))
            if is_feature_enabled('VALIDATION_UTILS_AVAILABLE'):
                staleness = get_price_staleness_indicator(datetime.now())
                st.write(f"📡 {staleness}")
        elif current_price:
            st.metric("Current Price", format_price(symbol, current_price))
            st.write("📊 Historical Data")
    
    with col3:
        if smart_entry_result:
            entry_price = smart_entry_result.get('smart_entry_price', current_price or 0)
            st.metric("Smart Entry", format_price(symbol, entry_price))
            
            risk_level = smart_entry_result.get('risk_level', 'UNKNOWN')
            risk_color = "🟢" if risk_level == "LOW" else "🟡" if risk_level == "MEDIUM" else "🔴"
            st.write(f"{risk_color} {risk_level} Risk")
    
    # Smart entry analysis
    if smart_entry_result and signal != 0:
        st.subheader("📊 Smart Entry Analysis")
        
        with st.expander("Entry Strategy Details", expanded=True):
            reasons = smart_entry_result.get('reasons', [])
            for reason in reasons:
                if "⚠️" in reason or "🔴" in reason:
                    st.warning(reason)
                elif "✅" in reason or "🟢" in reason:
                    st.success(reason)
                else:
                    st.info(reason)
            
            # Technical levels
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'support_level' in smart_entry_result:
                    st.metric("Support Level", format_price(symbol, smart_entry_result['support_level']))
            
            with col2:
                if 'resistance_level' in smart_entry_result:
                    st.metric("Resistance Level", format_price(symbol, smart_entry_result['resistance_level']))
            
            with col3:
                if 'atr_value' in smart_entry_result:
                    st.metric("ATR Value", format_price(symbol, smart_entry_result['atr_value']))
            
            # Fill probability
            fill_prob = smart_entry_result.get('fill_probability', 0)
            prob_color = "🟢" if fill_prob >= 0.8 else "🟡" if fill_prob >= 0.6 else "🔴"
            st.metric("Fill Probability", f"{fill_prob:.1%}")
            st.write(f"{prob_color} {'Very High' if fill_prob >= 0.9 else 'High' if fill_prob >= 0.8 else 'Medium' if fill_prob >= 0.6 else 'Low'} Likelihood")
    
    # Position sizing information
    if position_info and signal != 0:
        st.subheader("💰 Position Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Position Size", f"{position_info.get('position_size', 0):.4f} lots")
            st.metric("Required Margin", format_currency(position_info.get('required_margin', 0)))
        
        with col2:
            st.metric("Stop Loss", format_price(symbol, position_info.get('stop_loss_price', 0)))
            st.metric("Take Profit", format_price(symbol, position_info.get('take_profit_price', 0)))
        
        with col3:
            st.metric("Potential Loss", format_currency(position_info.get('potential_loss', 0)))
            st.metric("Potential Profit", format_currency(position_info.get('potential_profit', 0)))
        
        with col4:
            st.metric("Risk Amount", format_currency(position_info.get('risk_amount', 0)))
            rr_ratio = position_info.get('risk_reward_ratio', 0)
            rr_color = "🟢" if rr_ratio >= 2 else "🟡" if rr_ratio >= 1.5 else "🔴"
            st.metric("Risk:Reward", f"{rr_ratio:.2f}")
            st.write(f"{rr_color} {'Excellent' if rr_ratio >= 2 else 'Good' if rr_ratio >= 1.5 else 'Poor'} R:R")
        
        # Risk warnings
        risk_warning = position_info.get('risk_warning', '')
        if 'HIGH' in risk_warning:
            st.error(f"⚠️ {risk_warning}")
        elif 'MEDIUM' in risk_warning:
            st.warning(f"⚠️ {risk_warning}")
        else:
            st.success(f"✅ {risk_warning}")
    
    # Real-time validation if available
    if is_feature_enabled('VALIDATION_UTILS_AVAILABLE') and signal != 0:
        st.subheader("🔍 Real-time Validation")
        
        try:
            from ..validation_utils import validate_signal_realtime, get_signal_quality_score
            
            validation_result = validate_signal_realtime(signal, confidence, symbol)
            quality_score = get_signal_quality_score(signal, confidence, smart_entry_result)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if validation_result.get('valid', False):
                    st.success("✅ Signal passes real-time validation")
                else:
                    st.error("❌ Signal validation failed")
                    for issue in validation_result.get('issues', []):
                        st.warning(f"⚠️ {issue}")
            
            with col2:
                quality_indicator = format_quality_indicator(quality_score)
                st.metric("Signal Quality", quality_indicator)
                
        except Exception as e:
            st.warning(f"Validation utilities unavailable: {e}")


def create_trading_chart(data: pd.DataFrame, trades: Optional[List[Dict]] = None, 
                        title: str = "Trading Chart") -> go.Figure:
    """
    Create an interactive trading chart with OHLCV data and trade markers.
    
    Args:
        data: OHLCV DataFrame with index as datetime
        trades: Optional list of trade dictionaries
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=['Price & Volume', 'RSI', 'MACD'],
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Volume bars
    if 'volume' in data.columns:
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                opacity=0.3,
                yaxis='y2'
            ),
            row=1, col=1
        )
    
    # Add moving averages if available
    if 'EMA_10' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['EMA_10'],
                name='EMA 10',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'EMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['EMA_20'],
                name='EMA 20',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    # RSI
    if 'rsi' in data.columns or 'RSI_14' in data.columns:
        rsi_col = 'rsi' if 'rsi' in data.columns else 'RSI_14'
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[rsi_col],
                name='RSI',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'MACD_12_26_9' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD_12_26_9'],
                name='MACD',
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        
        if 'MACDs_12_26_9' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACDs_12_26_9'],
                    name='MACD Signal',
                    line=dict(color='red')
                ),
                row=3, col=1
            )
        
        if 'MACDh_12_26_9' in data.columns:
            colors = ['green' if val >= 0 else 'red' for val in data['MACDh_12_26_9']]
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['MACDh_12_26_9'],
                    name='MACD Histogram',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=3, col=1
            )
    
    # Add trade markers if provided
    if trades:
        for trade in trades:
            # Entry marker
            color = 'green' if trade.get('type') == 'BUY' else 'red'
            fig.add_trace(
                go.Scatter(
                    x=[trade.get('entry_time')],
                    y=[trade.get('entry_price')],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if trade.get('type') == 'BUY' else 'triangle-down',
                        size=12,
                        color=color
                    ),
                    name=f"{trade.get('type')} Entry",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Exit marker if available
            if trade.get('exit_time') and trade.get('exit_price'):
                fig.add_trace(
                    go.Scatter(
                        x=[trade.get('exit_time')],
                        y=[trade.get('exit_price')],
                        mode='markers',
                        marker=dict(
                            symbol='x',
                            size=10,
                            color=color
                        ),
                        name=f"{trade.get('type')} Exit",
                        showlegend=False
                    ),
                    row=1, col=1
                )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig


def display_metrics_grid(metrics: Dict[str, Any], columns: int = 4):
    """
    Display metrics in a responsive grid layout.
    
    Args:
        metrics: Dictionary of metric_name: value pairs
        columns: Number of columns in the grid
    """
    
    metric_items = list(metrics.items())
    rows = len(metric_items) // columns + (1 if len(metric_items) % columns else 0)
    
    for row in range(rows):
        cols = st.columns(columns)
        for col in range(columns):
            idx = row * columns + col
            if idx < len(metric_items):
                metric_name, metric_value = metric_items[idx]
                with cols[col]:
                    st.metric(metric_name, metric_value)


def create_expandable_section(title: str, content_func: callable, expanded: bool = False):
    """
    Create an expandable section with custom content.
    
    Args:
        title: Section title
        content_func: Function that renders the content
        expanded: Whether section is expanded by default
    """
    
    with st.expander(title, expanded=expanded):
        content_func()


def format_trade_summary(trades: List[Dict]) -> str:
    """
    Format a summary of trades for display.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Formatted trade summary string
    """
    
    if not trades:
        return "No trades executed."
    
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    avg_win = np.mean([t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]) if winning_trades > 0 else 0
    avg_loss = np.mean([t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]) if losing_trades > 0 else 0
    
    summary = f"""
    **Trade Summary:**
    - Total Trades: {total_trades}
    - Winning Trades: {winning_trades}
    - Losing Trades: {losing_trades}
    - Win Rate: {win_rate:.1f}%
    - Total P&L: {format_currency(total_pnl)}
    - Average Win: {format_currency(avg_win)}
    - Average Loss: {format_currency(avg_loss)}
    """
    
    return summary


def show_loading_spinner(text: str = "Loading..."):
    """Show a loading spinner with custom text."""
    return st.spinner(text)


def display_data_quality_info(data: pd.DataFrame, symbol: str):
    """
    Display data quality information in an info box.
    
    Args:
        data: The dataset to analyze
        symbol: Trading symbol
    """
    
    if data.empty:
        st.error("❌ No data available")
        return
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Points", len(data))
    
    with col2:
        st.metric("Features", len(data.columns))
    
    with col3:
        time_range = data.index[-1] - data.index[0]
        st.metric("Time Range", f"{time_range.days} days")
    
    with col4:
        latest_time = data.index[-1]
        age = datetime.now() - latest_time.to_pydatetime()
        st.metric("Data Age", f"{age.seconds // 3600}h {(age.seconds % 3600) // 60}m")
    
    # Missing data check
    missing_data = data.isnull().sum().sum()
    total_cells = len(data) * len(data.columns)
    missing_percent = (missing_data / total_cells) * 100
    
    if missing_percent > 5:
        st.warning(f"⚠️ {missing_percent:.1f}% missing data detected")
    elif missing_percent > 0:
        st.info(f"ℹ️ {missing_percent:.1f}% missing data (acceptable)")
    else:
        st.success("✅ No missing data")


def create_status_indicator(status: str, message: str = "") -> str:
    """
    Create a status indicator with color coding.
    
    Args:
        status: Status level ('success', 'warning', 'error', 'info')
        message: Optional message to display
        
    Returns:
        Formatted status string with emoji
    """
    
    status_map = {
        'success': '✅',
        'warning': '⚠️',
        'error': '❌',
        'info': 'ℹ️'
    }
    
    emoji = status_map.get(status.lower(), 'ℹ️')
    return f"{emoji} {message}" if message else emoji