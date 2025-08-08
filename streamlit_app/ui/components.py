# components.py - Common UI components
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from ..utils.formatters import format_price


def display_websocket_status(ws_manager, symbol, api_source):
    """Display WebSocket connection status and controls"""
    with st.expander("🌐 Real-time Connection", expanded=True):
        ws_cols = st.columns([2, 1, 1])
        
        # Status WebSocket
        ws_status = ws_manager.get_status(symbol)
        ws_price = ws_manager.get_price(symbol)
        
        if "✅" in ws_status:
            if api_source == "Twelve Data":
                ws_cols[0].success(f"🟢 Twelve Data WebSocket: {ws_status}")
            else:
                ws_cols[0].success(f"🟢 Binance WebSocket: {ws_status}")
        elif "Error" in ws_status:
            ws_cols[0].error(f"🔴 Status: {ws_status}")
        else:
            ws_cols[0].warning(f"🟡 Status: {ws_status}")
        
        # Live Price Display
        if ws_price and ws_price > 0:
            price_history = ws_manager.get_price_history(symbol, 5)
            if len(price_history) >= 2:
                prev_price = price_history[-2]['price']
                price_change = ws_price - prev_price
                ws_cols[1].metric(
                    "🔴 Live Price", 
                    format_price(symbol, ws_price),
                    delta=f"{format_price(symbol, price_change)}"
                )
            else:
                ws_cols[1].metric("🔴 Live Price", format_price(symbol, ws_price))
        else:
            ws_cols[1].info("⏳ Waiting for data...")
        
        return ws_cols, ws_price


def display_price_history_chart(ws_manager, symbol, api_source):
    """Display mini price history chart"""
    ws_price = ws_manager.get_price(symbol)
    if ws_price and len(ws_manager.get_price_history(symbol, 20)) > 5:
        price_history = ws_manager.get_price_history(symbol, 20)
        prices = [p['price'] for p in price_history]
        timestamps = [p['timestamp'] for p in price_history]
        
        price_df = pd.DataFrame({
            'Time': timestamps,
            'Price': prices
        })
        fig_mini = px.line(
            price_df, 
            x='Time', 
            y='Price', 
            title=f"📊 Live Price Feed - {api_source} (Last 20 updates)",
            height=300
        )
        fig_mini.update_layout(template="plotly_dark")
        st.plotly_chart(fig_mini, use_container_width=True)


def display_trading_controls():
    """Display trading controls and settings"""
    live_cols = st.columns([2, 1, 1])
    
    # Import timeframe mapping locally to avoid circular imports
    timeframe_mapping = {
        '1 minute': '1min',
        '5 minutes': '5min',
        '15 minutes': '15min',
        '30 minutes': '30min',
        '1 hour': '1h',
        '4 hours': '4h',
        '1 day': '1day'
    }
    
    live_tf_key = live_cols[0].selectbox(
        "Pilih Timeframe", 
        list(timeframe_mapping.keys()), 
        index=2, 
        key="live_tf_select", 
        help="Timeframe untuk analisis sinyal"
    )
    
    col_refresh = live_cols[1]
    col_auto = live_cols[2]
    auto_refresh_live = col_auto.checkbox("🔁 Auto Refresh", key="auto_live")
    refresh_live = col_refresh.button("🔄 Refresh", use_container_width=True)
    
    return live_tf_key, auto_refresh_live, refresh_live


def display_risk_management_controls():
    """Display risk management settings"""
    with st.expander("Pengaturan Risk Management", expanded=True):
        risk_cols = st.columns(3)
        live_sl = risk_cols[0].number_input("Stop Loss (pips)", min_value=5, value=20, key="live_sl")
        use_ai_tp = st.checkbox("Gunakan Take Profit dari AI (berdasarkan Support/Resistance)", value=True)
        live_tp = risk_cols[1].number_input("Take Profit (pips)", min_value=5, value=40, key="live_tp", disabled=use_ai_tp)
        live_risk = risk_cols[2].slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1, key="live_risk")
        
        return live_sl, live_tp, live_risk, use_ai_tp


def display_signal_buttons(websocket_available, api_source):
    """Display signal generation buttons"""
    signal_cols = st.columns(3)
    generate_signal = signal_cols[0].button("📊 Generate Signal", use_container_width=True, type="primary")
    
    # Live Stream Button (Enhanced untuk semua providers)
    if websocket_available and api_source in ["Binance", "Twelve Data"]:
        start_live_stream = signal_cols[1].button("🔴 Live Stream (30s)", use_container_width=True, type="secondary")
        use_ws_price = signal_cols[2].checkbox("📡 Use Real-time Price", value=True, help="Use real-time streaming price for signals")
    else:
        start_live_stream = False
        use_ws_price = False
    
    return generate_signal, start_live_stream, use_ws_price


def display_sidebar_controls():
    """Display sidebar controls for API configuration"""
    with st.sidebar:
        st.header("🔧 Pengaturan Dasar")
        api_source = st.selectbox("Pilih Sumber Data", ["Twelve Data", "Binance"])

        api_key_1 = None
        api_key_2 = None
        symbol_options = []

        if api_source == "Twelve Data":
            st.info("📈 Menyediakan data Forex, Komoditas, dan Kripto dengan WebSocket real-time.")
            api_key_1 = st.text_input("Masukkan API Key Twelve Data Anda", type="password", help="Dapatkan API key gratis di https://twelvedata.com/")
            symbol_options = ["XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD", "ETH/USD"]
            
            if api_key_1:
                st.success("✅ API Key detected")
                with st.expander("📊 WebSocket Features", expanded=False):
                    st.markdown("**🌐 Real-time WebSocket:**")
                    st.code("XAU/USD, EUR/USD, GBP/USD, BTC/USD, ETH/USD")
                    st.markdown("**🔄 Fallback Polling:**")
                    st.info("Auto fallback ke polling jika WebSocket gagal")
            
            if not api_key_1 or api_key_1.strip() == "":
                st.error("API Key Twelve Data diperlukan.")
                st.stop()

        elif api_source == "Binance":
            st.info("💱 Menyediakan data Kripto dari Bursa Binance dengan WebSocket real-time.")
            api_key_1 = st.text_input("Binance API Key", type="password")
            api_key_2 = st.text_input("Binance API Secret", type="password")
            symbol_options = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
            
            if not api_key_1 or not api_key_2 or api_key_1.strip() == "" or api_key_2.strip() == "":
                st.error("API Key & Secret Binance diperlukan.")
                st.stop()

        symbol = st.selectbox("Pilih Simbol Trading", symbol_options, key=f"symbol_input_{api_source.lower()}")

        st.header("💰 Pengaturan Modal")
        account_balance = st.number_input("Modal Awal ($)", min_value=100.0, value=1000.0, step=100.0)
        
        return api_source, api_key_1, api_key_2, symbol, account_balance


def display_market_status():
    """Display market status information"""
    st.header("📊 Status Pasar")
    # Add market status logic here
    pass


def display_system_status(new_system_available, websocket_available, data_manager=None, trading_manager=None):
    """Display system status indicators"""
    # Phase 3 System Status
    if new_system_available:
        st.success("🚀 Phase 3 Unified Data & Trading System Active")
        if data_manager and trading_manager:
            st.info(f"📊 DataManager: {len(data_manager.providers)} providers | 🎯 TradingManager: {trading_manager.strategy.name}")
    else:
        st.info("📦 Using Legacy Utils System")
    
    # WebSocket availability indicator
    if websocket_available:
        st.success("🌐 WebSocket support available for real-time trading")
    else:
        st.warning("⚠️ WebSocket not available. Install with: pip install websocket-client")