"""
WebSocket Integration Helper untuk Trading Bot v8.3
Menyederhanakan penggunaan WebSocket di main app
"""

import streamlit as st
from datetime import datetime

def display_websocket_panel(ws_manager, api_source, symbol, api_key_1=None, api_key_2=None):
    """Display WebSocket connection panel"""
    with st.expander("🌐 Real-time WebSocket Connection", expanded=True):
        ws_cols = st.columns([2, 1, 1])
        
        # Status display
        status = ws_manager.get_connection_status(api_source, symbol)
        price = ws_manager.get_latest_price(api_source, symbol)
        
        if "✅" in status:
            ws_cols[0].success(f"🟢 {api_source} WebSocket: {status}")
        elif "Error" in status:
            ws_cols[0].error(f"🔴 {api_source}: {status}")
        else:
            ws_cols[0].warning(f"🟡 {api_source}: {status}")
        
        # Price display
        if price and price > 0:
            price_history = ws_manager.get_price_history(api_source, symbol, 5)
            if len(price_history) >= 2:
                prev_price = price_history[-2]['price']
                price_change = price - prev_price
                delta_text = f"+{price_change:.4f}" if price_change > 0 else f"{price_change:.4f}"
                ws_cols[1].metric("💰 Live Price", f"{price:.4f}", delta=delta_text)
            else:
                ws_cols[1].metric("💰 Live Price", f"{price:.4f}")
        else:
            ws_cols[1].info("⏳ Waiting for data...")
        
        # Connect/Disconnect button
        is_connected = ws_manager.is_connected(api_source, symbol)
        button_text = "🔌 Disconnect" if is_connected else "🔗 Connect"
        
        if ws_cols[2].button(button_text, use_container_width=True):
            if is_connected:
                client_key = f"{api_source}_{symbol}"
                ws_manager.stop_websocket(client_key)
                st.info("🔌 WebSocket disconnected")
            else:
                success = ws_manager.start_websocket(api_source, symbol, api_key_1, api_key_2)
                if success:
                    st.success(f"🔗 {api_source} WebSocket initiated")
                else:
                    st.error("❌ Failed to start WebSocket")
            st.rerun()
            
        # Mini price chart
        if price and len(ws_manager.get_price_history(api_source, symbol, 20)) > 5:
            price_history = ws_manager.get_price_history(api_source, symbol, 20)
            if price_history:
                import pandas as pd
                import plotly.express as px
                
                prices = [p['price'] for p in price_history if p['price']]
                timestamps = [p['timestamp'] for p in price_history if p['price']]
                
                if prices and timestamps and len(prices) == len(timestamps):
                    price_df = pd.DataFrame({
                        'Time': timestamps,
                        'Price': prices
                    })
                    
                    fig = px.line(
                        price_df, 
                        x='Time', 
                        y='Price', 
                        title=f"📊 Live {symbol} - {api_source} (Last {len(prices)} updates)",
                        height=300
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        showlegend=False,
                        margin=dict(t=40, b=20, l=20, r=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)

def get_websocket_price_for_signal(ws_manager, api_source, symbol, use_ws_price=True):
    """Get WebSocket price for signal generation"""
    if not use_ws_price:
        return None
        
    ws_price = ws_manager.get_latest_price(api_source, symbol)
    if ws_price and ws_price > 0:
        return ws_price
    return None

def format_websocket_status(status):
    """Format WebSocket status for display"""
    if "Connected" in status and "✅" in status:
        return "🟢 Connected"
    elif "Connecting" in status:
        return "🟡 Connecting..."
    elif "Error" in status:
        return "🔴 Error"
    else:
        return "⚪ Disconnected"