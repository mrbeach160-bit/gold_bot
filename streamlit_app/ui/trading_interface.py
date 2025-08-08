# trading_interface.py - Trading interface components
import streamlit as st
import time

from ..config import WEBSOCKET_AVAILABLE
from ..ai import calculate_smart_entry_price, predict_with_models, display_smart_signal_results
from ..data import load_and_process_data_enhanced
from ..trading import calculate_position_info, calculate_ai_take_profit
from ..utils.formatters import format_price
from .components import (
    display_websocket_status, 
    display_price_history_chart,
    display_trading_controls,
    display_risk_management_controls,
    display_signal_buttons
)

# Check for validation utilities
try:
    from ..validation_utils import get_price_staleness_indicator
    VALIDATION_UTILS_AVAILABLE = True
except ImportError:
    VALIDATION_UTILS_AVAILABLE = False

# Import support resistance
try:
    from utils.indicators import get_support_resistance
except ImportError:
    pass


def display_trading_interface(api_source, symbol, api_key_1, api_key_2, account_balance):
    """Display the main trading interface"""
    st.header(f"💼 Panel Trading ({api_source}) - Enhanced Real-time")
    subtab_live, subtab_backtest, subtab_signals = st.tabs(["Sinyal Live", "Backtesting", "Analisis Sinyal"])

    with subtab_live:
        st.subheader("🔍 Dapatkan Sinyal Trading Terbaru")
        
        # ============= ENHANCED PANEL WEBSOCKET (TWELVE DATA & BINANCE) =============
        if WEBSOCKET_AVAILABLE and api_source in ["Binance", "Twelve Data"]:
            ws_cols, ws_price = display_websocket_status(st.session_state.ws_manager, symbol, api_source)
            
            # Connect/Disconnect Button
            if ws_cols[2].button("🔗 Connect" if not st.session_state.ws_manager.is_connected(symbol) else "🔌 Disconnect"):
                if st.session_state.ws_manager.is_connected(symbol):
                    st.session_state.ws_manager.disconnect(symbol)
                    st.info("🔌 Stream disconnected")
                else:
                    success = st.session_state.ws_manager.start_stream(api_source, symbol, api_key_1, api_key_2)
                    if success:
                        if api_source == "Twelve Data":
                            st.success(f"🔗 Twelve Data WebSocket stream initiated")
                        else:
                            st.success(f"🔗 {api_source} stream initiated")
                    else:
                        st.error("❌ Failed to start stream")
                time.sleep(1)
                st.rerun()
                        
            # Price History Chart (Mini)
            display_price_history_chart(st.session_state.ws_manager, symbol, api_source)
        
        # ============= PANEL TRADING (ENHANCED) =============
        live_tf_key, auto_refresh_live, refresh_live = display_trading_controls()
        live_sl, live_tp, live_risk, use_ai_tp = display_risk_management_controls()
        
        # ============= TOMBOL SIGNAL (ENHANCED) =============
        generate_signal, start_live_stream, use_ws_price = display_signal_buttons(WEBSOCKET_AVAILABLE, api_source)

        # ============= LOGIC SIGNAL ENHANCED =============
        if generate_signal or refresh_live:
            _process_signal_generation(
                api_source, symbol, api_key_1, api_key_2, live_tf_key, 
                account_balance, live_sl, live_tp, live_risk, use_ai_tp, 
                use_ws_price
            )

    with subtab_backtest:
        st.subheader("🔙 Analisis Backtesting")
        st.info("Backtesting functionality akan ditambahkan di sini.")
        # TODO: Implement backtesting interface

    with subtab_signals:
        st.subheader("📈 Analisis Sinyal Historis")
        st.info("Analisis sinyal historis akan ditambahkan di sini.")
        # TODO: Implement signal analysis interface


def _process_signal_generation(api_source, symbol, api_key_1, api_key_2, live_tf_key, 
                              account_balance, live_sl, live_tp, live_risk, use_ai_tp, use_ws_price):
    """Process signal generation logic"""
    try:
        # Import model loading function locally to avoid circular imports
        from ..utils.model_loader import load_all_models
        
        all_models_live = load_all_models(symbol, live_tf_key)
        if all_models_live is None or any(v is None for v in all_models_live.values()):
            st.error(f"❌ Model untuk {symbol} timeframe {live_tf_key} belum dilatih atau gagal dimuat.")
            st.info("💡 Silakan latih semua model terlebih dahulu di tab 'Pelatihan AI'.")
        else:
            with st.spinner(f"🔄 Menganalisis kondisi pasar dari {api_source}..."):
                # Timeframe mapping
                timeframe_mapping = {
                    '1 minute': '1min',
                    '5 minutes': '5min',
                    '15 minutes': '15min',
                    '30 minutes': '30min',
                    '1 hour': '1h',
                    '4 hours': '4h',
                    '1 day': '1day'
                }
                
                recent_data = load_and_process_data_enhanced(
                    api_source, symbol, timeframe_mapping[live_tf_key], 
                    api_key_1, api_key_2, outputsize=200
                )
                
                if recent_data is None or len(recent_data) <= 61:
                    st.error("❌ Data tidak cukup atau gagal dimuat. Minimal 62 candle diperlukan.")
                    st.info("💡 Coba refresh atau periksa koneksi internet dan API key.")
                else:
                    # ENHANCED: Prioritize real-time WebSocket price integration
                    current_api_price = recent_data['close'].iloc[-1]
                    ws_price = st.session_state.ws_manager.get_price(symbol) if use_ws_price else None
                    
                    # Real-time price validation and integration
                    real_time_price = current_api_price  # Default fallback
                    price_source = "API"
                    
                    if ws_price and ws_price > 0 and use_ws_price:
                        price_diff_pct = abs(ws_price - current_api_price) / current_api_price
                        
                        # Tightened threshold for WebSocket price acceptance (2% -> 1%)
                        if price_diff_pct < 0.01:  # Within 1%
                            real_time_price = ws_price
                            price_source = "WebSocket"
                            # Update the last row with real-time price for calculations
                            recent_data.loc[recent_data.index[-1], 'close'] = ws_price
                            st.success(f"📡 Using Real-time WebSocket Price: {format_price(symbol, ws_price)} (API: {format_price(symbol, current_api_price)}, Δ: {price_diff_pct:.2%})")
                        else:
                            st.warning(f"⚠️ WebSocket price deviation {price_diff_pct:.2%} > 1% threshold, using API price")
                            st.info(f"🔄 Real-time: {format_price(symbol, ws_price)} vs API: {format_price(symbol, current_api_price)}")
                    else:
                        if use_ws_price:
                            st.info(f"⏳ WebSocket data not available, using API price: {format_price(symbol, current_api_price)}")
                    
                    # Display data freshness indicator
                    if VALIDATION_UTILS_AVAILABLE:
                        staleness_info = get_price_staleness_indicator(current_api_price, ws_price)
                        if staleness_info['staleness_level'] != 'FRESH':
                            if staleness_info['color'] == 'red':
                                st.error(f"🚨 **Data Quality Issue**: {staleness_info['message']}")
                            elif staleness_info['color'] == 'orange':
                                st.warning(f"⚠️ **Data Quality**: {staleness_info['message']}")
                    
                    st.info(f"📊 Using **{price_source}** price: {format_price(symbol, real_time_price)} for signal generation")
                    
                    prediction_data = recent_data.iloc[:-1]
                    signal, confidence, predicted_price = predict_with_models(all_models_live, prediction_data)
                    
                    if signal is None:
                        st.error("❌ Model prediction gagal. Coba refresh atau latih ulang model.")
                    else:
                        # Smart AI Entry calculation
                        smart_entry_result = calculate_smart_entry_price(
                            signal=signal,
                            recent_data=recent_data,
                            predicted_price=predicted_price,
                            confidence=confidence,
                            symbol=symbol
                        )
                        
                        position_info = None
                        if signal != "HOLD":
                            # Import conversion rate function locally
                            try:
                                from ..utils.conversion import get_conversion_rate
                            except ImportError:
                                # Fallback conversion rate logic
                                def get_conversion_rate(quote_currency, api_source, api_key_1, api_key_2):
                                    return 1.0  # Simplified fallback
                            
                            quote_currency = "USD"
                            if "/" in symbol:
                                quote_currency = symbol.split('/')[1]
                            elif "USDT" in symbol.upper():
                                quote_currency = "USDT"
                            
                            conversion_rate = get_conversion_rate(quote_currency, api_source, api_key_1, api_key_2)

                            if conversion_rate is None or conversion_rate <= 0:
                                st.error(f"❌ Tidak dapat mengambil nilai tukar untuk {quote_currency}")
                                conversion_rate = 1.0  # Fallback
                            
                            ai_tp_price = None
                            if use_ai_tp:
                                try:
                                    supports, resistances = get_support_resistance(recent_data)
                                    atr_val = recent_data['ATR_14'].iloc[-1]
                                    # Get real-time price for TP validation
                                    real_time_price = ws_price if ws_price and ws_price > 0 else current_api_price
                                    ai_tp_price = calculate_ai_take_profit(signal, smart_entry_result['entry_price'], supports, resistances, atr_val, real_time_price)
                                except Exception as e:
                                    st.warning(f"⚠️ AI TP calculation failed: {e}. Using manual TP.")

                            position_info = calculate_position_info(
                                signal, symbol, smart_entry_result['entry_price'], 
                                live_sl, live_tp, account_balance, live_risk, 
                                conversion_rate, ai_tp_price
                            )

                        # Display results
                        display_smart_signal_results(
                            signal, confidence, smart_entry_result, position_info, 
                            symbol, ws_price, current_api_price
                        )

    except Exception as e:
        st.error(f"❌ Error in signal generation: {str(e)}")
        st.exception(e)