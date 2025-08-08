"""
Trading Panel Component
Handles trading signals, analysis, and position calculations
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Safe imports - these would need to be properly imported from the main app
try:
    # These functions would need to be imported from utils or defined here
    # For now, we'll create placeholder functions
    def get_support_resistance(data):
        """Placeholder for support/resistance calculation"""
        # Simple implementation - would be replaced with actual logic
        high_points = data['high'].rolling(window=20).max()
        low_points = data['low'].rolling(window=20).min()
        return low_points.dropna(), high_points.dropna()
    
    # Placeholder validation
    VALIDATION_UTILS_AVAILABLE = False
    
except ImportError:
    VALIDATION_UTILS_AVAILABLE = False
    
    def get_support_resistance(data):
        """Simple fallback implementation"""
        return pd.Series([]), pd.Series([])


class TradingPanel:
    """Trading Panel Component for signal analysis and position management"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def format_price(symbol, price):
        """Format price berdasarkan symbol"""
        try:
            price = float(price)
            if 'XAU' in symbol.upper() or 'GOLD' in symbol.upper():
                return f"${price:,.2f}"
            elif 'BTC' in symbol.upper():
                return f"${price:,.1f}"
            elif 'ETH' in symbol.upper():
                return f"${price:,.2f}"
            elif 'JPY' in symbol.upper():
                return f"{price:.3f}"
            else:
                return f"{price:.5f}"
        except (ValueError, TypeError):
            return str(price)
    
    def calculate_smart_entry_price(self, signal, recent_data, predicted_price, confidence, symbol="XAUUSD"):
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
                entry_price, reasons, risk = self._calculate_buy_entry(
                    current_price, predicted_price, supports, resistances,
                    rsi, atr, macd, macd_signal, confidence
                )
            elif signal == "SELL":
                entry_price, reasons, risk = self._calculate_sell_entry(
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
            max_deviation = 0.005  # 0.5% max deviation from current price
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

    def _calculate_buy_entry(self, current_price, predicted_price, supports, resistances, 
                            rsi, atr, macd, macd_signal, confidence):
        """Calculate optimal BUY entry price"""
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
            rsi_adjustment = atr * 0.15  # Wait for pullback
            strategy_reasons.append(f"RSI overbought - wait for pullback ({rsi:.1f})")
            risk_level = "HIGH"
        
        # 3. MACD momentum check
        macd_adjustment = 0
        if macd > macd_signal:  # Bullish momentum
            macd_adjustment = -atr * 0.05  # Slight entry advantage
            strategy_reasons.append("MACD bullish momentum")
        else:
            macd_adjustment = atr * 0.1  # Wait for momentum
            strategy_reasons.append("MACD bearish - higher entry needed")
            risk_level = "HIGH"
        
        # 4. AI confidence adjustment
        confidence_adjustment = (1 - confidence) * atr * 0.3  # Higher confidence = better entry
        if confidence > 0.8:
            strategy_reasons.append(f"High AI confidence ({confidence:.1%})")
        else:
            strategy_reasons.append(f"Medium AI confidence ({confidence:.1%})")
            risk_level = "MEDIUM"
        
        # Combine all adjustments
        final_entry = base_entry + rsi_adjustment + macd_adjustment + confidence_adjustment
        
        return final_entry, strategy_reasons, risk_level

    def _calculate_sell_entry(self, current_price, predicted_price, supports, resistances,
                             rsi, atr, macd, macd_signal, confidence):
        """Calculate optimal SELL entry price"""
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
            rsi_adjustment = -atr * 0.15  # Wait for bounce
            strategy_reasons.append(f"RSI oversold - wait for bounce ({rsi:.1f})")
            risk_level = "HIGH"
        
        # 3. MACD momentum check
        macd_adjustment = 0
        if macd < macd_signal:  # Bearish momentum
            macd_adjustment = atr * 0.05  # Slight entry advantage
            strategy_reasons.append("MACD bearish momentum")
        else:
            macd_adjustment = -atr * 0.1  # Wait for momentum
            strategy_reasons.append("MACD bullish - lower entry needed")
            risk_level = "HIGH"
        
        # 4. AI confidence adjustment
        confidence_adjustment = -(1 - confidence) * atr * 0.3  # Higher confidence = better entry
        if confidence > 0.8:
            strategy_reasons.append(f"High AI confidence ({confidence:.1%})")
        else:
            strategy_reasons.append(f"Medium AI confidence ({confidence:.1%})")
            risk_level = "MEDIUM"
        
        # Combine all adjustments
        final_entry = base_entry + rsi_adjustment + macd_adjustment + confidence_adjustment
        
        return final_entry, strategy_reasons, risk_level

    def validate_trading_inputs(self, symbol, balance, risk_percent, sl_pips, tp_pips):
        """Validate trading input parameters"""
        errors = []
        
        if balance <= 0:
            errors.append("Balance must be positive")
        
        if risk_percent <= 0 or risk_percent > 10:
            errors.append("Risk percent must be between 0.1% and 10%")
        
        if sl_pips <= 0:
            errors.append("Stop Loss must be positive")
        
        if tp_pips <= 0:
            errors.append("Take Profit must be positive")
        
        if tp_pips <= sl_pips:
            errors.append("Take Profit should be larger than Stop Loss for better RRR")
        
        return len(errors) == 0, errors

    def calculate_position_info(self, signal, symbol, entry_price, sl_pips, tp_pips, 
                               balance, risk_percent, conversion_rate_to_usd, 
                               take_profit_price=None, leverage=20):
        """Calculate position sizing and risk parameters"""
        try:
            def get_pip_value_improved(symbol, price):
                """Enhanced pip value calculation"""
                if 'XAU' in symbol.upper() or 'GOLD' in symbol.upper():
                    return 0.10  # Gold: $0.10 per pip for 1 oz
                elif 'JPY' in symbol.upper():
                    return 0.01 / price  # JPY pairs
                elif 'BTC' in symbol.upper():
                    return price * 0.0001  # Bitcoin: 0.01% = 1 pip
                elif 'ETH' in symbol.upper():
                    return price * 0.0001  # Ethereum
                else:
                    return 0.0001 * conversion_rate_to_usd  # Standard forex
            
            pip_value = get_pip_value_improved(symbol, entry_price)
            
            # Calculate position size based on risk
            risk_amount = balance * (risk_percent / 100)
            position_size = risk_amount / (sl_pips * pip_value)
            
            # Apply leverage constraint
            max_position = balance * leverage
            if position_size * entry_price > max_position:
                position_size = max_position / entry_price
                st.warning(f"⚠️ Position size limited by leverage ({leverage}:1)")
            
            # Calculate levels
            if signal == "BUY":
                stop_loss = entry_price - (sl_pips * pip_value)
                if take_profit_price is not None:
                    take_profit = take_profit_price
                    if take_profit <= entry_price:
                        st.warning("⚠️ TP price tidak valid untuk BUY signal, menggunakan default")
                        take_profit = entry_price + (tp_pips * pip_value)
                else:
                    take_profit = entry_price + (tp_pips * pip_value)
            else:  # SELL
                stop_loss = entry_price + (sl_pips * pip_value)
                if take_profit_price is not None:
                    take_profit = take_profit_price
                    if take_profit >= entry_price:
                        st.warning("⚠️ TP price tidak valid untuk SELL signal, menggunakan default")
                        take_profit = entry_price - (tp_pips * pip_value)
                else:
                    take_profit = entry_price - (tp_pips * pip_value)
            
            # Calculate potential P&L
            if signal == "BUY":
                potential_profit = (take_profit - entry_price) * position_size
                potential_loss = (entry_price - stop_loss) * position_size
            else:  # SELL
                potential_profit = (entry_price - take_profit) * position_size
                potential_loss = (stop_loss - entry_price) * position_size
            
            # Risk-Reward Ratio
            rrr = potential_profit / potential_loss if potential_loss > 0 else 0
            
            return {
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'potential_profit': potential_profit,
                'potential_loss': potential_loss,
                'risk_reward_ratio': rrr,
                'pip_value': pip_value,
                'risk_amount': risk_amount
            }
            
        except Exception as e:
            st.error(f"❌ Error calculating position: {e}")
            return None

    def display_smart_signal_results(self, signal, confidence, smart_entry_result, 
                                    position_info, symbol, ws_price=None, current_price=None):
        """Enhanced UI display dengan Smart AI strategy reasoning and real-time validation"""
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
        
        st.markdown(f"""
        ## {signal_color} **{signal} SIGNAL**
        **Confidence:** {confidence:.1%} `{confidence_bar}`
        """)
        
        # Smart Entry Information
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
                self.format_price(symbol, entry_price),
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
                help="Overall signal risk assessment"
            )
        
        # Real-time price comparison
        if ws_price and current_price:
            rt_cols = st.columns(2)
            rt_cols[0].metric("API Price", self.format_price(symbol, current_price))
            rt_cols[1].metric("WebSocket Price", self.format_price(symbol, ws_price))
        
        # Strategy reasoning
        with st.expander("🧠 AI Strategy Analysis", expanded=True):
            st.markdown("**Entry Strategy Factors:**")
            for reason in smart_entry_result['strategy_reasons']:
                st.markdown(f"• {reason}")
        
        # Position information
        if position_info:
            with st.expander("💰 Position Details", expanded=True):
                detail_cols = st.columns(3)
                detail_cols[0].metric("Position Size", f"{position_info['position_size']:.4f}")
                detail_cols[1].metric("Stop Loss", self.format_price(symbol, position_info['stop_loss']))
                detail_cols[2].metric("Take Profit", self.format_price(symbol, position_info['take_profit']))
                
                st.markdown(f"""
                **Risk Management:**
                - Risk Amount: ${position_info['risk_amount']:.2f}
                - Potential Profit: ${position_info['potential_profit']:.2f}
                - Potential Loss: ${position_info['potential_loss']:.2f}
                - Risk-Reward Ratio: {position_info['risk_reward_ratio']:.2f}
                """)

    def render_trading_controls(self, symbol, account_balance):
        """Render trading control panel"""
        with st.expander("Pengaturan Risk Management", expanded=True):
            risk_cols = st.columns(3)
            sl_pips = risk_cols[0].number_input("Stop Loss (pips)", min_value=5, value=20, key="sl_pips")
            use_ai_tp = st.checkbox("Gunakan Take Profit dari AI (berdasarkan Support/Resistance)", value=True)
            tp_pips = risk_cols[1].number_input("Take Profit (pips)", min_value=5, value=40, key="tp_pips", disabled=use_ai_tp)
            risk_percent = risk_cols[2].slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1, key="risk_percent")
            
            return {
                'sl_pips': sl_pips,
                'tp_pips': tp_pips,
                'risk_percent': risk_percent,
                'use_ai_tp': use_ai_tp
            }

    def render_signal_buttons(self):
        """Render signal generation buttons"""
        signal_cols = st.columns(3)
        generate_signal = signal_cols[0].button("📊 Generate Signal", use_container_width=True, type="primary")
        auto_refresh = signal_cols[1].checkbox("🔄 Auto Refresh", key="auto_refresh")
        refresh_rate = signal_cols[2].selectbox("Refresh Rate", ["30s", "1m", "2m", "5m"], index=1)
        
        return {
            'generate_signal': generate_signal,
            'auto_refresh': auto_refresh,
            'refresh_rate': refresh_rate
        }