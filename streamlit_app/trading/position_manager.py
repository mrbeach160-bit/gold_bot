# position_manager.py - Position and trading calculations
import streamlit as st


def calculate_position_info(signal, symbol, entry_price, sl_pips, tp_pips, balance, risk_percent, conversion_rate_to_usd, take_profit_price=None, leverage=20):
    """
    IMPROVED: Menghitung informasi posisi dengan validasi yang lebih baik dan error handling
    """
    if signal == "HOLD":
        return None
        
    if conversion_rate_to_usd is None or conversion_rate_to_usd <= 0:
        st.error(f"❌ Conversion rate invalid: {conversion_rate_to_usd}")
        return None
    
    if entry_price <= 0:
        st.error(f"❌ Entry price invalid: {entry_price}")
        return None

    try:
        # Tentukan pip_size
        def get_pip_value_improved(symbol, price):
            symbol_upper = symbol.replace('/', '').upper()
            if symbol_upper == "BTCUSDT":
                return 0.1
            elif 'JPY' in symbol_upper:
                return 0.01
            elif 'XAU' in symbol_upper:
                return 0.01
            elif 'ETH' in symbol_upper:
                return 0.01
            else:
                return 0.0001

        pip_size = get_pip_value_improved(symbol, entry_price)
        sl_distance = sl_pips * pip_size

        # Stop loss price calculation dengan validasi
        if signal == "BUY":
            stop_loss_price = entry_price - sl_distance
            if stop_loss_price <= 0:
                st.error("❌ Stop loss price tidak valid (≤ 0)")
                return None
        else:
            stop_loss_price = entry_price + sl_distance

        # Take profit price calculation
        if take_profit_price:
            tp_price = take_profit_price
            # Validasi TP price masuk akal
            if signal == "BUY" and tp_price <= entry_price:
                st.warning("⚠️ TP price tidak valid untuk BUY signal, menggunakan default")
                tp_price = entry_price + (tp_pips * pip_size)
            elif signal == "SELL" and tp_price >= entry_price:
                st.warning("⚠️ TP price tidak valid untuk SELL signal, menggunakan default")
                tp_price = entry_price - (tp_pips * pip_size)
        else:
            tp_price = entry_price + (tp_pips * pip_size) if signal == "BUY" else entry_price - (tp_pips * pip_size)

        # Position size calculation dengan validasi
        risk_amount = balance * (risk_percent / 100)
        sl_distance_price = abs(entry_price - stop_loss_price)
        
        if sl_distance_price == 0:
            st.error("❌ Stop loss distance tidak boleh 0")
            return None

        # BTCUSDT perpetual futures calculation
        if symbol.replace('/', '').upper() == "BTCUSDT":
            position_size = risk_amount / sl_distance_price
            max_position_size = (balance * leverage) / entry_price
            if position_size > max_position_size:
                position_size = max_position_size
                st.warning(f"⚠️ Position size dikurangi karena leverage limit: {position_size:.4f}")
        else:
            # Forex/Spot calculation
            risk_amount_usd = risk_amount
            position_size = risk_amount_usd / (sl_distance_price * conversion_rate_to_usd)
            max_position_value_usd = balance * leverage
            position_value_usd = position_size * entry_price * conversion_rate_to_usd
            
            if position_value_usd > max_position_value_usd:
                position_size = max_position_value_usd / (entry_price * conversion_rate_to_usd)
                st.warning(f"⚠️ Position size dikurangi karena leverage limit: {position_size:.4f}")

        # Final validation
        if position_size <= 0:
            st.error("❌ Position size tidak valid (≤ 0)")
            return None

        # Minimum position size check
        min_position_value = 10  # $10 minimum
        if position_size * entry_price * conversion_rate_to_usd < min_position_value:
            st.warning(f"⚠️ Position size sangat kecil (< ${min_position_value})")

        return {
            'signal': signal,
            'entry_price': entry_price,
            'stop_loss': stop_loss_price,
            'take_profit': tp_price,
            'position_size': position_size,
            'risk_amount': risk_amount,
            'pip_size': pip_size,
            'conversion_rate': conversion_rate_to_usd
        }

    except Exception as e:
        st.error(f"❌ Error dalam perhitungan posisi: {str(e)}")
        return None


def calculate_ai_take_profit(signal, entry_price, supports, resistances, atr_value, current_price=None):
    """
    Calculate intelligent take profit based on S/R levels with real-time validation
    
    Args:
        signal: 'BUY' or 'SELL'
        entry_price: Entry price for the trade
        supports: Support levels DataFrame
        resistances: Resistance levels DataFrame  
        atr_value: Average True Range value
        current_price: Current real-time market price for validation
    """
    try:
        buffer_percent = 0.001  # 0.1% buffer untuk avoid false breakouts
        
        if signal == "BUY":
            # Cari resistance terdekat di atas entry + buffer
            valid_resistances = resistances[resistances > entry_price * (1 + buffer_percent)]
            if not valid_resistances.empty:
                nearest_resistance = valid_resistances.min()
                # Ensure reasonable distance (at least 1.5x ATR)
                min_tp_distance = entry_price + (1.5 * atr_value)
                take_profit = max(nearest_resistance, min_tp_distance)
            else:
                # Fallback: 2x ATR
                take_profit = entry_price + (2.0 * atr_value)
            
            # REAL-TIME VALIDATION: TP should be above current price
            if current_price and take_profit <= current_price * (1 + buffer_percent):
                st.warning(f"⚠️ BUY TP {take_profit:.4f} already passed current price {current_price:.4f}")
                return None  # Reject signal
                
        elif signal == "SELL":
            # Cari support terdekat di bawah entry - buffer
            valid_supports = supports[supports < entry_price * (1 - buffer_percent)]
            if not valid_supports.empty:
                nearest_support = valid_supports.max()
                # Ensure reasonable distance
                max_tp_distance = entry_price - (1.5 * atr_value)
                take_profit = min(nearest_support, max_tp_distance)
            else:
                take_profit = entry_price - (2.0 * atr_value)
            
            # REAL-TIME VALIDATION: TP should be below current price
            if current_price and take_profit >= current_price * (1 - buffer_percent):
                st.warning(f"⚠️ SELL TP {take_profit:.4f} already passed current price {current_price:.4f}")
                return None  # Reject signal
        
        return take_profit
    
    except Exception as e:
        st.warning(f"AI TP calculation failed: {e}")
        return None


def execute_trade_exit_realistic(current_candle, active_trade, slippage=0.001):
    """
    Realistic trade exit dengan proper priority dan slippage simulation
    """
    try:
        high_price = current_candle['high']
        low_price = current_candle['low']
        
        if active_trade['type'] == 'BUY':
            # Check SL first (more likely dalam volatile market)
            if low_price <= active_trade['sl']:
                # Simulate slippage - worse fill
                exit_price = max(active_trade['sl'] * (1 - slippage), low_price)
                return exit_price, 'SL'
            # Then check TP
            elif high_price >= active_trade['tp']:
                # Simulate slippage - worse fill
                exit_price = min(active_trade['tp'] * (1 - slippage), high_price) 
                return exit_price, 'TP'
        
        elif active_trade['type'] == 'SELL':
            # Check SL first
            if high_price >= active_trade['sl']:
                exit_price = min(active_trade['sl'] * (1 + slippage), high_price)
                return exit_price, 'SL'
            elif low_price <= active_trade['tp']:
                exit_price = max(active_trade['tp'] * (1 + slippage), low_price)
                return exit_price, 'TP'
        
        return None, None
        
    except Exception as e:
        st.warning(f"Error in trade exit calculation: {e}")
        return None, None


def calculate_realistic_pnl(entry_price, exit_price, position_size, trade_type, symbol):
    """
    Calculate PnL dengan realistic transaction costs
    """
    try:
        # Basic PnL calculation
        if trade_type == 'BUY':
            gross_pnl = (exit_price - entry_price) * position_size
        else:
            gross_pnl = (entry_price - exit_price) * position_size
        
        # Transaction costs
        spread_cost = entry_price * 0.0001 * position_size  # 1 pip spread average
        
        # Commission calculation based on symbol type
        if 'USDT' in symbol.upper():
            commission = position_size * entry_price * 0.001  # 0.1% for crypto
        elif any(fx in symbol.upper() for fx in ['EUR', 'GBP', 'JPY']):
            commission = max(2.0, position_size * entry_price * 0.00005)  # Forex commission
        else:
            commission = max(1.0, position_size * entry_price * 0.0001)  # Default 0.01%
        
        # Swap/overnight cost (simplified - assume intraday for now)
        swap_cost = 0
        
        total_costs = spread_cost + commission + swap_cost
        net_pnl = gross_pnl - total_costs
        
        return {
            'net_pnl': net_pnl,
            'gross_pnl': gross_pnl,
            'spread_cost': spread_cost,
            'commission': commission,
            'swap_cost': swap_cost,
            'total_costs': total_costs
        }
        
    except Exception as e:
        st.warning(f"Error calculating PnL: {e}")
        return {
            'net_pnl': 0, 'gross_pnl': 0, 'spread_cost': 0,
            'commission': 0, 'swap_cost': 0, 'total_costs': 0
        }