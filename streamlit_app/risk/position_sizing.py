"""
Position Sizing Module

Contains position sizing calculations and risk management utilities.
"""

# Import streamlit for temporary compatibility (should be refactored to use proper logging)
try:
    import streamlit as st
except ImportError:
    # Fallback for when streamlit is not available (e.g., in tests)
    class MockStreamlit:
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
    st = MockStreamlit()


def calculate_position_info(signal, symbol, entry_price, sl_pips, tp_pips, balance, risk_percent, conversion_rate_to_usd, take_profit_price=None, leverage=20):
    """
    Calculate position information with improved validation and error handling
    
    Args:
        signal: 'BUY' or 'SELL' signal
        symbol: Trading symbol (e.g., 'XAUUSD', 'BTCUSDT')
        entry_price: Entry price for the position
        sl_pips: Stop loss in pips
        tp_pips: Take profit in pips
        balance: Account balance
        risk_percent: Risk percentage (1-5%)
        conversion_rate_to_usd: Conversion rate to USD
        take_profit_price: Optional explicit take profit price
        leverage: Leverage multiplier (default 20)
        
    Returns:
        dict: Position information or None if invalid
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
        # Determine pip size
        pip_size = get_pip_value_improved(symbol, entry_price)
        sl_distance = sl_pips * pip_size

        # Stop loss price calculation with validation
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
            # Validate TP price makes sense
            if signal == "BUY" and tp_price <= entry_price:
                st.warning("⚠️ TP price tidak valid untuk BUY signal, menggunakan default")
                tp_price = entry_price + (tp_pips * pip_size)
            elif signal == "SELL" and tp_price >= entry_price:
                st.warning("⚠️ TP price tidak valid untuk SELL signal, menggunakan default")
                tp_price = entry_price - (tp_pips * pip_size)
        else:
            tp_price = entry_price + (tp_pips * pip_size) if signal == "BUY" else entry_price - (tp_pips * pip_size)

        # Position size calculation with validation
        risk_amount = balance * (risk_percent / 100)
        sl_distance_price = abs(entry_price - stop_loss_price)
        
        if sl_distance_price == 0:
            st.error("❌ Stop loss distance tidak boleh 0")
            return None

        # Calculate position size based on symbol type
        position_size = _calculate_position_size(symbol, entry_price, risk_amount, sl_distance_price, balance, leverage, conversion_rate_to_usd)
        
        if position_size is None:
            return None

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


def get_pip_value_improved(symbol, price):
    """
    Get pip value for different trading symbols
    
    Args:
        symbol: Trading symbol
        price: Current price
        
    Returns:
        float: Pip value
    """
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


def _calculate_position_size(symbol, entry_price, risk_amount, sl_distance_price, balance, leverage, conversion_rate_to_usd):
    """
    Internal function to calculate position size based on symbol type
    
    Args:
        symbol: Trading symbol
        entry_price: Entry price
        risk_amount: Amount to risk
        sl_distance_price: Stop loss distance in price
        balance: Account balance
        leverage: Leverage multiplier
        conversion_rate_to_usd: USD conversion rate
        
    Returns:
        float: Position size or None if invalid
    """
    try:
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
        
        return position_size
        
    except Exception as e:
        st.error(f"❌ Error in position size calculation: {str(e)}")
        return None


def calculate_risk_reward_ratio(entry_price, stop_loss_price, take_profit_price, signal):
    """
    Calculate risk-reward ratio for a trade
    
    Args:
        entry_price: Entry price
        stop_loss_price: Stop loss price
        take_profit_price: Take profit price
        signal: 'BUY' or 'SELL'
        
    Returns:
        float: Risk-reward ratio
    """
    if signal == "BUY":
        risk = entry_price - stop_loss_price
        reward = take_profit_price - entry_price
    else:  # SELL
        risk = stop_loss_price - entry_price
        reward = entry_price - take_profit_price
    
    if risk <= 0:
        return 0
    
    return reward / risk


def validate_position_parameters(entry_price, stop_loss_price, take_profit_price, signal):
    """
    Validate position parameters make sense
    
    Args:
        entry_price: Entry price
        stop_loss_price: Stop loss price
        take_profit_price: Take profit price
        signal: 'BUY' or 'SELL'
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if entry_price <= 0:
        return False, "Entry price must be positive"
    
    if stop_loss_price <= 0:
        return False, "Stop loss price must be positive"
    
    if take_profit_price <= 0:
        return False, "Take profit price must be positive"
    
    if signal == "BUY":
        if stop_loss_price >= entry_price:
            return False, "Stop loss must be below entry price for BUY signal"
        if take_profit_price <= entry_price:
            return False, "Take profit must be above entry price for BUY signal"
    elif signal == "SELL":
        if stop_loss_price <= entry_price:
            return False, "Stop loss must be above entry price for SELL signal"
        if take_profit_price >= entry_price:
            return False, "Take profit must be below entry price for SELL signal"
    
    return True, "Valid parameters"