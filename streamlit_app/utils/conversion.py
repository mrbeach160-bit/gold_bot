# conversion.py - Currency conversion utilities
import streamlit as st


def get_conversion_rate(quote_currency, api_source, api_key_1, api_key_2):
    """Get conversion rate for quote currency to USD"""
    try:
        # For USD or USDT, no conversion needed
        if quote_currency.upper() in ['USD', 'USDT']:
            return 1.0
        
        # For common currencies, use approximate rates (simplified)
        # In production, this should use real-time exchange rates
        conversion_rates = {
            'EUR': 1.1,   # EUR/USD approximate
            'GBP': 1.3,   # GBP/USD approximate  
            'JPY': 0.007, # USD/JPY approximate (1/143)
            'CHF': 1.1,   # CHF/USD approximate
            'CAD': 0.74,  # CAD/USD approximate
            'AUD': 0.67,  # AUD/USD approximate
        }
        
        rate = conversion_rates.get(quote_currency.upper())
        if rate:
            return rate
        
        # Try to get real conversion rate using available API
        if api_source == "Twelve Data" and api_key_1:
            return _get_twelve_data_conversion_rate(quote_currency, api_key_1)
        
        # Fallback: assume 1:1 ratio and warn user
        st.warning(f"⚠️ Cannot get conversion rate for {quote_currency}, using 1:1 ratio")
        return 1.0
        
    except Exception as e:
        st.warning(f"⚠️ Conversion rate error: {e}, using fallback rate")
        return 1.0


def _get_twelve_data_conversion_rate(quote_currency, api_key):
    """Get conversion rate from Twelve Data API"""
    try:
        from utils.data import get_gold_data
        
        # Create conversion symbol (e.g., EUR/USD)
        conversion_symbol = f"{quote_currency}/USD"
        
        # Get recent data for conversion
        rate_data = get_gold_data(api_key, interval='1min', symbol=conversion_symbol, outputsize=1)
        
        if rate_data is not None and not rate_data.empty:
            return rate_data['close'].iloc[-1]
        else:
            return 1.0
            
    except Exception as e:
        st.warning(f"⚠️ Failed to get {quote_currency} conversion rate: {e}")
        return 1.0