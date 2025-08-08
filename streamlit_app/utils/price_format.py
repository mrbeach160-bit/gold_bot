"""
Price Formatting Utilities

Contains functions for formatting prices and financial values.
"""


def format_price(symbol, price):
    """
    Format price based on trading symbol
    
    Args:
        symbol: Trading symbol (e.g., 'XAUUSD', 'BTCUSDT')
        price: Price value to format
        
    Returns:
        str: Formatted price string
    """
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


def format_percentage(value, decimals=2):
    """
    Format value as percentage
    
    Args:
        value: Decimal value (e.g., 0.1234 for 12.34%)
        decimals: Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return str(value)


def format_currency(amount, currency="USD", decimals=2):
    """
    Format amount as currency
    
    Args:
        amount: Amount to format
        currency: Currency symbol (default "USD")
        decimals: Number of decimal places
        
    Returns:
        str: Formatted currency string
    """
    try:
        amount = float(amount)
        if currency == "USD":
            return f"${amount:,.{decimals}f}"
        else:
            return f"{amount:,.{decimals}f} {currency}"
    except (ValueError, TypeError):
        return str(amount)


def format_pips(pips, decimals=1):
    """
    Format pip value
    
    Args:
        pips: Pip value
        decimals: Number of decimal places
        
    Returns:
        str: Formatted pips string
    """
    try:
        return f"{float(pips):.{decimals}f} pips"
    except (ValueError, TypeError):
        return str(pips)


def format_volume(volume):
    """
    Format trading volume with appropriate units
    
    Args:
        volume: Volume value
        
    Returns:
        str: Formatted volume string
    """
    try:
        volume = float(volume)
        if volume >= 1_000_000:
            return f"{volume / 1_000_000:.2f}M"
        elif volume >= 1_000:
            return f"{volume / 1_000:.2f}K"
        else:
            return f"{volume:.2f}"
    except (ValueError, TypeError):
        return str(volume)