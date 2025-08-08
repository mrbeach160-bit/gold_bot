# formatters.py - Formatting utility functions
import re


def sanitize_filename(name):
    """Membersihkan string untuk digunakan sebagai nama file yang aman."""
    return re.sub(r'[^a-zA-Z0-9_-]', '', name.replace('/', '_').replace('\\', '_'))


def format_price(symbol, price):
    """
    Format price berdasarkan symbol
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