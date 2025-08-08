# validators.py - Validation utility functions


def validate_trading_inputs(symbol, balance, risk_percent, sl_pips, tp_pips):
    """
    Validasi input trading untuk mencegah error dan memberikan warning
    """
    issues = []
    
    if balance < 100:
        issues.append("⚠️ Balance terlalu kecil (< $100). Minimal $500 direkomendasikan.")
    
    if risk_percent > 10:
        issues.append("🚨 Risk per trade > 10% sangat berbahaya! Recommended: 1-3%")
    elif risk_percent > 5:
        issues.append("⚠️ Risk per trade > 5% cukup tinggi. Pertimbangkan untuk menurunkan.")
    
    if sl_pips < 5:
        issues.append("⚠️ Stop Loss terlalu ketat (< 5 pips). Mungkin sering terkena noise.")
    
    if tp_pips < sl_pips:
        issues.append("⚠️ Take Profit lebih kecil dari Stop Loss. R:R ratio tidak optimal.")
    
    if tp_pips > sl_pips * 5:
        issues.append("⚠️ Take Profit terlalu jauh. Mungkin jarang tercapai.")
    
    return issues