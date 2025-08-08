"""
Smart Entry Price Calculation module.

This module contains all logic for generating smart entry prices based on 
multi-factor analysis including support/resistance, RSI, MACD, ATR, and 
other technical indicators with risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from .config import is_feature_enabled

# Import indicators if available
if is_feature_enabled('UTILS_AVAILABLE'):
    from utils.indicators import get_support_resistance


def calculate_smart_entry_price(signal: int, recent_data: pd.DataFrame, predicted_price: float, 
                               confidence: float, symbol: str = "XAUUSD") -> Dict[str, Any]:
    """
    Menghitung smart entry price berdasarkan analisis multi-faktor:
    - S/R levels dengan ATR-based validation
    - RSI untuk konfirmasi momentum  
    - MACD untuk trend confirmation
    - ATR untuk volatility assessment
    - Fill probability estimation dengan realistic slippage
    
    Args:
        signal: Model prediction (1=BUY, -1=SELL, 0=HOLD)
        recent_data: DataFrame dengan historical data terbaru
        predicted_price: Predicted price dari model
        confidence: Model confidence (0-1)
        symbol: Trading symbol (default XAUUSD)
        
    Returns:
        Dict dengan smart_entry_price, reasons, risk_level, fill_probability, dll.
    """
    
    if recent_data.empty or len(recent_data) < 20:
        return {
            "smart_entry_price": predicted_price,
            "reasons": ["⚠️ Data insufficient for smart analysis"],
            "risk_level": "HIGH",
            "fill_probability": 0.3,
            "confidence_adjusted": confidence * 0.5,
            "entry_method": "FALLBACK"
        }
    
    current_price = recent_data['close'].iloc[-1]
    
    # Calculate indicators if not present
    df_analyzed = recent_data.copy()
    
    # Ensure we have RSI
    if 'rsi' not in df_analyzed.columns:
        if 'RSI_14' in df_analyzed.columns:
            df_analyzed['rsi'] = df_analyzed['RSI_14']
        else:
            # Simple RSI calculation
            delta = df_analyzed['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_analyzed['rsi'] = 100 - (100 / (1 + rs))
    
    # Ensure we have MACD
    if 'MACD_12_26_9' not in df_analyzed.columns:
        exp1 = df_analyzed['close'].ewm(span=12).mean()
        exp2 = df_analyzed['close'].ewm(span=26).mean()
        df_analyzed['MACD_12_26_9'] = exp1 - exp2
        df_analyzed['MACDh_12_26_9'] = df_analyzed['MACD_12_26_9'].ewm(span=9).mean()
    
    # Ensure we have ATR
    if 'ATR_14' not in df_analyzed.columns:
        high_low = df_analyzed['high'] - df_analyzed['low']
        high_close = np.abs(df_analyzed['high'] - df_analyzed['close'].shift())
        low_close = np.abs(df_analyzed['low'] - df_analyzed['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df_analyzed['ATR_14'] = true_range.rolling(14).mean()
    
    # Get support and resistance levels
    try:
        if is_feature_enabled('UTILS_AVAILABLE'):
            supports, resistances = get_support_resistance(df_analyzed, window=20)
        else:
            # Fallback S/R calculation
            supports, resistances = _calculate_simple_support_resistance(df_analyzed)
    except:
        supports, resistances = _calculate_simple_support_resistance(df_analyzed)
    
    # Current values
    current_rsi = df_analyzed['rsi'].iloc[-1] if not pd.isna(df_analyzed['rsi'].iloc[-1]) else 50
    current_macd = df_analyzed['MACD_12_26_9'].iloc[-1] if not pd.isna(df_analyzed['MACD_12_26_9'].iloc[-1]) else 0
    current_atr = df_analyzed['ATR_14'].iloc[-1] if not pd.isna(df_analyzed['ATR_14'].iloc[-1]) else current_price * 0.001
    
    if signal == 1:  # BUY
        return _calculate_buy_entry(current_price, predicted_price, supports, resistances,
                                  current_rsi, current_macd, current_atr, confidence, symbol)
    elif signal == -1:  # SELL
        return _calculate_sell_entry(current_price, predicted_price, supports, resistances,
                                   current_rsi, current_macd, current_atr, confidence, symbol)
    else:  # HOLD
        return {
            "smart_entry_price": current_price,
            "reasons": ["📊 Model suggests HOLD - market consolidation"],
            "risk_level": "MEDIUM",
            "fill_probability": 0.95,
            "confidence_adjusted": confidence,
            "entry_method": "HOLD"
        }


def _calculate_buy_entry(current_price: float, predicted_price: float, supports: List[float], 
                        resistances: List[float], current_rsi: float, current_macd: float,
                        current_atr: float, confidence: float, symbol: str) -> Dict[str, Any]:
    """Calculate smart entry price for BUY signal."""
    
    reasons = []
    risk_factors = []
    
    # Base entry price (conservative approach)
    base_entry = min(current_price, predicted_price)
    
    # ATR-based adjustments
    atr_adjustment = current_atr * 0.25  # 25% of ATR for entry optimization
    
    # Support level validation
    nearest_support = max([s for s in supports if s <= current_price * 1.02], default=current_price * 0.98)
    support_distance = current_price - nearest_support
    
    # Resistance level check
    nearest_resistance = min([r for r in resistances if r >= current_price], default=current_price * 1.05)
    resistance_distance = nearest_resistance - current_price
    
    # Smart entry calculation
    if support_distance <= current_atr * 0.5:
        # Very close to support - aggressive entry
        smart_entry = nearest_support + (current_atr * 0.1)
        reasons.append(f"💪 Strong support at {nearest_support:.5f} - aggressive entry")
    elif support_distance <= current_atr:
        # Near support - moderate entry
        smart_entry = current_price - (atr_adjustment * 0.5)
        reasons.append(f"🎯 Near support level - optimized entry")
    else:
        # Away from support - conservative entry
        smart_entry = current_price - atr_adjustment
        reasons.append(f"⚡ Standard entry with ATR adjustment")
    
    # RSI validation
    if current_rsi < 30:
        smart_entry = smart_entry - (current_atr * 0.1)
        reasons.append(f"📈 RSI oversold ({current_rsi:.1f}) - enhanced entry")
    elif current_rsi > 70:
        risk_factors.append(f"⚠️ RSI overbought ({current_rsi:.1f})")
        smart_entry = smart_entry + (current_atr * 0.1)
    else:
        reasons.append(f"✅ RSI neutral ({current_rsi:.1f})")
    
    # MACD validation
    if current_macd > 0:
        reasons.append("🚀 MACD bullish confirmation")
    else:
        risk_factors.append("📉 MACD bearish divergence")
    
    # Resistance check
    if resistance_distance < current_atr * 2:
        risk_factors.append(f"🔴 Near resistance at {nearest_resistance:.5f}")
        # Adjust entry to be more conservative
        smart_entry = smart_entry - (current_atr * 0.15)
    else:
        reasons.append("🟢 Clear path to resistance")
    
    # Calculate risk level and fill probability
    risk_level = "LOW"
    fill_probability = 0.85
    
    if len(risk_factors) >= 2:
        risk_level = "HIGH"
        fill_probability = 0.60
    elif len(risk_factors) == 1:
        risk_level = "MEDIUM"
        fill_probability = 0.75
    
    # Confidence adjustment
    confidence_adjusted = confidence * (0.9 if risk_level == "LOW" else 0.7 if risk_level == "MEDIUM" else 0.5)
    
    # Final validation
    smart_entry = max(smart_entry, current_price * 0.995)  # Don't go too far below current
    smart_entry = min(smart_entry, current_price * 1.001)  # Don't go above current for buy
    
    return {
        "smart_entry_price": smart_entry,
        "reasons": reasons + risk_factors,
        "risk_level": risk_level,
        "fill_probability": fill_probability,
        "confidence_adjusted": confidence_adjusted,
        "entry_method": "SMART_BUY",
        "support_level": nearest_support,
        "resistance_level": nearest_resistance,
        "atr_value": current_atr
    }


def _calculate_sell_entry(current_price: float, predicted_price: float, supports: List[float],
                         resistances: List[float], current_rsi: float, current_macd: float,
                         current_atr: float, confidence: float, symbol: str) -> Dict[str, Any]:
    """Calculate smart entry price for SELL signal."""
    
    reasons = []
    risk_factors = []
    
    # Base entry price (conservative approach)
    base_entry = max(current_price, predicted_price)
    
    # ATR-based adjustments
    atr_adjustment = current_atr * 0.25  # 25% of ATR for entry optimization
    
    # Resistance level validation
    nearest_resistance = min([r for r in resistances if r >= current_price * 0.98], default=current_price * 1.02)
    resistance_distance = nearest_resistance - current_price
    
    # Support level check
    nearest_support = max([s for s in supports if s <= current_price], default=current_price * 0.95)
    support_distance = current_price - nearest_support
    
    # Smart entry calculation
    if resistance_distance <= current_atr * 0.5:
        # Very close to resistance - aggressive entry
        smart_entry = nearest_resistance - (current_atr * 0.1)
        reasons.append(f"💪 Strong resistance at {nearest_resistance:.5f} - aggressive entry")
    elif resistance_distance <= current_atr:
        # Near resistance - moderate entry
        smart_entry = current_price + (atr_adjustment * 0.5)
        reasons.append(f"🎯 Near resistance level - optimized entry")
    else:
        # Away from resistance - conservative entry
        smart_entry = current_price + atr_adjustment
        reasons.append(f"⚡ Standard entry with ATR adjustment")
    
    # RSI validation
    if current_rsi > 70:
        smart_entry = smart_entry + (current_atr * 0.1)
        reasons.append(f"📉 RSI overbought ({current_rsi:.1f}) - enhanced entry")
    elif current_rsi < 30:
        risk_factors.append(f"⚠️ RSI oversold ({current_rsi:.1f})")
        smart_entry = smart_entry - (current_atr * 0.1)
    else:
        reasons.append(f"✅ RSI neutral ({current_rsi:.1f})")
    
    # MACD validation
    if current_macd < 0:
        reasons.append("🚀 MACD bearish confirmation")
    else:
        risk_factors.append("📈 MACD bullish divergence")
    
    # Support check
    if support_distance < current_atr * 2:
        risk_factors.append(f"🔴 Near support at {nearest_support:.5f}")
        # Adjust entry to be more conservative
        smart_entry = smart_entry + (current_atr * 0.15)
    else:
        reasons.append("🟢 Clear path to support")
    
    # Calculate risk level and fill probability
    risk_level = "LOW"
    fill_probability = 0.85
    
    if len(risk_factors) >= 2:
        risk_level = "HIGH"
        fill_probability = 0.60
    elif len(risk_factors) == 1:
        risk_level = "MEDIUM"
        fill_probability = 0.75
    
    # Confidence adjustment
    confidence_adjusted = confidence * (0.9 if risk_level == "LOW" else 0.7 if risk_level == "MEDIUM" else 0.5)
    
    # Final validation
    smart_entry = min(smart_entry, current_price * 1.005)  # Don't go too far above current
    smart_entry = max(smart_entry, current_price * 0.999)  # Don't go below current for sell
    
    return {
        "smart_entry_price": smart_entry,
        "reasons": reasons + risk_factors,
        "risk_level": risk_level,
        "fill_probability": fill_probability,
        "confidence_adjusted": confidence_adjusted,
        "entry_method": "SMART_SELL",
        "support_level": nearest_support,
        "resistance_level": nearest_resistance,
        "atr_value": current_atr
    }


def _calculate_simple_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[List[float], List[float]]:
    """Simple fallback support/resistance calculation."""
    
    if len(df) < window:
        current_price = df['close'].iloc[-1]
        return [current_price * 0.98, current_price * 0.96], [current_price * 1.02, current_price * 1.04]
    
    # Rolling min/max for support/resistance
    support_levels = df['low'].rolling(window=window).min().dropna().unique()
    resistance_levels = df['high'].rolling(window=window).max().dropna().unique()
    
    # Filter to reasonable levels
    current_price = df['close'].iloc[-1]
    supports = [s for s in support_levels if current_price * 0.90 <= s <= current_price * 1.05]
    resistances = [r for r in resistance_levels if current_price * 0.95 <= r <= current_price * 1.10]
    
    # Ensure we have at least some levels
    if not supports:
        supports = [current_price * 0.98, current_price * 0.96]
    if not resistances:
        resistances = [current_price * 1.02, current_price * 1.04]
    
    return sorted(supports), sorted(resistances)