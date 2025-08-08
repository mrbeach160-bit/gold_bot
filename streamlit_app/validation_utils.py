"""
Real-time Signal Validation Utilities
Enhanced validation pipeline for trading signals with real-time price integration
"""

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def validate_signal_realtime(signal, entry_price, take_profit, stop_loss, current_price, ws_price=None, confidence=0.0, symbol="XAUUSD"):
    """
    Comprehensive real-time signal validation pipeline
    
    Args:
        signal: 'BUY' or 'SELL'
        entry_price: Calculated entry price
        take_profit: Calculated take profit price
        stop_loss: Calculated stop loss price
        current_price: Current market price from API
        ws_price: Real-time WebSocket price (if available)
        confidence: AI confidence level (0-1)
        symbol: Trading symbol
        
    Returns:
        dict: {
            'is_valid': bool,
            'quality_score': float (0-100),
            'rejection_reason': str,
            'warnings': list,
            'real_time_price': float,
            'price_staleness_seconds': int
        }
    """
    try:
        # Use real-time price if available, fallback to current price
        real_time_price = ws_price if ws_price and ws_price > 0 else current_price
        
        # Initialize result
        result = {
            'is_valid': True,
            'quality_score': 100.0,
            'rejection_reason': '',
            'warnings': [],
            'real_time_price': real_time_price,
            'price_staleness_seconds': 0
        }
        
        # 1. Entry Price Validation - Tightened to 0.5%
        entry_deviation = abs(entry_price - real_time_price) / real_time_price
        max_entry_deviation = 0.005  # 0.5% maximum deviation
        
        if entry_deviation > max_entry_deviation:
            result['is_valid'] = False
            result['rejection_reason'] = f"Entry price deviation too high: {entry_deviation:.3%} > {max_entry_deviation:.1%}"
            return result
        
        # 2. Take Profit Real-time Validation
        tp_validation = validate_take_profit_realtime(signal, take_profit, real_time_price, entry_price)
        if not tp_validation['is_valid']:
            result['is_valid'] = False
            result['rejection_reason'] = tp_validation['reason']
            return result
        
        # 3. Fill Probability Assessment (minimum 70%)
        fill_probability = calculate_realistic_fill_probability(entry_price, real_time_price, signal)
        if fill_probability < 0.7:
            result['is_valid'] = False
            result['rejection_reason'] = f"Fill probability too low: {fill_probability:.1%} < 70%"
            return result
        
        # 4. Calculate Quality Score
        quality_score = get_signal_quality_score(
            entry_deviation, fill_probability, confidence, 
            tp_validation.get('tp_quality', 1.0)
        )
        result['quality_score'] = quality_score
        
        # 5. Add warnings for borderline cases
        if entry_deviation > 0.003:  # 0.3%
            result['warnings'].append(f"Entry price deviation moderate: {entry_deviation:.3%}")
        
        if fill_probability < 0.8:
            result['warnings'].append(f"Fill probability below optimal: {fill_probability:.1%}")
        
        if confidence < 0.7:
            result['warnings'].append(f"AI confidence below optimal: {confidence:.1%}")
        
        return result
        
    except Exception as e:
        st.error(f"❌ Signal validation error: {e}")
        return {
            'is_valid': False,
            'quality_score': 0.0,
            'rejection_reason': f'Validation error: {str(e)}',
            'warnings': [],
            'real_time_price': current_price,
            'price_staleness_seconds': 0
        }


def validate_take_profit_realtime(signal, take_profit, current_price, entry_price):
    """
    Validate take profit against current real-time price
    
    Args:
        signal: 'BUY' or 'SELL'
        take_profit: Calculated take profit price
        current_price: Current real-time market price
        entry_price: Entry price for the signal
        
    Returns:
        dict: {'is_valid': bool, 'reason': str, 'tp_quality': float}
    """
    try:
        buffer_percent = 0.001  # 0.1% buffer
        
        if signal == "BUY":
            # For BUY: TP should be above current price + buffer
            min_tp_required = current_price * (1 + buffer_percent)
            if take_profit <= min_tp_required:
                return {
                    'is_valid': False, 
                    'reason': f"BUY TP {take_profit:.4f} already passed current price {current_price:.4f}",
                    'tp_quality': 0.0
                }
            
            # Quality based on distance from current price
            tp_distance = (take_profit - current_price) / current_price
            tp_quality = min(1.0, tp_distance / 0.01)  # 1% distance = max quality
            
        elif signal == "SELL":
            # For SELL: TP should be below current price - buffer
            max_tp_required = current_price * (1 - buffer_percent)
            if take_profit >= max_tp_required:
                return {
                    'is_valid': False, 
                    'reason': f"SELL TP {take_profit:.4f} already passed current price {current_price:.4f}",
                    'tp_quality': 0.0
                }
            
            # Quality based on distance from current price
            tp_distance = (current_price - take_profit) / current_price
            tp_quality = min(1.0, tp_distance / 0.01)  # 1% distance = max quality
            
        else:
            return {'is_valid': False, 'reason': 'Invalid signal type', 'tp_quality': 0.0}
        
        return {'is_valid': True, 'reason': '', 'tp_quality': tp_quality}
        
    except Exception as e:
        return {'is_valid': False, 'reason': f'TP validation error: {str(e)}', 'tp_quality': 0.0}


def calculate_realistic_fill_probability(entry_price, current_price, signal):
    """
    Calculate realistic fill probability based on price distance
    
    Args:
        entry_price: Proposed entry price
        current_price: Current market price
        signal: 'BUY' or 'SELL'
        
    Returns:
        float: Fill probability (0.0 to 1.0)
    """
    try:
        price_distance = abs(entry_price - current_price) / current_price
        
        # More realistic fill probability calculation
        if price_distance <= 0.001:  # 0.1%
            return 0.95
        elif price_distance <= 0.002:  # 0.2%
            return 0.90
        elif price_distance <= 0.003:  # 0.3%
            return 0.80
        elif price_distance <= 0.005:  # 0.5%
            return 0.70
        else:
            # Exponential decay for larger distances
            return max(0.1, 0.7 * np.exp(-price_distance * 100))
        
    except Exception:
        return 0.5  # Default moderate probability


def get_signal_quality_score(entry_deviation, fill_probability, confidence, tp_quality):
    """
    Calculate signal quality score (0-100)
    
    Args:
        entry_deviation: Entry price deviation from current price (decimal)
        fill_probability: Calculated fill probability (0-1)
        confidence: AI confidence level (0-1)
        tp_quality: Take profit quality score (0-1)
        
    Returns:
        float: Quality score (0-100)
    """
    try:
        # Weight factors
        entry_weight = 0.3
        fill_weight = 0.3
        confidence_weight = 0.2
        tp_weight = 0.2
        
        # Score components (0-100)
        entry_score = max(0, 100 * (1 - entry_deviation / 0.005))  # 0.5% = 0 score
        fill_score = fill_probability * 100
        confidence_score = confidence * 100
        tp_score = tp_quality * 100
        
        # Weighted average
        quality_score = (
            entry_score * entry_weight +
            fill_score * fill_weight +
            confidence_score * confidence_weight +
            tp_score * tp_weight
        )
        
        return round(quality_score, 1)
        
    except Exception:
        return 50.0  # Default moderate score


def is_signal_expired(signal_timestamp, expiry_seconds=30):
    """
    Check if signal has expired based on timestamp
    
    Args:
        signal_timestamp: When signal was generated (datetime)
        expiry_seconds: Signal expiry time in seconds (default: 30)
        
    Returns:
        dict: {'is_expired': bool, 'age_seconds': int, 'remaining_seconds': int}
    """
    try:
        current_time = datetime.now()
        
        if isinstance(signal_timestamp, str):
            # Try to parse string timestamp
            signal_timestamp = datetime.fromisoformat(signal_timestamp.replace('Z', '+00:00'))
        
        age_seconds = (current_time - signal_timestamp).total_seconds()
        remaining_seconds = max(0, expiry_seconds - age_seconds)
        is_expired = age_seconds > expiry_seconds
        
        return {
            'is_expired': is_expired,
            'age_seconds': int(age_seconds),
            'remaining_seconds': int(remaining_seconds)
        }
        
    except Exception as e:
        st.warning(f"Signal expiry check failed: {e}")
        return {
            'is_expired': False,  # Don't expire on error
            'age_seconds': 0,
            'remaining_seconds': 30
        }


def get_price_staleness_indicator(api_price, ws_price, ws_timestamp=None):
    """
    Get price staleness information for UI display
    
    Args:
        api_price: Price from API data
        ws_price: Real-time WebSocket price
        ws_timestamp: WebSocket data timestamp
        
    Returns:
        dict: {'staleness_level': str, 'message': str, 'color': str}
    """
    try:
        if not ws_price or ws_price <= 0:
            return {
                'staleness_level': 'NO_REALTIME',
                'message': '🔴 No real-time data',
                'color': 'red'
            }
        
        price_diff_pct = abs(ws_price - api_price) / api_price
        
        if price_diff_pct <= 0.001:  # 0.1%
            return {
                'staleness_level': 'FRESH',
                'message': '🟢 Data is fresh',
                'color': 'green'
            }
        elif price_diff_pct <= 0.005:  # 0.5%
            return {
                'staleness_level': 'MODERATE',
                'message': f'🟡 Price deviation: {price_diff_pct:.2%}',
                'color': 'orange'
            }
        else:
            return {
                'staleness_level': 'STALE',
                'message': f'🔴 High price deviation: {price_diff_pct:.2%}',
                'color': 'red'
            }
            
    except Exception:
        return {
            'staleness_level': 'ERROR',
            'message': '⚠️ Price staleness check failed',
            'color': 'gray'
        }


def format_quality_indicator(quality_score):
    """
    Format quality score for UI display
    
    Args:
        quality_score: Quality score (0-100)
        
    Returns:
        dict: {'emoji': str, 'color': str, 'text': str}
    """
    if quality_score >= 80:
        return {
            'emoji': '🟢',
            'color': 'green',
            'text': f'Excellent ({quality_score:.0f}%)'
        }
    elif quality_score >= 60:
        return {
            'emoji': '🟡',
            'color': 'orange', 
            'text': f'Good ({quality_score:.0f}%)'
        }
    elif quality_score >= 40:
        return {
            'emoji': '🟠',
            'color': 'orange',
            'text': f'Fair ({quality_score:.0f}%)'
        }
    else:
        return {
            'emoji': '🔴',
            'color': 'red',
            'text': f'Poor ({quality_score:.0f}%)'
        }