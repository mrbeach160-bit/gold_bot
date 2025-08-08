"""
Market Microstructure Analysis

Advanced market microstructure analysis for optimal trade execution,
order flow analysis, and liquidity assessment in real-time trading.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import statistics

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class TickData:
    """Individual tick data structure"""
    timestamp: datetime
    bid: float
    ask: float
    last_price: float
    volume: int
    bid_size: int
    ask_size: int
    trade_direction: Optional[str] = None  # 'BUY', 'SELL', or None

@dataclass
class OrderBookLevel:
    """Order book level data"""
    price: float
    size: int
    orders: int

@dataclass
class OrderBook:
    """Full order book snapshot"""
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    
    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0
    
    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0
    
    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid if self.bids and self.asks else 0.0
    
    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2 if self.bids and self.asks else 0.0

class OrderFlowAnalyzer:
    """Analyze order flow and market pressure"""
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.tick_history = []
        self.trade_history = []
        self.volume_history = []
        
    def add_tick(self, tick: TickData):
        """Add new tick data"""
        self.tick_history.append(tick)
        
        # Keep only recent history
        if len(self.tick_history) > self.lookback_periods:
            self.tick_history = self.tick_history[-self.lookback_periods:]
        
        # Classify trade direction if not provided
        if tick.trade_direction is None:
            tick.trade_direction = self._classify_trade_direction(tick)
    
    def _classify_trade_direction(self, tick: TickData) -> str:
        """Classify trade direction using tick rule"""
        if len(self.tick_history) < 2:
            return 'UNKNOWN'
        
        prev_tick = self.tick_history[-2]
        
        # Compare with previous price
        if tick.last_price > prev_tick.last_price:
            return 'BUY'
        elif tick.last_price < prev_tick.last_price:
            return 'SELL'
        else:
            # Use bid/ask comparison for zero tick
            mid_price = (tick.bid + tick.ask) / 2
            if tick.last_price >= mid_price:
                return 'BUY'
            else:
                return 'SELL'
    
    def calculate_bid_ask_pressure(self) -> Dict[str, float]:
        """Calculate bid/ask pressure indicators"""
        try:
            if len(self.tick_history) < 10:
                return {'bid_pressure': 0.5, 'ask_pressure': 0.5, 'net_pressure': 0.0}
            
            recent_ticks = self.tick_history[-20:]  # Last 20 ticks
            
            # Calculate pressure based on size changes
            bid_pressure_sum = 0
            ask_pressure_sum = 0
            
            for i in range(1, len(recent_ticks)):
                curr_tick = recent_ticks[i]
                prev_tick = recent_ticks[i-1]
                
                # Bid pressure (positive when bid size increases)
                bid_change = curr_tick.bid_size - prev_tick.bid_size
                if curr_tick.bid >= prev_tick.bid:  # Same or better bid
                    bid_pressure_sum += max(0, bid_change)
                
                # Ask pressure (positive when ask size increases)
                ask_change = curr_tick.ask_size - prev_tick.ask_size
                if curr_tick.ask <= prev_tick.ask:  # Same or better ask
                    ask_pressure_sum += max(0, ask_change)
            
            total_pressure = bid_pressure_sum + ask_pressure_sum
            
            if total_pressure > 0:
                bid_pressure = bid_pressure_sum / total_pressure
                ask_pressure = ask_pressure_sum / total_pressure
            else:
                bid_pressure = ask_pressure = 0.5
            
            net_pressure = bid_pressure - ask_pressure
            
            return {
                'bid_pressure': bid_pressure,
                'ask_pressure': ask_pressure,
                'net_pressure': net_pressure,
                'total_pressure': total_pressure
            }
            
        except Exception as e:
            logger.error(f"Error calculating bid/ask pressure: {e}")
            return {'bid_pressure': 0.5, 'ask_pressure': 0.5, 'net_pressure': 0.0}
    
    def calculate_volume_imbalance(self) -> Dict[str, float]:
        """Calculate volume imbalance between buys and sells"""
        try:
            if len(self.tick_history) < 10:
                return {'buy_volume': 0, 'sell_volume': 0, 'imbalance': 0.0}
            
            recent_ticks = self.tick_history[-50:]  # Last 50 ticks
            
            buy_volume = sum(
                tick.volume for tick in recent_ticks 
                if tick.trade_direction == 'BUY'
            )
            
            sell_volume = sum(
                tick.volume for tick in recent_ticks 
                if tick.trade_direction == 'SELL'
            )
            
            total_volume = buy_volume + sell_volume
            
            if total_volume > 0:
                imbalance = (buy_volume - sell_volume) / total_volume
            else:
                imbalance = 0.0
            
            return {
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'total_volume': total_volume,
                'imbalance': imbalance,
                'buy_ratio': buy_volume / total_volume if total_volume > 0 else 0.5
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume imbalance: {e}")
            return {'buy_volume': 0, 'sell_volume': 0, 'imbalance': 0.0}

class LiquidityAnalyzer:
    """Analyze market liquidity and depth"""
    
    def __init__(self):
        self.orderbook_history = []
        self.spread_history = []
        self.depth_history = []
        
    def add_orderbook(self, orderbook: OrderBook):
        """Add order book snapshot"""
        self.orderbook_history.append(orderbook)
        
        # Keep limited history
        if len(self.orderbook_history) > 100:
            self.orderbook_history = self.orderbook_history[-100:]
        
        # Update derived metrics
        self.spread_history.append(orderbook.spread)
        if len(self.spread_history) > 100:
            self.spread_history = self.spread_history[-100:]
    
    def estimate_liquidity_depth(self, price_levels: int = 5) -> Dict[str, Any]:
        """Estimate market depth and liquidity"""
        try:
            if not self.orderbook_history:
                return {'bid_depth': 0, 'ask_depth': 0, 'total_depth': 0}
            
            latest_book = self.orderbook_history[-1]
            
            # Calculate depth within specified levels
            bid_depth = sum(
                level.size for level in latest_book.bids[:price_levels]
            )
            
            ask_depth = sum(
                level.size for level in latest_book.asks[:price_levels]
            )
            
            total_depth = bid_depth + ask_depth
            
            # Calculate depth imbalance
            depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            # Calculate weighted average prices
            bid_weighted_price = self._calculate_weighted_price(latest_book.bids[:price_levels])
            ask_weighted_price = self._calculate_weighted_price(latest_book.asks[:price_levels])
            
            return {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': total_depth,
                'depth_imbalance': depth_imbalance,
                'bid_weighted_price': bid_weighted_price,
                'ask_weighted_price': ask_weighted_price,
                'effective_spread': ask_weighted_price - bid_weighted_price
            }
            
        except Exception as e:
            logger.error(f"Error estimating liquidity depth: {e}")
            return {'bid_depth': 0, 'ask_depth': 0, 'total_depth': 0}
    
    def _calculate_weighted_price(self, levels: List[OrderBookLevel]) -> float:
        """Calculate volume-weighted price for order book levels"""
        if not levels:
            return 0.0
        
        total_volume = sum(level.size for level in levels)
        if total_volume == 0:
            return levels[0].price
        
        weighted_sum = sum(level.price * level.size for level in levels)
        return weighted_sum / total_volume
    
    def calculate_spread_metrics(self) -> Dict[str, float]:
        """Calculate spread-related metrics"""
        try:
            if len(self.spread_history) < 10:
                return {'avg_spread': 0, 'spread_volatility': 0, 'current_spread': 0}
            
            recent_spreads = self.spread_history[-20:]
            
            avg_spread = statistics.mean(recent_spreads)
            spread_volatility = statistics.stdev(recent_spreads) if len(recent_spreads) > 1 else 0
            current_spread = recent_spreads[-1]
            
            # Spread percentiles
            spread_p25 = np.percentile(recent_spreads, 25)
            spread_p75 = np.percentile(recent_spreads, 75)
            
            return {
                'current_spread': current_spread,
                'avg_spread': avg_spread,
                'spread_volatility': spread_volatility,
                'spread_p25': spread_p25,
                'spread_p75': spread_p75,
                'spread_zscore': (current_spread - avg_spread) / spread_volatility if spread_volatility > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating spread metrics: {e}")
            return {'avg_spread': 0, 'spread_volatility': 0, 'current_spread': 0}

class MarketImpactEstimator:
    """Estimate market impact of trades"""
    
    def __init__(self):
        self.impact_history = []
        self.trade_size_history = []
        
    def estimate_impact(self, 
                       trade_size: float, 
                       orderbook: OrderBook, 
                       direction: str) -> Dict[str, float]:
        """Estimate market impact of a trade"""
        try:
            if direction.upper() == 'BUY':
                levels = orderbook.asks
                reference_price = orderbook.best_ask
            else:
                levels = orderbook.bids
                reference_price = orderbook.best_bid
            
            if not levels or reference_price == 0:
                return {'estimated_impact': 0, 'avg_execution_price': reference_price}
            
            # Walk through order book to estimate execution
            remaining_size = trade_size
            total_cost = 0
            levels_consumed = 0
            
            for level in levels:
                if remaining_size <= 0:
                    break
                
                size_from_level = min(remaining_size, level.size)
                total_cost += size_from_level * level.price
                remaining_size -= size_from_level
                levels_consumed += 1
            
            if trade_size > 0:
                avg_execution_price = total_cost / (trade_size - remaining_size)
                if direction.upper() == 'BUY':
                    impact = (avg_execution_price - reference_price) / reference_price
                else:
                    impact = (reference_price - avg_execution_price) / reference_price
            else:
                avg_execution_price = reference_price
                impact = 0
            
            # Estimate additional impact from remaining size
            if remaining_size > 0:
                # Assume higher impact for size beyond available liquidity
                additional_impact = (remaining_size / trade_size) * 0.01  # 1% impact assumption
                impact += additional_impact
            
            return {
                'estimated_impact': impact,
                'avg_execution_price': avg_execution_price,
                'levels_consumed': levels_consumed,
                'unfilled_size': remaining_size,
                'fill_ratio': (trade_size - remaining_size) / trade_size if trade_size > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            return {'estimated_impact': 0, 'avg_execution_price': 0}
    
    def record_actual_impact(self, 
                           trade_size: float, 
                           pre_trade_price: float, 
                           post_trade_price: float):
        """Record actual market impact for calibration"""
        try:
            if pre_trade_price > 0:
                actual_impact = abs(post_trade_price - pre_trade_price) / pre_trade_price
                
                self.impact_history.append({
                    'trade_size': trade_size,
                    'impact': actual_impact,
                    'timestamp': datetime.now()
                })
                
                # Keep limited history
                if len(self.impact_history) > 1000:
                    self.impact_history = self.impact_history[-1000:]
                    
        except Exception as e:
            logger.error(f"Error recording actual impact: {e}")

class MicrostructureAnalyzer:
    """Main microstructure analysis system"""
    
    def __init__(self):
        self.order_flow_analyzer = OrderFlowAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.impact_estimator = MarketImpactEstimator()
        
        # Price movement analysis
        self.price_history = []
        self.volatility_history = []
        
        logger.info("Microstructure Analyzer initialized")
    
    def analyze_order_flow(self, tick_data: TickData) -> Dict[str, Any]:
        """Comprehensive order flow analysis"""
        try:
            # Add tick to analyzer
            self.order_flow_analyzer.add_tick(tick_data)
            
            # Calculate various metrics
            pressure_metrics = self.order_flow_analyzer.calculate_bid_ask_pressure()
            volume_metrics = self.order_flow_analyzer.calculate_volume_imbalance()
            
            # Price movement analysis
            self._update_price_analysis(tick_data.last_price)
            
            analysis = {
                'timestamp': tick_data.timestamp,
                'bid_ask_pressure': pressure_metrics['net_pressure'],
                'volume_imbalance': volume_metrics['imbalance'],
                'price_acceleration': self._calculate_price_acceleration(),
                'trade_intensity': self._calculate_trade_intensity(),
                'pressure_metrics': pressure_metrics,
                'volume_metrics': volume_metrics
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing order flow: {e}")
            return {'error': str(e)}
    
    def analyze_liquidity(self, orderbook: OrderBook) -> Dict[str, Any]:
        """Analyze market liquidity"""
        try:
            # Add orderbook to analyzer
            self.liquidity_analyzer.add_orderbook(orderbook)
            
            # Calculate liquidity metrics
            depth_metrics = self.liquidity_analyzer.estimate_liquidity_depth()
            spread_metrics = self.liquidity_analyzer.calculate_spread_metrics()
            
            # Overall liquidity score
            liquidity_score = self._calculate_liquidity_score(depth_metrics, spread_metrics)
            
            analysis = {
                'timestamp': orderbook.timestamp,
                'liquidity_depth': depth_metrics['total_depth'],
                'spread': orderbook.spread,
                'liquidity_score': liquidity_score,
                'depth_metrics': depth_metrics,
                'spread_metrics': spread_metrics
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity: {e}")
            return {'error': str(e)}
    
    def optimize_entry_timing(self, 
                             prediction: Dict[str, Any], 
                             microstructure: Dict[str, Any],
                             orderbook: Optional[OrderBook] = None) -> Dict[str, Any]:
        """Optimize trade entry timing based on microstructure"""
        try:
            direction = prediction.get('direction', 'HOLD')
            confidence = prediction.get('confidence', 0.5)
            
            if direction == 'HOLD':
                return {'action': 'HOLD', 'reason': 'No trade signal'}
            
            # Check liquidity conditions
            liquidity_depth = microstructure.get('liquidity_depth', 0)
            spread = microstructure.get('spread', float('inf'))
            
            if liquidity_depth < 1000:  # Minimum liquidity threshold
                return {
                    'action': 'WAIT',
                    'reason': 'Low liquidity',
                    'suggested_delay': 30  # seconds
                }
            
            if spread > 0.01:  # Maximum spread threshold (1%)
                return {
                    'action': 'WAIT',
                    'reason': 'Wide spread',
                    'suggested_delay': 10
                }
            
            # Check order flow alignment
            bid_ask_pressure = microstructure.get('bid_ask_pressure', 0)
            volume_imbalance = microstructure.get('volume_imbalance', 0)
            
            # For BUY signals, look for positive pressure/imbalance
            if direction == 'BUY':
                if bid_ask_pressure > 0.3 and volume_imbalance > 0.2:
                    return {
                        'action': 'EXECUTE_IMMEDIATE',
                        'reason': 'Strong buying pressure',
                        'urgency': 'HIGH'
                    }
                elif bid_ask_pressure > 0 and volume_imbalance > 0:
                    return {
                        'action': 'EXECUTE_GRADUAL',
                        'reason': 'Moderate buying pressure',
                        'slices': 3,
                        'delay_between_slices': 5
                    }
                elif bid_ask_pressure < -0.3:
                    return {
                        'action': 'WAIT',
                        'reason': 'Strong selling pressure',
                        'suggested_delay': 60
                    }
            
            # For SELL signals, look for negative pressure/imbalance
            elif direction == 'SELL':
                if bid_ask_pressure < -0.3 and volume_imbalance < -0.2:
                    return {
                        'action': 'EXECUTE_IMMEDIATE',
                        'reason': 'Strong selling pressure',
                        'urgency': 'HIGH'
                    }
                elif bid_ask_pressure < 0 and volume_imbalance < 0:
                    return {
                        'action': 'EXECUTE_GRADUAL',
                        'reason': 'Moderate selling pressure',
                        'slices': 3,
                        'delay_between_slices': 5
                    }
                elif bid_ask_pressure > 0.3:
                    return {
                        'action': 'WAIT',
                        'reason': 'Strong buying pressure',
                        'suggested_delay': 60
                    }
            
            # Default execution strategy
            return {
                'action': 'EXECUTE_GRADUAL',
                'reason': 'Neutral conditions',
                'slices': 2,
                'delay_between_slices': 10
            }
            
        except Exception as e:
            logger.error(f"Error optimizing entry timing: {e}")
            return {'action': 'WAIT', 'reason': f'Analysis error: {str(e)}'}
    
    def estimate_execution_cost(self, 
                               trade_size: float, 
                               direction: str,
                               orderbook: OrderBook) -> Dict[str, Any]:
        """Estimate total execution cost including market impact"""
        try:
            # Estimate market impact
            impact_analysis = self.impact_estimator.estimate_impact(
                trade_size, orderbook, direction
            )
            
            # Calculate spread cost
            spread_cost = orderbook.spread / 2  # Half spread
            
            # Calculate timing cost (based on volatility)
            timing_cost = self._estimate_timing_cost()
            
            # Total cost
            total_cost = (
                impact_analysis['estimated_impact'] + 
                spread_cost + 
                timing_cost
            )
            
            return {
                'market_impact': impact_analysis['estimated_impact'],
                'spread_cost': spread_cost,
                'timing_cost': timing_cost,
                'total_cost': total_cost,
                'avg_execution_price': impact_analysis['avg_execution_price'],
                'fill_ratio': impact_analysis['fill_ratio']
            }
            
        except Exception as e:
            logger.error(f"Error estimating execution cost: {e}")
            return {'total_cost': 0, 'error': str(e)}
    
    def _update_price_analysis(self, price: float):
        """Update price movement analysis"""
        try:
            timestamp = datetime.now()
            self.price_history.append({'price': price, 'timestamp': timestamp})
            
            # Keep limited history
            if len(self.price_history) > 200:
                self.price_history = self.price_history[-200:]
                
        except Exception as e:
            logger.error(f"Error updating price analysis: {e}")
    
    def _calculate_price_acceleration(self) -> float:
        """Calculate price acceleration"""
        try:
            if len(self.price_history) < 10:
                return 0.0
            
            recent_prices = [p['price'] for p in self.price_history[-10:]]
            
            # Calculate first and second derivatives
            first_diff = np.diff(recent_prices)
            if len(first_diff) < 2:
                return 0.0
            
            second_diff = np.diff(first_diff)
            
            # Return average acceleration
            return float(np.mean(second_diff))
            
        except Exception as e:
            logger.error(f"Error calculating price acceleration: {e}")
            return 0.0
    
    def _calculate_trade_intensity(self) -> float:
        """Calculate trade intensity (trades per minute)"""
        try:
            if len(self.order_flow_analyzer.tick_history) < 10:
                return 0.0
            
            # Count ticks in last minute
            cutoff_time = datetime.now() - timedelta(minutes=1)
            recent_ticks = [
                tick for tick in self.order_flow_analyzer.tick_history
                if tick.timestamp > cutoff_time
            ]
            
            return len(recent_ticks)  # Ticks per minute
            
        except Exception as e:
            logger.error(f"Error calculating trade intensity: {e}")
            return 0.0
    
    def _calculate_liquidity_score(self, 
                                  depth_metrics: Dict[str, Any], 
                                  spread_metrics: Dict[str, Any]) -> float:
        """Calculate overall liquidity score (0-1)"""
        try:
            # Normalize depth (higher is better)
            depth_score = min(1.0, depth_metrics.get('total_depth', 0) / 10000)
            
            # Normalize spread (lower is better)
            current_spread = spread_metrics.get('current_spread', float('inf'))
            spread_score = max(0.0, 1.0 - (current_spread / 0.01))  # 1% spread = 0 score
            
            # Combined score
            return (depth_score + spread_score) / 2
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {e}")
            return 0.5
    
    def _estimate_timing_cost(self) -> float:
        """Estimate timing cost based on volatility"""
        try:
            if len(self.price_history) < 20:
                return 0.001  # Default 0.1%
            
            recent_prices = [p['price'] for p in self.price_history[-20:]]
            returns = np.diff(np.log(recent_prices))
            volatility = np.std(returns) if len(returns) > 1 else 0.001
            
            # Timing cost proportional to volatility
            return volatility * 0.5  # 50% of volatility
            
        except Exception as e:
            logger.error(f"Error estimating timing cost: {e}")
            return 0.001
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get comprehensive microstructure analysis"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'tick_count': len(self.order_flow_analyzer.tick_history),
                'orderbook_count': len(self.liquidity_analyzer.orderbook_history),
                'latest_spread': self.liquidity_analyzer.spread_history[-1] if self.liquidity_analyzer.spread_history else 0,
                'trade_intensity': self._calculate_trade_intensity(),
                'price_acceleration': self._calculate_price_acceleration(),
                'analysis_active': True
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive analysis: {e}")
            return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    # Create analyzer
    analyzer = MicrostructureAnalyzer()
    
    # Simulate tick data
    tick = TickData(
        timestamp=datetime.now(),
        bid=2000.0,
        ask=2000.5,
        last_price=2000.2,
        volume=100,
        bid_size=1500,
        ask_size=1200
    )
    
    # Analyze order flow
    order_flow_analysis = analyzer.analyze_order_flow(tick)
    print("Order flow analysis:", order_flow_analysis)
    
    # Create mock orderbook
    orderbook = OrderBook(
        timestamp=datetime.now(),
        symbol="XAUUSD",
        bids=[
            OrderBookLevel(2000.0, 1500, 3),
            OrderBookLevel(1999.9, 1200, 2),
            OrderBookLevel(1999.8, 800, 1)
        ],
        asks=[
            OrderBookLevel(2000.5, 1200, 2),
            OrderBookLevel(2000.6, 900, 2),
            OrderBookLevel(2000.7, 600, 1)
        ]
    )
    
    # Analyze liquidity
    liquidity_analysis = analyzer.analyze_liquidity(orderbook)
    print("Liquidity analysis:", liquidity_analysis)
    
    # Test execution optimization
    prediction = {
        'direction': 'BUY',
        'confidence': 0.75,
        'symbol': 'XAUUSD'
    }
    
    microstructure = {
        'liquidity_depth': 5000,
        'spread': 0.0025,
        'bid_ask_pressure': 0.4,
        'volume_imbalance': 0.3
    }
    
    execution_strategy = analyzer.optimize_entry_timing(prediction, microstructure, orderbook)
    print("Execution strategy:", execution_strategy)
    
    # Estimate execution cost
    cost_analysis = analyzer.estimate_execution_cost(1000, 'BUY', orderbook)
    print("Cost analysis:", cost_analysis)