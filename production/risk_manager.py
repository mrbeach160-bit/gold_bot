"""
Enhanced Risk Management System

Production-ready risk management with advanced position sizing,
correlation monitoring, drawdown protection, and real-time risk assessment.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    
    def update_price(self, current_price: float):
        """Update current price and unrealized P&L"""
        self.current_price = current_price
        if self.direction == 'BUY':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:  # SELL
            self.unrealized_pnl = (self.entry_price - current_price) * self.size

@dataclass
class RiskLimits:
    """Risk management limits configuration"""
    max_risk_per_trade: float = 0.02  # 2% per trade
    max_portfolio_risk: float = 0.10  # 10% total portfolio risk
    max_drawdown: float = 0.15  # 15% maximum drawdown
    max_correlation: float = 0.7  # Maximum correlation between positions
    max_daily_trades: int = 20  # Maximum trades per day
    min_profit_factor: float = 1.5  # Minimum profit factor to continue trading
    position_size_limits: Dict[str, float] = None  # Symbol-specific limits
    
    def __post_init__(self):
        if self.position_size_limits is None:
            self.position_size_limits = {
                'XAUUSD': 0.05,  # 5% max position size for gold
                'default': 0.03  # 3% default max position size
            }

class PositionSizer:
    """Advanced position sizing based on risk and volatility"""
    
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.volatility_lookback = 20  # Days for volatility calculation
        self.price_history = {}
    
    def calculate(self, 
                 symbol: str,
                 direction: str,
                 confidence: float,
                 account_balance: float,
                 entry_price: float,
                 stop_loss: float,
                 market_volatility: Optional[float] = None) -> float:
        """Calculate optimal position size"""
        try:
            # Base risk amount (Kelly criterion influenced)
            base_risk = self.risk_limits.max_risk_per_trade * account_balance
            
            # Adjust for confidence
            confidence_adjusted_risk = base_risk * confidence
            
            # Calculate risk per share/unit
            if direction == 'BUY':
                risk_per_unit = abs(entry_price - stop_loss)
            else:  # SELL
                risk_per_unit = abs(stop_loss - entry_price)
            
            if risk_per_unit == 0:
                logger.warning("Zero stop loss distance - using default position size")
                return account_balance * 0.01  # 1% default
            
            # Base position size
            position_size = confidence_adjusted_risk / risk_per_unit
            
            # Apply volatility adjustment
            if market_volatility:
                volatility_multiplier = min(1.0, 0.2 / market_volatility)  # Target 20% volatility
                position_size *= volatility_multiplier
            
            # Apply symbol-specific limits
            max_position_value = account_balance * self.risk_limits.position_size_limits.get(
                symbol, self.risk_limits.position_size_limits['default']
            )
            max_position_size = max_position_value / entry_price
            position_size = min(position_size, max_position_size)
            
            # Minimum position size (avoid dust trades)
            min_position_size = account_balance * 0.001 / entry_price  # 0.1% minimum
            position_size = max(position_size, min_position_size)
            
            logger.debug(f"Position size calculated: {position_size:.4f} for {symbol}")
            return position_size
            
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return account_balance * 0.01 / entry_price  # Safe fallback

class StopLossManager:
    """Dynamic stop loss management"""
    
    def __init__(self):
        self.atr_period = 14
        self.trailing_multiplier = 2.0
        self.min_stop_distance = 0.001  # 0.1% minimum stop distance
    
    def calculate(self, 
                 prediction: Dict[str, Any],
                 price_data: Optional[pd.DataFrame] = None,
                 volatility: Optional[float] = None) -> Dict[str, float]:
        """Calculate optimal stop loss and take profit levels"""
        try:
            entry_price = prediction.get('price', 0)
            direction = prediction.get('direction', 'HOLD')
            confidence = prediction.get('confidence', 0.5)
            
            if not entry_price or direction == 'HOLD':
                return {'stop_loss': None, 'take_profit': None}
            
            # Calculate ATR-based stop distance
            if price_data is not None and len(price_data) >= self.atr_period:
                atr = self._calculate_atr(price_data)
                stop_distance = atr * self.trailing_multiplier
            elif volatility:
                stop_distance = entry_price * volatility * 2.0  # 2x volatility
            else:
                stop_distance = entry_price * 0.02  # 2% fallback
            
            # Ensure minimum stop distance
            stop_distance = max(stop_distance, entry_price * self.min_stop_distance)
            
            # Adjust for confidence (tighter stops for lower confidence)
            confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
            stop_distance *= confidence_multiplier
            
            if direction == 'BUY':
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + (stop_distance * 2)  # 2:1 reward/risk
            else:  # SELL
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - (stop_distance * 2)
            
            return {
                'stop_loss': round(stop_loss, 5),
                'take_profit': round(take_profit, 5),
                'stop_distance': stop_distance,
                'risk_reward_ratio': 2.0
            }
            
        except Exception as e:
            logger.error(f"Stop loss calculation error: {e}")
            return {'stop_loss': None, 'take_profit': None}
    
    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate Average True Range"""
        try:
            high = data['high']
            low = data['low']
            close = data['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=self.atr_period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.02
            
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return 0.02

class CorrelationMonitor:
    """Monitor correlation between positions to avoid overexposure"""
    
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.correlation_window = 50  # Period for correlation calculation
        self.price_data = {}
        
    def update_price_data(self, symbol: str, price: float, timestamp: datetime):
        """Update price history for correlation calculation"""
        if symbol not in self.price_data:
            self.price_data[symbol] = []
        
        self.price_data[symbol].append({'price': price, 'timestamp': timestamp})
        
        # Keep only recent data
        cutoff_time = timestamp - timedelta(days=self.correlation_window)
        self.price_data[symbol] = [
            data for data in self.price_data[symbol] 
            if data['timestamp'] > cutoff_time
        ]
    
    def calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols"""
        try:
            if symbol1 not in self.price_data or symbol2 not in self.price_data:
                return 0.0
            
            data1 = pd.DataFrame(self.price_data[symbol1])
            data2 = pd.DataFrame(self.price_data[symbol2])
            
            if len(data1) < 10 or len(data2) < 10:
                return 0.0
            
            # Merge on timestamp and calculate returns
            merged = pd.merge(data1, data2, on='timestamp', suffixes=('_1', '_2'))
            
            if len(merged) < 10:
                return 0.0
            
            returns1 = merged['price_1'].pct_change().dropna()
            returns2 = merged['price_2'].pct_change().dropna()
            
            correlation = returns1.corr(returns2)
            return correlation if not pd.isna(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Correlation calculation error: {e}")
            return 0.0
    
    def exceeds_limits(self, new_symbol: str, new_direction: str, 
                      current_positions: List[Position]) -> bool:
        """Check if new position would exceed correlation limits"""
        try:
            # Group positions by direction
            same_direction_positions = [
                pos for pos in current_positions 
                if pos.direction == new_direction
            ]
            
            # Check correlation with existing positions
            for position in same_direction_positions:
                correlation = self.calculate_correlation(new_symbol, position.symbol)
                
                if abs(correlation) > self.risk_limits.max_correlation:
                    logger.warning(f"High correlation detected: {new_symbol} vs {position.symbol} "
                                 f"({correlation:.2f})")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Correlation limit check error: {e}")
            return False

class DrawdownMonitor:
    """Monitor portfolio drawdown and implement protection"""
    
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.equity_history = []
        self.peak_equity = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown_reached = 0.0
        
    def update(self, current_equity: float):
        """Update drawdown calculations"""
        try:
            timestamp = datetime.now()
            self.equity_history.append({'equity': current_equity, 'timestamp': timestamp})
            
            # Update peak equity
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            
            # Calculate current drawdown
            if self.peak_equity > 0:
                self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
                self.max_drawdown_reached = max(self.max_drawdown_reached, self.current_drawdown)
            
            # Keep only recent history (last 1000 updates)
            if len(self.equity_history) > 1000:
                self.equity_history = self.equity_history[-1000:]
                
        except Exception as e:
            logger.error(f"Drawdown update error: {e}")
    
    def should_reduce_risk(self) -> Tuple[bool, float]:
        """Check if risk should be reduced due to drawdown"""
        if self.current_drawdown > self.risk_limits.max_drawdown:
            # Gradually reduce risk as drawdown increases
            risk_multiplier = max(0.1, 1.0 - (self.current_drawdown / self.risk_limits.max_drawdown))
            return True, risk_multiplier
        
        return False, 1.0
    
    def should_stop_trading(self) -> bool:
        """Check if trading should be stopped"""
        return self.current_drawdown > (self.risk_limits.max_drawdown * 1.2)  # 20% above limit

class ProductionRiskManager:
    """Main production risk management system"""
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.position_sizer = PositionSizer(self.risk_limits)
        self.stop_loss_manager = StopLossManager()
        self.correlation_monitor = CorrelationMonitor(self.risk_limits)
        self.drawdown_monitor = DrawdownMonitor(self.risk_limits)
        
        # Portfolio state
        self.current_positions: List[Position] = []
        self.daily_trades = 0
        self.last_trade_date = None
        self.account_balance = 10000.0  # Default starting balance
        
        # Performance tracking
        self.trades_history = []
        self.daily_pnl = []
        
        logger.info("Production Risk Manager initialized")
    
    def update_account_balance(self, balance: float):
        """Update account balance"""
        self.account_balance = balance
        self.drawdown_monitor.update(balance)
    
    def evaluate_trade(self, 
                      prediction: Dict[str, Any],
                      current_positions: Optional[List[Position]] = None,
                      market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Comprehensive trade evaluation with risk management"""
        try:
            if current_positions is None:
                current_positions = self.current_positions
            
            symbol = prediction.get('symbol', 'XAUUSD')
            direction = prediction.get('direction', 'HOLD')
            confidence = prediction.get('confidence', 0.5)
            price = prediction.get('price', 0)
            
            # Check if trading should be stopped
            if self.drawdown_monitor.should_stop_trading():
                return {
                    'action': 'STOP_TRADING',
                    'reason': f'Maximum drawdown exceeded: {self.drawdown_monitor.current_drawdown:.2%}'
                }
            
            # Check daily trade limits
            today = datetime.now().date()
            if self.last_trade_date != today:
                self.daily_trades = 0
                self.last_trade_date = today
            
            if self.daily_trades >= self.risk_limits.max_daily_trades:
                return {
                    'action': 'SKIP',
                    'reason': f'Daily trade limit reached: {self.daily_trades}'
                }
            
            # Check for HOLD signals
            if direction == 'HOLD' or confidence < 0.55:
                return {
                    'action': 'SKIP',
                    'reason': f'Low confidence signal: {confidence:.2f}'
                }
            
            # Check correlation limits
            if self.correlation_monitor.exceeds_limits(symbol, direction, current_positions):
                return {
                    'action': 'SKIP',
                    'reason': 'Correlation limit exceeded'
                }
            
            # Check drawdown and adjust risk
            should_reduce_risk, risk_multiplier = self.drawdown_monitor.should_reduce_risk()
            if should_reduce_risk:
                logger.warning(f"Reducing risk due to drawdown: {risk_multiplier:.2f}")
            
            # Calculate stop loss and take profit
            stop_loss_info = self.stop_loss_manager.calculate(
                prediction, market_data, 
                volatility=prediction.get('volatility')
            )
            
            if not stop_loss_info.get('stop_loss'):
                return {
                    'action': 'SKIP',
                    'reason': 'Could not calculate stop loss'
                }
            
            # Calculate position size
            position_size = self.position_sizer.calculate(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                account_balance=self.account_balance,
                entry_price=price,
                stop_loss=stop_loss_info['stop_loss'],
                market_volatility=prediction.get('volatility')
            )
            
            # Apply risk reduction if necessary
            position_size *= risk_multiplier
            
            # Final validation
            if position_size <= 0:
                return {
                    'action': 'SKIP',
                    'reason': 'Position size too small'
                }
            
            return {
                'action': 'EXECUTE',
                'symbol': symbol,
                'direction': direction,
                'position_size': position_size,
                'entry_price': price,
                'stop_loss': stop_loss_info['stop_loss'],
                'take_profit': stop_loss_info['take_profit'],
                'confidence': confidence,
                'risk_reward_ratio': stop_loss_info.get('risk_reward_ratio', 2.0),
                'risk_amount': abs(price - stop_loss_info['stop_loss']) * position_size,
                'risk_multiplier': risk_multiplier
            }
            
        except Exception as e:
            logger.error(f"Trade evaluation error: {e}")
            return {
                'action': 'SKIP',
                'reason': f'Evaluation error: {str(e)}'
            }
    
    def add_position(self, position: Position):
        """Add new position to portfolio"""
        self.current_positions.append(position)
        self.daily_trades += 1
        
        # Update correlation monitor
        self.correlation_monitor.update_price_data(
            position.symbol, position.entry_price, position.entry_time
        )
        
        logger.info(f"Position added: {position.symbol} {position.direction} "
                   f"size={position.size:.4f} @ {position.entry_price}")
    
    def close_position(self, position: Position, exit_price: float, exit_time: datetime):
        """Close position and calculate P&L"""
        try:
            if position in self.current_positions:
                self.current_positions.remove(position)
            
            # Calculate realized P&L
            if position.direction == 'BUY':
                pnl = (exit_price - position.entry_price) * position.size
            else:  # SELL
                pnl = (position.entry_price - exit_price) * position.size
            
            # Update account balance
            self.account_balance += pnl
            self.drawdown_monitor.update(self.account_balance)
            
            # Record trade
            trade_record = {
                'symbol': position.symbol,
                'direction': position.direction,
                'size': position.size,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'entry_time': position.entry_time,
                'exit_time': exit_time,
                'pnl': pnl,
                'duration': (exit_time - position.entry_time).total_seconds() / 3600  # hours
            }
            
            self.trades_history.append(trade_record)
            
            logger.info(f"Position closed: {position.symbol} P&L={pnl:.2f}")
            
            return pnl
            
        except Exception as e:
            logger.error(f"Position close error: {e}")
            return 0.0
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            # Update current position P&L
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.current_positions)
            
            # Calculate performance metrics
            if self.trades_history:
                realized_pnl = sum(trade['pnl'] for trade in self.trades_history)
                winning_trades = [t for t in self.trades_history if t['pnl'] > 0]
                losing_trades = [t for t in self.trades_history if t['pnl'] < 0]
                
                win_rate = len(winning_trades) / len(self.trades_history)
                avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            else:
                realized_pnl = 0
                win_rate = 0
                profit_factor = 0
            
            return {
                'account_balance': self.account_balance,
                'total_positions': len(self.current_positions),
                'realized_pnl': realized_pnl,
                'unrealized_pnl': total_unrealized_pnl,
                'total_pnl': realized_pnl + total_unrealized_pnl,
                'current_drawdown': self.drawdown_monitor.current_drawdown,
                'max_drawdown': self.drawdown_monitor.max_drawdown_reached,
                'daily_trades': self.daily_trades,
                'total_trades': len(self.trades_history),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'direction': pos.direction,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'stop_loss': pos.stop_loss,
                        'take_profit': pos.take_profit
                    }
                    for pos in self.current_positions
                ]
            }
            
        except Exception as e:
            logger.error(f"Portfolio summary error: {e}")
            return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    # Create risk manager
    risk_manager = ProductionRiskManager()
    
    # Example prediction
    prediction = {
        'symbol': 'XAUUSD',
        'direction': 'BUY',
        'confidence': 0.75,
        'price': 2000.0,
        'volatility': 0.015
    }
    
    # Evaluate trade
    result = risk_manager.evaluate_trade(prediction)
    print("Trade evaluation result:", result)
    
    if result['action'] == 'EXECUTE':
        # Create and add position
        position = Position(
            symbol=result['symbol'],
            direction=result['direction'],
            size=result['position_size'],
            entry_price=result['entry_price'],
            entry_time=datetime.now(),
            stop_loss=result['stop_loss'],
            take_profit=result['take_profit']
        )
        
        risk_manager.add_position(position)
        
        # Get portfolio summary
        summary = risk_manager.get_portfolio_summary()
        print("Portfolio summary:", summary)