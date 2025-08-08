"""
Risk Service for the modular application.
Handles risk management calculations and validations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class RiskService:
    """Service for risk management and calculations."""
    
    def __init__(self):
        self.max_risk_per_trade = 10.0  # Maximum risk percentage per trade
        self.max_daily_loss = 20.0  # Maximum daily loss percentage
        self.max_portfolio_risk = 25.0  # Maximum portfolio risk
    
    def validate_trading_inputs(self, symbol: str, balance: float, risk_percent: float,
                              sl_pips: float, tp_pips: float) -> List[str]:
        """
        Validate trading inputs for safety.
        
        Args:
            symbol: Trading symbol
            balance: Account balance
            risk_percent: Risk percentage per trade
            sl_pips: Stop loss in pips
            tp_pips: Take profit in pips
            
        Returns:
            List of validation issues (empty if all valid)
        """
        issues = []
        
        # Validate balance
        if balance <= 0:
            issues.append("Balance must be positive")
        elif balance < 100:
            issues.append("Balance is very low - consider paper trading first")
        
        # Validate risk percentage
        if risk_percent <= 0:
            issues.append("Risk percentage must be positive")
        elif risk_percent > self.max_risk_per_trade:
            issues.append(f"Risk percentage too high (max {self.max_risk_per_trade}%)")
        
        # Validate stop loss
        if sl_pips <= 0:
            issues.append("Stop loss must be positive")
        elif sl_pips > 1000:
            issues.append("Stop loss seems too large")
        
        # Validate take profit
        if tp_pips <= 0:
            issues.append("Take profit must be positive")
        elif tp_pips > 2000:
            issues.append("Take profit seems too large")
        
        # Risk-reward ratio
        if tp_pips < sl_pips:
            issues.append("Take profit should typically be larger than stop loss")
        
        return issues
    
    def calculate_position_size(self, balance: float, risk_percent: float,
                              entry_price: float, sl_price: float) -> Dict[str, Any]:
        """
        Calculate position size based on risk management.
        
        Args:
            balance: Account balance
            risk_percent: Risk percentage
            entry_price: Entry price
            sl_price: Stop loss price
            
        Returns:
            Dictionary with position size calculations
        """
        try:
            risk_amount = balance * (risk_percent / 100)
            sl_distance = abs(entry_price - sl_price)
            
            if sl_distance <= 0:
                return {
                    'position_size': 0,
                    'risk_amount': risk_amount,
                    'sl_distance': 0,
                    'error': 'Invalid stop loss distance'
                }
            
            # Calculate position size
            position_size = risk_amount / sl_distance
            
            # Limit position size to available balance
            max_position = balance * 0.95  # Use max 95% of balance
            if position_size > max_position:
                position_size = max_position
            
            return {
                'position_size': position_size,
                'risk_amount': risk_amount,
                'sl_distance': sl_distance,
                'effective_risk_pct': (position_size * sl_distance / balance) * 100,
                'max_loss': position_size * sl_distance,
                'error': None
            }
            
        except Exception as e:
            return {
                'position_size': 0,
                'risk_amount': 0,
                'sl_distance': 0,
                'error': str(e)
            }
    
    def calculate_risk_reward_ratio(self, entry_price: float, sl_price: float, 
                                  tp_price: float) -> float:
        """
        Calculate risk-reward ratio.
        
        Args:
            entry_price: Entry price
            sl_price: Stop loss price
            tp_price: Take profit price
            
        Returns:
            Risk-reward ratio
        """
        try:
            risk = abs(entry_price - sl_price)
            reward = abs(tp_price - entry_price)
            
            if risk <= 0:
                return 0
            
            return reward / risk
            
        except:
            return 0
    
    def assess_signal_risk(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk level of a trading signal.
        
        Args:
            prediction_result: Result from prediction service
            
        Returns:
            Dictionary with risk assessment
        """
        try:
            ensemble = prediction_result.get('ensemble', {})
            individual = prediction_result.get('individual', {})
            
            direction = ensemble.get('direction', 'HOLD')
            confidence = ensemble.get('confidence', 0.5)
            model_count = ensemble.get('model_count', 0)
            
            # Base risk assessment
            if direction == 'HOLD':
                risk_level = 'LOW'
                risk_score = 0.2
            elif confidence >= 0.8:
                risk_level = 'LOW'
                risk_score = 0.3
            elif confidence >= 0.65:
                risk_level = 'MEDIUM'
                risk_score = 0.5
            elif confidence >= 0.55:
                risk_level = 'MEDIUM'
                risk_score = 0.6
            else:
                risk_level = 'HIGH'
                risk_score = 0.8
            
            # Adjust for model consensus
            if model_count >= 4:
                risk_score *= 0.9  # Lower risk with more models
            elif model_count <= 2:
                risk_score *= 1.2  # Higher risk with fewer models
            
            # Check for model disagreement
            directions = [pred.get('direction') for pred in individual.values()]
            unique_directions = set(directions)
            if len(unique_directions) > 2:
                risk_score *= 1.3  # Higher risk with disagreement
                risk_level = 'HIGH'
            
            # Finalize risk level based on score
            if risk_score <= 0.3:
                risk_level = 'LOW'
            elif risk_score <= 0.6:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'HIGH'
            
            return {
                'risk_level': risk_level,
                'risk_score': min(1.0, risk_score),
                'confidence': confidence,
                'model_consensus': len(unique_directions),
                'model_count': model_count,
                'recommended_risk_pct': self._get_recommended_risk_pct(risk_level),
                'warnings': self._generate_risk_warnings(prediction_result)
            }
            
        except Exception as e:
            return {
                'risk_level': 'HIGH',
                'risk_score': 1.0,
                'confidence': 0.0,
                'model_consensus': 0,
                'model_count': 0,
                'recommended_risk_pct': 1.0,
                'warnings': [f'Risk assessment error: {str(e)}']
            }
    
    def _get_recommended_risk_pct(self, risk_level: str) -> float:
        """Get recommended risk percentage based on risk level."""
        if risk_level == 'LOW':
            return 2.0
        elif risk_level == 'MEDIUM':
            return 1.5
        else:  # HIGH
            return 1.0
    
    def _generate_risk_warnings(self, prediction_result: Dict[str, Any]) -> List[str]:
        """Generate risk warnings based on prediction result."""
        warnings = []
        
        load_errors = prediction_result.get('load_errors', {})
        individual = prediction_result.get('individual', {})
        ensemble = prediction_result.get('ensemble', {})
        
        # Check for model loading issues
        error_count = len([e for e in load_errors.values() if 'optional' not in e.lower()])
        if error_count > 0:
            warnings.append(f"{error_count} models failed to load")
        
        # Check for low model count
        if len(individual) < 3:
            warnings.append("Less than 3 models available for prediction")
        
        # Check for low confidence
        confidence = ensemble.get('confidence', 0)
        if confidence < 0.6:
            warnings.append("Low ensemble confidence")
        
        # Check for prediction errors
        error_count = sum(1 for pred in individual.values() 
                         if pred.get('raw', {}).get('error'))
        if error_count > 0:
            warnings.append(f"{error_count} individual predictions had errors")
        
        return warnings
    
    def calculate_portfolio_metrics(self, trades: List[Dict[str, Any]], 
                                  initial_balance: float) -> Dict[str, Any]:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            trades: List of completed trades
            initial_balance: Initial balance
            
        Returns:
            Dictionary with portfolio metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_consecutive_losses': 0,
                'avg_trade_risk': 0,
                'sharpe_ratio': 0
            }
        
        # Basic metrics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(trades) * 100
        
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for trade in trades:
            if trade['pnl'] <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # Average trade risk (as percentage of balance)
        avg_trade_risk = np.mean([abs(t['pnl']) / initial_balance * 100 for t in trades])
        
        # Simple Sharpe ratio approximation
        returns = [t['pnl'] / initial_balance for t in trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_trade_risk': avg_trade_risk,
            'sharpe_ratio': sharpe_ratio,
            'total_return_pct': (sum(t['pnl'] for t in trades) / initial_balance) * 100
        }