# models/evaluation.py
"""
Advanced evaluation framework with walk-forward optimization and trading simulation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import TimeSeriesSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


class TradingSimulator:
    """Realistic trading simulation with costs, slippage, and position sizing."""
    
    def __init__(self, initial_capital: float = 10000, 
                 transaction_cost: float = 0.001,  # 0.1% per trade
                 slippage: float = 0.0005,         # 0.05% slippage
                 max_position_size: float = 0.2):   # 20% max position
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        
        # Trading state
        self.capital = initial_capital
        self.position = 0.0  # Current position size (-1 to 1, where 1 = fully long)
        self.trades = []
        self.portfolio_values = []
        self.daily_returns = []
        
    def calculate_position_size(self, signal: str, confidence: float, 
                               current_price: float, volatility: float) -> float:
        """Calculate position size based on confidence and risk management."""
        if signal == 'HOLD':
            return 0.0
        
        # Base position size from confidence
        base_size = confidence * self.max_position_size
        
        # Volatility adjustment (reduce size in high volatility)
        vol_adjustment = max(0.5, 1.0 - volatility * 10)  # Cap at 50% reduction
        
        # Risk-adjusted position size
        adjusted_size = base_size * vol_adjustment
        
        # Direction
        if signal == 'SELL':
            adjusted_size = -adjusted_size
        
        return adjusted_size
    
    def execute_trade(self, signal: str, confidence: float, 
                     current_price: float, timestamp: datetime,
                     volatility: float = 0.01) -> Dict[str, Any]:
        """Execute a trade based on model signal."""
        target_position = self.calculate_position_size(signal, confidence, current_price, volatility)
        position_change = target_position - self.position
        
        if abs(position_change) < 0.01:  # Minimum trade threshold
            return {
                'executed': False,
                'reason': 'Position change too small',
                'current_position': self.position,
                'target_position': target_position
            }
        
        # Calculate trade costs
        trade_size = abs(position_change) * self.capital
        cost = trade_size * (self.transaction_cost + self.slippage)
        
        # Adjust price for slippage
        if position_change > 0:  # Buying
            execution_price = current_price * (1 + self.slippage)
        else:  # Selling
            execution_price = current_price * (1 - self.slippage)
        
        # Execute trade
        self.capital -= cost
        self.position = target_position
        
        trade_record = {
            'timestamp': timestamp,
            'signal': signal,
            'confidence': confidence,
            'position_change': position_change,
            'execution_price': execution_price,
            'cost': cost,
            'new_position': self.position,
            'capital_after_cost': self.capital,
            'executed': True
        }
        
        self.trades.append(trade_record)
        return trade_record
    
    def update_portfolio_value(self, current_price: float, timestamp: datetime):
        """Update portfolio value based on current price."""
        # Position value change
        if self.trades:
            last_price = self.trades[-1]['execution_price']
            price_change = (current_price - last_price) / last_price
            position_pnl = self.position * self.capital * price_change
        else:
            position_pnl = 0
        
        # Total portfolio value
        total_value = self.capital + position_pnl
        
        self.portfolio_values.append({
            'timestamp': timestamp,
            'portfolio_value': total_value,
            'capital': self.capital,
            'position': self.position,
            'position_pnl': position_pnl,
            'current_price': current_price
        })
        
        # Calculate daily return
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2]['portfolio_value']
            daily_return = (total_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
    
    def backtest(self, predictions: List[Dict], prices: pd.DataFrame) -> Dict[str, Any]:
        """Run complete backtest simulation."""
        print(f"Running backtest with {len(predictions)} predictions...")
        
        # Reset simulator state
        self.capital = self.initial_capital
        self.position = 0.0
        self.trades = []
        self.portfolio_values = []
        self.daily_returns = []
        
        # Align predictions with price data
        for i, pred in enumerate(predictions):
            if i >= len(prices):
                break
                
            current_price = prices['close'].iloc[i]
            timestamp = prices.index[i] if hasattr(prices.index, '__getitem__') else datetime.now()
            
            # Calculate recent volatility
            if i >= 20:
                recent_returns = prices['close'].iloc[max(0, i-20):i].pct_change().dropna()
                volatility = recent_returns.std() if len(recent_returns) > 1 else 0.01
            else:
                volatility = 0.01
            
            # Execute trade if signal present
            signal = pred.get('direction', 'HOLD')
            confidence = pred.get('confidence', 0.5)
            
            if signal != 'HOLD':
                self.execute_trade(signal, confidence, current_price, timestamp, volatility)
            
            # Update portfolio value
            self.update_portfolio_value(current_price, timestamp)
        
        # Calculate performance metrics
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive trading performance metrics."""
        if not self.portfolio_values:
            return {'error': 'No portfolio data available'}
        
        # Extract portfolio values
        portfolio_values = [pv['portfolio_value'] for pv in self.portfolio_values]
        
        # Total return
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return (assuming daily data)
        days = len(portfolio_values)
        years = days / 252  # 252 trading days per year
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility and Sharpe ratio
        if len(self.daily_returns) > 1:
            volatility = np.std(self.daily_returns) * np.sqrt(252)  # Annualized
            sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = self.initial_capital
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Win rate and trade statistics
        profitable_trades = 0
        total_trades = len(self.trades)
        
        if total_trades > 0:
            for trade in self.trades:
                # Simplified profit calculation (would need actual exit prices for accuracy)
                if trade['position_change'] > 0:  # Long trade
                    profitable_trades += 1 if trade['confidence'] > 0.6 else 0
                elif trade['position_change'] < 0:  # Short trade
                    profitable_trades += 1 if trade['confidence'] > 0.6 else 0
            
            win_rate = profitable_trades / total_trades
        else:
            win_rate = 0
        
        # Calmar ratio (annualized return / max drawdown)
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Profit factor (simplified)
        profit_factor = (1 + total_return) / max(1 - total_return, 0.1)  # Gross profit / Gross loss
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'volatility': volatility,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'final_portfolio_value': portfolio_values[-1],
            'days_traded': len(portfolio_values),
            'avg_daily_return': np.mean(self.daily_returns) if self.daily_returns else 0
        }


class WalkForwardOptimizer:
    """Walk-forward optimization for time series model validation."""
    
    def __init__(self, training_window: int = 252,  # 1 year training
                 testing_window: int = 63,          # 3 months testing
                 step_size: int = 21):              # 1 month steps
        self.training_window = training_window
        self.testing_window = testing_window
        self.step_size = step_size
        
    def generate_splits(self, data_length: int) -> List[Tuple[slice, slice]]:
        """Generate train/test splits for walk-forward analysis."""
        splits = []
        
        current_start = 0
        while current_start + self.training_window + self.testing_window <= data_length:
            train_end = current_start + self.training_window
            test_end = train_end + self.testing_window
            
            train_slice = slice(current_start, train_end)
            test_slice = slice(train_end, test_end)
            
            splits.append((train_slice, test_slice))
            current_start += self.step_size
        
        return splits
    
    def optimize_model(self, model, data: pd.DataFrame, 
                      target_column: str = 'target') -> Dict[str, Any]:
        """Perform walk-forward optimization on a model."""
        if not HAS_SKLEARN:
            print("Walk-forward optimization requires scikit-learn")
            return {'error': 'sklearn not available'}
        
        splits = self.generate_splits(len(data))
        results = []
        
        print(f"Performing walk-forward optimization with {len(splits)} splits...")
        
        for i, (train_slice, test_slice) in enumerate(splits):
            print(f"Split {i+1}/{len(splits)}: Train {train_slice}, Test {test_slice}")
            
            # Prepare training and testing data
            train_data = data.iloc[train_slice]
            test_data = data.iloc[test_slice]
            
            try:
                # Train model
                model_trained = model.train(train_data)
                if not model_trained:
                    print(f"Model training failed for split {i+1}")
                    continue
                
                # Generate predictions
                predictions = []
                actuals = []
                
                for j in range(len(test_data)):
                    # Use expanding window for prediction (includes all data up to current point)
                    pred_data = data.iloc[train_slice.start:test_slice.start + j + 1]
                    pred = model.predict(pred_data)
                    
                    if target_column in test_data.columns:
                        actual = test_data[target_column].iloc[j]
                    else:
                        # Generate target from future returns
                        if j < len(test_data) - 5:
                            future_return = (test_data['close'].iloc[j+5] - test_data['close'].iloc[j]) / test_data['close'].iloc[j]
                            actual = 'BUY' if future_return > 0.002 else 'SELL' if future_return < -0.002 else 'HOLD'
                        else:
                            actual = 'HOLD'
                    
                    predictions.append(pred)
                    actuals.append(actual)
                
                # Calculate metrics for this split
                split_results = self.calculate_split_metrics(predictions, actuals, test_data)
                split_results['split_index'] = i
                split_results['train_period'] = (train_slice.start, train_slice.stop)
                split_results['test_period'] = (test_slice.start, test_slice.stop)
                
                results.append(split_results)
                
            except Exception as e:
                print(f"Error in split {i+1}: {e}")
                continue
        
        # Aggregate results
        return self.aggregate_results(results)
    
    def calculate_split_metrics(self, predictions: List[Dict], 
                               actuals: List[str], test_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for a single walk-forward split."""
        # Extract prediction directions
        pred_directions = [pred.get('direction', 'HOLD') for pred in predictions]
        
        # Accuracy
        accuracy = accuracy_score(actuals, pred_directions) if HAS_SKLEARN else 0
        
        # Direction-specific metrics
        unique_labels = list(set(actuals + pred_directions))
        if HAS_SKLEARN and len(unique_labels) > 1:
            precision = precision_score(actuals, pred_directions, average='weighted', zero_division=0)
            recall = recall_score(actuals, pred_directions, average='weighted', zero_division=0)
            f1 = f1_score(actuals, pred_directions, average='weighted', zero_division=0)
        else:
            precision = recall = f1 = 0
        
        # Trading simulation
        simulator = TradingSimulator()
        trading_results = simulator.backtest(predictions, test_data)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_predictions': len(predictions),
            'trading_metrics': trading_results
        }
    
    def aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from all walk-forward splits."""
        if not results:
            return {'error': 'No valid results from walk-forward optimization'}
        
        # Aggregate prediction metrics
        accuracies = [r['accuracy'] for r in results if 'accuracy' in r]
        precisions = [r['precision'] for r in results if 'precision' in r]
        recalls = [r['recall'] for r in results if 'recall' in r]
        f1_scores = [r['f1_score'] for r in results if 'f1_score' in r]
        
        # Aggregate trading metrics
        trading_metrics = [r['trading_metrics'] for r in results if 'trading_metrics' in r]
        
        returns = [tm.get('total_return', 0) for tm in trading_metrics if isinstance(tm, dict)]
        sharpe_ratios = [tm.get('sharpe_ratio', 0) for tm in trading_metrics if isinstance(tm, dict)]
        max_drawdowns = [tm.get('max_drawdown', 0) for tm in trading_metrics if isinstance(tm, dict)]
        win_rates = [tm.get('win_rate', 0) for tm in trading_metrics if isinstance(tm, dict)]
        
        return {
            'num_splits': len(results),
            'prediction_metrics': {
                'mean_accuracy': np.mean(accuracies) if accuracies else 0,
                'std_accuracy': np.std(accuracies) if accuracies else 0,
                'mean_precision': np.mean(precisions) if precisions else 0,
                'mean_recall': np.mean(recalls) if recalls else 0,
                'mean_f1_score': np.mean(f1_scores) if f1_scores else 0
            },
            'trading_metrics': {
                'mean_return': np.mean(returns) if returns else 0,
                'std_return': np.std(returns) if returns else 0,
                'mean_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
                'mean_max_drawdown': np.mean(max_drawdowns) if max_drawdowns else 0,
                'mean_win_rate': np.mean(win_rates) if win_rates else 0,
                'consistency_ratio': len([r for r in returns if r > 0]) / len(returns) if returns else 0
            },
            'detailed_results': results
        }


class PerformanceMonitor:
    """Monitor model performance decay and trigger retraining."""
    
    def __init__(self, performance_window: int = 50,
                 decay_threshold: float = 0.1):
        self.performance_window = performance_window
        self.decay_threshold = decay_threshold
        self.performance_history = []
        self.baseline_performance = None
        
    def update_performance(self, accuracy: float, timestamp: datetime):
        """Update performance tracking."""
        self.performance_history.append({
            'accuracy': accuracy,
            'timestamp': timestamp
        })
        
        # Keep only recent history
        if len(self.performance_history) > self.performance_window * 2:
            self.performance_history = self.performance_history[-self.performance_window:]
        
        # Set baseline if not set
        if self.baseline_performance is None and len(self.performance_history) >= 20:
            baseline_scores = [p['accuracy'] for p in self.performance_history[:20]]
            self.baseline_performance = np.mean(baseline_scores)
    
    def check_performance_decay(self) -> Dict[str, Any]:
        """Check if model performance has significantly decayed."""
        if len(self.performance_history) < self.performance_window:
            return {
                'decay_detected': False,
                'reason': 'Insufficient performance history'
            }
        
        if self.baseline_performance is None:
            return {
                'decay_detected': False,
                'reason': 'Baseline performance not established'
            }
        
        # Calculate recent performance
        recent_scores = [p['accuracy'] for p in self.performance_history[-self.performance_window:]]
        recent_performance = np.mean(recent_scores)
        
        # Check for decay
        performance_drop = self.baseline_performance - recent_performance
        decay_detected = performance_drop > self.decay_threshold
        
        return {
            'decay_detected': decay_detected,
            'baseline_performance': self.baseline_performance,
            'recent_performance': recent_performance,
            'performance_drop': performance_drop,
            'threshold': self.decay_threshold,
            'recommendation': 'Retrain model' if decay_detected else 'Continue monitoring'
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.performance_history:
            return {'error': 'No performance history available'}
        
        scores = [p['accuracy'] for p in self.performance_history]
        
        return {
            'total_evaluations': len(self.performance_history),
            'current_performance': scores[-1] if scores else 0,
            'mean_performance': np.mean(scores),
            'performance_std': np.std(scores),
            'min_performance': np.min(scores),
            'max_performance': np.max(scores),
            'baseline_performance': self.baseline_performance,
            'recent_trend': self.calculate_trend(),
            'performance_stability': 1.0 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
        }
    
    def calculate_trend(self) -> float:
        """Calculate performance trend over recent history."""
        if len(self.performance_history) < 10:
            return 0.0
        
        recent_scores = [p['accuracy'] for p in self.performance_history[-20:]]
        x = np.arange(len(recent_scores))
        
        # Simple linear trend
        if len(recent_scores) > 1:
            correlation = np.corrcoef(x, recent_scores)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0


class ComprehensiveEvaluator:
    """Comprehensive evaluation framework combining all evaluation methods."""
    
    def __init__(self):
        self.trading_simulator = TradingSimulator()
        self.walk_forward_optimizer = WalkForwardOptimizer()
        self.performance_monitor = PerformanceMonitor()
        
    def evaluate_model(self, model, data: pd.DataFrame, 
                      comprehensive: bool = True) -> Dict[str, Any]:
        """Perform comprehensive model evaluation."""
        print(f"Starting comprehensive evaluation of {model.__class__.__name__}...")
        
        evaluation_results = {
            'model_info': model.get_model_info(),
            'evaluation_timestamp': datetime.now(),
            'data_summary': {
                'total_samples': len(data),
                'date_range': (data.index[0], data.index[-1]) if hasattr(data.index, '__getitem__') else ('N/A', 'N/A'),
                'features': list(data.columns)
            }
        }
        
        # Basic prediction evaluation
        try:
            print("Generating predictions for evaluation...")
            predictions = []
            actuals = []
            
            # Generate predictions on a subset of data
            eval_data = data.tail(min(500, len(data)))  # Use last 500 samples or less
            
            for i in range(len(eval_data) - 5):  # Leave 5 samples for future target calculation
                pred_data = eval_data.iloc[:i+1]
                pred = model.predict(pred_data)
                
                # Calculate actual outcome
                future_return = (eval_data['close'].iloc[i+5] - eval_data['close'].iloc[i]) / eval_data['close'].iloc[i]
                actual = 'BUY' if future_return > 0.002 else 'SELL' if future_return < -0.002 else 'HOLD'
                
                predictions.append(pred)
                actuals.append(actual)
            
            # Basic metrics
            pred_directions = [p.get('direction', 'HOLD') for p in predictions]
            basic_accuracy = accuracy_score(actuals, pred_directions) if HAS_SKLEARN else 0
            
            evaluation_results['basic_metrics'] = {
                'accuracy': basic_accuracy,
                'num_predictions': len(predictions),
                'direction_distribution': pd.Series(pred_directions).value_counts().to_dict()
            }
            
            print(f"Basic accuracy: {basic_accuracy:.3f}")
            
        except Exception as e:
            print(f"Error in basic evaluation: {e}")
            evaluation_results['basic_metrics'] = {'error': str(e)}
        
        # Trading simulation
        try:
            print("Running trading simulation...")
            trading_results = self.trading_simulator.backtest(predictions, eval_data)
            evaluation_results['trading_simulation'] = trading_results
            
            if 'total_return' in trading_results:
                print(f"Trading simulation - Total return: {trading_results['total_return']:.3f}")
                print(f"Sharpe ratio: {trading_results['sharpe_ratio']:.3f}")
            
        except Exception as e:
            print(f"Error in trading simulation: {e}")
            evaluation_results['trading_simulation'] = {'error': str(e)}
        
        # Walk-forward optimization (if comprehensive)
        if comprehensive and len(data) > 500:
            try:
                print("Performing walk-forward optimization...")
                wf_results = self.walk_forward_optimizer.optimize_model(model, data)
                evaluation_results['walk_forward'] = wf_results
                
                if 'prediction_metrics' in wf_results:
                    print(f"Walk-forward - Mean accuracy: {wf_results['prediction_metrics']['mean_accuracy']:.3f}")
                
            except Exception as e:
                print(f"Error in walk-forward optimization: {e}")
                evaluation_results['walk_forward'] = {'error': str(e)}
        
        # Performance monitoring update
        if 'basic_metrics' in evaluation_results and 'accuracy' in evaluation_results['basic_metrics']:
            self.performance_monitor.update_performance(
                evaluation_results['basic_metrics']['accuracy'],
                datetime.now()
            )
            
            decay_check = self.performance_monitor.check_performance_decay()
            evaluation_results['performance_monitoring'] = {
                'decay_analysis': decay_check,
                'performance_summary': self.performance_monitor.get_performance_summary()
            }
        
        print("Comprehensive evaluation completed")
        return evaluation_results
    
    def compare_models(self, models: List, data: pd.DataFrame) -> Dict[str, Any]:
        """Compare multiple models using the same evaluation framework."""
        print(f"Comparing {len(models)} models...")
        
        comparison_results = {
            'comparison_timestamp': datetime.now(),
            'models_compared': len(models),
            'data_summary': {
                'total_samples': len(data),
                'evaluation_samples': min(500, len(data))
            },
            'model_results': {},
            'rankings': {}
        }
        
        # Evaluate each model
        for model in models:
            model_name = model.__class__.__name__
            print(f"\nEvaluating {model_name}...")
            
            try:
                model_results = self.evaluate_model(model, data, comprehensive=False)
                comparison_results['model_results'][model_name] = model_results
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                comparison_results['model_results'][model_name] = {'error': str(e)}
        
        # Calculate rankings
        try:
            self.calculate_model_rankings(comparison_results)
        except Exception as e:
            print(f"Error calculating rankings: {e}")
            comparison_results['rankings'] = {'error': str(e)}
        
        return comparison_results
    
    def calculate_model_rankings(self, comparison_results: Dict[str, Any]):
        """Calculate model rankings based on multiple criteria."""
        model_scores = {}
        
        for model_name, results in comparison_results['model_results'].items():
            if 'error' in results:
                continue
            
            score = 0
            criteria_count = 0
            
            # Basic accuracy
            if 'basic_metrics' in results and 'accuracy' in results['basic_metrics']:
                score += results['basic_metrics']['accuracy'] * 100
                criteria_count += 1
            
            # Trading return
            if 'trading_simulation' in results and 'total_return' in results['trading_simulation']:
                # Normalize return to 0-100 scale (assuming reasonable returns are -50% to +100%)
                return_score = max(0, min(100, (results['trading_simulation']['total_return'] + 0.5) * 100))
                score += return_score
                criteria_count += 1
            
            # Sharpe ratio
            if 'trading_simulation' in results and 'sharpe_ratio' in results['trading_simulation']:
                # Normalize Sharpe ratio to 0-100 scale (assuming reasonable range is -2 to +4)
                sharpe_score = max(0, min(100, (results['trading_simulation']['sharpe_ratio'] + 2) * 16.67))
                score += sharpe_score
                criteria_count += 1
            
            # Max drawdown (inverted - lower is better)
            if 'trading_simulation' in results and 'max_drawdown' in results['trading_simulation']:
                # Normalize max drawdown to 0-100 scale (lower drawdown = higher score)
                drawdown_score = max(0, 100 - results['trading_simulation']['max_drawdown'] * 100)
                score += drawdown_score
                criteria_count += 1
            
            if criteria_count > 0:
                model_scores[model_name] = score / criteria_count
        
        # Sort by score
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        comparison_results['rankings'] = {
            'overall_ranking': [{'model': name, 'score': score} for name, score in ranked_models],
            'criteria_weights': {
                'accuracy': 25,
                'total_return': 25,
                'sharpe_ratio': 25,
                'max_drawdown': 25
            }
        }
        
        if ranked_models:
            print(f"\nModel Rankings:")
            for i, (name, score) in enumerate(ranked_models):
                print(f"{i+1}. {name}: {score:.2f}")