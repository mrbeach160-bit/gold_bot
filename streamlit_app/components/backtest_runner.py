"""
Backtest Runner Component
Handles strategy backtesting and historical performance analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

class BacktestRunner:
    """Component for running strategy backtests and performance analysis"""
    
    def __init__(self):
        self.backtest_results = None
        self.trade_history = []
        
    def run_backtest(self, symbol, data, initial_balance, risk_percent, sl_pips, tp_pips, 
                     predict_func, all_models, api_source, api_key_1=None, api_key_2=None, 
                     use_ai_tp=False):
        """
        Run comprehensive backtest with smart entry logic
        """
        try:
            if data is None or data.empty:
                st.error("❌ No data available for backtesting")
                return None
            
            # Initialize backtest parameters
            balance = initial_balance
            trades = []
            active_trade = None
            equity_curve = [balance]
            trade_count = 0
            winning_trades = 0
            losing_trades = 0
            
            st.info(f"🔄 Running backtest on {len(data)} data points...")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Backtest loop
            for i in range(100, len(data)):  # Start from 100 to have enough history
                progress = i / len(data)
                progress_bar.progress(progress)
                status_text.text(f"Processing candle {i}/{len(data)} - Balance: ${balance:.2f}")
                
                current_candle = data.iloc[i]
                recent_data = data.iloc[max(0, i-100):i+1]  # Last 100 candles for analysis
                
                # Generate prediction and signal
                try:
                    if predict_func and all_models:
                        prediction_result = predict_func(all_models, recent_data)
                        if prediction_result:
                            signal = prediction_result.get('ensemble_signal', 'HOLD')
                            confidence = prediction_result.get('ensemble_confidence', 0.5)
                            predicted_price = prediction_result.get('ensemble_prediction', current_candle['close'])
                        else:
                            signal, confidence, predicted_price = 'HOLD', 0.5, current_candle['close']
                    else:
                        # Fallback to simple signal generation
                        signal, confidence, predicted_price = self._generate_simple_signal(recent_data)
                
                except Exception as e:
                    signal, confidence, predicted_price = 'HOLD', 0.5, current_candle['close']
                
                # Handle active trade exit
                if active_trade:
                    exit_result = self._execute_trade_exit(current_candle, active_trade)
                    if exit_result:
                        trade_result = exit_result
                        trade_result['exit_time'] = current_candle.name
                        trade_result['duration'] = i - active_trade['entry_index']
                        
                        trades.append(trade_result)
                        balance += trade_result['pnl']
                        equity_curve.append(balance)
                        
                        if trade_result['pnl'] > 0:
                            winning_trades += 1
                        else:
                            losing_trades += 1
                        
                        active_trade = None
                        trade_count += 1
                
                # Enter new trade if no active trade and valid signal
                if not active_trade and signal in ['BUY', 'SELL'] and confidence > 0.6:
                    # Calculate smart entry using the trading panel logic
                    from .trading_panel import TradingPanel
                    trading_panel = TradingPanel()
                    
                    smart_entry_result = trading_panel.calculate_smart_entry_price(
                        signal, recent_data, predicted_price, confidence, symbol
                    )
                    
                    if smart_entry_result['risk_level'] != 'REJECTED':
                        entry_price = smart_entry_result['entry_price']
                        
                        # Calculate position info
                        position_info = trading_panel.calculate_position_info(
                            signal, symbol, entry_price, sl_pips, tp_pips,
                            balance, risk_percent, conversion_rate_to_usd=1.0,
                            leverage=20
                        )
                        
                        if position_info:
                            active_trade = {
                                'signal': signal,
                                'entry_price': entry_price,
                                'entry_time': current_candle.name,
                                'entry_index': i,
                                'stop_loss': position_info['stop_loss'],
                                'take_profit': position_info['take_profit'],
                                'position_size': position_info['position_size'],
                                'confidence': confidence,
                                'risk_amount': position_info['risk_amount']
                            }
            
            # Close any remaining active trade
            if active_trade:
                final_candle = data.iloc[-1]
                exit_result = self._execute_trade_exit(final_candle, active_trade, force_exit=True)
                if exit_result:
                    exit_result['exit_time'] = final_candle.name
                    exit_result['duration'] = len(data) - 1 - active_trade['entry_index']
                    trades.append(exit_result)
                    balance += exit_result['pnl']
                    equity_curve.append(balance)
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(
                trades, initial_balance, balance, winning_trades, losing_trades
            )
            
            progress_bar.progress(1.0)
            status_text.text(f"Backtest completed - Final Balance: ${balance:.2f}")
            
            self.backtest_results = {
                'trades': trades,
                'equity_curve': equity_curve,
                'performance': performance,
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_return': ((balance - initial_balance) / initial_balance) * 100
            }
            
            return self.backtest_results
            
        except Exception as e:
            st.error(f"❌ Backtest error: {e}")
            return None
    
    def _generate_simple_signal(self, data):
        """Generate simple signal for fallback"""
        if len(data) < 20:
            return 'HOLD', 0.5, data['close'].iloc[-1]
        
        # Simple moving average crossover
        short_ma = data['close'].rolling(5).mean().iloc[-1]
        long_ma = data['close'].rolling(20).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        if short_ma > long_ma:
            return 'BUY', 0.7, current_price * 1.001
        elif short_ma < long_ma:
            return 'SELL', 0.7, current_price * 0.999
        else:
            return 'HOLD', 0.5, current_price
    
    def _execute_trade_exit(self, current_candle, active_trade, force_exit=False):
        """Execute trade exit logic"""
        current_price = current_candle['close']
        signal = active_trade['signal']
        entry_price = active_trade['entry_price']
        stop_loss = active_trade['stop_loss']
        take_profit = active_trade['take_profit']
        position_size = active_trade['position_size']
        
        exit_reason = None
        exit_price = current_price
        
        if force_exit:
            exit_reason = "Force Exit"
        elif signal == 'BUY':
            if current_price <= stop_loss:
                exit_reason = "Stop Loss"
                exit_price = stop_loss
            elif current_price >= take_profit:
                exit_reason = "Take Profit" 
                exit_price = take_profit
        elif signal == 'SELL':
            if current_price >= stop_loss:
                exit_reason = "Stop Loss"
                exit_price = stop_loss
            elif current_price <= take_profit:
                exit_reason = "Take Profit"
                exit_price = take_profit
        
        if exit_reason:
            # Calculate P&L
            if signal == 'BUY':
                pnl = (exit_price - entry_price) * position_size
            else:  # SELL
                pnl = (entry_price - exit_price) * position_size
            
            return {
                'signal': signal,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'position_size': position_size,
                'pnl': pnl,
                'return_pct': (pnl / active_trade['risk_amount']) * 100 if active_trade['risk_amount'] > 0 else 0
            }
        
        return None
    
    def _calculate_performance_metrics(self, trades, initial_balance, final_balance, 
                                     winning_trades, losing_trades):
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {}
        
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate average win/loss
        winning_pnls = [t['pnl'] for t in trades if t['pnl'] > 0]
        losing_pnls = [t['pnl'] for t in trades if t['pnl'] < 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        
        # Risk-Reward Ratio
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Maximum drawdown
        equity_curve = [initial_balance]
        running_balance = initial_balance
        peak = initial_balance
        max_drawdown = 0
        
        for trade in trades:
            running_balance += trade['pnl']
            equity_curve.append(running_balance)
            
            if running_balance > peak:
                peak = running_balance
            
            drawdown = ((peak - running_balance) / peak) * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio (simplified)
        returns = [t['return_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'risk_reward_ratio': risk_reward,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_return': ((final_balance - initial_balance) / initial_balance) * 100
        }
    
    def render_backtest_controls(self, symbol, initial_balance=1000):
        """Render backtest configuration controls"""
        st.subheader("⚙️ Backtest Configuration")
        
        config_cols = st.columns(2)
        
        with config_cols[0]:
            st.markdown("**💰 Capital Settings**")
            balance = st.number_input("Initial Balance ($)", min_value=100, value=initial_balance, step=100)
            risk_percent = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1)
            leverage = st.selectbox("Leverage", [1, 5, 10, 20, 50], index=3)
        
        with config_cols[1]:
            st.markdown("**📊 Risk Management**")
            sl_pips = st.number_input("Stop Loss (pips)", min_value=5, value=20, step=5)
            tp_pips = st.number_input("Take Profit (pips)", min_value=5, value=40, step=5)
            use_ai_tp = st.checkbox("Use AI Take Profit", value=True)
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings", expanded=False):
            adv_cols = st.columns(2)
            
            with adv_cols[0]:
                min_confidence = st.slider("Minimum Confidence", 0.5, 0.9, 0.6, 0.05)
                max_trades = st.number_input("Max Concurrent Trades", min_value=1, value=1, max_value=5)
            
            with adv_cols[1]:
                commission = st.number_input("Commission ($)", min_value=0.0, value=2.0, step=0.5)
                slippage = st.number_input("Slippage (pips)", min_value=0, value=1, step=1)
        
        return {
            'initial_balance': balance,
            'risk_percent': risk_percent,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips,
            'use_ai_tp': use_ai_tp,
            'leverage': leverage,
            'min_confidence': min_confidence,
            'max_trades': max_trades,
            'commission': commission,
            'slippage': slippage
        }
    
    def render_backtest_results(self):
        """Render comprehensive backtest results"""
        if not self.backtest_results:
            st.info("No backtest results available. Run a backtest first.")
            return
        
        results = self.backtest_results
        performance = results['performance']
        
        st.subheader("📊 Backtest Results")
        
        # Summary metrics
        summary_cols = st.columns(4)
        
        with summary_cols[0]:
            st.metric(
                "Total Return",
                f"{results['total_return']:.2f}%",
                delta=f"${results['final_balance'] - results['initial_balance']:.2f}"
            )
        
        with summary_cols[1]:
            st.metric(
                "Win Rate",
                f"{performance.get('win_rate', 0):.1f}%",
                delta=f"{performance.get('winning_trades', 0)}/{performance.get('total_trades', 0)} trades"
            )
        
        with summary_cols[2]:
            st.metric(
                "Risk-Reward",
                f"{performance.get('risk_reward_ratio', 0):.2f}",
                delta=f"Avg Win: ${performance.get('avg_win', 0):.2f}"
            )
        
        with summary_cols[3]:
            st.metric(
                "Max Drawdown",
                f"{performance.get('max_drawdown', 0):.2f}%",
                delta=f"Sharpe: {performance.get('sharpe_ratio', 0):.2f}"
            )
        
        # Equity curve chart
        self._render_equity_curve()
        
        # Trade analysis
        self._render_trade_analysis()
    
    def _render_equity_curve(self):
        """Render equity curve chart"""
        if not self.backtest_results:
            return
        
        equity_curve = self.backtest_results['equity_curve']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=equity_curve,
            mode='lines',
            name='Equity Curve',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_hline(
            y=self.backtest_results['initial_balance'],
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Balance"
        )
        
        fig.update_layout(
            title="📈 Equity Curve",
            xaxis_title="Trade Number",
            yaxis_title="Balance ($)",
            height=400,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_trade_analysis(self):
        """Render detailed trade analysis"""
        if not self.backtest_results or not self.backtest_results['trades']:
            return
        
        trades_df = pd.DataFrame(self.backtest_results['trades'])
        
        with st.expander("📋 Trade Details", expanded=False):
            # Trade statistics
            stat_cols = st.columns(3)
            
            with stat_cols[0]:
                st.markdown("**📊 Trade Distribution**")
                buy_trades = len(trades_df[trades_df['signal'] == 'BUY'])
                sell_trades = len(trades_df[trades_df['signal'] == 'SELL'])
                st.write(f"BUY trades: {buy_trades}")
                st.write(f"SELL trades: {sell_trades}")
            
            with stat_cols[1]:
                st.markdown("**💰 P&L Analysis**")
                total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
                total_loss = trades_df[trades_df['pnl'] < 0]['pnl'].sum()
                st.write(f"Total Profit: ${total_profit:.2f}")
                st.write(f"Total Loss: ${total_loss:.2f}")
            
            with stat_cols[2]:
                st.markdown("**⏱️ Duration Analysis**")
                if 'duration' in trades_df.columns:
                    avg_duration = trades_df['duration'].mean()
                    st.write(f"Avg Duration: {avg_duration:.1f} candles")
                    st.write(f"Max Duration: {trades_df['duration'].max()} candles")
            
            # Trade table
            st.markdown("**📋 Trade History**")
            display_df = trades_df.copy()
            display_df['pnl'] = display_df['pnl'].round(2)
            display_df['return_pct'] = display_df['return_pct'].round(2)
            
            st.dataframe(
                display_df[['signal', 'entry_price', 'exit_price', 'exit_reason', 'pnl', 'return_pct']],
                use_container_width=True
            )
    
    def export_backtest_results(self, format='csv'):
        """Export backtest results"""
        if not self.backtest_results:
            return None
        
        trades_df = pd.DataFrame(self.backtest_results['trades'])
        
        if format == 'csv':
            return trades_df.to_csv(index=False)
        elif format == 'json':
            return trades_df.to_json(orient='records', date_format='iso')
        else:
            return trades_df
    
    def render_export_controls(self):
        """Render export controls for backtest results"""
        if self.backtest_results:
            with st.expander("💾 Export Results", expanded=False):
                export_format = st.selectbox("Export Format", ["CSV", "JSON"], index=0)
                
                if st.button("📥 Download Results"):
                    format_lower = export_format.lower()
                    data = self.export_backtest_results(format_lower)
                    
                    if data:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"backtest_results_{timestamp}.{format_lower}"
                        
                        st.download_button(
                            label=f"Download {export_format}",
                            data=data,
                            file_name=filename,
                            mime=f"text/{format_lower}" if format_lower == 'csv' else 'application/json'
                        )