"""
Backtest Panel for the modular application.
Handles backtesting interface and results display.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Optional


class BacktestPanel:
    """Panel for backtesting interface and results."""
    
    def __init__(self, backtest_service, data_service, feature_service):
        self.backtest_service = backtest_service
        self.data_service = data_service
        self.feature_service = feature_service
    
    def render(self, symbol: str, timeframe: str) -> None:
        """
        Render the backtest panel interface.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        """
        st.header("📈 Backtesting")
        
        # Backtest configuration
        backtest_config = self._render_backtest_config()
        
        # Data preparation
        backtest_data = self._render_backtest_data_section(symbol, timeframe, backtest_config)
        
        # Backtest execution
        if backtest_data is not None and not backtest_data.empty:
            self._render_backtest_execution(backtest_data, symbol, timeframe, backtest_config)
        else:
            st.warning("⚠️ No data available for backtesting. Please fetch historical data first.")
    
    def _render_backtest_config(self) -> Dict[str, Any]:
        """Render backtest configuration controls."""
        st.subheader("⚙️ Backtest Configuration")
        
        # Basic settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            initial_balance = st.number_input(
                "Initial Balance ($)",
                min_value=1000.0,
                max_value=1000000.0,
                value=10000.0,
                step=1000.0,
                help="Starting capital for backtest"
            )
        
        with col2:
            risk_per_trade = st.slider(
                "Risk per Trade (%)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1,
                help="Risk percentage per individual trade"
            )
        
        with col3:
            history_size = st.number_input(
                "History Size",
                min_value=500,
                max_value=5000,
                value=1500,
                step=100,
                help="Number of historical bars for backtesting"
            )
        
        # Risk management settings
        st.subheader("🛡️ Risk Management")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sl_percentage = st.number_input(
                "Stop Loss (%)",
                min_value=0.1,
                max_value=5.0,
                value=0.5,
                step=0.1,
                help="Stop loss percentage"
            )
        
        with col2:
            tp_percentage = st.number_input(
                "Take Profit (%)",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Take profit percentage"
            )
        
        with col3:
            use_atr_sl_tp = st.checkbox(
                "Use ATR-based SL/TP",
                value=False,
                help="Use ATR (Average True Range) for stop loss and take profit levels"
            )
        
        with col4:
            evaluation_freq = st.number_input(
                "Evaluation Frequency",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                help="Make predictions every N bars"
            )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                confidence_threshold = st.slider(
                    "Signal Confidence Threshold",
                    min_value=0.5,
                    max_value=0.9,
                    value=0.55,
                    step=0.05,
                    help="Minimum ensemble confidence to enter trades"
                )
                
                max_trades = st.number_input(
                    "Maximum Trades",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    help="Maximum number of trades to execute"
                )
            
            with col2:
                commission = st.number_input(
                    "Commission per Trade ($)",
                    min_value=0.0,
                    max_value=50.0,
                    value=5.0,
                    step=0.5,
                    help="Trading commission per trade"
                )
                
                slippage = st.number_input(
                    "Slippage (pips)",
                    min_value=0.0,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    help="Expected slippage in pips"
                )
        
        return {
            'initial_balance': initial_balance,
            'risk_per_trade': risk_per_trade,
            'history_size': history_size,
            'sl_percentage': sl_percentage,
            'tp_percentage': tp_percentage,
            'use_atr_sl_tp': use_atr_sl_tp,
            'evaluation_freq': evaluation_freq,
            'confidence_threshold': confidence_threshold,
            'max_trades': max_trades,
            'commission': commission,
            'slippage': slippage
        }
    
    def _render_backtest_data_section(self, symbol: str, timeframe: str, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Render backtest data fetching section."""
        st.subheader("📊 Historical Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Fetch Backtest Data", type="primary", use_container_width=True):
                return self._fetch_backtest_data(symbol, timeframe, config)
        
        with col2:
            api_key = st.text_input(
                "API Key (Optional)",
                type="password",
                help="Twelve Data API key"
            )
            if api_key:
                config['api_key'] = api_key
        
        with col3:
            if st.button("🔄 Refresh Data", use_container_width=True):
                data_key = f"backtest_data_{symbol}_{timeframe}"
                if data_key in st.session_state:
                    del st.session_state[data_key]
                return self._fetch_backtest_data(symbol, timeframe, config)
        
        # Check for cached data
        data_key = f"backtest_data_{symbol}_{timeframe}"
        if data_key in st.session_state:
            backtest_data = st.session_state[data_key]
            
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Points", len(backtest_data))
            with col2:
                price_range = f"{backtest_data['low'].min():.2f} - {backtest_data['high'].max():.2f}"
                st.metric("Price Range", price_range)
            with col3:
                try:
                    date_range = f"{backtest_data.index[0].strftime('%Y-%m-%d')} to {backtest_data.index[-1].strftime('%Y-%m-%d')}"
                except:
                    date_range = "N/A"
                st.metric("Date Range", date_range)
            with col4:
                validity = "✅ Valid" if self.data_service.validate_data(backtest_data, 500) else "❌ Invalid"
                st.metric("Data Validity", validity)
            
            return backtest_data
        
        return None
    
    def _fetch_backtest_data(self, symbol: str, timeframe: str, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Fetch and prepare data for backtesting."""
        try:
            with st.spinner("📥 Fetching historical data for backtesting..."):
                # Fetch raw data
                raw_data = self.data_service.fetch_data(
                    symbol, timeframe, config['history_size'], config.get('api_key')
                )
                
                if raw_data is None or raw_data.empty:
                    st.error("Failed to fetch historical data")
                    return None
                
                # Add technical indicators
                with st.spinner("🔧 Adding technical indicators..."):
                    data_with_indicators = self.feature_service.add_technical_indicators(raw_data)
                
                # Add engineered features
                with st.spinner("⚙️ Engineering features..."):
                    final_data = self.feature_service.add_engineered_features(data_with_indicators)
                
                # Validate data
                if not self.data_service.validate_data(final_data, 500):
                    st.error("Backtest data validation failed")
                    return None
                
                # Cache data
                data_key = f"backtest_data_{symbol}_{timeframe}"
                st.session_state[data_key] = final_data
                
                st.success(f"✅ Successfully prepared {len(final_data)} bars for backtesting")
                return final_data
                
        except Exception as e:
            st.error(f"Error fetching backtest data: {str(e)}")
            return None
    
    def _render_backtest_execution(self, data: pd.DataFrame, symbol: str, timeframe: str, config: Dict[str, Any]) -> None:
        """Render backtest execution section."""
        st.subheader("🚀 Backtest Execution")
        
        # Update prediction service confidence threshold
        self.backtest_service.prediction_service.ensemble_threshold = config['confidence_threshold']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Run Backtest", type="primary", use_container_width=True):
                self._execute_backtest(data, symbol, timeframe, config)
        
        with col2:
            if st.button("🧹 Clear Results", use_container_width=True):
                result_key = f"backtest_results_{symbol}_{timeframe}"
                if result_key in st.session_state:
                    del st.session_state[result_key]
                st.rerun()
        
        # Display results if available
        result_key = f"backtest_results_{symbol}_{timeframe}"
        if result_key in st.session_state:
            results = st.session_state[result_key]
            self._display_backtest_results(results, symbol, timeframe)
    
    def _execute_backtest(self, data: pd.DataFrame, symbol: str, timeframe: str, config: Dict[str, Any]) -> None:
        """Execute the backtest."""
        try:
            st.info("📈 Starting backtest execution...")
            
            # Run backtest
            results = self.backtest_service.run_backtest(
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                initial_balance=config['initial_balance'],
                risk_per_trade=config['risk_per_trade'],
                sl_percentage=config['sl_percentage'],
                tp_percentage=config['tp_percentage'],
                use_atr_sl_tp=config['use_atr_sl_tp'],
                evaluation_freq=config['evaluation_freq']
            )
            
            # Store results
            result_key = f"backtest_results_{symbol}_{timeframe}"
            st.session_state[result_key] = results
            
            # Display results
            self._display_backtest_results(results, symbol, timeframe)
            
        except Exception as e:
            st.error(f"❌ Backtest execution failed: {str(e)}")
    
    def _display_backtest_results(self, results: Dict[str, Any], symbol: str, timeframe: str) -> None:
        """Display backtest results."""
        if not results.get('success', False):
            st.error(f"❌ Backtest failed: {results.get('message', 'Unknown error')}")
            return
        
        st.success("✅ Backtest completed successfully!")
        
        # Key metrics
        self._render_key_metrics(results)
        
        # Equity curve
        self._render_equity_curve(results)
        
        # Trade analysis
        self._render_trade_analysis(results)
        
        # Performance details
        self._render_performance_details(results)
    
    def _render_key_metrics(self, results: Dict[str, Any]) -> None:
        """Render key performance metrics."""
        st.subheader("📊 Key Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Return",
                f"{results['total_return_pct']:.2f}%",
                delta=f"{results['total_return_pct']:.2f}%"
            )
        
        with col2:
            st.metric(
                "Final Balance",
                f"${results['final_balance']:,.2f}",
                delta=f"${results['final_balance'] - results['initial_balance']:,.2f}"
            )
        
        with col3:
            st.metric(
                "Total Trades",
                results['total_trades'],
                delta=None
            )
        
        with col4:
            st.metric(
                "Win Rate",
                f"{results['win_rate']:.1f}%",
                delta=None
            )
        
        with col5:
            st.metric(
                "Profit Factor",
                f"{results['profit_factor']:.2f}",
                delta=None
            )
        
        # Additional metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Max Drawdown",
                f"{results['max_drawdown']:.2f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                "Winning Trades",
                results['winning_trades'],
                delta=None
            )
        
        with col3:
            st.metric(
                "Losing Trades",
                results['losing_trades'],
                delta=None
            )
        
        with col4:
            st.metric(
                "Avg Win",
                f"${results['avg_win']:,.2f}",
                delta=None
            )
        
        with col5:
            st.metric(
                "Avg Loss",
                f"${results['avg_loss']:,.2f}",
                delta=None
            )
    
    def _render_equity_curve(self, results: Dict[str, Any]) -> None:
        """Render equity curve chart."""
        st.subheader("📈 Equity Curve")
        
        equity_curve = results.get('equity_curve', [])
        if not equity_curve:
            st.warning("No equity curve data available")
            return
        
        try:
            # Create equity curve chart
            fig = go.Figure()
            
            times = [point['time'] for point in equity_curve]
            equity_values = [point['equity'] for point in equity_curve]
            
            fig.add_trace(go.Scatter(
                x=times,
                y=equity_values,
                mode='lines',
                name='Portfolio Equity',
                line=dict(color='blue', width=2)
            ))
            
            # Add initial balance line
            fig.add_hline(
                y=results['initial_balance'],
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Balance"
            )
            
            fig.update_layout(
                title="Portfolio Equity Over Time",
                xaxis_title="Time",
                yaxis_title="Equity ($)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering equity curve: {str(e)}")
    
    def _render_trade_analysis(self, results: Dict[str, Any]) -> None:
        """Render trade analysis section."""
        st.subheader("💼 Trade Analysis")
        
        trades = results.get('trades', [])
        if not trades:
            st.info("No trades were executed during the backtest period")
            return
        
        # Trade summary table
        st.write("**Sample Trades (First 10):**")
        
        trade_data = []
        for i, trade in enumerate(trades[:10]):
            trade_data.append({
                'Trade #': i + 1,
                'Side': trade['side'],
                'Entry Price': f"${trade['entry_price']:.2f}",
                'Exit Price': f"${trade['exit_price']:.2f}",
                'P&L': f"${trade['pnl']:,.2f}",
                'Exit Reason': trade['exit_reason'],
                'Duration (bars)': trade['duration_bars']
            })
        
        if trade_data:
            df = pd.DataFrame(trade_data)
            
            # Style the dataframe
            def style_pnl(val):
                if '$-' in val:
                    return 'background-color: #f8d7da; color: #721c24;'
                elif '$' in val and '-' not in val:
                    return 'background-color: #d4edda; color: #155724;'
                return ''
            
            def style_side(val):
                if val == 'BUY':
                    return 'background-color: #d4edda; color: #155724;'
                elif val == 'SELL':
                    return 'background-color: #f8d7da; color: #721c24;'
                return ''
            
            styled_df = df.style.applymap(style_pnl, subset=['P&L']).applymap(style_side, subset=['Side'])
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Trade distribution chart
        if len(trades) > 1:
            with st.expander("📊 P&L Distribution", expanded=False):
                pnl_values = [trade['pnl'] for trade in trades]
                
                fig = go.Figure(data=[go.Histogram(x=pnl_values, nbinsx=20)])
                fig.update_layout(
                    title="Trade P&L Distribution",
                    xaxis_title="P&L ($)",
                    yaxis_title="Number of Trades",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_details(self, results: Dict[str, Any]) -> None:
        """Render detailed performance analysis."""
        with st.expander("📋 Detailed Performance Analysis", expanded=False):
            
            # Risk metrics
            st.write("**Risk Metrics:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"• Max Drawdown: {results['max_drawdown']:.2f}%")
                st.write(f"• Profit Factor: {results['profit_factor']:.2f}")
                st.write(f"• Win Rate: {results['win_rate']:.1f}%")
            
            with col2:
                st.write(f"• Total Trades: {results['total_trades']}")
                st.write(f"• Evaluation Frequency: {results.get('evaluation_frequency', 'N/A')} bars")
                st.write(f"• Initial Balance: ${results['initial_balance']:,.2f}")
            
            # Strategy performance
            st.write("**Strategy Performance:**")
            total_return = results['total_return_pct']
            if total_return > 0:
                st.success(f"✅ Profitable strategy with {total_return:.2f}% return")
            else:
                st.error(f"❌ Losing strategy with {total_return:.2f}% return")
            
            # Export results
            st.write("**Export Results:**")
            if st.button("📥 Download Backtest Results"):
                self._export_backtest_results(results)
    
    def _export_backtest_results(self, results: Dict[str, Any]) -> None:
        """Export backtest results as JSON."""
        try:
            import json
            
            # Prepare export data
            export_data = {
                'backtest_summary': {
                    'initial_balance': results['initial_balance'],
                    'final_balance': results['final_balance'],
                    'total_return_pct': results['total_return_pct'],
                    'total_trades': results['total_trades'],
                    'win_rate': results['win_rate'],
                    'profit_factor': results['profit_factor'],
                    'max_drawdown': results['max_drawdown']
                },
                'trades': results.get('trades', []),
                'equity_curve': results.get('equity_curve', [])
            }
            
            # Convert to JSON string
            json_str = json.dumps(export_data, indent=2, default=str)
            
            # Provide download
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json_str,
                file_name="backtest_results.json",
                mime="application/json"
            )
            
            st.success("✅ Backtest results ready for download")
            
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")