"""
Analytics Panel for the modular application.
Provides analytics and performance insights.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, List


class AnalyticsPanel:
    """Panel for analytics and performance insights."""
    
    def __init__(self, model_registry, prediction_service, risk_service):
        self.model_registry = model_registry
        self.prediction_service = prediction_service
        self.risk_service = risk_service
    
    def render(self, symbol: str, timeframe: str) -> None:
        """
        Render the analytics panel interface.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        """
        st.header("📊 Analytics & Insights")
        
        # Analytics navigation
        analytics_tabs = st.tabs([
            "System Overview",
            "Model Performance", 
            "Trading Analytics",
            "Risk Analysis",
            "Data Insights"
        ])
        
        with analytics_tabs[0]:
            self._render_system_overview(symbol, timeframe)
        
        with analytics_tabs[1]:
            self._render_model_performance(symbol, timeframe)
        
        with analytics_tabs[2]:
            self._render_trading_analytics(symbol, timeframe)
        
        with analytics_tabs[3]:
            self._render_risk_analysis(symbol, timeframe)
        
        with analytics_tabs[4]:
            self._render_data_insights(symbol, timeframe)
    
    def _render_system_overview(self, symbol: str, timeframe: str) -> None:
        """Render system overview analytics."""
        st.subheader("🖥️ System Overview")
        
        # System status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Services Status**")
            services = [
                ("Model Registry", hasattr(st.session_state, 'model_registry')),
                ("Prediction Service", hasattr(st.session_state, 'prediction_service')),
                ("Training Service", hasattr(st.session_state, 'training_service')),
                ("Backtest Service", hasattr(st.session_state, 'backtest_service')),
                ("Risk Service", hasattr(st.session_state, 'risk_service'))
            ]
            
            for service_name, available in services:
                status = "✅" if available else "❌"
                st.write(f"{status} {service_name}")
        
        with col2:
            st.write("**Model Availability**")
            try:
                available, total = self.model_registry.count_available_models(symbol, timeframe)
                st.metric("Models Available", f"{available}/{total}")
                st.metric("Availability Rate", f"{(available/total*100):.1f}%" if total > 0 else "0%")
                
                # Model types breakdown
                metadata = self.model_registry.scan(symbol, timeframe)
                model_status = {}
                for model_type, info in metadata.items():
                    if model_type in ['lstm', 'xgb', 'cnn', 'svc', 'nb']:
                        model_status[model_type.upper()] = "✅" if info['exists'] else "❌"
                
                for model, status in model_status.items():
                    st.write(f"{status} {model}")
                
            except Exception as e:
                st.error(f"Error checking models: {str(e)}")
        
        with col3:
            st.write("**Session Data**")
            session_keys = list(st.session_state.keys())
            data_keys = [k for k in session_keys if any(x in k for x in ['data', 'result', 'training'])]
            
            st.metric("Session Variables", len(session_keys))
            st.metric("Data Objects", len(data_keys))
            
            if data_keys:
                st.write("**Cached Data:**")
                for key in data_keys[:5]:  # Show first 5
                    if len(key) > 25:
                        display_key = key[:22] + "..."
                    else:
                        display_key = key
                    st.write(f"• {display_key}")
        
        # System performance metrics
        self._render_system_performance()
    
    def _render_system_performance(self) -> None:
        """Render system performance metrics."""
        st.subheader("⚡ System Performance")
        
        # Session state analysis
        session_data = {}
        for key, value in st.session_state.items():
            try:
                if hasattr(value, '__len__') and not isinstance(value, str):
                    session_data[key] = len(value)
                elif isinstance(value, (int, float)):
                    session_data[key] = value
                else:
                    session_data[key] = 1
            except:
                session_data[key] = 1
        
        if session_data:
            # Create a simple chart of session data
            fig = go.Figure(data=[
                go.Bar(
                    x=list(session_data.keys())[:10],  # Top 10 items
                    y=list(session_data.values())[:10],
                    name="Session Data Size"
                )
            ])
            
            fig.update_layout(
                title="Session State Overview",
                xaxis_title="Object",
                yaxis_title="Size/Count",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_model_performance(self, symbol: str, timeframe: str) -> None:
        """Render model performance analytics."""
        st.subheader("🤖 Model Performance")
        
        # Model file analysis
        try:
            metadata = self.model_registry.scan(symbol, timeframe)
            
            # Model size analysis
            model_sizes = {}
            for model_type, info in metadata.items():
                if info['exists'] and model_type in ['lstm', 'xgb', 'cnn', 'svc', 'nb']:
                    model_sizes[model_type.upper()] = info['size_mb']
            
            if model_sizes:
                # Model size chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(model_sizes.keys()),
                        y=list(model_sizes.values()),
                        name="Model Size (MB)"
                    )
                ])
                
                fig.update_layout(
                    title="Model File Sizes",
                    xaxis_title="Model Type",
                    yaxis_title="Size (MB)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Model modification times
            mod_times = {}
            for model_type, info in metadata.items():
                if info['exists'] and info['modified'] and model_type in ['lstm', 'xgb', 'cnn', 'svc', 'nb']:
                    mod_times[model_type.upper()] = info['modified']
            
            if mod_times:
                st.write("**Model Last Modified:**")
                for model, mod_time in sorted(mod_times.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"• {model}: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        except Exception as e:
            st.error(f"Error analyzing model performance: {str(e)}")
        
        # Prediction performance (if available)
        self._render_prediction_performance(symbol, timeframe)
    
    def _render_prediction_performance(self, symbol: str, timeframe: str) -> None:
        """Render prediction performance metrics."""
        signal_key = f"signal_result_{symbol}_{timeframe}"
        
        if signal_key in st.session_state:
            st.write("**Latest Prediction Analysis:**")
            
            prediction_result = st.session_state[signal_key]
            ensemble = prediction_result.get('ensemble', {})
            individual = prediction_result.get('individual', {})
            
            # Model consensus analysis
            if individual:
                directions = [pred.get('direction', 'UNKNOWN') for pred in individual.values()]
                confidences = [pred.get('confidence', 0) for pred in individual.values()]
                
                # Direction consensus
                direction_counts = pd.Series(directions).value_counts()
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=direction_counts.index,
                        values=direction_counts.values,
                        title="Model Direction Consensus"
                    )
                ])
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence distribution
                if confidences:
                    fig = go.Figure(data=[
                        go.Histogram(
                            x=confidences,
                            nbinsx=10,
                            name="Confidence Distribution"
                        )
                    ])
                    
                    fig.update_layout(
                        title="Model Confidence Distribution",
                        xaxis_title="Confidence",
                        yaxis_title="Count",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent predictions available. Generate a signal to see prediction analytics.")
    
    def _render_trading_analytics(self, symbol: str, timeframe: str) -> None:
        """Render trading analytics."""
        st.subheader("💼 Trading Analytics")
        
        # Check for backtest results
        result_key = f"backtest_results_{symbol}_{timeframe}"
        
        if result_key in st.session_state:
            results = st.session_state[result_key]
            
            if results.get('success'):
                self._analyze_backtest_results(results)
            else:
                st.warning("Latest backtest was not successful.")
        else:
            st.info("No backtest results available. Run a backtest to see trading analytics.")
        
        # Training analytics
        self._render_training_analytics(symbol, timeframe)
    
    def _analyze_backtest_results(self, results: Dict[str, Any]) -> None:
        """Analyze backtest results in detail."""
        trades = results.get('trades', [])
        equity_curve = results.get('equity_curve', [])
        
        if trades:
            # Trade timing analysis
            trade_durations = [trade['duration_bars'] for trade in trades]
            if trade_durations:
                fig = go.Figure(data=[
                    go.Histogram(
                        x=trade_durations,
                        nbinsx=20,
                        name="Trade Duration Distribution"
                    )
                ])
                
                fig.update_layout(
                    title="Trade Duration Analysis",
                    xaxis_title="Duration (bars)",
                    yaxis_title="Number of Trades",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # P&L analysis by side
            buy_trades = [t for t in trades if t['side'] == 'BUY']
            sell_trades = [t for t in trades if t['side'] == 'SELL']
            
            col1, col2 = st.columns(2)
            
            with col1:
                buy_pnl = sum(t['pnl'] for t in buy_trades)
                buy_count = len(buy_trades)
                st.metric("BUY Trades", buy_count)
                st.metric("BUY P&L", f"${buy_pnl:,.2f}")
                if buy_count > 0:
                    st.metric("Avg BUY P&L", f"${buy_pnl/buy_count:,.2f}")
            
            with col2:
                sell_pnl = sum(t['pnl'] for t in sell_trades)
                sell_count = len(sell_trades)
                st.metric("SELL Trades", sell_count)
                st.metric("SELL P&L", f"${sell_pnl:,.2f}")
                if sell_count > 0:
                    st.metric("Avg SELL P&L", f"${sell_pnl/sell_count:,.2f}")
        
        # Drawdown analysis
        if equity_curve:
            equity_values = [point['equity'] for point in equity_curve]
            drawdowns = []
            peak = equity_values[0]
            
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak * 100
                drawdowns.append(drawdown)
            
            if drawdowns:
                fig = go.Figure(data=[
                    go.Scatter(
                        y=drawdowns,
                        mode='lines',
                        name='Drawdown %',
                        fill='tonexty'
                    )
                ])
                
                fig.update_layout(
                    title="Drawdown Analysis",
                    xaxis_title="Time",
                    yaxis_title="Drawdown %",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_training_analytics(self, symbol: str, timeframe: str) -> None:
        """Render training analytics."""
        training_key = f"training_results_{symbol}_{timeframe}"
        
        if training_key in st.session_state:
            st.write("**Training Analytics:**")
            
            results = st.session_state[training_key]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Method Used:** {results.get('method_used', 'Unknown')}")
                st.write(f"**Success:** {'✅' if results.get('success') else '❌'}")
                st.write(f"**Models Trained:** {len(results.get('models_trained', []))}")
            
            with col2:
                if results.get('evaluation'):
                    eval_data = results['evaluation']
                    st.write(f"**Data Size:** {eval_data.get('data_size', 'N/A')}")
                    st.write(f"**Models Available:** {eval_data.get('models_available', 0)}/{eval_data.get('total_models', 0)}")
                    st.write(f"**Total Size:** {eval_data.get('total_size_mb', 0):.2f} MB")
        else:
            st.info("No training results available for analysis.")
    
    def _render_risk_analysis(self, symbol: str, timeframe: str) -> None:
        """Render risk analysis."""
        st.subheader("⚠️ Risk Analysis")
        
        # Signal risk analysis
        signal_key = f"signal_result_{symbol}_{timeframe}"
        
        if signal_key in st.session_state:
            prediction_result = st.session_state[signal_key]
            
            try:
                risk_assessment = self.risk_service.assess_signal_risk(prediction_result)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_level = risk_assessment['risk_level']
                    if risk_level == 'LOW':
                        st.success(f"🟢 Risk Level: {risk_level}")
                    elif risk_level == 'MEDIUM':
                        st.warning(f"🟡 Risk Level: {risk_level}")
                    else:
                        st.error(f"🔴 Risk Level: {risk_level}")
                
                with col2:
                    st.metric("Risk Score", f"{risk_assessment['risk_score']:.2f}")
                    st.metric("Model Count", risk_assessment['model_count'])
                
                with col3:
                    st.metric("Recommended Risk", f"{risk_assessment['recommended_risk_pct']:.1f}%")
                    st.metric("Consensus Level", risk_assessment['model_consensus'])
                
                # Risk warnings
                warnings = risk_assessment.get('warnings', [])
                if warnings:
                    st.write("**Risk Warnings:**")
                    for warning in warnings:
                        st.warning(f"⚠️ {warning}")
                
            except Exception as e:
                st.error(f"Risk analysis failed: {str(e)}")
        
        # Portfolio risk analysis (if backtest available)
        result_key = f"backtest_results_{symbol}_{timeframe}"
        
        if result_key in st.session_state:
            results = st.session_state[result_key]
            if results.get('success'):
                trades = results.get('trades', [])
                if trades:
                    portfolio_metrics = self.risk_service.calculate_portfolio_metrics(
                        trades, results['initial_balance']
                    )
                    
                    st.write("**Portfolio Risk Metrics:**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Win Rate", f"{portfolio_metrics['win_rate']:.1f}%")
                        st.metric("Profit Factor", f"{portfolio_metrics['profit_factor']:.2f}")
                    
                    with col2:
                        st.metric("Max Consecutive Losses", portfolio_metrics['max_consecutive_losses'])
                        st.metric("Avg Trade Risk", f"{portfolio_metrics['avg_trade_risk']:.2f}%")
                    
                    with col3:
                        st.metric("Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.2f}")
                        st.metric("Total Return", f"{portfolio_metrics['total_return_pct']:.2f}%")
    
    def _render_data_insights(self, symbol: str, timeframe: str) -> None:
        """Render data insights and analysis."""
        st.subheader("📈 Data Insights")
        
        # Check for available data
        data_sources = [
            (f"training_data_{symbol}_{timeframe}", "Training Data"),
            (f"signal_data_{symbol}_{timeframe}", "Signal Data"),
            (f"backtest_data_{symbol}_{timeframe}", "Backtest Data")
        ]
        
        available_data = []
        for key, name in data_sources:
            if key in st.session_state:
                available_data.append((key, name, st.session_state[key]))
        
        if not available_data:
            st.info("No data available for analysis. Fetch some data first.")
            return
        
        # Data selection
        selected_data = st.selectbox(
            "Select Data for Analysis",
            options=[name for _, name, _ in available_data],
            help="Choose which dataset to analyze"
        )
        
        # Find selected dataset
        selected_df = None
        for key, name, df in available_data:
            if name == selected_data:
                selected_df = df
                break
        
        if selected_df is not None:
            self._analyze_dataset(selected_df, selected_data, symbol, timeframe)
    
    def _analyze_dataset(self, df: pd.DataFrame, data_name: str, symbol: str, timeframe: str) -> None:
        """Analyze a specific dataset."""
        st.write(f"**Analysis for {data_name}:**")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Points", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        # Price statistics
        if 'close' in df.columns:
            price_stats = df['close'].describe()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Price Statistics:**")
                st.write(f"• Min: ${price_stats['min']:.2f}")
                st.write(f"• Max: ${price_stats['max']:.2f}")
                st.write(f"• Mean: ${price_stats['mean']:.2f}")
                st.write(f"• Std: ${price_stats['std']:.2f}")
            
            with col2:
                # Price distribution
                fig = go.Figure(data=[
                    go.Histogram(
                        x=df['close'],
                        nbinsx=30,
                        name="Price Distribution"
                    )
                ])
                
                fig.update_layout(
                    title="Price Distribution",
                    xaxis_title="Price",
                    yaxis_title="Frequency",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        if len(df.select_dtypes(include=['float64', 'int64']).columns) > 1:
            with st.expander("📊 Correlation Analysis", expanded=False):
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                corr_matrix = df[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Feature Correlation Matrix"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if not missing_data.empty:
            with st.expander("🔍 Missing Data Analysis", expanded=False):
                fig = go.Figure(data=[
                    go.Bar(
                        x=missing_data.index,
                        y=missing_data.values,
                        name="Missing Values"
                    )
                ])
                
                fig.update_layout(
                    title="Missing Data by Column",
                    xaxis_title="Column",
                    yaxis_title="Missing Count",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)