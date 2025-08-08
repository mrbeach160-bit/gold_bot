"""
Signal Panel for the modular application.
Handles live signal generation and display.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Optional


class SignalPanel:
    """Panel for live signal generation and display."""
    
    def __init__(self, prediction_service, data_service, feature_service, risk_service):
        self.prediction_service = prediction_service
        self.data_service = data_service
        self.feature_service = feature_service
        self.risk_service = risk_service
    
    def render(self, symbol: str, timeframe: str) -> None:
        """
        Render the signal panel interface.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        """
        st.header("🔮 Live Trading Signals")
        
        # Signal configuration
        signal_config = self._render_signal_config()
        
        # Data fetching for signals
        signal_data = self._render_signal_data_section(symbol, timeframe, signal_config)
        
        # Signal generation and display
        if signal_data is not None and not signal_data.empty:
            self._render_signal_generation(signal_data, symbol, timeframe, signal_config)
        else:
            st.warning("⚠️ No data available for signal generation. Please fetch data first.")
    
    def _render_signal_config(self) -> Dict[str, Any]:
        """Render signal configuration controls."""
        st.subheader("⚙️ Signal Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            data_bars = st.number_input(
                "Data Bars for Analysis",
                min_value=100,
                max_value=1000,
                value=500,
                step=50,
                help="Number of historical bars to fetch for signal analysis"
            )
        
        with col2:
            confidence_threshold = st.slider(
                "Minimum Confidence",
                min_value=0.5,
                max_value=0.95,
                value=0.55,
                step=0.05,
                help="Minimum ensemble confidence for signal"
            )
        
        with col3:
            auto_refresh = st.checkbox(
                "Auto Refresh",
                value=False,
                help="Automatically refresh signals (be mindful of API limits)"
            )
        
        # Advanced signal settings
        with st.expander("🔧 Advanced Signal Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                show_individual = st.checkbox(
                    "Show Individual Model Predictions",
                    value=True,
                    help="Display predictions from each model"
                )
                
                show_risk_assessment = st.checkbox(
                    "Show Risk Assessment",
                    value=True,
                    help="Display risk analysis for the signal"
                )
            
            with col2:
                show_raw_data = st.checkbox(
                    "Show Raw Prediction Data",
                    value=False,
                    help="Display raw prediction outputs (debug)"
                )
                
                api_key = st.text_input(
                    "Custom API Key",
                    type="password",
                    help="Override default API key",
                    key="signal_api_key"
                )
        
        return {
            'data_bars': data_bars,
            'confidence_threshold': confidence_threshold,
            'auto_refresh': auto_refresh,
            'show_individual': show_individual,
            'show_risk_assessment': show_risk_assessment,
            'show_raw_data': show_raw_data,
            'api_key': api_key if api_key else None
        }
    
    def _render_signal_data_section(self, symbol: str, timeframe: str, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Render signal data fetching section."""
        st.subheader("📊 Live Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📡 Fetch Latest Data", type="primary", use_container_width=True):
                return self._fetch_signal_data(symbol, timeframe, config)
        
        with col2:
            if st.button("🔄 Refresh Signal", use_container_width=True):
                data_key = f"signal_data_{symbol}_{timeframe}"
                if data_key in st.session_state:
                    return st.session_state[data_key]
                else:
                    st.warning("No data cached. Please fetch data first.")
                    return None
        
        with col3:
            if st.button("🧹 Clear Cache", use_container_width=True):
                data_key = f"signal_data_{symbol}_{timeframe}"
                if data_key in st.session_state:
                    del st.session_state[data_key]
                st.rerun()
        
        # Check for cached data
        data_key = f"signal_data_{symbol}_{timeframe}"
        if data_key in st.session_state:
            signal_data = st.session_state[data_key]
            
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Points", len(signal_data))
            with col2:
                st.metric("Latest Price", f"{signal_data['close'].iloc[-1]:.2f}")
            with col3:
                try:
                    last_update = signal_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                except:
                    last_update = "N/A"
                st.metric("Last Update", last_update)
            with col4:
                validity = "✅ Valid" if self.data_service.validate_data(signal_data, 100) else "❌ Invalid"
                st.metric("Data Validity", validity)
            
            return signal_data
        
        return None
    
    def _fetch_signal_data(self, symbol: str, timeframe: str, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Fetch and prepare data for signal generation."""
        try:
            with st.spinner("📡 Fetching latest market data..."):
                # Fetch raw data
                raw_data = self.data_service.fetch_data(
                    symbol, timeframe, config['data_bars'], config['api_key']
                )
                
                if raw_data is None or raw_data.empty:
                    st.error("Failed to fetch market data")
                    return None
                
                # Add technical indicators
                with st.spinner("🔧 Adding technical indicators..."):
                    data_with_indicators = self.feature_service.add_technical_indicators(raw_data)
                
                # Add engineered features
                with st.spinner("⚙️ Engineering features..."):
                    final_data = self.feature_service.add_engineered_features(data_with_indicators)
                
                # Validate data
                if not self.data_service.validate_data(final_data, 100):
                    st.error("Signal data validation failed")
                    return None
                
                # Cache data
                data_key = f"signal_data_{symbol}_{timeframe}"
                st.session_state[data_key] = final_data
                
                st.success(f"✅ Successfully fetched {len(final_data)} bars for signal analysis")
                return final_data
                
        except Exception as e:
            st.error(f"Error fetching signal data: {str(e)}")
            return None
    
    def _render_signal_generation(self, data: pd.DataFrame, symbol: str, timeframe: str, config: Dict[str, Any]) -> None:
        """Render signal generation and results."""
        st.subheader("🎯 Signal Analysis")
        
        # Generate signal button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔮 Generate Signal", type="primary", use_container_width=True):
                self._generate_and_display_signal(data, symbol, timeframe, config)
        
        with col2:
            if st.button("📊 Show Price Chart", use_container_width=True):
                self._render_price_chart(data, symbol, timeframe)
        
        # Display cached signal if available
        signal_key = f"signal_result_{symbol}_{timeframe}"
        if signal_key in st.session_state:
            signal_result = st.session_state[signal_key]
            self._display_signal_results(signal_result, config)
    
    def _generate_and_display_signal(self, data: pd.DataFrame, symbol: str, timeframe: str, config: Dict[str, Any]) -> None:
        """Generate and display trading signal."""
        try:
            with st.spinner("🔮 Generating trading signal..."):
                # Make predictions
                prediction_result = self.prediction_service.make_predictions(data, symbol, timeframe)
                
                # Store result
                signal_key = f"signal_result_{symbol}_{timeframe}"
                st.session_state[signal_key] = prediction_result
                
                # Display results
                self._display_signal_results(prediction_result, config)
                
        except Exception as e:
            st.error(f"Signal generation failed: {str(e)}")
    
    def _display_signal_results(self, prediction_result: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Display the prediction results."""
        ensemble = prediction_result.get('ensemble', {})
        individual = prediction_result.get('individual', {})
        load_errors = prediction_result.get('load_errors', {})
        
        # Main signal display
        st.subheader("🎯 Trading Signal")
        
        direction = ensemble.get('direction', 'HOLD')
        confidence = ensemble.get('confidence', 0.0)
        message = ensemble.get('message', '')
        
        # Signal strength indicator
        if direction == 'HOLD':
            signal_color = "blue"
            signal_icon = "🔵"
        elif direction == 'BUY':
            signal_color = "green"
            signal_icon = "🟢"
        else:  # SELL
            signal_color = "red"
            signal_icon = "🔴"
        
        # Display main signal
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Direction", f"{signal_icon} {direction}", delta=None)
        
        with col2:
            st.metric("Confidence", f"{confidence:.1%}", delta=None)
        
        with col3:
            model_count = ensemble.get('model_count', 0)
            st.metric("Models Used", model_count, delta=None)
        
        # Signal message
        if message:
            if confidence >= config['confidence_threshold']:
                st.success(f"✅ {message}")
            else:
                st.warning(f"⚠️ {message}")
        
        # Risk assessment
        if config['show_risk_assessment']:
            self._render_risk_assessment(prediction_result)
        
        # Individual model predictions
        if config['show_individual'] and individual:
            self._render_individual_predictions(individual)
        
        # Load errors
        if load_errors:
            with st.expander("⚠️ Model Loading Issues", expanded=False):
                for model_name, error in load_errors.items():
                    if 'optional' not in error.lower():
                        st.error(f"**{model_name}:** {error}")
                    else:
                        st.info(f"**{model_name}:** {error}")
        
        # Raw data
        if config['show_raw_data']:
            with st.expander("🔍 Raw Prediction Data", expanded=False):
                st.json(prediction_result)
    
    def _render_individual_predictions(self, individual: Dict[str, Any]) -> None:
        """Render individual model predictions."""
        st.subheader("🤖 Individual Model Predictions")
        
        # Create table for individual predictions
        individual_data = []
        for model_name, prediction in individual.items():
            individual_data.append({
                'Model': model_name.upper(),
                'Direction': prediction.get('direction', 'UNKNOWN'),
                'Confidence': f"{prediction.get('confidence', 0):.1%}",
                'Status': '✅ Success' if 'error' not in prediction.get('raw', {}) else '❌ Error'
            })
        
        if individual_data:
            df = pd.DataFrame(individual_data)
            
            # Style the dataframe
            def style_direction(val):
                if val == 'BUY':
                    return 'background-color: #d4edda; color: #155724;'
                elif val == 'SELL':
                    return 'background-color: #f8d7da; color: #721c24;'
                elif val == 'HOLD':
                    return 'background-color: #cce7ff; color: #004085;'
                return ''
            
            def style_status(val):
                if '✅' in val:
                    return 'background-color: #d4edda; color: #155724;'
                elif '❌' in val:
                    return 'background-color: #f8d7da; color: #721c24;'
                return ''
            
            styled_df = df.style.applymap(style_direction, subset=['Direction']).applymap(style_status, subset=['Status'])
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    def _render_risk_assessment(self, prediction_result: Dict[str, Any]) -> None:
        """Render risk assessment for the signal."""
        st.subheader("⚠️ Risk Assessment")
        
        try:
            risk_assessment = self.risk_service.assess_signal_risk(prediction_result)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                risk_level = risk_assessment['risk_level']
                if risk_level == 'LOW':
                    st.success(f"🟢 Risk: {risk_level}")
                elif risk_level == 'MEDIUM':
                    st.warning(f"🟡 Risk: {risk_level}")
                else:
                    st.error(f"🔴 Risk: {risk_level}")
            
            with col2:
                st.metric("Risk Score", f"{risk_assessment['risk_score']:.2f}")
            
            with col3:
                st.metric("Model Consensus", risk_assessment['model_consensus'])
            
            with col4:
                st.metric("Recommended Risk", f"{risk_assessment['recommended_risk_pct']:.1f}%")
            
            # Risk warnings
            warnings = risk_assessment.get('warnings', [])
            if warnings:
                st.write("**Risk Warnings:**")
                for warning in warnings:
                    st.warning(f"⚠️ {warning}")
            
        except Exception as e:
            st.error(f"Risk assessment failed: {str(e)}")
    
    def _render_price_chart(self, data: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Render price chart with recent data."""
        try:
            # Create candlestick chart
            fig = go.Figure(data=go.Candlestick(
                x=data.index[-100:],  # Last 100 bars
                open=data['open'].iloc[-100:],
                high=data['high'].iloc[-100:],
                low=data['low'].iloc[-100:],
                close=data['close'].iloc[-100:],
                name=f"{symbol} {timeframe}"
            ))
            
            # Add moving averages if available
            if 'EMA_10' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index[-100:],
                    y=data['EMA_10'].iloc[-100:],
                    mode='lines',
                    name='EMA 10',
                    line=dict(color='blue', width=1)
                ))
            
            if 'EMA_20' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index[-100:],
                    y=data['EMA_20'].iloc[-100:],
                    mode='lines',
                    name='EMA 20',
                    line=dict(color='red', width=1)
                ))
            
            fig.update_layout(
                title=f"{symbol} {timeframe} - Recent Price Action",
                xaxis_title="Time",
                yaxis_title="Price",
                height=400,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Chart rendering failed: {str(e)}")