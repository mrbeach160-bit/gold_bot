"""
Live Stream Manager Component
Handles live stream analysis and real-time signal updates
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
from collections import deque

class LiveStreamManager:
    """Component for managing live streaming analysis and signal updates"""
    
    def __init__(self, ws_manager=None):
        self.ws_manager = ws_manager
        self.is_streaming = False
        self.stream_thread = None
        self.live_signals = deque(maxlen=50)  # Store last 50 signals
        self.performance_metrics = {
            'total_signals': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'last_update': None
        }
        
    def start_live_analysis(self, symbol, timeframe, models, auto_refresh_interval=30):
        """Start live streaming analysis"""
        if self.is_streaming:
            return False
            
        self.is_streaming = True
        
        def live_analysis_loop():
            while self.is_streaming:
                try:
                    # Get current price from WebSocket or API
                    current_price = None
                    if self.ws_manager:
                        current_price = self.ws_manager.get_price(symbol)
                    
                    if current_price:
                        # Generate signal with current data
                        signal_result = self._generate_live_signal(
                            symbol, current_price, models, timeframe
                        )
                        
                        if signal_result:
                            self.live_signals.append(signal_result)
                            self._update_performance_metrics(signal_result)
                    
                    # Wait for next update
                    time.sleep(auto_refresh_interval)
                    
                except Exception as e:
                    st.error(f"Live analysis error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start analysis thread
        self.stream_thread = threading.Thread(target=live_analysis_loop, daemon=True)
        self.stream_thread.start()
        
        return True
    
    def stop_live_analysis(self):
        """Stop live streaming analysis"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread = None
        return True
    
    def _generate_live_signal(self, symbol, current_price, models, timeframe):
        """Generate signal for current market conditions"""
        try:
            # This would integrate with the actual model prediction logic
            # For now, return a mock signal structure
            signal_result = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'price': current_price,
                'signal': np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4]),
                'confidence': np.random.uniform(0.6, 0.95),
                'timeframe': timeframe,
                'source': 'live_stream'
            }
            
            return signal_result
            
        except Exception as e:
            return None
    
    def _update_performance_metrics(self, signal_result):
        """Update performance tracking metrics"""
        self.performance_metrics['total_signals'] += 1
        self.performance_metrics['last_update'] = signal_result['timestamp']
        
        # Simple accuracy calculation (would need actual implementation)
        # This is a placeholder - real implementation would track signal outcomes
        if signal_result['confidence'] > 0.8:
            self.performance_metrics['correct_predictions'] += 1
        
        if self.performance_metrics['total_signals'] > 0:
            self.performance_metrics['accuracy'] = (
                self.performance_metrics['correct_predictions'] / 
                self.performance_metrics['total_signals']
            )
    
    def get_live_signals(self, limit=10):
        """Get recent live signals"""
        return list(self.live_signals)[-limit:]
    
    def get_streaming_status(self):
        """Get current streaming status"""
        return {
            'is_streaming': self.is_streaming,
            'total_signals': self.performance_metrics['total_signals'],
            'accuracy': self.performance_metrics['accuracy'],
            'last_update': self.performance_metrics['last_update']
        }
    
    def render_live_stream_panel(self, symbol, timeframe):
        """Render live streaming control panel"""
        st.subheader("📡 Live Stream Analysis")
        
        # Streaming controls
        control_cols = st.columns([2, 1, 1])
        
        with control_cols[0]:
            if not self.is_streaming:
                if st.button("🚀 Start Live Analysis", type="primary", use_container_width=True):
                    success = self.start_live_analysis(symbol, timeframe, models=None)
                    if success:
                        st.success("✅ Live analysis started")
                        st.rerun()
                    else:
                        st.error("❌ Failed to start live analysis")
            else:
                if st.button("⏹️ Stop Live Analysis", type="secondary", use_container_width=True):
                    self.stop_live_analysis()
                    st.info("🔴 Live analysis stopped")
                    st.rerun()
        
        with control_cols[1]:
            refresh_interval = st.selectbox(
                "Update Interval",
                ["10s", "30s", "1m", "2m", "5m"],
                index=1,
                key="live_refresh_interval"
            )
        
        with control_cols[2]:
            auto_scroll = st.checkbox("Auto Scroll", value=True, key="auto_scroll")
        
        # Status indicators
        status = self.get_streaming_status()
        
        if self.is_streaming:
            st.success("🟢 Live Analysis Active")
            
            # Performance metrics
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric("Total Signals", status['total_signals'])
            
            with metric_cols[1]:
                st.metric("Accuracy", f"{status['accuracy']:.1%}")
            
            with metric_cols[2]:
                if status['last_update']:
                    time_diff = datetime.now() - status['last_update']
                    st.metric("Last Update", f"{time_diff.seconds}s ago")
                else:
                    st.metric("Last Update", "Never")
            
            with metric_cols[3]:
                # Connection status from WebSocket
                if self.ws_manager:
                    ws_status = self.ws_manager.get_status(symbol)
                    if "✅" in ws_status:
                        st.metric("Data Source", "🟢 Real-time")
                    else:
                        st.metric("Data Source", "🟡 Polling")
                else:
                    st.metric("Data Source", "📊 API")
        else:
            st.info("⚪ Live Analysis Stopped")
    
    def render_live_signals_feed(self, limit=10):
        """Render live signals feed"""
        recent_signals = self.get_live_signals(limit)
        
        if not recent_signals:
            st.info("📡 No live signals yet. Start live analysis to see real-time signals.")
            return
        
        st.subheader("🔄 Live Signals Feed")
        
        # Display signals in reverse chronological order
        for i, signal in enumerate(reversed(recent_signals)):
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
                
                # Timestamp
                time_str = signal['timestamp'].strftime("%H:%M:%S")
                col1.text(time_str)
                
                # Signal
                signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
                col2.markdown(f"{signal_emoji.get(signal['signal'], '⚪')} {signal['signal']}")
                
                # Price
                col3.text(f"${signal['price']:.2f}")
                
                # Confidence
                confidence_bar = "█" * int(signal['confidence'] * 5) + "░" * (5 - int(signal['confidence'] * 5))
                col4.text(f"{signal['confidence']:.1%}")
                
                # Confidence bar
                col5.text(confidence_bar)
                
                # Add separator except for last item
                if i < len(recent_signals) - 1:
                    st.divider()
    
    def render_live_metrics_chart(self, hours=1):
        """Render live performance metrics chart"""
        if not self.live_signals:
            return
        
        # Get signals from last hour
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_signals = [s for s in self.live_signals if s['timestamp'] > cutoff_time]
        
        if not recent_signals:
            return
        
        # Create DataFrame for plotting
        df_signals = pd.DataFrame(recent_signals)
        
        # Simple signal distribution chart
        signal_counts = df_signals['signal'].value_counts()
        
        import plotly.express as px
        
        fig = px.pie(
            values=signal_counts.values,
            names=signal_counts.index,
            title=f"Signal Distribution (Last {hours}h)",
            color_discrete_map={
                'BUY': '#00ff00',
                'SELL': '#ff0000',
                'HOLD': '#ffff00'
            }
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_signal_quality_metrics(self):
        """Render signal quality and performance metrics"""
        if not self.live_signals:
            return
        
        with st.expander("📊 Signal Quality Analysis", expanded=False):
            recent_signals = list(self.live_signals)
            
            if len(recent_signals) >= 5:
                # Calculate various metrics
                confidence_values = [s['confidence'] for s in recent_signals]
                signal_types = [s['signal'] for s in recent_signals]
                
                # Average confidence
                avg_confidence = np.mean(confidence_values)
                min_confidence = np.min(confidence_values)
                max_confidence = np.max(confidence_values)
                
                # Signal distribution
                signal_dist = pd.Series(signal_types).value_counts(normalize=True)
                
                # Display metrics
                qual_cols = st.columns(3)
                
                with qual_cols[0]:
                    st.metric("Average Confidence", f"{avg_confidence:.1%}")
                    st.metric("Min Confidence", f"{min_confidence:.1%}")
                
                with qual_cols[1]:
                    st.metric("Max Confidence", f"{max_confidence:.1%}")
                    st.metric("Signal Count", len(recent_signals))
                
                with qual_cols[2]:
                    # Most frequent signal
                    most_frequent = signal_dist.index[0] if len(signal_dist) > 0 else "N/A"
                    st.metric("Dominant Signal", most_frequent)
                    st.metric("Signal Frequency", f"{signal_dist.iloc[0]:.1%}" if len(signal_dist) > 0 else "N/A")
            
            else:
                st.info("Need at least 5 signals for quality analysis")
    
    def export_signals_data(self, format='csv'):
        """Export live signals data"""
        if not self.live_signals:
            st.warning("No signals to export")
            return None
        
        df = pd.DataFrame(self.live_signals)
        
        if format == 'csv':
            return df.to_csv(index=False)
        elif format == 'json':
            return df.to_json(orient='records', date_format='iso')
        else:
            return df
    
    def render_export_controls(self):
        """Render data export controls"""
        if self.live_signals:
            with st.expander("💾 Export Data", expanded=False):
                export_format = st.selectbox("Export Format", ["CSV", "JSON"], index=0)
                
                if st.button("📥 Download Signals Data"):
                    format_lower = export_format.lower()
                    data = self.export_signals_data(format_lower)
                    
                    if data:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"live_signals_{timestamp}.{format_lower}"
                        
                        st.download_button(
                            label=f"Download {export_format}",
                            data=data,
                            file_name=filename,
                            mime=f"text/{format_lower}" if format_lower == 'csv' else 'application/json'
                        )
    
    def clear_signals_history(self):
        """Clear signals history"""
        self.live_signals.clear()
        self.performance_metrics = {
            'total_signals': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'last_update': None
        }
    
    def render_history_controls(self):
        """Render history management controls"""
        with st.expander("🗑️ History Management", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧹 Clear History", help="Clear all stored signals"):
                    self.clear_signals_history()
                    st.success("✅ History cleared")
                    st.rerun()
            
            with col2:
                st.metric("Stored Signals", len(self.live_signals))
                st.caption(f"Max: {self.live_signals.maxlen} signals")