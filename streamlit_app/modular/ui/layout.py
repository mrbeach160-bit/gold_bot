"""
Layout for the modular application.
Handles the main UI structure and navigation.
"""

import streamlit as st
from typing import Dict, Any


class Layout:
    """Main layout manager for the modular application."""
    
    def __init__(self):
        self.tabs = [
            "Model Training",
            "Model Status", 
            "Live Signals",
            "Backtesting",
            "Analytics"
        ]
    
    def setup_page_config(self):
        """Set up the Streamlit page configuration."""
        st.set_page_config(
            page_title="Gold Trading Bot - Modular",
            page_icon="🥇",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_header(self):
        """Render the main header."""
        st.title("🥇 Gold Trading Bot - Modular Architecture")
        st.markdown("---")
        
        # Show system status in sidebar
        with st.sidebar:
            st.header("📊 System Status")
            self._render_system_status()
    
    def _render_system_status(self):
        """Render system status in sidebar."""
        try:
            # Check if services are initialized
            if hasattr(st.session_state, 'services_initialized') and st.session_state.services_initialized:
                st.success("✅ Services Initialized")
                
                # Show model registry status
                if hasattr(st.session_state, 'model_registry'):
                    st.info("📁 Model Registry: Ready")
                
                # Show prediction service status
                if hasattr(st.session_state, 'prediction_service'):
                    st.info("🔮 Prediction Service: Ready")
                
                # Show training service status
                if hasattr(st.session_state, 'training_service'):
                    st.info("🏗️ Training Service: Ready")
                
                # Show backtest service status
                if hasattr(st.session_state, 'backtest_service'):
                    st.info("📈 Backtest Service: Ready")
                
            else:
                st.warning("⚠️ Services Not Initialized")
                
        except Exception as e:
            st.error(f"Status error: {str(e)}")
    
    def render_navigation(self) -> str:
        """
        Render the main navigation tabs.
        
        Returns:
            Selected tab name
        """
        return st.tabs(self.tabs)
    
    def render_symbol_timeframe_selector(self) -> tuple:
        """
        Render symbol and timeframe selector.
        
        Returns:
            Tuple of (symbol, timeframe)
        """
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.selectbox(
                "Trading Symbol",
                options=["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD"],
                index=0,
                help="Select the trading symbol for analysis"
            )
        
        with col2:
            timeframe = st.selectbox(
                "Timeframe",
                options=["5m", "15m", "30m", "1h", "4h", "1d"],
                index=1,  # Default to 15m
                help="Select the timeframe for analysis"
            )
        
        return symbol, timeframe
    
    def render_data_fetch_controls(self) -> Dict[str, Any]:
        """
        Render data fetching controls.
        
        Returns:
            Dictionary with data fetch parameters
        """
        st.subheader("📊 Data Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            data_size = st.number_input(
                "Data Size",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Number of historical bars to fetch"
            )
        
        with col2:
            api_key = st.text_input(
                "API Key (Optional)",
                type="password",
                help="Twelve Data API key (leave empty to use config)"
            )
        
        with col3:
            auto_fetch = st.checkbox(
                "Auto-fetch on change",
                value=True,
                help="Automatically fetch data when symbol/timeframe changes"
            )
        
        return {
            'data_size': data_size,
            'api_key': api_key if api_key else None,
            'auto_fetch': auto_fetch
        }
    
    def render_error_container(self, error_message: str):
        """Render error message in a container."""
        st.error(f"❌ {error_message}")
    
    def render_warning_container(self, warning_message: str):
        """Render warning message in a container."""
        st.warning(f"⚠️ {warning_message}")
    
    def render_success_container(self, success_message: str):
        """Render success message in a container."""
        st.success(f"✅ {success_message}")
    
    def render_info_container(self, info_message: str):
        """Render info message in a container."""
        st.info(f"ℹ️ {info_message}")
    
    def render_loading_spinner(self, message: str = "Loading..."):
        """Render loading spinner with message."""
        return st.spinner(message)
    
    def render_progress_bar(self, progress: float, message: str = ""):
        """Render progress bar."""
        if message:
            st.write(message)
        return st.progress(progress)
    
    def render_metric_cards(self, metrics: Dict[str, Any]):
        """
        Render metric cards in columns.
        
        Args:
            metrics: Dictionary with metric name -> (value, delta) pairs
        """
        if not metrics:
            return
        
        num_metrics = len(metrics)
        cols = st.columns(num_metrics)
        
        for i, (name, data) in enumerate(metrics.items()):
            with cols[i]:
                if isinstance(data, tuple) and len(data) == 2:
                    value, delta = data
                    st.metric(name, value, delta)
                else:
                    st.metric(name, data)
    
    def render_expandable_section(self, title: str, content_func, expanded: bool = False):
        """
        Render an expandable section.
        
        Args:
            title: Section title
            content_func: Function to render content
            expanded: Whether section is expanded by default
        """
        with st.expander(title, expanded=expanded):
            content_func()
    
    def render_two_column_layout(self, left_func, right_func, left_width: int = 1, right_width: int = 1):
        """
        Render two-column layout.
        
        Args:
            left_func: Function to render left column content
            right_func: Function to render right column content
            left_width: Relative width of left column
            right_width: Relative width of right column
        """
        col1, col2 = st.columns([left_width, right_width])
        
        with col1:
            left_func()
        
        with col2:
            right_func()
    
    def render_footer(self):
        """Render footer with information."""
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: gray; font-size: 12px;'>
                Gold Trading Bot - Modular Architecture v1.0<br>
                Built with Streamlit | Real ML Predictions | Advanced Risk Management
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def render_debug_info(self, debug_data: Dict[str, Any]):
        """
        Render debug information in an expandable section.
        
        Args:
            debug_data: Dictionary with debug information
        """
        with st.expander("🐛 Debug Information", expanded=False):
            st.json(debug_data)
    
    def center_content(self, content_func):
        """
        Center content using columns.
        
        Args:
            content_func: Function to render centered content
        """
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            content_func()
    
    def render_status_indicator(self, status: str, message: str = ""):
        """
        Render status indicator with appropriate color and icon.
        
        Args:
            status: Status level ('success', 'warning', 'error', 'info')
            message: Optional message to display
        """
        if status == 'success':
            st.success(f"✅ {message}" if message else "✅ Success")
        elif status == 'warning':
            st.warning(f"⚠️ {message}" if message else "⚠️ Warning")
        elif status == 'error':
            st.error(f"❌ {message}" if message else "❌ Error")
        elif status == 'info':
            st.info(f"ℹ️ {message}" if message else "ℹ️ Info")
        else:
            st.write(f"🔵 {message}" if message else "🔵 Status")