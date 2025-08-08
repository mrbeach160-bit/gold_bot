"""
Modular Streamlit Application Entry Point.
A clean, layered architecture for the Gold Trading Bot.
"""

import streamlit as st
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modular components
from streamlit_app.modular.core.data_service import DataService
from streamlit_app.modular.core.feature_service import FeatureService
from streamlit_app.modular.core.training_service import TrainingService
from streamlit_app.modular.core.prediction_service import PredictionService
from streamlit_app.modular.core.model_registry import ModelRegistry
from streamlit_app.modular.core.backtest_service import BacktestService
from streamlit_app.modular.core.risk_service import RiskService

from streamlit_app.modular.ui.layout import Layout
from streamlit_app.modular.ui.components.training_panel import TrainingPanel
from streamlit_app.modular.ui.components.model_status_panel import ModelStatusPanel
from streamlit_app.modular.ui.components.signal_panel import SignalPanel
from streamlit_app.modular.ui.components.backtest_panel import BacktestPanel
from streamlit_app.modular.ui.components.analytics_panel import AnalyticsPanel

from streamlit_app.modular.legacy.legacy_bridge import LegacyBridge


def initialize_services():
    """Initialize all core services."""
    try:
        # Core services
        data_service = DataService()
        feature_service = FeatureService()
        model_registry = ModelRegistry()
        training_service = TrainingService()
        risk_service = RiskService()
        
        # Prediction service (depends on model_registry and feature_service)
        prediction_service = PredictionService(model_registry, feature_service)
        
        # Backtest service (depends on prediction_service)
        backtest_service = BacktestService(prediction_service)
        
        # Legacy bridge
        legacy_bridge = LegacyBridge()
        
        # Store in session state
        st.session_state.data_service = data_service
        st.session_state.feature_service = feature_service
        st.session_state.model_registry = model_registry
        st.session_state.training_service = training_service
        st.session_state.prediction_service = prediction_service
        st.session_state.backtest_service = backtest_service
        st.session_state.risk_service = risk_service
        st.session_state.legacy_bridge = legacy_bridge
        
        # Mark as initialized
        st.session_state.services_initialized = True
        
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to initialize services: {str(e)}")
        return False


def initialize_ui_components():
    """Initialize UI components."""
    try:
        # Main layout
        layout = Layout()
        
        # UI panels
        training_panel = TrainingPanel(
            st.session_state.training_service,
            st.session_state.data_service,
            st.session_state.feature_service
        )
        
        model_status_panel = ModelStatusPanel(
            st.session_state.model_registry
        )
        
        signal_panel = SignalPanel(
            st.session_state.prediction_service,
            st.session_state.data_service,
            st.session_state.feature_service,
            st.session_state.risk_service
        )
        
        backtest_panel = BacktestPanel(
            st.session_state.backtest_service,
            st.session_state.data_service,
            st.session_state.feature_service
        )
        
        analytics_panel = AnalyticsPanel(
            st.session_state.model_registry,
            st.session_state.prediction_service,
            st.session_state.risk_service
        )
        
        # Store in session state
        st.session_state.layout = layout
        st.session_state.training_panel = training_panel
        st.session_state.model_status_panel = model_status_panel
        st.session_state.signal_panel = signal_panel
        st.session_state.backtest_panel = backtest_panel
        st.session_state.analytics_panel = analytics_panel
        
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to initialize UI components: {str(e)}")
        return False


def main():
    """Main application entry point."""
    
    # Initialize services if not already done
    if not hasattr(st.session_state, 'services_initialized') or not st.session_state.services_initialized:
        with st.spinner("🔧 Initializing modular services..."):
            if not initialize_services():
                st.stop()
    
    # Initialize UI components if not already done
    if not hasattr(st.session_state, 'layout'):
        with st.spinner("🎨 Initializing UI components..."):
            if not initialize_ui_components():
                st.stop()
    
    # Get components from session state
    layout = st.session_state.layout
    training_panel = st.session_state.training_panel
    model_status_panel = st.session_state.model_status_panel
    signal_panel = st.session_state.signal_panel
    backtest_panel = st.session_state.backtest_panel
    analytics_panel = st.session_state.analytics_panel
    
    # Setup page configuration
    layout.setup_page_config()
    
    # Render header
    layout.render_header()
    
    # Symbol and timeframe selection
    symbol, timeframe = layout.render_symbol_timeframe_selector()
    
    # Main navigation
    tab1, tab2, tab3, tab4, tab5 = layout.render_navigation()
    
    # Render tabs
    with tab1:
        try:
            training_panel.render(symbol, timeframe)
        except Exception as e:
            st.error(f"Training panel error: {str(e)}")
            st.exception(e)
    
    with tab2:
        try:
            model_status_panel.render(symbol, timeframe)
        except Exception as e:
            st.error(f"Model status panel error: {str(e)}")
            st.exception(e)
    
    with tab3:
        try:
            signal_panel.render(symbol, timeframe)
        except Exception as e:
            st.error(f"Signal panel error: {str(e)}")
            st.exception(e)
    
    with tab4:
        try:
            backtest_panel.render(symbol, timeframe)
        except Exception as e:
            st.error(f"Backtest panel error: {str(e)}")
            st.exception(e)
    
    with tab5:
        try:
            analytics_panel.render(symbol, timeframe)
        except Exception as e:
            st.error(f"Analytics panel error: {str(e)}")
            st.exception(e)
    
    # Render footer
    layout.render_footer()
    
    # Debug information (in development)
    if st.sidebar.checkbox("🐛 Debug Mode", value=False):
        with st.sidebar.expander("Debug Information", expanded=False):
            debug_info = {
                "Symbol": symbol,
                "Timeframe": timeframe,
                "Services Initialized": st.session_state.get('services_initialized', False),
                "Session State Keys": len(st.session_state.keys()),
                "Legacy Bridge Available": st.session_state.legacy_bridge.is_legacy_available() if hasattr(st.session_state, 'legacy_bridge') else False
            }
            
            # Add model availability
            try:
                available, total = st.session_state.model_registry.count_available_models(symbol, timeframe)
                debug_info["Models Available"] = f"{available}/{total}"
            except:
                debug_info["Models Available"] = "Error"
            
            st.json(debug_info)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.exception(e)
        
        # Emergency fallback information
        st.markdown("---")
        st.info("""
        **Emergency Fallback:**
        
        If this modular app fails, you can still use the legacy app:
        ```bash
        streamlit run streamlit_app/app.py
        ```
        
        **Troubleshooting:**
        1. Check if all dependencies are installed: `pip install -r requirements.txt`
        2. Ensure the `utils/` directory and legacy `app.py` exist
        3. Check API keys and configuration
        4. Clear Streamlit cache: `streamlit cache clear`
        """)