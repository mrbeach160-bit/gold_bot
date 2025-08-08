"""
Model Status Panel for the modular application.
Displays model availability and metadata.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any


class ModelStatusPanel:
    """Panel for displaying model status and metadata."""
    
    def __init__(self, model_registry):
        self.model_registry = model_registry
    
    def render(self, symbol: str, timeframe: str) -> None:
        """
        Render the model status panel.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        """
        st.header("📊 Model Status")
        
        # Model overview
        self._render_model_overview(symbol, timeframe)
        
        # Detailed model table
        self._render_model_table(symbol, timeframe)
        
        # Model actions
        self._render_model_actions(symbol, timeframe)
    
    def _render_model_overview(self, symbol: str, timeframe: str) -> None:
        """Render model overview metrics."""
        try:
            available, total = self.model_registry.count_available_models(symbol, timeframe)
            metadata = self.model_registry.scan(symbol, timeframe)
            
            # Calculate total size
            total_size_mb = sum(
                info.get('size_mb', 0) for info in metadata.values() 
                if info.get('exists', False)
            )
            
            # Calculate availability rate
            availability_rate = (available / total * 100) if total > 0 else 0
            
            # Determine status
            if available == 0:
                status = "❌ No Models"
                status_color = "red"
            elif available == total:
                status = "✅ All Available"
                status_color = "green"
            else:
                status = "⚠️ Partial"
                status_color = "orange"
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Models Available",
                    f"{available}/{total}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Availability Rate",
                    f"{availability_rate:.1f}%",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Total Size",
                    f"{total_size_mb:.2f} MB",
                    delta=None
                )
            
            with col4:
                st.metric(
                    "Symbol/Timeframe",
                    f"{symbol}/{timeframe}",
                    delta=None
                )
            
            with col5:
                # Status indicator
                if status_color == "green":
                    st.success(status)
                elif status_color == "orange":
                    st.warning(status)
                else:
                    st.error(status)
            
        except Exception as e:
            st.error(f"Error loading model overview: {str(e)}")
    
    def _render_model_table(self, symbol: str, timeframe: str) -> None:
        """Render detailed model status table."""
        st.subheader("📋 Detailed Model Status")
        
        try:
            # Get model summary table
            summary_df = self.model_registry.get_summary_table(symbol, timeframe)
            
            if summary_df.empty:
                st.warning("No model information available")
                return
            
            # Add styling to the dataframe
            def style_availability(val):
                if val == '✅':
                    return 'background-color: #d4edda; color: #155724;'
                elif val == '❌':
                    return 'background-color: #f8d7da; color: #721c24;'
                return ''
            
            def style_status(val):
                if val == 'Ready':
                    return 'background-color: #d4edda; color: #155724;'
                elif val == 'Missing':
                    return 'background-color: #f8d7da; color: #721c24;'
                else:
                    return 'background-color: #fff3cd; color: #856404;'
            
            # Apply styling
            styled_df = summary_df.style.applymap(
                style_availability, subset=['Available']
            ).applymap(
                style_status, subset=['Status']
            ).format({
                'Size (MB)': '{:.2f}',
                'Last Modified': lambda x: x if x != 'N/A' else x
            })
            
            # Display table
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Show additional info
            self._render_model_details(symbol, timeframe)
            
        except Exception as e:
            st.error(f"Error rendering model table: {str(e)}")
    
    def _render_model_details(self, symbol: str, timeframe: str) -> None:
        """Render additional model details."""
        try:
            metadata = self.model_registry.scan(symbol, timeframe)
            
            # Model file paths
            with st.expander("📂 Model File Paths", expanded=False):
                for model_type, info in metadata.items():
                    status_icon = "✅" if info['exists'] else "❌"
                    st.write(f"{status_icon} **{model_type.upper()}:** `{info['path']}`")
            
            # Model dependencies
            with st.expander("🔗 Model Dependencies", expanded=False):
                dependencies = {
                    'LSTM': ['tensorflow/keras', 'numpy', 'scikit-learn'],
                    'XGBoost': ['xgboost', 'numpy', 'pandas'],
                    'CNN': ['tensorflow/keras', 'numpy', 'scikit-learn'],
                    'SVC': ['scikit-learn', 'numpy'],
                    'Naive Bayes': ['scikit-learn', 'numpy'],
                    'Scalers': ['scikit-learn', 'joblib']
                }
                
                for model_type, deps in dependencies.items():
                    st.write(f"**{model_type}:** {', '.join(deps)}")
            
            # Model specifications
            with st.expander("📐 Model Specifications", expanded=False):
                specs = {
                    'LSTM': 'Sequence length: 60, Features: close prices',
                    'XGBoost': 'Features: OHLC + technicals + engineered',
                    'CNN': 'Window size: 20, Features: OHLC + basic technicals',
                    'SVC': 'Features: OHLC + technicals (scaled)',
                    'Naive Bayes': 'Features: OHLC + technicals',
                    'Meta Learner': 'Ensemble of all model predictions'
                }
                
                for model_type, spec in specs.items():
                    available = metadata.get(model_type.lower().replace(' ', '_'), {}).get('exists', False)
                    status_icon = "✅" if available else "❌"
                    st.write(f"{status_icon} **{model_type}:** {spec}")
            
        except Exception as e:
            st.warning(f"Could not load model details: {str(e)}")
    
    def _render_model_actions(self, symbol: str, timeframe: str) -> None:
        """Render model management actions."""
        st.subheader("🔧 Model Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Refresh Status", use_container_width=True):
                # Clear any cached data
                if hasattr(self.model_registry, '_cached_metadata'):
                    delattr(self.model_registry, '_cached_metadata')
                st.rerun()
        
        with col2:
            if st.button("📊 Load Models", use_container_width=True):
                self._test_model_loading(symbol, timeframe)
        
        with col3:
            if st.button("🧹 Clean Cache", use_container_width=True):
                self._clean_model_cache()
        
        with col4:
            if st.button("📋 Export Info", use_container_width=True):
                self._export_model_info(symbol, timeframe)
    
    def _test_model_loading(self, symbol: str, timeframe: str) -> None:
        """Test loading models to verify they work."""
        try:
            with st.spinner("🔄 Testing model loading..."):
                models, load_errors = self.model_registry.load_for_prediction(symbol, timeframe)
                
                if models:
                    st.success(f"✅ Successfully loaded {len(models)} models")
                    
                    # Show loaded models
                    st.write("**Loaded models:**")
                    for model_name in models.keys():
                        st.write(f"• {model_name}")
                else:
                    st.warning("⚠️ No models could be loaded")
                
                # Show load errors
                if load_errors:
                    st.write("**Load errors:**")
                    for model_name, error in load_errors.items():
                        if 'optional' not in error.lower():
                            st.error(f"• {model_name}: {error}")
                        else:
                            st.info(f"• {model_name}: {error}")
                
        except Exception as e:
            st.error(f"❌ Model loading test failed: {str(e)}")
    
    def _clean_model_cache(self) -> None:
        """Clean model cache and temporary files."""
        try:
            # Clear session state cache
            cache_keys = [key for key in st.session_state.keys() if 'model' in key.lower()]
            for key in cache_keys:
                if key.startswith('training_data_') or key.startswith('model_'):
                    del st.session_state[key]
            
            # Clear Streamlit cache
            st.cache_data.clear()
            
            st.success("✅ Model cache cleaned")
            
        except Exception as e:
            st.error(f"❌ Cache cleaning failed: {str(e)}")
    
    def _export_model_info(self, symbol: str, timeframe: str) -> None:
        """Export model information as downloadable data."""
        try:
            # Get model information
            metadata = self.model_registry.scan(symbol, timeframe)
            summary_df = self.model_registry.get_summary_table(symbol, timeframe)
            
            # Create export data
            export_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'scan_metadata': metadata,
                'summary': summary_df.to_dict('records') if not summary_df.empty else []
            }
            
            # Convert to JSON string
            import json
            json_str = json.dumps(export_data, indent=2, default=str)
            
            # Provide download
            st.download_button(
                label="📥 Download Model Info (JSON)",
                data=json_str,
                file_name=f"model_info_{symbol}_{timeframe}.json",
                mime="application/json"
            )
            
            st.success("✅ Model information ready for download")
            
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
    
    def render_quick_status(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Render a quick status view for use in other panels.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Dictionary with status information
        """
        try:
            available, total = self.model_registry.count_available_models(symbol, timeframe)
            
            if available == 0:
                st.error(f"❌ No models available for {symbol} {timeframe}")
                status = 'none'
            elif available == total:
                st.success(f"✅ All {total} models available for {symbol} {timeframe}")
                status = 'complete'
            else:
                st.warning(f"⚠️ {available}/{total} models available for {symbol} {timeframe}")
                status = 'partial'
            
            return {
                'status': status,
                'available': available,
                'total': total,
                'symbol': symbol,
                'timeframe': timeframe
            }
            
        except Exception as e:
            st.error(f"❌ Status check failed: {str(e)}")
            return {
                'status': 'error',
                'available': 0,
                'total': 0,
                'symbol': symbol,
                'timeframe': timeframe,
                'error': str(e)
            }