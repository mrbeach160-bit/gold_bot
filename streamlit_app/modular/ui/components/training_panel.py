"""
Training Panel for the modular application.
Handles model training interface and controls.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional


class TrainingPanel:
    """Panel for model training interface."""
    
    def __init__(self, training_service, data_service, feature_service):
        self.training_service = training_service
        self.data_service = data_service
        self.feature_service = feature_service
    
    def render(self, symbol: str, timeframe: str) -> None:
        """
        Render the training panel interface.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        """
        st.header("🏗️ Model Training")
        
        # Get training parameters
        params = self.training_service.get_training_parameters()
        
        # Training configuration
        self._render_training_config(params)
        
        # Data fetching and preparation
        training_data = self._render_data_section(symbol, timeframe, params)
        
        # Training execution
        if training_data is not None and not training_data.empty:
            self._render_training_execution(training_data, symbol, timeframe, params)
        else:
            st.warning("⚠️ No training data available. Please fetch data first.")
    
    def _render_training_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Render training configuration controls."""
        st.subheader("⚙️ Training Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prefer_unified = st.checkbox(
                "Prefer Unified Training",
                value=params['prefer_unified'],
                help="Try unified ModelManager first before falling back to legacy"
            )
        
        with col2:
            evaluate_after = st.checkbox(
                "Evaluate After Training",
                value=params['evaluate_after'],
                help="Run model evaluation after training completes"
            )
        
        with col3:
            force_retrain = st.checkbox(
                "Force Retrain",
                value=False,
                help="Retrain even if models already exist"
            )
        
        # Advanced settings in expander
        with st.expander("🔧 Advanced Settings", expanded=False):
            min_data_required = st.number_input(
                "Minimum Data Required",
                min_value=50,
                max_value=500,
                value=params['min_data_required'],
                help="Minimum number of data points required for training"
            )
            
            st.write("**Supported Model Types:**")
            for model_type in params['model_types']:
                st.write(f"• {model_type}")
        
        return {
            'prefer_unified': prefer_unified,
            'evaluate_after': evaluate_after,
            'force_retrain': force_retrain,
            'min_data_required': min_data_required
        }
    
    def _render_data_section(self, symbol: str, timeframe: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Render data fetching and preparation section."""
        st.subheader("📊 Training Data")
        
        # Data size configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            data_size = st.number_input(
                "Training Data Size",
                min_value=params['min_data_required'],
                max_value=5000,
                value=params['data_size'],
                step=100,
                help="Number of historical bars for training"
            )
        
        with col2:
            api_key = st.text_input(
                "API Key (Optional)",
                type="password",
                help="Twelve Data API key",
                key="training_api_key"
            )
        
        with col3:
            if st.button("📥 Fetch Training Data", type="primary"):
                return self._fetch_and_prepare_data(symbol, timeframe, data_size, api_key)
        
        # Check if data exists in session state
        data_key = f"training_data_{symbol}_{timeframe}"
        if data_key in st.session_state:
            training_data = st.session_state[data_key]
            
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Points", len(training_data))
            with col2:
                st.metric("Features", len(training_data.columns))
            with col3:
                date_range = "N/A"
                if hasattr(training_data.index, 'min') and hasattr(training_data.index, 'max'):
                    try:
                        start_date = training_data.index.min().strftime('%Y-%m-%d')
                        end_date = training_data.index.max().strftime('%Y-%m-%d')
                        date_range = f"{start_date} to {end_date}"
                    except:
                        pass
                st.metric("Date Range", date_range)
            with col4:
                validity = "✅ Valid" if self.data_service.validate_data(training_data) else "❌ Invalid"
                st.metric("Data Validity", validity)
            
            # Show sample data
            with st.expander("👀 Preview Training Data", expanded=False):
                st.write("**Latest 10 rows:**")
                st.dataframe(training_data.tail(10))
                
                st.write("**Available columns:**")
                st.write(list(training_data.columns))
            
            return training_data
        
        return None
    
    def _fetch_and_prepare_data(self, symbol: str, timeframe: str, data_size: int, api_key: Optional[str]) -> Optional[pd.DataFrame]:
        """Fetch and prepare training data."""
        try:
            with st.spinner("📥 Fetching training data..."):
                # Fetch raw data
                raw_data = self.data_service.fetch_data(symbol, timeframe, data_size, api_key)
                
                if raw_data is None or raw_data.empty:
                    st.error("Failed to fetch data")
                    return None
                
                # Add technical indicators
                with st.spinner("🔧 Adding technical indicators..."):
                    data_with_indicators = self.feature_service.add_technical_indicators(raw_data)
                
                # Add engineered features
                with st.spinner("⚙️ Engineering features..."):
                    final_data = self.feature_service.add_engineered_features(data_with_indicators)
                
                # Validate data
                if not self.data_service.validate_data(final_data):
                    st.error("Data validation failed")
                    return None
                
                # Store in session state
                data_key = f"training_data_{symbol}_{timeframe}"
                st.session_state[data_key] = final_data
                
                st.success(f"✅ Successfully prepared {len(final_data)} data points for training")
                return final_data
                
        except Exception as e:
            st.error(f"Error preparing training data: {str(e)}")
            return None
    
    def _render_training_execution(self, training_data: pd.DataFrame, symbol: str, timeframe: str, config: Dict[str, Any]) -> None:
        """Render training execution controls."""
        st.subheader("🚀 Training Execution")
        
        # Check if models already exist
        if hasattr(st.session_state, 'model_registry'):
            available, total = st.session_state.model_registry.count_available_models(symbol, timeframe)
            
            if available > 0 and not config.get('force_retrain', False):
                st.warning(f"⚠️ {available}/{total} models already exist for {symbol} {timeframe}")
                st.info("💡 Enable 'Force Retrain' to retrain existing models")
        
        # Training button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏗️ Start Training", type="primary", use_container_width=True):
                self._execute_training(training_data, symbol, timeframe, config)
        
        with col2:
            if st.button("🔄 Reset Training Data", use_container_width=True):
                data_key = f"training_data_{symbol}_{timeframe}"
                if data_key in st.session_state:
                    del st.session_state[data_key]
                st.rerun()
        
        # Show training history if available
        self._render_training_history(symbol, timeframe)
    
    def _execute_training(self, training_data: pd.DataFrame, symbol: str, timeframe: str, config: Dict[str, Any]) -> None:
        """Execute the training process."""
        try:
            # Validate data one more time
            if not self.data_service.validate_data(training_data, config['min_data_required']):
                st.error("❌ Training data validation failed")
                return
            
            st.info(f"🏗️ Starting training for {symbol} {timeframe}...")
            
            # Execute training
            results = self.training_service.train_models(
                data=training_data,
                symbol=symbol,
                timeframe=timeframe,
                prefer_unified=config['prefer_unified'],
                evaluate_after=config['evaluate_after']
            )
            
            # Display results
            self._display_training_results(results, symbol, timeframe)
            
            # Refresh model status if training was successful
            if results['success']:
                # Clear any cached model data to force refresh
                if hasattr(st.session_state, 'model_registry'):
                    st.session_state.model_registry.__dict__.pop('_cached_metadata', None)
                
                # Store training results
                training_key = f"training_results_{symbol}_{timeframe}"
                st.session_state[training_key] = results
                
                st.balloons()
            
        except Exception as e:
            st.error(f"❌ Training execution failed: {str(e)}")
    
    def _display_training_results(self, results: Dict[str, Any], symbol: str, timeframe: str) -> None:
        """Display training results."""
        if results['success']:
            st.success(f"✅ Training completed successfully using {results['method_used']} method")
            
            # Show trained models
            if results['models_trained']:
                st.write("**Models trained:**")
                for model in results['models_trained']:
                    st.write(f"• {model}")
            
            # Show evaluation results
            if results.get('evaluation'):
                eval_data = results['evaluation']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Models Available", f"{eval_data['models_available']}/{eval_data['total_models']}")
                with col2:
                    st.metric("Availability Rate", f"{eval_data['availability_rate']:.1%}")
                with col3:
                    st.metric("Data Size", eval_data['data_size'])
                with col4:
                    st.metric("Total Size (MB)", eval_data.get('total_size_mb', 0))
        
        else:
            st.error("❌ Training failed")
            
            # Show errors
            if results['errors']:
                st.write("**Errors encountered:**")
                for error in results['errors']:
                    st.error(f"• {error}")
    
    def _render_training_history(self, symbol: str, timeframe: str) -> None:
        """Render training history if available."""
        training_key = f"training_results_{symbol}_{timeframe}"
        
        if training_key in st.session_state:
            with st.expander("📈 Training History", expanded=False):
                results = st.session_state[training_key]
                
                st.write(f"**Last Training Method:** {results.get('method_used', 'Unknown')}")
                st.write(f"**Success:** {'✅' if results.get('success') else '❌'}")
                
                if results.get('evaluation'):
                    st.write("**Evaluation Results:**")
                    st.json(results['evaluation'])
                
                if results.get('errors'):
                    st.write("**Errors:**")
                    for error in results['errors']:
                        st.write(f"• {error}")