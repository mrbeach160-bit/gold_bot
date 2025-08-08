"""
Model Status Display Component
Handles AI model status, availability checking, and validation
"""

import streamlit as st
import os
import joblib
from datetime import datetime
import pandas as pd

class ModelStatusDisplay:
    """Component for displaying AI model status and information"""
    
    def __init__(self, model_dir='model'):
        self.model_dir = model_dir
        self.model_types = ['LSTM', 'XGB', 'CNN', 'SVC', 'NB']  # Naive Bayes
        
    def sanitize_filename(self, symbol):
        """Sanitize symbol for filename"""
        return symbol.replace('/', '_').replace('\\', '_')
    
    def check_model_availability(self, symbol, timeframe):
        """Check which models are available for the symbol/timeframe"""
        symbol_fn = self.sanitize_filename(symbol)
        available_models = {}
        
        for model_type in self.model_types:
            if model_type == 'LSTM':
                model_path = f"{self.model_dir}/{symbol_fn}_{timeframe}_lstm_model.h5"
                scaler_path = f"{self.model_dir}/{symbol_fn}_{timeframe}_lstm_scaler.pkl"
                available_models[model_type] = os.path.exists(model_path) and os.path.exists(scaler_path)
            elif model_type == 'XGB':
                model_path = f"{self.model_dir}/{symbol_fn}_{timeframe}_xgb_model.pkl"
                available_models[model_type] = os.path.exists(model_path)
            elif model_type == 'CNN':
                model_path = f"{self.model_dir}/{symbol_fn}_{timeframe}_cnn_model.h5"
                available_models[model_type] = os.path.exists(model_path)
            elif model_type == 'SVC':
                model_path = f"{self.model_dir}/{symbol_fn}_{timeframe}_svc_model.pkl"
                scaler_path = f"{self.model_dir}/{symbol_fn}_{timeframe}_svc_scaler.pkl"
                available_models[model_type] = os.path.exists(model_path) and os.path.exists(scaler_path)
            elif model_type == 'NB':
                model_path = f"{self.model_dir}/{symbol_fn}_{timeframe}_nb_model.pkl"
                available_models[model_type] = os.path.exists(model_path)
        
        return available_models
    
    def get_model_file_info(self, symbol, timeframe):
        """Get file size and modification date for models"""
        symbol_fn = self.sanitize_filename(symbol)
        model_info = {}
        
        for model_type in self.model_types:
            info = {'exists': False, 'size': 0, 'modified': None}
            
            if model_type == 'LSTM':
                model_path = f"{self.model_dir}/{symbol_fn}_{timeframe}_lstm_model.h5"
                scaler_path = f"{self.model_dir}/{symbol_fn}_{timeframe}_lstm_scaler.pkl"
                
                if os.path.exists(model_path):
                    info['exists'] = True
                    info['size'] = os.path.getsize(model_path)
                    info['modified'] = datetime.fromtimestamp(os.path.getmtime(model_path))
                    
                    if os.path.exists(scaler_path):
                        info['size'] += os.path.getsize(scaler_path)
                        
            elif model_type in ['XGB', 'SVC', 'NB']:
                model_path = f"{self.model_dir}/{symbol_fn}_{timeframe}_{model_type.lower()}_model.pkl"
                
                if os.path.exists(model_path):
                    info['exists'] = True
                    info['size'] = os.path.getsize(model_path)
                    info['modified'] = datetime.fromtimestamp(os.path.getmtime(model_path))
                    
                    if model_type == 'SVC':
                        scaler_path = f"{self.model_dir}/{symbol_fn}_{timeframe}_svc_scaler.pkl"
                        if os.path.exists(scaler_path):
                            info['size'] += os.path.getsize(scaler_path)
                            
            elif model_type == 'CNN':
                model_path = f"{self.model_dir}/{symbol_fn}_{timeframe}_cnn_model.h5"
                
                if os.path.exists(model_path):
                    info['exists'] = True
                    info['size'] = os.path.getsize(model_path)
                    info['modified'] = datetime.fromtimestamp(os.path.getmtime(model_path))
            
            model_info[model_type] = info
        
        return model_info
    
    def format_file_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def get_model_ensemble_readiness(self, symbol, timeframe):
        """Check if ensemble prediction is possible"""
        available_models = self.check_model_availability(symbol, timeframe)
        available_count = sum(available_models.values())
        
        readiness = {
            'total_models': len(self.model_types),
            'available_count': available_count,
            'missing_models': [model for model, available in available_models.items() if not available],
            'ensemble_ready': available_count >= 3,  # Need at least 3 models for ensemble
            'all_models_ready': available_count == len(self.model_types)
        }
        
        return readiness
    
    def render_model_status_panel(self, symbol, timeframe):
        """Render comprehensive model status panel"""
        st.subheader("🤖 Model Status Dashboard")
        
        # Check model availability
        available_models = self.check_model_availability(symbol, timeframe)
        model_info = self.get_model_file_info(symbol, timeframe)
        ensemble_readiness = self.get_model_ensemble_readiness(symbol, timeframe)
        
        # Overall status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if ensemble_readiness['all_models_ready']:
                st.success(f"✅ All Models Ready")
                st.metric("Models Available", f"{ensemble_readiness['available_count']}/{ensemble_readiness['total_models']}")
            elif ensemble_readiness['ensemble_ready']:
                st.warning(f"⚠️ Ensemble Ready")
                st.metric("Models Available", f"{ensemble_readiness['available_count']}/{ensemble_readiness['total_models']}")
            else:
                st.error(f"❌ Insufficient Models")
                st.metric("Models Available", f"{ensemble_readiness['available_count']}/{ensemble_readiness['total_models']}")
        
        with col2:
            # Calculate total model size
            total_size = sum(info['size'] for info in model_info.values() if info['exists'])
            st.metric("Total Model Size", self.format_file_size(total_size))
        
        with col3:
            # Find latest model modification date
            latest_modified = None
            for info in model_info.values():
                if info['exists'] and info['modified']:
                    if latest_modified is None or info['modified'] > latest_modified:
                        latest_modified = info['modified']
            
            if latest_modified:
                st.metric("Last Updated", latest_modified.strftime("%Y-%m-%d %H:%M"))
            else:
                st.metric("Last Updated", "No models")
        
        # Detailed model status table
        with st.expander("📊 Detailed Model Information", expanded=False):
            model_data = []
            
            for model_type in self.model_types:
                info = model_info[model_type]
                available = available_models[model_type]
                
                status_emoji = "✅" if available else "❌"
                size_str = self.format_file_size(info['size']) if info['exists'] else "-"
                modified_str = info['modified'].strftime("%Y-%m-%d %H:%M") if info['modified'] else "-"
                
                model_data.append({
                    'Model': model_type,
                    'Status': f"{status_emoji} {'Available' if available else 'Missing'}",
                    'Size': size_str,
                    'Last Modified': modified_str
                })
            
            df_models = pd.DataFrame(model_data)
            st.dataframe(df_models, use_container_width=True, hide_index=True)
        
        # Recommendations
        if not ensemble_readiness['ensemble_ready']:
            st.warning("⚠️ **Recommendation:** Train more models for better ensemble predictions")
            st.info(f"Missing models: {', '.join(ensemble_readiness['missing_models'])}")
        
        # Training recommendations
        if ensemble_readiness['available_count'] > 0:
            with st.expander("🎯 Model Performance Tips", expanded=False):
                st.markdown("""
                **Ensemble Prediction Strategy:**
                - **3+ models**: Basic ensemble capability
                - **4+ models**: Good ensemble diversity
                - **5 models**: Optimal ensemble performance
                
                **Model Specialties:**
                - **LSTM**: Excellent for trend patterns and sequences
                - **XGBoost**: Great for feature interactions and non-linear patterns
                - **CNN**: Effective for pattern recognition in price movements
                - **SVC**: Strong for classification boundaries
                - **Naive Bayes**: Fast for probabilistic predictions
                
                **Training Recommendations:**
                - Retrain models weekly for optimal performance
                - Use different data sizes for model diversity
                - Monitor individual model accuracy
                """)
        
        return ensemble_readiness
    
    def render_training_status(self, is_training=False, current_step=None, total_steps=None):
        """Render training progress status"""
        if is_training:
            st.info("🏗️ Model Training in Progress...")
            if current_step and total_steps:
                progress = current_step / total_steps
                st.progress(progress)
                st.caption(f"Step {current_step} of {total_steps}")
        else:
            st.success("✅ Training Complete")
    
    def render_model_validation_results(self, validation_results=None):
        """Render model validation results if available"""
        if validation_results:
            with st.expander("📈 Model Validation Results", expanded=False):
                for model_name, results in validation_results.items():
                    st.markdown(f"**{model_name}:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        accuracy = results.get('accuracy', 0)
                        st.metric(f"{model_name} Accuracy", f"{accuracy:.2%}")
                    
                    with col2:
                        precision = results.get('precision', 0)
                        st.metric("Precision", f"{precision:.3f}")
                    
                    with col3:
                        recall = results.get('recall', 0)
                        st.metric("Recall", f"{recall:.3f}")
    
    def check_model_freshness(self, symbol, timeframe, max_age_days=7):
        """Check if models are fresh enough for trading"""
        model_info = self.get_model_file_info(symbol, timeframe)
        current_time = datetime.now()
        
        freshness_status = {}
        
        for model_type, info in model_info.items():
            if info['exists'] and info['modified']:
                age_days = (current_time - info['modified']).days
                is_fresh = age_days <= max_age_days
                
                freshness_status[model_type] = {
                    'age_days': age_days,
                    'is_fresh': is_fresh,
                    'status': 'Fresh' if is_fresh else f'Stale ({age_days}d old)'
                }
            else:
                freshness_status[model_type] = {
                    'age_days': None,
                    'is_fresh': False,
                    'status': 'Missing'
                }
        
        return freshness_status
    
    def render_freshness_warning(self, symbol, timeframe, max_age_days=7):
        """Render warning if models are stale"""
        freshness = self.check_model_freshness(symbol, timeframe, max_age_days)
        stale_models = [model for model, status in freshness.items() 
                       if not status['is_fresh'] and status['age_days'] is not None]
        
        if stale_models:
            st.warning(f"⚠️ Some models are older than {max_age_days} days: {', '.join(stale_models)}")
            st.info("Consider retraining for optimal performance")
            
            return True
        return False