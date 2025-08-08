# training_interface.py - Model training interface components
import streamlit as st

from ..ai import train_and_save_all_models
from ..data import load_and_process_data_enhanced
from .components import display_trading_controls


def display_training_interface(api_source, symbol, api_key_1, api_key_2):
    """Display the model training interface"""
    st.header("🧠 Pelatihan Model AI")
    
    st.markdown("""
    **Instruksi:**
    1. Pilih timeframe untuk data training
    2. Tentukan jumlah data yang akan digunakan
    3. Klik tombol training untuk melatih semua model
    4. Tunggu hingga proses selesai (bisa memakan waktu beberapa menit)
    """)
    
    # Training controls
    train_cols = st.columns([2, 1, 1])
    
    # Timeframe mapping
    timeframe_mapping = {
        '1 minute': '1min',
        '5 minutes': '5min',
        '15 minutes': '15min',
        '30 minutes': '30min',
        '1 hour': '1h',
        '4 hours': '4h',
        '1 day': '1day'
    }
    
    train_tf_key = train_cols[0].selectbox(
        "Pilih Timeframe Training", 
        list(timeframe_mapping.keys()), 
        index=4,  # Default to 1 hour
        key="train_tf_select"
    )
    
    data_size = train_cols[1].number_input(
        "Jumlah Data", 
        min_value=500, 
        max_value=5000, 
        value=1500, 
        step=100,
        help="Lebih banyak data = training lebih lama tapi mungkin lebih akurat"
    )
    
    start_training = train_cols[2].button(
        "🚀 Start Training", 
        use_container_width=True, 
        type="primary"
    )
    
    # Information about models
    with st.expander("ℹ️ Informasi Model AI", expanded=False):
        st.markdown("""
        **Model yang akan dilatih:**
        
        1. **LSTM (Long Short-Term Memory)**
           - Neural network untuk prediksi harga
           - Baik untuk pattern sequential
        
        2. **XGBoost (Extreme Gradient Boosting)**
           - Tree-based ensemble model
           - Cepat dan akurat untuk klasifikasi
        
        3. **CNN (Convolutional Neural Network)**
           - Deep learning untuk pattern recognition
           - Deteksi pattern dalam data time series
        
        4. **SVC (Support Vector Classifier)**
           - Machine learning klasik
           - Pemisahan data dengan margin optimal
        
        5. **Naive Bayes**
           - Probabilistic classifier
           - Cepat dan efisien
        
        6. **Meta Learner**
           - Mengkombinasikan prediksi semua model
           - Memberikan keputusan final
        """)
    
    # Training process
    if start_training:
        _process_model_training(api_source, symbol, api_key_1, api_key_2, train_tf_key, timeframe_mapping, data_size)
    
    # Display training status
    _display_training_status(symbol)


def _process_model_training(api_source, symbol, api_key_1, api_key_2, train_tf_key, timeframe_mapping, data_size):
    """Process the model training"""
    try:
        with st.spinner("📊 Mengunduh data training..."):
            train_data = load_and_process_data_enhanced(
                api_source, symbol, timeframe_mapping[train_tf_key], 
                api_key_1, api_key_2, outputsize=data_size
            )
        
        if train_data is None or len(train_data) < 100:
            st.error("❌ Data training tidak cukup. Minimal 100 data points diperlukan.")
            return
        
        st.success(f"✅ Data berhasil dimuat: {len(train_data)} candles")
        
        # Start training process
        with st.spinner("🏗️ Memulai proses training..."):
            trained_models = train_and_save_all_models(train_data, symbol, train_tf_key)
        
        if trained_models:
            st.success("🎉 Training berhasil! Semua model telah disimpan.")
            st.balloons()
            
            # Display training summary
            st.markdown("### 📊 Training Summary")
            model_count = sum(1 for v in trained_models.values() if v is not None)
            st.metric("Models Trained", f"{model_count}/6")
            
            with st.expander("📋 Model Details", expanded=False):
                for model_name, model in trained_models.items():
                    status = "✅ Success" if model is not None else "❌ Failed"
                    st.write(f"**{model_name.upper()}**: {status}")
        else:
            st.error("❌ Training gagal. Silakan coba lagi.")
    
    except Exception as e:
        st.error(f"❌ Error during training: {str(e)}")
        st.exception(e)


def _display_training_status(symbol):
    """Display current training status and saved models"""
    st.markdown("### 💾 Status Model Tersimpan")
    
    import os
    import glob
    from ..utils.formatters import sanitize_filename
    
    model_dir = 'model'
    symbol_fn = sanitize_filename(symbol)
    
    if os.path.exists(model_dir):
        # Find all model files for this symbol
        model_patterns = [
            f'lstm_model_{symbol_fn}_*.keras',
            f'xgb_model_{symbol_fn}_*.pkl',
            f'cnn_model_{symbol_fn}_*.keras',
            f'svc_model_{symbol_fn}_*.pkl',
            f'nb_model_{symbol_fn}_*.pkl',
            f'scaler_{symbol_fn}_*.pkl'
        ]
        
        found_models = {}
        for pattern in model_patterns:
            files = glob.glob(os.path.join(model_dir, pattern))
            if files:
                model_type = pattern.split('_')[0]
                found_models[model_type] = len(files)
        
        if found_models:
            st.success(f"📁 Model directory: `{model_dir}/`")
            
            cols = st.columns(3)
            model_names = ['lstm', 'xgb', 'cnn', 'svc', 'nb', 'scaler']
            
            for i, model_name in enumerate(model_names):
                col_idx = i % 3
                count = found_models.get(model_name, 0)
                status = "✅" if count > 0 else "❌"
                cols[col_idx].metric(f"{status} {model_name.upper()}", f"{count} files")
        else:
            st.info(f"ℹ️ Belum ada model untuk {symbol}. Silakan training terlebih dahulu.")
    else:
        st.info("ℹ️ Folder model belum ada. Akan dibuat otomatis saat training.")


def display_model_management():
    """Display model management interface"""
    st.markdown("### 🛠️ Manajemen Model")
    
    mgmt_cols = st.columns(3)
    
    if mgmt_cols[0].button("🗑️ Clear All Models", help="Hapus semua model yang tersimpan"):
        _clear_all_models()
    
    if mgmt_cols[1].button("📊 Model Statistics", help="Tampilkan statistik model"):
        _show_model_statistics()
    
    if mgmt_cols[2].button("🔄 Refresh Status", help="Refresh status model"):
        st.rerun()


def _clear_all_models():
    """Clear all saved models"""
    import os
    import glob
    
    model_dir = 'model'
    if os.path.exists(model_dir):
        model_files = glob.glob(os.path.join(model_dir, '*'))
        if model_files:
            for file_path in model_files:
                try:
                    os.remove(file_path)
                except Exception as e:
                    st.error(f"Failed to remove {file_path}: {e}")
            st.success(f"🗑️ Removed {len(model_files)} model files")
        else:
            st.info("ℹ️ No model files to remove")
    else:
        st.info("ℹ️ Model directory doesn't exist")


def _show_model_statistics():
    """Show model statistics"""
    import os
    import glob
    from datetime import datetime
    
    model_dir = 'model'
    if os.path.exists(model_dir):
        model_files = glob.glob(os.path.join(model_dir, '*'))
        if model_files:
            st.markdown("#### 📈 Model File Statistics")
            
            file_data = []
            total_size = 0
            
            for file_path in model_files:
                stat = os.stat(file_path)
                size_mb = stat.st_size / (1024 * 1024)
                modified = datetime.fromtimestamp(stat.st_mtime)
                
                file_data.append({
                    'File': os.path.basename(file_path),
                    'Size (MB)': f"{size_mb:.2f}",
                    'Modified': modified.strftime("%Y-%m-%d %H:%M")
                })
                total_size += size_mb
            
            import pandas as pd
            df = pd.DataFrame(file_data)
            st.dataframe(df, use_container_width=True)
            st.metric("Total Size", f"{total_size:.2f} MB")
        else:
            st.info("ℹ️ No model files found")
    else:
        st.info("ℹ️ Model directory doesn't exist")