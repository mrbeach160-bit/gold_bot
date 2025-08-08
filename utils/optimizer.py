# utils/optimizer.py

import pandas as pd
from itertools import product
from .backtester import run_backtest # Mengimpor fungsi backtest yang sudah ada

def run_optimization(data, initial_capital, sl_range, tp_range):
    """
    Menjalankan optimisasi parameter SL dan TP.
    
    sl_range: Tuple (start, stop, step) untuk Stop Loss. Contoh: (50, 151, 10)
    tp_range: Tuple (start, stop, step) untuk Take Profit. Contoh: (100, 201, 10)
    """
    
    sl_values = range(sl_range[0], sl_range[1], sl_range[2])
    tp_values = range(tp_range[0], tp_range[1], tp_range[2])
    
    # Buat semua kemungkinan kombinasi dari SL dan TP
    param_grid = list(product(sl_values, tp_values))
    
    all_results = []
    
    # Buat progress bar di Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, params in enumerate(param_grid):
        sl, tp = params
        
        # Jalankan backtest untuk setiap kombinasi
        metrics, _ = run_backtest(data.copy(), initial_capital, sl, tp)
        
        # Simpan hasil
        result = {
            'Stop Loss (pips)': sl,
            'Take Profit (pips)': tp,
            'Total P/L (points)': metrics.get("Total Profit/Loss (points)", 0),
            'Win Rate (%)': metrics.get("Win Rate (%)", 0),
            'Total Trades': metrics.get("Total Trades", 0),
            'Max Drawdown (%)': metrics.get("Max Drawdown (%)", 0),
            'Profit Factor': metrics.get("Profit Factor", 0)
        }
        all_results.append(result)
        
        # Update progress bar
        progress_percentage = (i + 1) / len(param_grid)
        progress_bar.progress(progress_percentage)
        status_text.text(f"Menguji kombinasi {i+1}/{len(param_grid)} (SL: {sl}, TP: {tp})...")

    status_text.text("Optimisasi selesai!")
    
    return pd.DataFrame(all_results)

# Perlu st di sini untuk progress bar, jadi kita import
try:
    import streamlit as st
except ImportError:
    # Jika file dijalankan di luar streamlit, buat objek dummy
    class DummyStreamlit:
        def progress(self, *args, **kwargs): return self
        def empty(self, *args, **kwargs): return self
        def text(self, *args, **kwargs): pass
    st = DummyStreamlit()
