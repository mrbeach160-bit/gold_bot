# data_loader.py - Data loading and processing functions
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import data utilities
try:
    from utils.data import get_gold_data
except ImportError:
    st.error("utils/data.py not found. Please ensure data utilities are available.")

# Import indicators
try:
    from utils.indicators import add_indicators
except ImportError:
    st.error("utils/indicators.py not found. Please ensure indicators module is available.")

# Check for Binance availability
try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False


def get_binance_data(api_key, api_secret, interval, symbol, outputsize=500):
    """
    Mengambil data historis dari Binance dan memformatnya.
    Kini mendukung pengambilan data > 1000 dengan pagination.
    """
    try:
        client = Client(api_key, api_secret)
        
        # Map interval untuk Binance
        interval_map = {
            '1min': Client.KLINE_INTERVAL_1MINUTE,
            '5min': Client.KLINE_INTERVAL_5MINUTE,
            '15min': Client.KLINE_INTERVAL_15MINUTE,
            '30min': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1day': Client.KLINE_INTERVAL_1DAY
        }
        
        binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_1HOUR)
        
        # Batasan Binance: maksimal 1000 per request
        # Untuk data > 1000, kita akan menggunakan pagination
        all_klines = []
        limit_per_request = min(outputsize, 1000)
        
        if outputsize <= 1000:
            klines = client.get_klines(symbol=symbol, interval=binance_interval, limit=limit_per_request)
            all_klines.extend(klines)
        else:
            # Pagination untuk data > 1000
            remaining = outputsize
            end_time = None
            
            while remaining > 0:
                current_limit = min(remaining, 1000)
                
                if end_time:
                    klines = client.get_klines(
                        symbol=symbol, 
                        interval=binance_interval, 
                        limit=current_limit,
                        endTime=end_time
                    )
                else:
                    klines = client.get_klines(
                        symbol=symbol, 
                        interval=binance_interval, 
                        limit=current_limit
                    )
                
                if not klines:
                    break
                    
                all_klines.extend(klines)
                remaining -= len(klines)
                
                # Set end_time ke timestamp kline pertama untuk request berikutnya
                end_time = int(klines[0][0]) - 1
        
        # Convert ke DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                   'close_time', 'quote_asset_volume', 'number_of_trades',
                   'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        
        df = pd.DataFrame(all_klines, columns=columns)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Convert tipe data
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Reverse agar data terbaru di bawah (untuk konsistensi dengan Twelve Data)
        df = df.iloc[::-1].reset_index(drop=True)
        
        # Validasi data
        if df.empty:
            st.error(f"❌ Data Binance kosong untuk {symbol}")
            return None
            
        if len(df) < outputsize * 0.5:  # Jika data < 50% dari yang diminta
            st.warning(f"⚠️ Data yang diperoleh ({len(df)}) kurang dari yang diminta ({outputsize})")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error mengambil data Binance: {str(e)}")
        return None


def load_and_process_data_enhanced(api_source, symbol, interval, api_key_1, api_key_2=None, outputsize=500):
    """Enhanced data loading v8.3 - Simplified untuk Twelve Data dan Binance saja"""
    data = None
    try:
        if api_source == 'Twelve Data':
            if not api_key_1 or api_key_1.strip() == "":
                raise ValueError("API Key Twelve Data tidak valid")
            data = get_gold_data(api_key_1, interval=interval, symbol=symbol, outputsize=outputsize)
        
        elif api_source == 'Binance':
            if not BINANCE_AVAILABLE:
                raise ValueError("Binance library tidak tersedia. Install dengan: pip install python-binance")
            if not api_key_1 or not api_key_2 or api_key_1.strip() == "" or api_key_2.strip() == "":
                raise ValueError("API Key & Secret Binance tidak valid")
            symbol_binance = symbol.replace('/', '')
            data = get_binance_data(api_key_1, api_key_2, interval, symbol_binance, outputsize)

        if data is None or data.empty:
            raise ValueError(f"Data kosong dari {api_source}. Periksa koneksi, API Key, atau format simbol.")

        # Rest of the processing remains the same
        required_cols = ['open', 'high', 'low', 'close']
        if 'volume' in data.columns:
            required_cols.append('volume')

        for col in required_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            else:
                raise ValueError(f"Data tidak lengkap, kolom '{col}' tidak ditemukan.")

        data.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

        if data.empty:
            raise ValueError("Data menjadi kosong setelah pembersihan. Kemungkinan format data dari API salah atau tidak ada data valid.")

        if 'XAU' in symbol.upper() and (data['close'] > 10000).any():
            st.warning(f"Peringatan: Terdeteksi harga Emas > $10,000. Data mungkin tidak akurat.")
        if 'EUR' in symbol.upper() and (data['close'] > 2).any():
            st.warning(f"Peringatan: Terdeteksi harga EUR/USD > 2.0. Data mungkin tidak akurat.")

        data = add_indicators(data)

        if data.empty:
            raise ValueError("Data kosong setelah menambahkan indikator teknis.")

        st.success(f"✅ Data berhasil dimuat: {len(data)} candles dari {api_source}")

        return data

    except Exception as e:
        error_msg = str(e)
        
        if "API key" in error_msg.lower():
            st.error(f"🔑 {error_msg}")
        elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            st.error(f"🌐 Masalah koneksi: {error_msg}")
        elif "rate limit" in error_msg.lower():
            st.error(f"⏱️ Rate limit tercapai: {error_msg}")
        elif "symbol" in error_msg.lower():
            st.error(f"📊 Masalah simbol: {error_msg}")
        else:
            st.error(f"❌ Error: {error_msg}")
        
        return None