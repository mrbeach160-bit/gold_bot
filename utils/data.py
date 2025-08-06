# utils/data.py

import requests
import pandas as pd
from datetime import datetime
import time
import os
import sys

# Add project root to path for config import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import ConfigManager, get_config

def get_gold_data(api_key=None, interval=None, symbol=None, outputsize=500):
    """
    Mengambil data dari Twelve Data dengan integrasi configuration system.
    FINAL VERSION: Logika paginasi diperbaiki untuk mematuhi limit 5000 per permintaan.
    
    Args:
        api_key: API key (optional, will use config if not provided)
        interval: Data interval (optional, will use config if not provided)
        symbol: Trading symbol (optional, will use config if not provided)
        outputsize: Number of data points to fetch
    """
    # Get configuration values if not provided
    try:
        config = get_config()
        
        if api_key is None:
            api_key = config.api.twelve_data_key
            if not api_key:
                raise ValueError("Twelve Data API key not configured. Please set TWELVE_DATA_API_KEY environment variable or pass api_key parameter.")
        
        if interval is None:
            interval = config.trading.timeframe
        
        if symbol is None:
            symbol = config.trading.symbol
            
        # Use configured timeout and retries
        timeout = config.api.api_timeout
        max_retries = config.api.max_retries
        
    except RuntimeError:
        # Configuration not loaded, use parameters as provided
        if api_key is None:
            raise ValueError("API key is required when configuration is not loaded")
        if interval is None:
            raise ValueError("Interval is required when configuration is not loaded")
        if symbol is None:
            raise ValueError("Symbol is required when configuration is not loaded")
        
        # Use default values
        timeout = 30
        max_retries = 3
    
    all_data_list = []
    end_date = None
    remaining_size = outputsize

    while remaining_size > 0:
        # Pastikan ukuran permintaan saat ini tidak pernah melebihi 5000
        current_batch_size = min(remaining_size, 5000)
        
        api_url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={current_batch_size}&apikey={api_key}"
        
        if end_date:
            api_url += f"&end_date={end_date}"

        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.get(api_url, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                break  # Success, exit retry loop
                
            except requests.exceptions.Timeout:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Request timeout after {max_retries} attempts")
                    return None
                time.sleep(2 ** retry_count)  # Exponential backoff
                continue
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Error koneksi saat mengambil data dari Twelve Data: {e}")
                    return None
                time.sleep(2 ** retry_count)  # Exponential backoff
                continue

        try:
            if 'values' not in data:
                if data.get('code') == 429:
                    print("Batas API per menit tercapai. Menunggu 60 detik...")
                    time.sleep(60)
                    continue # Coba lagi permintaan yang sama setelah menunggu
                
                print("Respons API tidak mengandung 'values' atau ada error lain. Pesan dari server:")
                print(data)
                break

            df = pd.DataFrame(data['values'])
            
            if df.empty:
                break
            
            # Tambahkan batch data ke dalam list
            all_data_list.append(df)
            
            # Update end_date untuk iterasi berikutnya
            end_date = df['datetime'].min()
            
            # Kurangi sisa data yang perlu diambil
            remaining_size -= len(df)
            
            # Beri jeda 1 detik untuk API gratis
            time.sleep(1)
            break  # Successfully processed this batch, exit retry loop

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Error saat memproses data: {e}")
                return None
            time.sleep(2 ** retry_count)  # Exponential backoff
            
    if not all_data_list:
        return None

    # Gabungkan semua DataFrame dalam list menjadi satu
    all_data = pd.concat(all_data_list)
    
    # Hapus duplikat dan format DataFrame
    all_data.drop_duplicates(subset='datetime', keep='first', inplace=True)
    all_data['datetime'] = pd.to_datetime(all_data['datetime'])
    all_data.set_index('datetime', inplace=True)
    
    numeric_cols = ['open', 'high', 'low', 'close']
    if 'volume' not in all_data.columns:
        all_data['volume'] = 0  # fallback jika tidak tersedia
    numeric_cols.append('volume')
    for col in numeric_cols:
        all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
        
    all_data.sort_index(inplace=True)
    final_data = all_data.tail(outputsize)

    return final_data