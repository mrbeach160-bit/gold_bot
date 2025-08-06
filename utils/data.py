# utils/data.py

import requests
import pandas as pd
from datetime import datetime
import time

def get_gold_data(api_key, interval, symbol, outputsize=500):
    """
    Mengambil data dari Twelve Data.
    FINAL VERSION: Logika paginasi diperbaiki untuk mematuhi limit 5000 per permintaan.
    """
    all_data_list = []
    end_date = None
    remaining_size = outputsize

    while remaining_size > 0:
        # Pastikan ukuran permintaan saat ini tidak pernah melebihi 5000
        current_batch_size = min(remaining_size, 5000)
        
        api_url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={current_batch_size}&apikey={api_key}"
        
        if end_date:
            api_url += f"&end_date={end_date}"

        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()

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

        except requests.exceptions.RequestException as e:
            print(f"Error koneksi saat mengambil data dari Twelve Data: {e}")
            return None
        except Exception as e:
            print(f"Error saat memproses data: {e}")
            return None
            
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