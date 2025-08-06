# train_models.py
# Jalankan file ini dari terminal untuk melatih semua model AI.
# Contoh: python train_models.py --symbol "XAU/USD" --timeframe "5m" --apikey "YOUR_API_KEY"

import os
import sys
import argparse

# --- KODE PERBAIKAN PATH ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data import get_gold_data
from utils.model_manager import train_and_save_all_models

def main(args):
    """Fungsi utama untuk menjalankan proses training."""
    
    print(f"Memulai proses training untuk simbol: {args.symbol}, timeframe: {args.timeframe}")
    
    # Konversi timeframe display ke format API
    tf_map = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h'}
    api_timeframe = tf_map.get(args.timeframe)
    
    if not api_timeframe:
        print(f"Error: Timeframe '{args.timeframe}' tidak valid. Pilihan: {list(tf_map.keys())}")
        return

    print("Mengunduh data historis dalam jumlah besar...")
    training_data = get_gold_data(args.apikey, interval=api_timeframe, symbol=args.symbol, outputsize=5000)
    
    if training_data is not None and len(training_data) > 60:
        print("Data berhasil diunduh. Memulai training semua model...")
        # Timeframe key (e.g., '5m') digunakan untuk penamaan file model
        train_and_save_all_models(training_data, args.timeframe)
        print("\n=====================================")
        print(" SEMUA MODEL BERHASIL DILATIH!")
        print("=====================================")
    else:
        print("Error: Gagal mengunduh data yang cukup untuk training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skrip untuk melatih model AI trading.")
    parser.add_argument("--symbol", type=str, required=True, help="Simbol trading (e.g., 'XAU/USD')")
    parser.add_argument("--timeframe", type=str, required=True, help="Timeframe (e.g., '5m', '1h')")
    parser.add_argument("--apikey", type=str, required=True, help="API Key Twelve Data Anda")
    
    args = parser.parse_args()
    main(args)
