# main.py - Dengan Perbaikan Path
import sys
import os

# --- Kode Perbaikan Dimulai Di Sini ---
# 1. Dapatkan path absolut dari direktori tempat main.py berada (yaitu, /root/gold_bot/)
project_root = os.path.dirname(os.path.abspath(__file__))

# 2. Tambahkan path ini ke daftar tempat Python mencari modul.
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- Kode Perbaikan Selesai ---

# Sekarang, import Anda akan berjalan seperti biasa
from streamlit_app.app import main

if __name__ == "__main__":
    main()
