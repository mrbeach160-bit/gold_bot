import requests
import pandas as pd
from datetime import datetime

def fetch_gold_data(api_key, interval='1h', outputsize=500):
    symbol = 'XAU/USD'
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    if 'values' not in data:
        raise Exception(f"API Error: {data.get('message', 'Unknown error')}")

    df = pd.DataFrame(data['values'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.astype({
        'open': 'float',
        'high': 'float',
        'low': 'float',
        'close': 'float'
    })
    return df