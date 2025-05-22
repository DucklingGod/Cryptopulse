"""
Append today's Exorde sentiment to the merged price+sentiment file for real-time prediction.
"""
import os
import pandas as pd
from datetime import datetime
import requests

# Paths
BASE_DIR = os.path.dirname(__file__)
EXORDE_CSV = os.path.join(BASE_DIR, 'exorde_btc_sentiment.csv')
MERGED_CSV = os.path.abspath(os.path.join(BASE_DIR, '../data/Bitstamp_BTCUSD_d_ml_with_sentiment.csv'))
OUT_CSV = os.path.abspath(os.path.join(BASE_DIR, '../data/Bitstamp_BTCUSD_d_ml_with_sentiment_realtime.csv'))

# Load Exorde sentiment (get latest row)
exorde_df = pd.read_csv(EXORDE_CSV)
latest_exorde = exorde_df.iloc[-1]

# Load merged file
merged_df = pd.read_csv(MERGED_CSV)

def fetch_yesterday_close_price():
    # Use CoinGecko API to get yesterday's close price for Bitcoin
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': 2,
        'interval': 'daily'
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"[ERROR] Failed to fetch price from CoinGecko: {resp.status_code} {resp.text}")
        return None
    data = resp.json()
    # Get the last full day's close price (second-to-last entry)
    prices = data.get('prices', [])
    if len(prices) < 2:
        print("[ERROR] Not enough price data from CoinGecko.")
        return None
    # Each price is [timestamp, price], get the second-to-last
    close_price = prices[-2][1]
    return close_price

# Prepare new row: use latest available price row, but update date and Exorde columns
today = datetime.utcnow().strftime('%Y-%m-%d')
if today in merged_df['date'].astype(str).values:
    print(f"[INFO] Today's date ({today}) already exists in merged file. No row appended.")
else:
    # Use last available price row as template
    price_cols = [col for col in merged_df.columns if col not in exorde_df.columns or col == 'date']
    last_price_row = merged_df.iloc[-1][price_cols].copy()
    new_row = last_price_row.copy()
    new_row['date'] = today
    # Fetch yesterday's close price
    close_price = fetch_yesterday_close_price()
    if close_price is not None:
        # Set all price columns to yesterday's close
        for col in price_cols:
            if 'close' in col.lower() or 'price' in col.lower():
                new_row[col] = close_price
    # Fill in Exorde columns
    for col in exorde_df.columns:
        new_row[col] = latest_exorde[col]
    # Append and save
    merged_df = pd.concat([merged_df, pd.DataFrame([new_row])], ignore_index=True)
    # Sort by date descending (newest first)
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df = merged_df.sort_values('date', ascending=False)
    merged_df['date'] = merged_df['date'].dt.strftime('%Y-%m-%d')
    merged_df.to_csv(OUT_CSV, index=False)
    print(f"[INFO] Appended today's Exorde sentiment and yesterday's close price to {OUT_CSV} (sorted by date descending)")
