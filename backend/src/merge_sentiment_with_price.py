"""
Merge Twitter sentiment CSV with price CSV for ML training.
Adds avg_sentiment_score to each date in the price CSV.
"""
import os
import pandas as pd

# Paths (edit as needed)
PRICE_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/Bitstamp_BTCUSD_d_ml.csv"))
SENTIMENT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/bitcoin_twitter_sentiment.csv"))
EXORDE_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "exorde_btc_sentiment.csv"))
OUT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/Bitstamp_BTCUSD_d_ml_with_sentiment.csv"))

# Load data
price_df = pd.read_csv(PRICE_CSV)
sentiment_df = pd.read_csv(SENTIMENT_CSV)
exorde_df = pd.read_csv(EXORDE_CSV)

# Ensure date columns are in the same format
def normalize_date(df, col):
    df[col] = pd.to_datetime(df[col]).dt.date
    return df

price_df = normalize_date(price_df, 'date')
sentiment_df = normalize_date(sentiment_df, 'date')
exorde_df = normalize_date(exorde_df, 'date')

# Merge Twitter and Exorde sentiment into price data
merged = pd.merge(price_df, sentiment_df, on='date', how='left')
merged = pd.merge(merged, exorde_df, on='date', how='left', suffixes=('', '_exorde'))

# Fill missing sentiment with 0 (neutral) or NaN for other Exorde columns
merged['avg_sentiment_score'] = merged['avg_sentiment_score'].fillna(0)
for col in ['sentiment', 'percentagePositivePosts', 'percentageNegativePosts', 'percentageNeutralPosts', 'postsCount']:
    if col in merged:
        merged[col] = merged[col].fillna(0)

# Save merged file
merged.to_csv(OUT_CSV, index=False)
print(f"[INFO] Merged file saved to {OUT_CSV}")
