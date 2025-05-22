"""
Fetch Bitcoin sentiment and social metrics from Exorde API and print insights.
"""
import requests
import os
import json
import csv
from datetime import datetime, timedelta

API_KEY = "dfa8f057-3ffe-4947-956e-0cc2f13bb0f7"
BASE_URL = "https://api.exorde.network/v1"
ASSET = "bitcoin"

CACHE_FILE = os.path.join(os.path.dirname(__file__), 'exorde_btc_sentiment_cache.json')
CACHE_TTL_HOURS = 24
CSV_FILE = os.path.join(os.path.dirname(__file__), 'exorde_btc_sentiment.csv')

def load_cache():
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
    ts = datetime.fromisoformat(cache.get('timestamp'))
    if datetime.utcnow() - ts < timedelta(hours=CACHE_TTL_HOURS):
        return cache['data']
    return None

def save_cache(data):
    with open(CACHE_FILE, 'w') as f:
        json.dump({'timestamp': datetime.utcnow().isoformat(), 'data': data}, f)

def extract_sentiment_row(data):
    sentiment = data.get('sentiment', {})
    if not isinstance(sentiment, dict):
        sentiment = {'sentiment': sentiment}
    # Use endDate if available, else current UTC date
    date = sentiment.get('endDate') or datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    row = {
        'date': date[:10],
        'sentiment': sentiment.get('sentiment'),
        'percentagePositivePosts': sentiment.get('percentagePositivePosts'),
        'percentageNegativePosts': sentiment.get('percentageNegativePosts'),
        'percentageNeutralPosts': sentiment.get('percentageNeutralPosts'),
        'postsCount': sentiment.get('postsCount'),
    }
    return row

def save_sentiment_to_csv(row, csv_file=CSV_FILE):
    file_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# Example endpoint for Exorde (docs: https://docs.exorde.network/)
# We'll use the /sentiment endpoint for the asset
url = "https://api.exorde.io/sentiment"

headers = {
    "X-Exorde-Api-Version": "v1",
    "Accept": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Try to load from cache first
cached_data = load_cache()
if cached_data:
    data = cached_data
    print("[INFO] Loaded Exorde sentiment data from cache (no API call made).\n")
else:
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"[ERROR] Failed to fetch data: {response.status_code} {response.text}")
        exit(1)
    data = response.json()
    save_cache(data)
    print("[INFO] Fetched Exorde sentiment data from API and cached it.\n")

print("Exorde Bitcoin Sentiment & Social Metrics:\n")
if 'sentiment' in data:
    sentiment = data['sentiment']
    if isinstance(sentiment, dict):
        # Print all keys/values in the sentiment dict
        print("Sentiment details:")
        for k, v in sentiment.items():
            print(f"  {k}: {v}")
        # Try to print a main score if present
        main_score = sentiment.get('score') or sentiment.get('value') or sentiment.get('sentiment')
        if main_score is not None:
            print(f"Current Sentiment Score: {main_score:.3f} (range: -1=very negative, 0=neutral, 1=very positive)")
    else:
        print(f"Current Sentiment Score: {sentiment:.3f} (range: -1=very negative, 0=neutral, 1=very positive)")
if 'volume' in data:
    print(f"Social Volume (posts analyzed): {data['volume']:,}")
if 'sources' in data:
    print(f"Sources: {', '.join(data['sources'])}")
if 'updated_at' in data:
    print(f"Last Updated: {data['updated_at']}")

# Print more details if available
for key in data:
    if key not in ['sentiment', 'volume', 'sources', 'updated_at']:
        print(f"{key.capitalize()}: {data[key]}")

row = extract_sentiment_row(data)
save_sentiment_to_csv(row)
print(f"[INFO] Saved sentiment row to {CSV_FILE}\n")
