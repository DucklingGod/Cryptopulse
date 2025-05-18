import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Base URL for our own backend API
BACKEND_API_BASE_URL = "http://127.0.0.1:5000/api"

def fetch_historical_prices(symbol="BTC/USD", interval="1day", outputsize="365"):
    """Fetches historical price data from our backend API."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize
    }
    try:
        response = requests.get(f"{BACKEND_API_BASE_URL}/twelvedata/time_series", params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "ok" and "values" in data:
            df = pd.DataFrame(data["values"])
            if df.empty:
                print("Fetched empty DataFrame for historical prices.")
                return pd.DataFrame()
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            
            expected_cols = ["open", "high", "low", "close", "volume"]
            for col in expected_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    print(f"Warning: Column 	'{col}'	 not found in historical price data. Filling with 0 or NaN.")
                    if col == 'volume':
                        df[col] = 0 
                    else: 
                        df[col] = float('nan') 
            return df.sort_index()
        else:
            print(f"Error fetching historical prices: {data.get('message', 'No values returned or status not ok')}")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"Request failed for historical prices: {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"Error processing JSON for historical prices: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred in fetch_historical_prices: {e}")
        return pd.DataFrame()

def fetch_news_with_sentiment(currencies=None, limit_pages=1):
    """Fetches news data with sentiment from our backend API."""
    all_news = []
    page = 1
    next_page_url = f"{BACKEND_API_BASE_URL}/cryptopanic/posts"
    initial_params = {"public": "true", "regions": "en"}
    if currencies:
        initial_params["currencies"] = currencies

    current_url = next_page_url
    current_params_for_get = initial_params.copy()

    while current_url and page <= limit_pages:
        try:
            print(f"Fetching news page: {page}, URL: {current_url} with params: {current_params_for_get if page == 1 else 'None (using full next_page_url)'}")
            if page > 1 and next_page_url: 
                response = requests.get(next_page_url)
                current_params_for_get = {} 
            else:
                response = requests.get(current_url, params=current_params_for_get)
            
            response.raise_for_status()
            data = response.json()
            if "results" in data:
                all_news.extend(data["results"])
            
            next_page_url = data.get("next") 
            current_url = next_page_url 
            page += 1
            if next_page_url: 
                time.sleep(1)  
        except requests.exceptions.RequestException as e:
            print(f"Request failed for news: {e}")
            break
        except ValueError as e:
            print(f"Error processing JSON for news: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred while fetching news: {e}")
            break
            
    df = pd.DataFrame(all_news)
    if not df.empty and "published_at" in df.columns and "sentiment" in df.columns and "title" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"])
        # Ensure published_at is timezone-naive before further processing
        if df["published_at"].dt.tz is not None:
            df["published_at"] = df["published_at"].dt.tz_localize(None)
        df["sentiment_score"] = df["sentiment"].apply(lambda x: x.get("compound") if isinstance(x, dict) else None)
        return df[["published_at", "title", "sentiment_score"]].sort_values(by="published_at")
    else:
        print("News DataFrame is empty or missing required columns (published_at, sentiment, title).")
        return pd.DataFrame()

def prepare_ml_data(symbol="BTC/USD", price_interval="1day", price_outputsize="365", news_currencies="BTC", news_pages=5):
    print(f"Starting data preparation for {symbol}...")
    prices_df = fetch_historical_prices(symbol=symbol, interval=price_interval, outputsize=price_outputsize)
    if prices_df.empty:
        print("Failed to fetch price data. Aborting ML data preparation.")
        return pd.DataFrame()
    print(f"Fetched {len(prices_df)} price records.")

    news_df = fetch_news_with_sentiment(currencies=news_currencies, limit_pages=news_pages)
    if news_df.empty:
        print("Failed to fetch news data or news data is unusable. Proceeding with price data only for ML features.")
    else:
        print(f"Fetched {len(news_df)} news records.")

    # Ensure prices_df index is DatetimeIndex and timezone-naive
    prices_df.index = prices_df.index.normalize() # Normalize to midnight
    if prices_df.index.tz is not None:
        prices_df.index = prices_df.index.tz_localize(None)

    if not news_df.empty and 'published_at' in news_df.columns and 'sentiment_score' in news_df.columns:
        # 'published_at' in news_df is already made timezone-naive in fetch_news_with_sentiment
        news_df["date"] = news_df["published_at"].dt.normalize() # This will be tz-naive
        daily_sentiment = news_df.groupby("date")["sentiment_score"].mean().rename("avg_sentiment_score")
        # daily_sentiment.index will be tz-naive because 'date' column (derived from tz-naive 'published_at') is tz-naive
        
        combined_df = prices_df.join(daily_sentiment, how="left")
        combined_df["avg_sentiment_score"] = combined_df["avg_sentiment_score"].fillna(0) 
    else:
        combined_df = prices_df.copy()
        combined_df["avg_sentiment_score"] = 0

    print("Combined DataFrame head (before feature engineering):")
    print(combined_df.head())

    if 'close' not in combined_df.columns:
        print("Error: 'close' column missing from price data. Cannot proceed with feature engineering.")
        return pd.DataFrame()
        
    combined_df["price_change_pct"] = combined_df["close"].pct_change() * 100
    for lag in [1, 3, 7]:
        combined_df[f'close_lag_{lag}'] = combined_df['close'].shift(lag)
        if 'avg_sentiment_score' in combined_df.columns:
             combined_df[f'sentiment_lag_{lag}'] = combined_df['avg_sentiment_score'].shift(lag)

    combined_df.dropna(inplace=True)

    print("DataFrame head after feature engineering and dropna:")
    print(combined_df.head())

    output_filename = f"/home/ubuntu/crypto_dashboard/backend/data/ml_prepared_data_{symbol.replace('/', '_')}.csv"
    try:
        import os
        os.makedirs("/home/ubuntu/crypto_dashboard/backend/data", exist_ok=True)
        combined_df.to_csv(output_filename)
        print(f"Prepared data saved to {output_filename}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")
        
    return combined_df

if __name__ == "__main__":
    print("Make sure the Flask backend server (src/main.py) is running on http://127.0.0.1:5000 before running this script.")
    prepared_data = prepare_ml_data(symbol="BTC/USD", price_outputsize="90", news_currencies="BTC", news_pages=2) 
    if not prepared_data.empty:
        print("\n--- Sample of Prepared Data ---")
        print(prepared_data.head())
        print(f"\nTotal prepared records: {len(prepared_data)}")
    else:
        print("Data preparation failed or returned an empty DataFrame.")
