"""
ML Prediction Module

This module provides an implementation of the ML prediction functionality
for the crypto dashboard using LSTM neural networks. It includes data loading,
preprocessing, model training, and price prediction.

How to use:
1. Place your historical data CSV files in the 'data' directory. Each file should
   be named 'ml_prepared_data_<SYMBOL>.csv', where <SYMBOL> is the trading pair
   symbol (e.g., 'BTC_USD').
2. Run the 'train_model' function to train the LSTM model on the data.
3. Use the 'predict_price' function to predict the next price for a given symbol.
"""

import os
from dotenv import load_dotenv
import json
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import requests
import joblib

# Load environment variables from .env file
load_dotenv()

# Define directories using environment variables or default relative paths
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "../data"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "../models"))

# Use environment variable for API key
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")

# --- Alpha Vantage Data Fetching ---
def fetch_and_save_alphavantage(symbol="BTC_USD", market="USD", outputsize="full"):
    """
    Fetches daily historical price data from Alpha Vantage and saves as CSV for ML pipeline.
    """
    # Alpha Vantage crypto endpoint: https://www.alphavantage.co/documentation/#digital-currency-daily
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "DIGITAL_CURRENCY_DAILY",
        "symbol": symbol.split("_")[0],  # e.g., BTC
        "market": market,  # e.g., USD
        "apikey": ALPHAVANTAGE_API_KEY
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if "Time Series (Digital Currency Daily)" not in data:
        raise ValueError(f"Alpha Vantage API error or limit reached: {data}")
    ts = data["Time Series (Digital Currency Daily)"]
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()    # Use '4a. close (USD)', '4b. close (USD)', or fallback to '4. close' as the close price
    close_col = None
    for candidate in ["4a. close (USD)", "4b. close (USD)", "4. close"]:
        if candidate in df.columns:
            close_col = candidate
            break
    if not close_col:
        print(f"[ERROR] No close price in USD found for {symbol}. Available columns: {list(df.columns)}")
        raise ValueError(f"No close price in USD found for {symbol} from Alpha Vantage.")
    df_out = pd.DataFrame({
        "date": df.index,
        "close": df[close_col].astype(float)
    })
    out_path = os.path.join(DATA_DIR, f"ml_prepared_data_{symbol}.csv")
    os.makedirs(DATA_DIR, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"[INFO] Saved Alpha Vantage data to {out_path}")
    return out_path

# --- Data Loading (updated) ---
def load_and_preprocess_data(symbol="BTC_USD", sequence_length=60, train_split=0.8, for_prediction=False, last_n_rows_for_pred=None):
    # Always use absolute path for Bitstamp_BTCUSD_d_ml_with_sentiment_realtime.csv for BTC_USD real-time
    if symbol == "BTC_USD":
        # Use the real-time merged file if it exists
        realtime_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/Bitstamp_BTCUSD_d_ml_with_sentiment_realtime.csv"))
        fallback_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/Bitstamp_BTCUSD_d_ml_with_sentiment.csv"))
        data_path = realtime_path if os.path.exists(realtime_path) else fallback_path
    else:
        data_path = os.path.join(DATA_DIR, f"ml_prepared_data_{symbol}.csv")
    print(f"[DEBUG] Loading data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[DEBUG] Columns in {data_path}: {list(df.columns)}")
    # Drop rows where 'close' is not a number
    df = df[pd.to_numeric(df['close'], errors='coerce').notnull()].reset_index(drop=True)
    df['close'] = df['close'].astype(float)
    # If data is in reverse chronological order (newest first), reverse it to chronological (oldest first)
    if pd.to_datetime(df.iloc[0][df.columns[0]]) > pd.to_datetime(df.iloc[-1][df.columns[0]]):
        print("[INFO] Detected reverse chronological order. Reversing DataFrame to chronological order.")
        df = df.iloc[::-1].reset_index(drop=True)
    # Use all numeric columns except 'date' as features
    features_to_use = [col for col in df.columns if col != 'date' and pd.api.types.is_numeric_dtype(df[col])]
    if 'close' not in features_to_use:
        raise ValueError("CSV must contain a 'close' column.")
    data_for_scaling = df[features_to_use].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data_for_scaling)
    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i, :])
        y.append(scaled[i, features_to_use.index('close')])  # Predicting 'close'
    X, y = np.array(X), np.array(y)
    split = int(len(X) * train_split)
    if for_prediction:
        X_pred = scaled[-sequence_length:]
        X_pred = np.reshape(X_pred, (1, sequence_length, len(features_to_use)))
        return None, None, None, X_pred, scaler, df
    else:
        return X[:split], y[:split], X[split:], y[split:], scaler, df

# --- Model Building ---
def build_lstm_model(input_shape):
    model = keras.Sequential([
        keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(50),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Training ---
def train_model(symbol="BTC_USD", sequence_length=60, epochs=50, batch_size=32):
    X_train, y_train, X_test, y_test, scaler, df = load_and_preprocess_data(symbol, sequence_length)
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-5)
    history = model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_data=(X_test, y_test), verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, f"lstm_model_{symbol}.h5"))
    # Save the entire scaler object using joblib
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{symbol}.joblib"))
    return model, scaler, history

# --- Fetch latest price from CoinGecko ---
def fetch_latest_price_from_coingecko(symbol="BTC_USD"):
    # Only supports BTC_USD for now
    if symbol != "BTC_USD":
        raise NotImplementedError("Only BTC_USD is supported for real-time price fetch.")
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "bitcoin", "vs_currencies": "usd"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return float(data["bitcoin"]["usd"])
    except Exception as e:
        print(f"[ERROR] Failed to fetch real-time price from CoinGecko: {e}")
        return None

# --- Prediction with MC Dropout and real-time price ---
def predict_price(symbol="BTC_USD", sequence_length=60, mc_dropout_passes=30):
    model_path = os.path.join(MODEL_DIR, f"lstm_model_{symbol}.h5")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{symbol}.joblib")
    print(f"[DEBUG] Loading model from: {model_path}")
    print(f"[DEBUG] File exists: {os.path.exists(model_path)}")
    print(f"[DEBUG] File size: {os.path.getsize(model_path) if os.path.exists(model_path) else 'N/A'}")
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    # Load data
    _, _, _, X_pred, _, df = load_and_preprocess_data(symbol, sequence_length, for_prediction=True)
    # Fetch real-time price and replace last close
    latest_price = fetch_latest_price_from_coingecko(symbol)
    if latest_price is not None:
        print(f"[INFO] Using real-time price from CoinGecko: {latest_price}")
        df = df.copy()
        df.loc[df.index[-1], 'close'] = latest_price
        # Recreate X_pred with updated close
        features_to_use = [f for f in df.columns if f != 'date']
        data_for_scaling = df[features_to_use].values
        scaler.fit(data_for_scaling)
        scaled_input_sequence = scaler.transform(data_for_scaling[-sequence_length:])
        X_pred = np.reshape(scaled_input_sequence, (1, sequence_length, len(features_to_use)))
    else:
        print("[WARN] Falling back to last CSV close price for prediction.")
    # MC Dropout: run prediction multiple times with dropout enabled
    preds = []
    for _ in range(mc_dropout_passes):
        pred_scaled = model(X_pred, training=True).numpy()[0][0]
        dummy_array = np.zeros((1, X_pred.shape[2]))
        dummy_array[0, 0] = pred_scaled
        pred_price = scaler.inverse_transform(dummy_array)[0][0]
        preds.append(pred_price)
    preds = np.array(preds)
    pred_mean = float(np.mean(preds))
    pred_std = float(np.std(preds))
    last_actual_close = df['close'].iloc[-1]
    trend = "bullish" if pred_mean > last_actual_close else "bearish"
    pred_change = abs(pred_mean - last_actual_close)
    confidence = 1.0 - min(pred_std / (pred_change + 1e-8), 1.0) if pred_change > 0 else 0.0
    return {
        "predicted_next_close": pred_mean,
        "trend": trend,
        "last_actual_close": float(last_actual_close),
        "confidence": confidence,
        "uncertainty": pred_std
    }

# --- Utility Functions ---
def prepare_all_coins(symbols, market="USD"):
    """
    Bulk fetch and save historical price data for all given symbols.
    Example: prepare_all_coins(["BTC_USD", "ETH_USD", "SOL_USD"])
    """
    for symbol in symbols:
        try:
            print(f"[INFO] Preparing data for {symbol}...")
            fetch_and_save_alphavantage(symbol, market=market)
        except Exception as e:
            print(f"[ERROR] Could not prepare data for {symbol}: {e}")

# For testing
if __name__ == "__main__":
    # Always retrain the model before prediction to ensure input shape matches new features
    print("\n[INFO] Training ML model for BTC_USD with current features...")
    model, scaler, history = train_model("BTC_USD")
    print("Training complete. Model and scaler saved.")
    print("\n[INFO] Running real-time prediction using Bitstamp_BTCUSD_d_ml_with_sentiment_realtime.csv...")
    result = predict_price("BTC_USD")
    print("\n[RESULT] Real-time ML Prediction for BTC/USD:")
    print(f"  Last Actual Close: {result['last_actual_close']}")
    print(f"  Predicted Next Close: {result['predicted_next_close']}")
    print(f"  Trend: {result['trend']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Uncertainty (std): {result['uncertainty']:.2f}")
