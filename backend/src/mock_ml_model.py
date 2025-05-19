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
    data_path = os.path.join(DATA_DIR, f"ml_prepared_data_{symbol}.csv")
    if not os.path.exists(data_path):
        print(f"[INFO] Data file not found for {symbol}, fetching from Alpha Vantage...")
        fetch_and_save_alphavantage(symbol)
    df = pd.read_csv(data_path)
    # --- Use all relevant features for ML, including Elliott Wave-inspired features ---
    # List of features to use (add more as needed)
    features_to_use = [
        'close',
        'avg_sentiment_score',
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_high', 'bb_low', 'bb_mid', 'bb_width',
        'ema_20', 'ema_50', 'sma_20', 'sma_50',
        'doji', 'hammer', 'bullish_engulfing',
        'price_change_pct', 'close_lag_1', 'close_lag_3', 'close_lag_7',
        'sentiment_lag_1', 'sentiment_lag_3', 'sentiment_lag_7',
        'swing_high', 'swing_low', 'zigzag', 'wave_count'
    ]
    # Only keep features that exist in the DataFrame
    features_to_use = [f for f in features_to_use if f in df.columns]
    if 'close' not in features_to_use:
        raise ValueError("CSV must contain a 'close' column.")
    data_for_scaling = df[features_to_use].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data_for_scaling)
    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i, :])
        y.append(scaled[i, 0])  # Predicting 'close'
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
    model.save(os.path.join(MODEL_DIR, f"lstm_model_{symbol}.keras"))
    np.save(os.path.join(MODEL_DIR, f"scaler_{symbol}.npy"), scaler.min_, allow_pickle=True)
    np.save(os.path.join(MODEL_DIR, f"scaler_scale_{symbol}.npy"), scaler.scale_, allow_pickle=True)
    return model, scaler, history

# --- Prediction with MC Dropout ---
def predict_price(symbol="BTC_USD", sequence_length=60, mc_dropout_passes=30):
    model_path = os.path.join(MODEL_DIR, f"lstm_model_{symbol}.keras")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{symbol}.npy")
    scaler_scale_path = os.path.join(MODEL_DIR, f"scaler_scale_{symbol}.npy")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    model = keras.models.load_model(model_path)
    scaler = MinMaxScaler()
    scaler.min_ = np.load(scaler_path, allow_pickle=True)
    scaler.scale_ = np.load(scaler_scale_path, allow_pickle=True)
    _, _, _, X_pred, _, df = load_and_preprocess_data(symbol, sequence_length, for_prediction=True)
    # MC Dropout: run prediction multiple times with dropout enabled
    # Enable dropout at inference by setting training=True
    preds = []
    for _ in range(mc_dropout_passes):
        pred_scaled = model(X_pred, training=True).numpy()[0][0]
        # Create a dummy array with the same shape as our feature set
        dummy_array = np.zeros((1, X_pred.shape[2]))
        dummy_array[0, 0] = pred_scaled  # Set the first feature (close price)
        pred_price = scaler.inverse_transform(dummy_array)[0][0]
        preds.append(pred_price)
    preds = np.array(preds)
    pred_mean = float(np.mean(preds))
    pred_std = float(np.std(preds))
    last_actual_close = df['close'].iloc[-1]
    trend = "bullish" if pred_mean > last_actual_close else "bearish"
    # Confidence: 1 - (uncertainty / |predicted change|), clipped to [0,1]
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
    print("Training ML model for BTC_USD...")
    model, scaler, history = train_model("BTC_USD")
    print("Training complete. Model and scaler saved.")
