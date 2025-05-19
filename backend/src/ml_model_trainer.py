import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model # Added load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from dotenv import load_dotenv
import argparse
import requests
from datetime import datetime

load_dotenv()

# Define directories using environment variables or default relative paths
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "../data"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "../models"))

# Use environment variable for API key
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_preprocess_data(symbol="BTC_USD", sequence_length=60, train_split=0.8, for_prediction=False, last_n_rows_for_pred=None, data_path_override=None):
    """Loads prepared data, scales it, and creates sequences for LSTM.
       If for_prediction is True, it loads the full dataset to get the latest sequence and the scaler.
       last_n_rows_for_pred: If provided and for_prediction is True, uses the last N rows of the original data to form the input sequence.
       data_path_override: If provided, use this path instead of default logic.
    """
    if data_path_override:
        data_path = data_path_override
    elif symbol == "BTC_USD":
        data_path = os.path.join(DATA_DIR, "Bitstamp_BTCUSD_d.csv")
    else:
        data_path = os.path.join(DATA_DIR, f"ml_prepared_data_{symbol}.csv")
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None, None, None, None, None, None

    df = pd.read_csv(data_path)
    print(f"[DEBUG] Loaded data from: {data_path}")
    print(f"[DEBUG] Columns found: {list(df.columns)}")
    # Find the close column, case-insensitive
    close_col = next((col for col in df.columns if col.lower() == 'close'), None)
    if not close_col:
        print("Critical feature 'close' is missing (case-insensitive search). Cannot proceed.")
        return None, None, None, None, None, None
    # If avg_sentiment_score is missing, handle as before
    sentiment_col = next((col for col in df.columns if col.lower() == 'avg_sentiment_score'), None)
    features_to_use = [close_col]
    if sentiment_col:
        features_to_use.append(sentiment_col)
    else:
        print("[DEBUG] Sentiment column not found, proceeding with only close price.")

    data_for_scaling = df[features_to_use].values
    scaler = MinMaxScaler(feature_range=(0, 1))

    if for_prediction:
        scaler.fit(data_for_scaling)
        if last_n_rows_for_pred is not None and last_n_rows_for_pred >= sequence_length:
            input_data_for_pred = df[features_to_use].iloc[-last_n_rows_for_pred:].values
        else:
            input_data_for_pred = data_for_scaling[-sequence_length:]

        if len(input_data_for_pred) < sequence_length:
            print(f"Not enough data for prediction sequence. Required {sequence_length}, got {len(input_data_for_pred)}.")
            return None, None, None, None, scaler, None

        scaled_input_sequence = scaler.transform(input_data_for_pred)
        X_pred = np.array([scaled_input_sequence[-sequence_length:]])
        return None, None, None, X_pred, scaler, df[features_to_use].iloc[-sequence_length:]

    scaled_data = scaler.fit_transform(data_for_scaling)
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, :])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    training_data_len = int(np.ceil(len(X) * train_split))
    X_train, X_test = X[:training_data_len], X[training_data_len:]
    y_train, y_test = y[:training_data_len], y[training_data_len:]

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test, scaler, df[features_to_use]

def build_lstm_model(input_shape):
    """Builds an LSTM model."""
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_model(symbol="BTC_USD", sequence_length=60, epochs=50, batch_size=32, data_path_override=None):
    """Trains the LSTM model and saves it."""
    X_train, y_train, X_test, y_test, scaler, _ = load_and_preprocess_data(
        symbol=symbol,
        sequence_length=sequence_length,
        for_prediction=False,
        data_path_override=data_path_override
    )

    if X_train is None or X_train.shape[0] == 0 or X_test is None or X_test.shape[0] == 0:
        print("Not enough training or testing data available. Skipping model training.")
        return None, None, None

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001)

    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    print("Model training finished.")

    model_path = os.path.join(MODEL_DIR, f"lstm_model_{symbol}.keras")
    model.save(model_path)
    np.save(os.path.join(MODEL_DIR, f"scaler_{symbol}.npy"), scaler.min_, allow_pickle=True)
    np.save(os.path.join(MODEL_DIR, f"scaler_scale_{symbol}.npy"), scaler.scale_, allow_pickle=True)
    print(f"Model and scaler saved to {model_path}")
    return model, scaler, history

# --- Prediction with MC Dropout ---
def predict_price(symbol="BTC_USD", sequence_length=60, data_path_override=None):
    model_path = os.path.join(MODEL_DIR, f"lstm_model_{symbol}.keras")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{symbol}.npy")
    scaler_scale_path = os.path.join(MODEL_DIR, f"scaler_scale_{symbol}.npy")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}. Train the model first.")
        print("Attempting to train the model now...")
        train_model(symbol=symbol, sequence_length=sequence_length, epochs=10, batch_size=16, data_path_override=data_path_override)
        if not os.path.exists(model_path):
            print("Failed to train and find model. Cannot predict.")
            return None
    model = load_model(model_path)
    _, _, _, X_pred_sequence, scaler, last_sequence_df = load_and_preprocess_data(
        symbol=symbol,
        sequence_length=sequence_length,
        for_prediction=True,
        last_n_rows_for_pred=sequence_length,
        data_path_override=data_path_override
    )
    if X_pred_sequence is None or scaler is None or last_sequence_df is None:
        print("Failed to load or preprocess data for prediction.")
        return None
    predicted_scaled_price = model.predict(X_pred_sequence)
    num_features = scaler.n_features_in_
    dummy_array = np.zeros((len(predicted_scaled_price), num_features))
    dummy_array[:, 0] = predicted_scaled_price.ravel()
    predicted_price = scaler.inverse_transform(dummy_array)[:, 0]
    last_actual_price = last_sequence_df["close"].iloc[-1]
    trend = "bullish" if predicted_price[0] > last_actual_price else "bearish" if predicted_price[0] < last_actual_price else "neutral"
    return {"predicted_next_close": float(predicted_price[0]), "trend": trend, "last_actual_close": float(last_actual_price)}

def fetch_latest_price_from_coingecko():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    response = requests.get(url)
    if response.status_code == 200:
        price = response.json()["bitcoin"]["usd"]
        print(f"[DEBUG] Latest BTC price from CoinGecko: {price}")
        return price
    else:
        print("[ERROR] Failed to fetch price from CoinGecko")
        return None

def predict_with_realtime_price(symbol="BTC_USD", sequence_length=60, data_path_override=None):
    # Load historical data as before
    if data_path_override:
        data_path = data_path_override
    elif symbol == "BTC_USD":
        data_path = os.path.join(DATA_DIR, "Bitstamp_BTCUSD_d.csv")
    else:
        data_path = os.path.join(DATA_DIR, f"ml_prepared_data_{symbol}.csv")
    df = pd.read_csv(data_path)
    # Find the close column, case-insensitive
    close_col = next((col for col in df.columns if col.lower() == 'close'), None)
    if not close_col:
        print("Critical feature 'close' is missing (case-insensitive search). Cannot proceed.")
        return None
    # Fetch latest price
    latest_price = fetch_latest_price_from_coingecko()
    if latest_price is None:
        return None
    # Replace the last close value with the real-time price (do not append a new row)
    df_copy = df.copy()
    df_copy.loc[df_copy.index[-1], close_col] = latest_price
    # Use only the last `sequence_length` rows for prediction
    df_recent = df_copy.tail(sequence_length)
    # Save to a temp CSV and use as input
    temp_path = "temp_realtime_input.csv"
    df_recent.to_csv(temp_path, index=False)
    return predict_price(symbol=symbol, sequence_length=sequence_length, data_path_override=temp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTC_USD", help="Symbol to train/predict")
    parser.add_argument("--data_path", type=str, default=None, help="Path to data file (overrides default)")
    parser.add_argument("--realtime", action="store_true", help="Fetch latest price from CoinGecko for prediction")
    args = parser.parse_args()
    print("Ensure ml_data_preparation.py has been run to generate the data file.")
    if args.realtime:
        prediction_result = predict_with_realtime_price(symbol=args.symbol, data_path_override=args.data_path)
    else:
        prediction_result = predict_price(symbol=args.symbol, data_path_override=args.data_path)
    if prediction_result:
        print(f"\nPrediction for {args.symbol}: {prediction_result}")
    else:
        print(f"\nFailed to get prediction for {args.symbol}.")

