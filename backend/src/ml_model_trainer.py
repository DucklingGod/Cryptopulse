import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model # Added load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# Define the path to the prepared data
DATA_DIR = "/home/ubuntu/crypto_dashboard/backend/data"
MODEL_DIR = "/home/ubuntu/crypto_dashboard/backend/models"

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_preprocess_data(symbol="BTC_USD", sequence_length=60, train_split=0.8, for_prediction=False, last_n_rows_for_pred=None):
    """Loads prepared data, scales it, and creates sequences for LSTM.
       If for_prediction is True, it loads the full dataset to get the latest sequence and the scaler.
       last_n_rows_for_pred: If provided and for_prediction is True, uses the last N rows of the original data to form the input sequence.
    """
    data_path = os.path.join(DATA_DIR, f"ml_prepared_data_{symbol}.csv")
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None, None, None, None, None, None

    df = pd.read_csv(data_path, index_col="datetime", parse_dates=True)

    features_to_use = ["close", "avg_sentiment_score"]
    if not all(feature in df.columns for feature in features_to_use):
        print(f"Missing one or more required features ({features_to_use}) in the data.")
        if "close" in df.columns:
            print("Proceeding with only close price.")
            features_to_use = ["close"]
        else:
            print("Critical feature close is missing. Cannot proceed.")
            return None, None, None, None, None, None

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

def train_model(symbol="BTC_USD", sequence_length=60, epochs=50, batch_size=32):
    """Trains the LSTM model and saves it."""
    X_train, y_train, X_test, y_test, scaler, _ = load_and_preprocess_data(
        symbol=symbol,
        sequence_length=sequence_length,
        for_prediction=False
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
    print(f"Model saved to {model_path}")
    return model, scaler, history

def predict_price(symbol="BTC_USD", sequence_length=60):
    """Loads a trained model and predicts the next price point."""
    model_path = os.path.join(MODEL_DIR, f"lstm_model_{symbol}.keras")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}. Train the model first.")
        print("Attempting to train the model now...")
        train_model(symbol=symbol, sequence_length=sequence_length, epochs=10, batch_size=16)
        if not os.path.exists(model_path):
            print("Failed to train and find model. Cannot predict.")
            return None

    model = load_model(model_path)
    
    _, _, _, X_pred_sequence, scaler, last_sequence_df = load_and_preprocess_data(
        symbol=symbol,
        sequence_length=sequence_length,
        for_prediction=True,
        last_n_rows_for_pred=sequence_length
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

if __name__ == "__main__":
    print("Ensure ml_data_preparation.py has been run to generate the data file.")
    # train_model(symbol="BTC_USD", epochs=2)
    prediction_result = predict_price(symbol="BTC_USD")
    if prediction_result:
        print(f"\nPrediction for BTC_USD: {prediction_result}")
    else:
        print("\nFailed to get prediction for BTC_USD.")

