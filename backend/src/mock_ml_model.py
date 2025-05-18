"""
Mock ML Prediction Module

This module provides a mock implementation of the ML prediction functionality
for the crypto dashboard. It's designed with a clear interface that can be
easily replaced with a real ML implementation in the future.

How to replace with real implementation:
1. Keep the function signatures the same
2. Replace the mock logic with actual ML model loading and prediction
3. Ensure the return format matches what's expected by the risk assessment module
"""

import os
import json
import random
from datetime import datetime, timedelta

# Define constants
MODEL_DIR = "/home/ubuntu/crypto_dashboard/backend/models"
DATA_DIR = "/home/ubuntu/crypto_dashboard/backend/data"

# Cache for mock predictions to ensure consistency between calls
PREDICTION_CACHE = {}

def load_and_preprocess_data(symbol="BTC_USD", sequence_length=60, train_split=0.8, 
                            for_prediction=False, last_n_rows_for_pred=None):
    """
    Mock implementation of data loading and preprocessing.
    
    In a real implementation, this would:
    1. Load historical price data
    2. Scale the data
    3. Create sequences for LSTM input
    
    Args:
        symbol: Trading pair symbol (e.g., "BTC_USD")
        sequence_length: Number of time steps in each sequence
        train_split: Proportion of data to use for training
        for_prediction: Whether data is being prepared for prediction
        last_n_rows_for_pred: Number of rows to use for prediction
        
    Returns:
        Tuple containing mock data structures that would normally be used for ML
    """
    print(f"[MOCK] Loading and preprocessing data for {symbol}")
    
    # Check if we have real data available (for reference only)
    data_path = os.path.join(DATA_DIR, f"ml_prepared_data_{symbol}.csv")
    has_real_data = os.path.exists(data_path)
    
    if has_real_data:
        print(f"[MOCK] Real data file exists at {data_path} (not used in mock implementation)")
    
    # Return mock objects that mimic what would be returned by the real implementation
    if for_prediction:
        # For prediction mode, return None for training data and mock prediction sequence
        mock_X_pred = [[random.uniform(0.4, 0.6) for _ in range(2)] for _ in range(sequence_length)]
        mock_scaler = "MOCK_SCALER"
        mock_df = {"close": [random.uniform(90000, 100000) for _ in range(sequence_length)]}
        return None, None, None, mock_X_pred, mock_scaler, mock_df
    else:
        # For training mode, return mock training and test data
        mock_X_train = [[random.uniform(0.4, 0.6) for _ in range(2)] for _ in range(100)]
        mock_y_train = [random.uniform(0.4, 0.6) for _ in range(100)]
        mock_X_test = [[random.uniform(0.4, 0.6) for _ in range(2)] for _ in range(20)]
        mock_y_test = [random.uniform(0.4, 0.6) for _ in range(20)]
        mock_scaler = "MOCK_SCALER"
        mock_df = {"close": [random.uniform(90000, 100000) for _ in range(120)]}
        return mock_X_train, mock_y_train, mock_X_test, mock_y_test, mock_scaler, mock_df

def build_lstm_model(input_shape):
    """
    Mock implementation of LSTM model building.
    
    In a real implementation, this would:
    1. Create a Sequential model with LSTM layers
    2. Compile the model
    
    Args:
        input_shape: Shape of input data for the model
        
    Returns:
        Mock model object
    """
    print(f"[MOCK] Building LSTM model with input shape {input_shape}")
    return "MOCK_MODEL"

def train_model(symbol="BTC_USD", sequence_length=60, epochs=50, batch_size=32):
    """
    Mock implementation of model training.
    
    In a real implementation, this would:
    1. Load and preprocess data
    2. Build the LSTM model
    3. Train the model on the data
    4. Save the trained model
    
    Args:
        symbol: Trading pair symbol (e.g., "BTC_USD")
        sequence_length: Number of time steps in each sequence
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Tuple containing mock model, scaler, and history
    """
    print(f"[MOCK] Training model for {symbol} with {epochs} epochs and batch size {batch_size}")
    
    # Create mock model file to simulate successful training
    model_path = os.path.join(MODEL_DIR, f"mock_lstm_model_{symbol}.json")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    with open(model_path, 'w') as f:
        json.dump({
            "model_type": "MOCK_LSTM",
            "symbol": symbol,
            "sequence_length": sequence_length,
            "created_at": datetime.now().isoformat()
        }, f)
    
    print(f"[MOCK] Model saved to {model_path}")
    return "MOCK_MODEL", "MOCK_SCALER", "MOCK_HISTORY"

def predict_price(symbol="BTC_USD", sequence_length=60):
    """
    Mock implementation of price prediction.
    
    In a real implementation, this would:
    1. Load the trained model
    2. Prepare the latest data sequence
    3. Make a prediction using the model
    4. Convert the prediction back to the original scale
    
    Args:
        symbol: Trading pair symbol (e.g., "BTC_USD")
        sequence_length: Number of time steps in each sequence
        
    Returns:
        Dictionary with prediction results
    """
    print(f"[MOCK] Predicting price for {symbol}")
    
    # Check if we have a cached prediction for this symbol
    cache_key = f"{symbol}_{sequence_length}"
    if cache_key in PREDICTION_CACHE:
        # Use cached prediction if it's less than 5 minutes old
        cached_data = PREDICTION_CACHE[cache_key]
        if datetime.now() - cached_data["timestamp"] < timedelta(minutes=5):
            print(f"[MOCK] Using cached prediction for {symbol}")
            return cached_data["prediction"]
    
    # Get the last actual price (use a realistic value for BTC)
    if symbol == "BTC_USD":
        last_price = random.uniform(90000, 100000)
    else:
        last_price = random.uniform(1000, 5000)
    
    # Generate a random prediction within a realistic range
    # For demonstration, we'll make it slightly bullish or bearish
    direction = random.choice([-1, 1, 1])  # Slightly biased toward bullish
    change_pct = random.uniform(0.5, 2.0) * direction
    predicted_price = last_price * (1 + change_pct/100)
    
    # Determine trend based on price change
    trend = "bullish" if predicted_price > last_price else "bearish"
    
    # Create prediction result
    prediction = {
        "predicted_next_close": float(predicted_price),
        "trend": trend,
        "last_actual_close": float(last_price),
        "mock_implementation": True,
        "confidence": random.uniform(0.6, 0.9)
    }
    
    # Cache the prediction
    PREDICTION_CACHE[cache_key] = {
        "timestamp": datetime.now(),
        "prediction": prediction
    }
    
    return prediction

# For testing
if __name__ == "__main__":
    print("Testing mock ML prediction module")
    prediction = predict_price("BTC_USD")
    print(f"Mock prediction for BTC_USD: {prediction}")
