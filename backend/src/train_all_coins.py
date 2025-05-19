"""
Bulk model trainer for all coins in your project.
Run this script to train LSTM models for all supported coins.
"""

from mock_ml_model import prepare_all_coins, train_model

# List all symbols you want to support here:
COIN_SYMBOLS = [
    "BTC_USD",
    "ETH_USD",
    "SOL_USD",
    # Add more symbols as needed
]

def main():
    print("[INFO] Downloading historical data for all coins...")
    prepare_all_coins(COIN_SYMBOLS)
    print("[INFO] Training models for all coins...")
    for symbol in COIN_SYMBOLS:
        try:
            print(f"[INFO] Training model for {symbol}...")
            train_model(symbol)
        except Exception as e:
            print(f"[ERROR] Could not train model for {symbol}: {e}")
    print("[INFO] All done.")

if __name__ == "__main__":
    main()
