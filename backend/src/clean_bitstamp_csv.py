import pandas as pd
import sys

# Path to your Bitstamp data file (edit if needed)
data_path = '../data/Bitstamp_BTCUSD_d.csv'
output_path = '../data/Bitstamp_BTCUSD_d_cleaned.csv'

def clean_bitstamp_csv(input_path, output_path):
    df = pd.read_csv(input_path)
    # Drop rows where 'close' is not a number
    df = df[pd.to_numeric(df['close'], errors='coerce').notnull()].reset_index(drop=True)
    df['close'] = df['close'].astype(float)
    # Remove duplicate header rows if present
    df = df[df['date'] != 'date']
    df.to_csv(output_path, index=False)
    print(f"Cleaned file saved to: {output_path}")
    print(f"Rows: {len(df)}")

if __name__ == '__main__':
    clean_bitstamp_csv(data_path, output_path)
