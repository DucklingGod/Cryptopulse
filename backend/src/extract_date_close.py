import pandas as pd

# Input and output paths (edit if needed)
input_path = '../data/Bitstamp_BTCUSD_d.csv'
output_path = '../data/Bitstamp_BTCUSD_d_ml.csv'

def extract_date_close(input_path, output_path):
    df = pd.read_csv(input_path)
    # Only keep 'date' and 'close' columns
    df_out = df[['date', 'close']].copy()
    # Drop rows where 'close' is not a number
    df_out = df_out[pd.to_numeric(df_out['close'], errors='coerce').notnull()].reset_index(drop=True)
    df_out['close'] = df_out['close'].astype(float)
    # Remove duplicate header rows if present
    df_out = df_out[df_out['date'] != 'date']
    df_out.to_csv(output_path, index=False)
    print(f"Extracted file saved to: {output_path}")
    print(df_out.head())

if __name__ == '__main__':
    extract_date_close(input_path, output_path)
