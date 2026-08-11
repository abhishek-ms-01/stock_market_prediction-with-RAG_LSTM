import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_project_root():
    """Returns absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_dataset(filename="final_dataset.csv"):
    """Loads CSV dataset from data directory using root-relative pathing."""
    root = get_project_root()
    path = os.path.join(root, "data", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)

def engineer_features(df):
    """
    Computes standard 8 relative indicators used by LSTM model:
    ['RSI', 'MACD', 'Return', 'MA_20_ratio', 'Close_Open', 'High_Low', 'Volume_ratio', 'Momentum_Factor']
    """
    df = df.copy()

    # Flatten columns if multi-indexed (e.g. from yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Convert to 1D Series if needed
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns and isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    if 'Return' not in df.columns:
        df['Return'] = df['Close'].pct_change()

    if 'MA_20' in df.columns and 'MA_20_ratio' not in df.columns:
        df['MA_20_ratio'] = df['Close'] / df['MA_20'] - 1

    if 'Close_Open' not in df.columns:
        df['Close_Open'] = df['Close'] / df['Open'] - 1

    if 'High_Low' not in df.columns:
        df['High_Low'] = df['High'] / df['Low'] - 1

    if 'Volume_ratio' not in df.columns and 'Volume' in df.columns:
        df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(10).mean() - 1

    if 'Volatility' not in df.columns:
        df['Volatility'] = df['Return'].rolling(10).std()

    if 'Momentum_Factor' not in df.columns:
        df['Momentum_Factor'] = df['Return'].apply(lambda x: 1 if x > 0 else 0)

    return df

def prepare_lstm_input(df, features, seq_len=5):
    """
    Scales features and creates 3D sequence array of shape (batch, seq_len, num_features).
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    if len(scaled_data) < seq_len:
        raise ValueError(f"Insufficient rows ({len(scaled_data)}) for sequence length {seq_len}")

    latest_seq = scaled_data[-seq_len:]
    latest_seq_3d = np.expand_dims(latest_seq, axis=0)

    return latest_seq_3d, scaler
