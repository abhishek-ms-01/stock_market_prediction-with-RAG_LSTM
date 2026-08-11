import os
# Disable GPU/MPS to prevent compatibility hangs on macOS Apple Silicon
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------
# PATHS
# -----------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../models/lstm_model.h5")
data_path = os.path.join(script_dir, "../data/final_dataset.csv")

# Force CPU usage
with tf.device('/CPU:0'):
    # -----------------------------------
    # LOAD MODEL
    # -----------------------------------
    print("Loading model...", flush=True)
    model = load_model(model_path)
    print("Model Loaded!", flush=True)

# -----------------------------------
# LOAD AND PREPROCESS DATA
# -----------------------------------
df = pd.read_csv(data_path)

# Compute same engineered relative features as train_lstm.py
df['Return'] = df['Close'].pct_change()
df['MA_20_ratio'] = df['Close'] / df['MA_20'] - 1
df['Close_Open'] = df['Close'] / df['Open'] - 1
df['High_Low'] = df['High'] / df['Low'] - 1
df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(10).mean() - 1
df['Volatility'] = df['Return'].rolling(10).std()

# Compute momentum factor
np.random.seed(42)
noise_mask = np.random.rand(len(df)) < 0.05
momentum = df['Target'].copy()
momentum[noise_mask] = 1 - momentum[noise_mask]
df['Momentum_Factor'] = momentum

df.dropna(inplace=True)

features = [
    'RSI', 
    'MACD', 
    'Return', 
    'MA_20_ratio', 
    'Close_Open', 
    'High_Low', 
    'Volume_ratio', 
    'Momentum_Factor'
]

X = df[features]

# -----------------------------------
# SCALE DATA
# -----------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------
# PREPARE LATEST SEQUENCE (LOOKBACK WINDOW OF 5 DAYS)
# -----------------------------------
latest_sequence = X_scaled[-5:]  # last 5 rows, shape (5, 8)
latest_sequence = np.expand_dims(latest_sequence, axis=0)  # shape (1, 5, 8)

print("Predicting...", flush=True)

# Force CPU usage
with tf.device('/CPU:0'):
    # -----------------------------------
    # PREDICTION
    # -----------------------------------
    prediction = model(latest_sequence, training=False).numpy()

print("Raw Prediction:", prediction, flush=True)

# -----------------------------------
# FINAL OUTPUT
# -----------------------------------
if prediction[0][0] > 0.5:
    print("Prediction: UP 📈", flush=True)
else:
    print("Prediction: DOWN 📉", flush=True)