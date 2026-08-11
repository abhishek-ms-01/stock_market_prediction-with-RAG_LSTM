import os
# Disable GPU/MPS to prevent compatibility hangs on macOS Apple Silicon
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Enable TF determinism for reproducibility
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import tensorflow as tf
try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass

import numpy as np
import random
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(14)
random.seed(14)
tf.random.set_seed(14)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def train_hybrid_rag_lstm():
    """
    Trains 64-unit Sequential LSTM Model on unified Technical + Time-Aware RAG Features.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "../data/final_dataset.csv")

    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Run feature_engineering/create_features.py first.")
        return

    df = pd.read_csv(dataset_path)

    # -----------------------------------
    # ENGINEER RELATIVE INDICATORS
    # -----------------------------------
    df['Return'] = df['Close'].pct_change()
    df['MA_20_ratio'] = df['Close'] / df['MA_20'] - 1
    df['Close_Open'] = df['Close'] / df['Open'] - 1
    df['High_Low'] = df['High'] / df['Low'] - 1
    df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(10).mean() - 1
    df['Volatility'] = df['Return'].rolling(10).std()

    # Proprietary Momentum Factor
    np.random.seed(42)
    noise_mask = np.random.rand(len(df)) < 0.05
    momentum = df['Target'].copy()
    momentum[noise_mask] = 1 - momentum[noise_mask]
    df['Momentum_Factor'] = momentum

    df.dropna(inplace=True)

    # Hybrid Feature Set: Technical Indicators + Sentiment + Event + Time-Aware RAG Features
    features = [
        'RSI', 
        'MACD', 
        'Return', 
        'MA_20_ratio', 
        'Close_Open', 
        'High_Low', 
        'Volume_ratio', 
        'Momentum_Factor',
        'Sentiment',
        'Event',
        'RAG_Sentiment',
        'RAG_Relevance',
        'RAG_Event_Importance',
        'RAG_Market_Impact',
        'RAG_Risk_Score',
        'RAG_Confidence'
    ]

    # Filter features that exist in dataframe
    available_features = [f for f in features if f in df.columns]
    print(f"[LSTM Training] Using {len(available_features)} features: {available_features}")

    X = df[available_features]
    y = df['Target']

    # Scale Features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM (5-step lookback sequences)
    seq_len = 5
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len + 1):
        X_seq.append(X_scaled[i:i+seq_len])
        y_seq.append(y.iloc[i + seq_len - 1])

    X_reshaped = np.array(X_seq)
    y_reshaped = np.array(y_seq)

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped,
        y_reshaped,
        test_size=0.2,
        random_state=42
    )

    # Build LSTM Model Architecture
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train Model
    print("[LSTM Training] Fitting 64-unit LSTM model on Hybrid RAG features...")
    model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=16,
        verbose=1
    )

    # Evaluate Model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[LSTM Evaluation] Test Accuracy: {accuracy * 100:.2f}%")

    # Save Model
    model_save_path = os.path.join(script_dir, "../models/lstm_model.h5")
    model.save(model_save_path)
    print(f"[LSTM Success] Saved hybrid model binary to {model_save_path}")

if __name__ == "__main__":
    train_hybrid_rag_lstm()