import os
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

# Ensure project root in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# TensorFlow CPU Determinism
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from data_ingestion.live_news_fetcher import LiveNewsFetcher, compute_weighted_composite_sentiment

def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Binary Focal Loss for handling class imbalance during trend training:
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        epsilon = tf.keras.backend.epsilon()
        pt_1 = tf.clip_by_value(pt_1, epsilon, 1.0 - epsilon)
        pt_0 = tf.clip_by_value(pt_0, epsilon, 1.0 - epsilon)
        
        loss_1 = -alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1)
        loss_0 = -(1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0)
        return tf.reduce_mean(loss_1 + loss_0)
        
    return focal_loss_fixed

class DynamicLSTMOnlineTrainer:
    """
    Dynamic Real-Time Online Trainer & Fine-Tuner for Stock Market LSTM Models.
    Downloads up-to-date price data & live news sentiment for any ticker,
    engineers 5-step sequence lookbacks, trains/fine-tunes a 64-unit LSTM,
    uses Focal Loss for trend imbalance, and caches trained model instances per ticker.
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        self.live_fetcher = LiveNewsFetcher()
        self.cache_ttl = cache_ttl_seconds
        self.model_cache = {}  # ticker -> { 'model': model, 'scaler': scaler, 'meta': dict, 'timestamp': float }

    def build_lstm_architecture(self, input_shape: tuple) -> Sequential:
        """Constructs 64-unit Sequential LSTM classification architecture with Focal Loss."""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.005), loss=binary_focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])
        return model

    def prepare_up_to_date_dataset(self, ticker: str, period: str = "1y"):
        """Downloads up-to-date price history & live news sentiment, engineering relative indicators."""
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            raise ValueError(f"No price data available for ticker: {ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        
        # Engineer Relative Indicators
        df['Return'] = df['Close'].pct_change()
        
        # MA_20
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_20_ratio'] = (df['Close'] / df['MA_20']) - 1.0
        
        df['Close_Open'] = (df['Close'] / df['Open']) - 1.0
        df['High_Low'] = (df['High'] / df['Low']) - 1.0
        
        vol_mean = df['Volume'].rolling(10).mean().replace(0, np.nan)
        df['Volume_ratio'] = (df['Volume'] / vol_mean) - 1.0
        
        df['Volatility'] = df['Return'].rolling(10).std()
        
        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50.0)

        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26

        # Target: 1 if Next Day Return > 0 else 0
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        # Live Breaking News Sentiment Integration with Credibility & Time Decay
        ticker_clean = ticker.replace(".NS", "").replace(".BO", "")
        df_live_news = self.live_fetcher.get_live_news_for_ticker(ticker_clean, ticker)
        
        weighted_sentiment = compute_weighted_composite_sentiment(df_live_news)
        avg_sentiment = float(df_live_news['sentiment'].mean()) if not df_live_news.empty and 'sentiment' in df_live_news.columns else 0.0
            
        df['Sentiment'] = weighted_sentiment
        df['Momentum_Factor'] = (df['Return'] > 0).astype(float)

        # Feature Set
        features = [
            'RSI', 'MACD', 'Return', 'MA_20_ratio', 
            'Close_Open', 'High_Low', 'Volume_ratio', 'Volatility'
        ]
        
        df.dropna(subset=features, inplace=True)
        if len(df) < 10:
            raise ValueError(f"Insufficient historical data after feature engineering for {ticker}.")

        return df, features, weighted_sentiment, avg_sentiment

    def train_up_to_date_model(self, ticker: str, period: str = "1y", force_retrain: bool = False):
        """
        Dynamically trains/fine-tunes the LSTM on up-to-date data for the given ticker.
        Caches model with TTL to maximize responsiveness.
        """
        now_ts = time.time()
        
        # Check Cache
        if not force_retrain and ticker in self.model_cache:
            entry = self.model_cache[ticker]
            if (now_ts - entry['timestamp']) < self.cache_ttl:
                meta = entry['meta'].copy()
                meta['cache_hit'] = True
                return entry['model'], entry['scaler'], meta

        print(f"[Dynamic LSTM Online Trainer] Retraining model on up-to-date data for {ticker}...")
        df, features, weighted_sentiment, avg_sentiment = self.prepare_up_to_date_dataset(ticker, period=period)

        X_df = df[features]
        y_df = df['Target']

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_df)

        seq_len = 5
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - seq_len + 1):
            X_seq.append(X_scaled[i:i+seq_len])
            y_seq.append(y_df.iloc[i + seq_len - 1])

        X_reshaped = np.array(X_seq, dtype=np.float32)
        y_reshaped = np.array(y_seq, dtype=np.float32)

        input_shape = (seq_len, len(features))
        model = self.build_lstm_architecture(input_shape)

        # Train for 10 epochs on CPU using Focal Loss
        with tf.device('/CPU:0'):
            history = model.fit(
                X_reshaped, y_reshaped,
                epochs=10,
                batch_size=16,
                verbose=0
            )

        train_loss = float(history.history['loss'][-1]) if 'loss' in history.history else 0.5
        train_acc = float(history.history['accuracy'][-1]) if 'accuracy' in history.history else 0.5

        # Transaction Friction Deduction Breakdown (0.15% brokerage + 0.05% slippage)
        friction_rate = 0.0020
        last_return = float(df['Return'].iloc[-1]) if not df.empty else 0.0
        net_expected_return = round(last_return - friction_rate, 4)

        meta = {
            "ticker": ticker,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_points": len(df),
            "seq_count": len(X_reshaped),
            "features": features,
            "live_sentiment_score": round(weighted_sentiment, 3),
            "raw_mean_sentiment": round(avg_sentiment, 3),
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "focal_loss_enabled": True,
            "estimated_friction_rate": friction_rate,
            "net_expected_return": net_expected_return,
            "model_trained_on_latest_data": True,
            "cache_hit": False
        }

        self.model_cache[ticker] = {
            "model": model,
            "scaler": scaler,
            "meta": meta,
            "timestamp": now_ts
        }

        print(f"[Dynamic LSTM Online Trainer Success] Model for {ticker} trained on {len(df)} up-to-date data points (Focal Loss: {train_loss:.4f}, Acc: {train_acc:.4f}).")
        return model, scaler, meta

if __name__ == "__main__":
    trainer = DynamicLSTMOnlineTrainer()
    model, scaler, meta = trainer.train_up_to_date_model("RELIANCE.NS", period="6mo", force_retrain=True)
    print("Training Metadata:", meta)

