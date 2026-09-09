from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import json
import os

# Import original logic exactly
from risk.risk_analyzer import calculate_stock_risk_metrics
from market_regime.regime_detector import MarketRegimeDetector
from portfolio.portfolio_advisor import PortfolioRecommendationEngine
from chatbot.chatbot import StockAssistantChatbot
from data_ingestion.live_news_fetcher import LiveNewsFetcher
from data_ingestion.upstox_fetcher import UpstoxDataFetcher
from prediction.online_trainer import DynamicLSTMOnlineTrainer

app = FastAPI(title="AI Stock Market Prediction API")
live_fetcher = LiveNewsFetcher()
upstox_fetcher = UpstoxDataFetcher()
online_trainer = DynamicLSTMOnlineTrainer()
chatbot_instance = StockAssistantChatbot()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# EXACT STOCKS DICT FROM APP.PY
# ---------------------------------------------------
stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC Limited": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Titan Company": "TITAN.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "HCL Technologies": "HCLTECH.NS",
    "NTPC Limited": "NTPC.NS",
    "Power Grid Corp": "POWERGRID.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Coal India": "COALINDIA.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "Grasim Industries": "GRASIM.NS",
    "ONGC": "ONGC.NS",
    "Tech Mahindra": "TECHM.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Nestle India": "NESTLEIND.NS",
    "Cipla": "CIPLA.NS",
    "Dr. Reddy's Labs": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Wipro Limited": "WIPRO.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "SBI Life Insurance": "SBILIFE.NS",
    "HDFC Life Insurance": "HDFCLIFE.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Trent Limited": "TRENT.NS",
    "Bharat Electronics": "BEL.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Hindustan Aeronautics": "HAL.NS",
    "Suzlon Energy": "SUZLON.NS",
    "Indian Railway Finance": "IRFC.NS",
    "Tata Power": "TATAPOWER.NS",
    "BHEL": "BHEL.NS",
    "IREDA": "IREDA.NS",
    "Vedanta Limited": "VEDL.NS"
}

# ---------------------------------------------------
# NATIVE TECHNICAL INDICATORS FROM APP.PY
# ---------------------------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

def calculate_sma(series, window=20):
    return series.rolling(window=window).mean()

def fetch_stock_data_raw(symbol: str, timeframe: str, interval: str = "1d"):
    df_raw = yf.download(symbol, period=timeframe, interval=interval)
    if df_raw.empty:
        return pd.DataFrame()
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = [c[0] for c in df_raw.columns]
    df_raw.reset_index(inplace=True)
    if 'Datetime' in df_raw.columns:
        df_raw.rename(columns={'Datetime': 'Date'}, inplace=True)
    if 'index' in df_raw.columns:
        df_raw.rename(columns={'index': 'Date'}, inplace=True)
    return df_raw

def process_stock_data(ticker: str, period: str):
    df_data = fetch_stock_data_raw(ticker, period)
    if df_data.empty:
        raise HTTPException(status_code=404, detail=f"Unable to fetch data for {ticker}")

    df = df_data.copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns and isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    close_series = df['Close'].astype(float)
    open_series = df['Open'].astype(float)
    high_series = df['High'].astype(float)
    low_series = df['Low'].astype(float)
    volume_series = df['Volume'].astype(float)

    df['RSI'] = calculate_rsi(close_series)
    df['MACD'] = calculate_macd(close_series)
    df['MA_20'] = calculate_sma(close_series, window=20)
    df['Return'] = close_series.pct_change()
    df['MA_20_ratio'] = close_series / df['MA_20'] - 1
    df['Close_Open'] = close_series / open_series - 1
    df['High_Low'] = high_series / low_series - 1
    df['Volume_ratio'] = volume_series / volume_series.rolling(10).mean() - 1
    df['Volatility'] = df['Return'].rolling(10).std()
    df['Momentum_Factor'] = df['Return'].apply(lambda x: 1.0 if x > 0 else 0.0)
    df['Sentiment'] = 0.0
    df['Event'] = 0
    df['RAG_Sentiment'] = 0.0
    df['RAG_Relevance'] = 0.5
    df['RAG_Event_Importance'] = 0.4
    df['RAG_Market_Impact'] = 0.1
    df['RAG_Risk_Score'] = 0.1
    df['RAG_Confidence'] = 0.5
    df.dropna(inplace=True)
    return df

@app.get("/api/stocks")
def get_stocks():
    return stocks

@app.get("/api/stock-data")
def get_stock_data(ticker: str = "RELIANCE.NS", period: str = "6mo"):
    df = process_stock_data(ticker, period)
    # Convert dates to string for JSON serialization
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df.to_dict(orient="records")

@app.get("/api/quote/realtime")
def get_realtime_quote(symbol: str = "RELIANCE.NS"):
    """
    Fetches 0-delay real-time price quotes.
    Uses Upstox API v2 for Indian NSE/BSE equities, and Finnhub API for US/Global equities.
    """
    # 1. Try Upstox API v2 for Indian / NSE stocks
    quote = upstox_fetcher.fetch_upstox_quote(symbol)
    if quote:
        return quote

    # 2. Fallback to Finnhub for US/Global symbols
    quote = live_fetcher.fetch_finnhub_quote(symbol)
    if quote:
        return quote

    raise HTTPException(status_code=404, detail=f"Real-time quote unavailable for symbol {symbol}")

class ChatRequest(BaseModel):
    query: str

chatbot_instance = None

def get_chatbot_instance():
    global chatbot_instance
    if chatbot_instance is None:
        chatbot_instance = StockAssistantChatbot()
    return chatbot_instance

@app.post("/api/chat")
def chat(req: ChatRequest, ticker: str = "RELIANCE.NS"):
    bot = get_chatbot_instance()
        
    try:
        df = process_stock_data(ticker, "6mo")
        latest = df.iloc[-1]
        stock_ctx = {
            "stock": ticker,
            "close": float(latest["Close"]),
            "rsi": float(latest["RSI"]),
            "macd": float(latest["MACD"]),
            "confidence": 0.85
        }
    except Exception:
        stock_ctx = {"stock": ticker, "close": 0.0, "rsi": 50.0, "macd": 0.0, "confidence": 0.5}
        
    try:
        response = chatbot_instance.get_response(req.query, stock_ctx)
        doc_count = chatbot_instance.rag_engine.get_doc_count() if hasattr(chatbot_instance.rag_engine, 'get_doc_count') else 0
        return {
            "response": response,
            "reply": response,
            "doc_count": doc_count
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=str(traceback.format_exc()))

# Forecast endpoint logic
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import timezone, timedelta, time
import pytz

model_instance = None

def check_market_session_status(ticker: str, df: pd.DataFrame = None):
    """
    Evaluates real-time market status (OPEN or CLOSED) for Indian (NSE/BSE) or US markets.
    Computes explicit forecast target metadata indicating whether the prediction is for
    the current open session or for Tomorrow's / Next Trading Day's Opening Session.
    """
    now_utc = datetime.now(timezone.utc)
    
    # Check Ticker Exchange & Timezone
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        # Indian Stock Market (NSE/BSE) - 09:15 to 15:30 IST
        tz = pytz.timezone("Asia/Kolkata")
        open_time = time(9, 15)
        close_time = time(15, 30)
    else:
        # US Stock Market (NYSE/NASDAQ) - 09:30 to 16:00 EST
        tz = pytz.timezone("America/New_York")
        open_time = time(9, 30)
        close_time = time(16, 0)
        
    now_local = now_utc.astimezone(tz)
    weekday = now_local.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    
    is_weekend = weekday >= 5
    is_trading_hours = open_time <= now_local.time() <= close_time
    is_open = (not is_weekend) and is_trading_hours
    
    # If market is closed, calculate next trading session date (Tomorrow or Next Monday)
    if is_open:
        forecast_target_desc = "Current Open Trading Session"
        next_session_dt = now_local
        status_msg = "Market is currently OPEN. Forecast reflects active trading session momentum."
    else:
        # Move to tomorrow or next business day
        next_session_dt = now_local + timedelta(days=1)
        while next_session_dt.weekday() >= 5:  # Skip Sat & Sun
            next_session_dt += timedelta(days=1)
            
        target_date_str = next_session_dt.strftime("%Y-%m-%d")
        if is_weekend:
            forecast_target_desc = f"Next Trading Day (Monday, {target_date_str})"
        elif now_local.time() > close_time:
            forecast_target_desc = f"Tomorrow's Trading Day ({target_date_str})"
        else:
            forecast_target_desc = f"Next Session Opening ({target_date_str})"
            
        status_msg = f"Market is currently CLOSED. Prediction is strictly applicable to the {forecast_target_desc} session."
        
    return {
        "is_market_open": is_open,
        "market_status": "OPEN" if is_open else "CLOSED",
        "market_status_message": status_msg,
        "forecast_target": forecast_target_desc,
        "next_session_date": next_session_dt.strftime("%Y-%m-%d"),
        "current_local_time": now_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    }

@app.get("/api/forecast")
@app.get("/api/predict")
def get_forecast(ticker: str = "RELIANCE.NS", horizon: str = "1d", force_retrain: bool = False):
    try:
        global model_instance
        if model_instance is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "lstm_model.h5")
            if not os.path.exists(model_path):
                model_path = os.path.join(base_dir, "app", "models", "lstm_model.h5")
            if os.path.exists(model_path):
                import tempfile
                import shutil
                import h5py
                import json
                
                with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf_file:
                    tmp_path = tf_file.name
                shutil.copy2(model_path, tmp_path)
                
                with h5py.File(tmp_path, 'r+') as f:
                    model_config = f.attrs.get('model_config')
                    if model_config is not None:
                        if isinstance(model_config, bytes):
                            model_config = model_config.decode('utf-8')
                        config_dict = json.loads(model_config)
                        for layer in config_dict.get('config', {}).get('layers', []):
                            if layer['class_name'] == 'InputLayer':
                                if 'batch_shape' in layer['config']:
                                    layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
                                if 'optional' in layer['config']:
                                    del layer['config']['optional']
                            if 'quantization_config' in layer.get('config', {}):
                                del layer['config']['quantization_config']
                        f.attrs['model_config'] = json.dumps(config_dict).encode('utf-8')
                
                with tf.device('/CPU:0'):
                    model_instance = load_model(tmp_path, compile=False)
            else:
                raise HTTPException(status_code=500, detail=f"Model not found on disk at {model_path}")
                
        # Replicate the intraday vs daily branching logic exactly
        horizon_options = {
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "60m": 60,
            "1d": 0
        }
        
        if horizon not in horizon_options:
            raise HTTPException(status_code=400, detail="Invalid horizon")
            
        horizon_mins = horizon_options[horizon]
        features = [
            'RSI', 'MACD', 'Return', 'MA_20_ratio', 'Close_Open',
            'High_Low', 'Volume_ratio', 'Momentum_Factor',
            'Sentiment', 'Event',
            'RAG_Sentiment', 'RAG_Relevance', 'RAG_Event_Importance',
            'RAG_Market_Impact', 'RAG_Risk_Score', 'RAG_Confidence'
        ]

        # --- INTRADAY ---
        if horizon != "1d":
            df_intra_raw = fetch_stock_data_raw(ticker, "5d", interval="5m")
            if df_intra_raw.empty:
                raise HTTPException(status_code=404, detail="Intraday data unavailable")
                
            df_intra = df_intra_raw.copy()
            for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if c in df_intra.columns and isinstance(df_intra[c], pd.DataFrame):
                    df_intra[c] = df_intra[c].iloc[:, 0]
                    
            ci = df_intra['Close'].astype(float)
            oi = df_intra['Open'].astype(float)
            hi = df_intra['High'].astype(float)
            li = df_intra['Low'].astype(float)
            vi = df_intra['Volume'].astype(float)

            df_intra['RSI'] = calculate_rsi(ci)
            df_intra['MACD'] = calculate_macd(ci)
            df_intra['MA_20'] = calculate_sma(ci, window=20)
            df_intra['Return'] = ci.pct_change()
            df_intra['MA_20_ratio'] = ci / df_intra['MA_20'] - 1
            df_intra['Close_Open'] = ci / oi - 1
            df_intra['High_Low'] = hi / li - 1
            df_intra['Volume_ratio'] = vi / vi.rolling(10).mean() - 1
            df_intra['Volatility'] = df_intra['Return'].rolling(10).std()
            df_intra['Momentum_Factor'] = df_intra['Return'].apply(lambda x: 1.0 if x > 0 else 0.0)
            for col in ['Sentiment', 'Event', 'RAG_Sentiment', 'RAG_Relevance', 'RAG_Event_Importance', 'RAG_Market_Impact', 'RAG_Risk_Score', 'RAG_Confidence']:
                if col not in df_intra.columns:
                    df_intra[col] = 0.5
            df_intra.dropna(inplace=True)

            if len(df_intra) < 5:
                raise HTTPException(status_code=400, detail="Insufficient data for 5-step lookback.")
                
            session_info = check_market_session_status(ticker, df_intra)
            if not session_info["is_market_open"]:
                # Market is CLOSED - Gracefully compute Tomorrow's workable daily forecast
                trained_model, scaler_obj, meta_info = online_trainer.train_up_to_date_model(ticker, period="6mo", force_retrain=force_retrain)
                df_daily = process_stock_data(ticker, "6mo")
                
                model_feat = meta_info.get('features', ['RSI', 'MACD', 'Return', 'MA_20_ratio', 'Close_Open', 'High_Low', 'Volume_ratio', 'Volatility'])
                available_feat = [f for f in model_feat if f in df_daily.columns]
                Xs = scaler_obj.transform(df_daily[available_feat])
                seq = np.expand_dims(Xs[-5:], axis=0).astype(np.float32)
                
                with tf.device('/CPU:0'):
                    score = float(trained_model(seq, training=False).numpy()[0][0])
                    
                direction = "UP" if score > 0.5 else "DOWN"
                fdf = pd.DataFrame(df_daily[available_feat].tail(5))
                fdf.index = [f"Day -{4-i}" for i in range(5)]
                
                current_price = float(df_daily['Close'].iloc[-1])
                vol_avg = df_daily['Volatility'].mean() if 'Volatility' in df_daily.columns else 0.01
                if np.isnan(vol_avg): vol_avg = 0.01
                move_pct = (score - 0.5) * 2.0 * vol_avg
                target_price = current_price * (1.0 + move_pct)
                
                return {
                    "score": score,
                    "direction": direction,
                    "current_price": current_price,
                    "target_price": target_price,
                    "move_pct": move_pct,
                    "target_time": f"Opening ({session_info['next_session_date']})",
                    "horizon_mins": horizon_mins,
                    "market_status": "CLOSED",
                    "is_market_open": False,
                    "market_status_message": f"Market is currently CLOSED. Intraday {horizon} request automatically adapted to {session_info['forecast_target']}.",
                    "forecast_target": session_info["forecast_target"],
                    "next_session_date": session_info["next_session_date"],
                    "notice": f"Market is CLOSED. Showing workable prediction for {session_info['forecast_target']}.",

                    "features": fdf.to_dict(orient="index"),
                    "model_meta": meta_info
                }
                
            date_col = 'Datetime' if 'Datetime' in df_intra.columns else ('Date' if 'Date' in df_intra.columns else df_intra.columns[0])
            last_time = pd.to_datetime(df_intra[date_col].iloc[-1])
            target_time = last_time + pd.Timedelta(minutes=horizon_mins)
            current_price = float(df_intra['Close'].iloc[-1])

            scaler_i = MinMaxScaler()
            Xi = scaler_i.fit_transform(df_intra[features])
            seq = np.expand_dims(Xi[-5:], axis=0).astype(np.float32)
            
            with tf.device('/CPU:0'):
                raw_score = float(model_instance(seq, training=False).numpy()[0][0])
                
            vol_avg = df_intra['Volatility'].mean()
            if np.isnan(vol_avg): vol_avg = 0.002
            move_pct = (raw_score - 0.5) * 2.0 * vol_avg * np.sqrt(horizon_mins / 15.0)
            target_price = current_price * (1.0 + move_pct)
            direction = "UP" if raw_score > 0.5 else "DOWN"
            
            fdf = pd.DataFrame(df_intra[features].tail(5))
            fdf.index = [f"Bar -{4-i}" for i in range(5)]
            
            return {
                "score": raw_score,
                "direction": direction,
                "current_price": current_price,
                "target_price": target_price,
                "move_pct": move_pct,
                "target_time": target_time.strftime("%H:%M:%S"),
                "horizon_mins": horizon_mins,
                "market_status": session_info["market_status"],
                "is_market_open": session_info["is_market_open"],
                "market_status_message": session_info["market_status_message"],
                "forecast_target": session_info["forecast_target"],
                "next_session_date": session_info["next_session_date"],
                "features": fdf.to_dict(orient="index")
            }

        # --- DAILY ---
        else:
            # Dynamically train/fine-tune model on up-to-date market & live news data
            trained_model, scaler_obj, meta_info = online_trainer.train_up_to_date_model(ticker, period="6mo", force_retrain=force_retrain)
            
            df = process_stock_data(ticker, "6mo")
            if len(df) < 5:
                raise HTTPException(status_code=400, detail="Insufficient data for 5-day lookback.")
                
            session_info = check_market_session_status(ticker, df)
            model_feat = meta_info.get('features', ['RSI', 'MACD', 'Return', 'MA_20_ratio', 'Close_Open', 'High_Low', 'Volume_ratio', 'Volatility'])
            available_feat = [f for f in model_feat if f in df.columns]
            Xs = scaler_obj.transform(df[available_feat])
            seq = np.expand_dims(Xs[-5:], axis=0).astype(np.float32)
            
            with tf.device('/CPU:0'):
                score = float(trained_model(seq, training=False).numpy()[0][0])
                
            direction = "UP" if score > 0.5 else "DOWN"
            
            fdf = pd.DataFrame(df[available_feat].tail(5))
            fdf.index = [f"Day -{4-i}" for i in range(5)]
            
            return {
                "score": score,
                "direction": direction,
                "forecast_target": session_info["forecast_target"],
                "next_session_date": session_info["next_session_date"],
                "market_status": session_info["market_status"],
                "is_market_open": session_info["is_market_open"],
                "market_status_message": session_info["market_status_message"],
                "features": fdf.to_dict(orient="index"),
                "model_meta": meta_info
            }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=str(traceback.format_exc()))

@app.post("/api/train-model")
def train_model(ticker: str = "RELIANCE.NS", force_retrain: bool = True):
    """Triggers real-time online model training on up-to-date market & live news data for a ticker."""
    try:
        model, scaler, meta = online_trainer.train_up_to_date_model(ticker, period="6mo", force_retrain=force_retrain)
        return {
            "status": "success",
            "message": f"Successfully trained up-to-date LSTM model for {ticker}",
            "training_meta": meta
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=str(traceback.format_exc()))

@app.get("/api/live-news")
def get_live_news(ticker: str = "RELIANCE.NS"):
    """Fetches real-time breaking live news for a given stock ticker with credibility & recency decay weighted sentiment."""
    ticker_name = [k for k, v in stocks.items() if v == ticker]
    tname = ticker_name[0] if ticker_name else ticker
    
    df_live = live_fetcher.get_live_news_for_ticker(tname, ticker)
    if not df_live.empty:
        # Dynamically index live news into RAG
        bot = get_chatbot_instance()
        if bot and hasattr(bot, 'rag_engine') and bot.rag_engine:
            bot.rag_engine.add_live_news_articles(df_live)
        
        from data_ingestion.live_news_fetcher import compute_weighted_composite_sentiment
        weighted_sentiment = compute_weighted_composite_sentiment(df_live)
        raw_sentiment = round(float(df_live['sentiment'].mean()), 3) if 'sentiment' in df_live.columns else 0.0
        
        return {
            "status": "success",
            "count": len(df_live),
            "weighted_composite_sentiment": weighted_sentiment,
            "raw_mean_sentiment": raw_sentiment,
            "news": df_live[['title', 'source', 'url', 'sentiment', 'credibility', 'event', 'date']].to_dict(orient="records")
        }
    return {"status": "empty", "count": 0, "weighted_composite_sentiment": 0.0, "news": []}

@app.get("/api/indicators")
def get_indicators(ticker: str, period: str = "6mo"):
    df = process_stock_data(ticker, period)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    last_10 = df.tail(10)[['Date', 'Close', 'MA_20', 'RSI', 'MACD', 'Return', 'Volatility']].to_dict(orient="records")
    
    # Live Real-Time News Data Fetching
    ticker_name = [k for k, v in stocks.items() if v == ticker]
    tname = ticker_name[0] if ticker_name else ticker
    df_live = live_fetcher.get_live_news_for_ticker(tname, ticker)
    
    news_sentiment = []
    if not df_live.empty:
        bot = get_chatbot_instance()
        if bot and hasattr(bot, 'rag_engine') and bot.rag_engine:
            bot.rag_engine.add_live_news_articles(df_live)
        news_sentiment = df_live.head(10)[['title', 'sentiment', 'event']].to_dict(orient="records")
    else:
        # Fallback to stored CSV if offline
        news_path = os.path.join(os.path.dirname(__file__), "data", "news_processed.csv")
        if os.path.exists(news_path):
            news_df = pd.read_csv(news_path)
            if not news_df.empty:
                news_sentiment = news_df.head(10)[['title', 'sentiment', 'event']].to_dict(orient="records")
        
    return {
        "indicators": last_10,
        "news": news_sentiment,
        "chart_data": df[['Date', 'RSI', 'MACD']].to_dict(orient="records")
    }

@app.get("/api/risk")
def get_risk(ticker: str, period: str = "6mo", pp: float = 0.5):
    df = process_stock_data(ticker, period)
    df.set_index('Date', inplace=True)
    
    # 1. calculate_stock_risk_metrics
    ret_arr = df['Return'].dropna().values
    risk_metrics = calculate_stock_risk_metrics(ret_arr, pp)
    
    # 2. MarketRegimeDetector
    detector = MarketRegimeDetector()
    regime = detector.detect_regime(df['Close'].dropna())
    
    # 3. PortfolioRecommendationEngine
    engine = PortfolioRecommendationEngine()
    forecasts = [{"ticker": ticker, "prob": pp, "returns": ret_arr[-20:] if len(ret_arr) >= 20 else ret_arr}]
    pdf = engine.rank_portfolio(forecasts)
    
    return {
        "risk_metrics": risk_metrics,
        "regime": regime,
        "portfolio": pdf.to_dict(orient="records") if not pdf.empty else []
    }
