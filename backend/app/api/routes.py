from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import yfinance as yf

# Import existing backend modules
from app.services.chatbot.chatbot import StockAssistantChatbot
from app.services.risk.risk_analyzer import calculate_stock_risk_metrics
from app.services.market_regime.regime_detector import MarketRegimeDetector
from app.services.portfolio.portfolio_advisor import PortfolioRecommendationEngine
from app.services.indicators.rsi import calculate_rsi
from app.services.indicators.macd import calculate_macd
from app.services.indicators.moving_average import calculate_ma

router = APIRouter()

chatbot = StockAssistantChatbot()
regime_detector = MarketRegimeDetector()
portfolio_advisor = PortfolioRecommendationEngine()

class ChatRequest(BaseModel):
    query: str
    context: dict = None

class ForecastRequest(BaseModel):
    symbol: str

@router.get("/stock/{symbol}")
def get_stock_data(symbol: str):
    """
    Fetch OHLCV data and calculate indicators.
    """
    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False)
        if df.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")
        
        # Flatten multi-level columns if necessary
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        
        # Calculate indicators using existing modules
        df = calculate_rsi(df)
        df = calculate_macd(df)
        df = calculate_ma(df)
        
        # Handle NaN values for JSON serialization
        df = df.fillna(0)
        
        # Rename 'Date' to string format for JSON
        if 'Date' in df.columns:
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            
        return {"symbol": symbol, "data": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast")
def get_forecast(request: ForecastRequest):
    """
    Run LSTM model inference.
    """
    # TODO: Implement actual model inference logic here. 
    # For now, returning a stubbed response as requested in Phase 1 if complex.
    # The actual inference requires 5-day sequence of 16 features.
    import random
    prob = random.uniform(0.3, 0.8)
    return {
        "symbol": request.symbol,
        "prediction_prob": prob,
        "signal": "UP" if prob > 0.5 else "DOWN"
    }

@router.get("/risk/{symbol}")
def get_risk_metrics(symbol: str):
    """
    Calculate risk metrics and market regime.
    """
    # TODO: Fetch actual recent returns for the symbol instead of dummy data
    dummy_returns = np.random.normal(0, 0.02, 30)
    metrics = calculate_stock_risk_metrics(dummy_returns, 0.65)
    
    dummy_prices = pd.Series(np.linspace(100, 150, 30))
    regime = regime_detector.detect_regime(dummy_prices)
    
    return {
        "symbol": symbol,
        "risk_metrics": metrics,
        "market_regime": regime
    }

@router.post("/chat")
def get_chat_response(request: ChatRequest):
    """
    RAG chatbot response.
    """
    try:
        reply = chatbot.get_response(request.query, request.context)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio")
def get_portfolio():
    """
    Portfolio recommendations.
    """
    dummy_forecasts = [
        {"ticker": "RELIANCE.NS", "prob": 0.75, "returns": np.array([0.01, 0.02])},
        {"ticker": "TCS.NS", "prob": 0.40, "returns": np.array([-0.01, 0.01])}
    ]
    df_rank = portfolio_advisor.rank_portfolio(dummy_forecasts)
    return {"recommendations": df_rank.to_dict(orient="records")}
