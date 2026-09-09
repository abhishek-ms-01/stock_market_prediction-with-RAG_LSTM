import re
from chatbot.rag_engine import FinancialRAGEngine

class StockAssistantChatbot:
    """
    RAG-Augmented Financial AI Assistant.
    Combines rule-based technical knowledge, live quantitative stock metrics,
    and a TF-IDF / Cosine Similarity Vector RAG engine to retrieve real market news.
    """

    def __init__(self):
        self.rag_engine = FinancialRAGEngine()

        self.knowledge_base = {
            r"(how.*(system|platform|alphatrade|it).*work|system design|architecture|explain.*system|workflow|methodology)": (
                "📘 **AlphaTrade System Architecture & Methodology**:\n\n"
                "AlphaTrade is a Time-Aware Multi-Modal Financial Intelligence Platform built in 4 tiers:\n\n"
                "1. **Presentation Tier (Next.js 14 & TradingView)**: Institutional dark-mode dashboard rendering real-time HTML5 candlestick charts, direction forecast cards, and risk meters.\n"
                "2. **API Gateway (FastAPI Python 3.12)**: Asynchronous microservice with an in-memory 5-minute TTL cache and a timezone-aware **Market Session Evaluator** (NSE 09:15–15:30 IST / US 09:30–16:00 EST) that smoothly adapts off-market queries to Tomorrow's Session Opening forecast.\n"
                "3. **Multi-Source Ingestion Engine**: Zero-delay quotes from **Upstox API v2** (NSE/BSE) and **Finnhub** (US), 1-year OHLCV bars via **yfinance**, and live news scraping from Google News RSS & Yahoo Finance with corporate event NER deduplication.\n"
                "4. **AI & Quantitative Intelligence Core**:\n"
                "   • **Dynamic Online LSTM (64 units)**: Trained on-the-fly with **Binary Focal Loss (γ=2.0)** to eliminate trend class imbalance.\n"
                "   • **Event-Driven Dense RAG**: 384-dim semantic embeddings (`all-MiniLM-L6-v2`) indexed in a **FAISS** vector store with publisher credibility weighting and exponential time-decay ($e^{-\\lambda \\Delta t}$).\n"
                "   • **Quantitative Risk Engine**: Live Value at Risk (VaR 95%), Sharpe Ratio, and 5-cluster **K-Means Market Regime Detection** (Bull, Bear, Sideways, High Volatility, Low Volatility)."
            ),
            r"(how.*(predict|forecast)|forecast|prediction logic|target price|focal loss|lstm model|deep learning)": (
                "🧠 **Prediction & Deep Learning Methodology**:\n\n"
                "• **Input Tensor**: 5-step lookback sequence across 8 relative stationary features: `[Batch, 5, 8]`.\n"
                "• **8 Features**: RSI(14), MACD(12,26,9), Close/MA_20 ratio, Daily Return, Close/Open ratio, High/Low ratio, Volume ratio, and 10-day Volatility.\n"
                "• **Neural Network**: 64-unit LSTM -> Dropout (0.2) -> Dense (32, ReLU) -> Dense (1, Sigmoid).\n"
                "• **Binary Focal Loss (γ=2.0, α=0.25)**: Down-weights easy trend continuation samples to prioritize critical market inflection points.\n"
                "• **Volatility-Scaled Target Price**: Mathematically calibrated as: `Target = Current_Price * [1 + (Score - 0.5) * 2.0 * Volatility * sqrt(Horizon/15)]`."
            ),
            r"\b(var|value at risk|sharpe|risk|regime|market regime|drawdown)\b": (
                "🛡️ **Quantitative Risk & Market Regime Analysis**:\n\n"
                "• **Value at Risk (VaR 95%)**: Represents maximum expected 1-day percentage loss at 95% statistical confidence.\n"
                "• **Sharpe Ratio**: Annualized risk-adjusted excess return over the risk-free benchmark.\n"
                "• **Market Regime Detection**: Unsupervised K-Means clustering (k=5) over 20-day SMA ratio and return volatility, identifying:\n"
                "  1. Bull Market 📈 | 2. Bear Market 📉 | 3. Sideways Market ↔️ | 4. High Volatility ⚡ | 5. Low Volatility 🟢"
            ),
            r"(off.?market|market closed|weekend|tomm?orr?ow|after hours|next session)": (
                "⏰ **Off-Market Hours & Tomorrow's Forecast Engine**:\n\n"
                "AlphaTrade implements a timezone-aware session evaluator for NSE/BSE (`Asia/Kolkata`) and US (`America/New_York`).\n"
                "When queried during evenings, nights, or weekends, the system automatically detects that the market is CLOSED, switches from 5-minute intraday to a daily momentum model, and labels the forecast as **Tomorrow's Session Opening Forecast** (or Next Monday) without throwing server errors."
            ),
            r"\b(upstox|finnhub|data source|data feed|news source|sources)\b": (
                "📥 **Data Ingestion Feeds**:\n\n"
                "• **Upstox API v2**: Official zero-delay live equity quotes for Indian NSE/BSE stocks.\n"
                "• **Finnhub**: Sub-second real-time quotes for US equities.\n"
                "• **yfinance**: Comprehensive historical daily & 5-minute intraday OHLCV candles.\n"
                "• **Google News RSS & Yahoo Finance**: Scraped in real-time, categorized into corporate events, and indexed into FAISS."
            ),
            r"\b(buy|sell|hold|recommendation|portfolio advisor)\b": (
                "🎯 **Portfolio Recommendation Rules**:\n\n"
                "• **BUY 🟢**: Confidence Probability P >= 0.65 AND Expected Return > +0.5%.\n"
                "• **SELL 🔴**: Confidence Probability P <= 0.35 OR Expected Return < -0.5%.\n"
                "• **HOLD 🟡**: Balanced risk-reward profile (0.35 < P < 0.65)."
            ),
            r"\b(rsi|relative strength index)\b": (
                "📈 **Relative Strength Index (RSI 14)**:\n"
                "RSI measures momentum on a scale of 0 to 100 based on average gains vs. losses over 14 periods.\n"
                "• RSI > 70: Overbought condition (potential pullback/drop).\n"
                "• RSI < 30: Oversold condition (potential rebound/buying opportunity)."
            ),
            r"\b(macd|moving average convergence divergence)\b": (
                "📊 **MACD (12, 26, 9)**:\n"
                "MACD tracks trend direction and momentum using short (12-period) and long (26-period) EMAs.\n"
                "• Bullish Crossover: MACD line crosses above the 9-day signal line.\n"
                "• Bearish Crossover: MACD line crosses below the 9-day signal line."
            ),
            r"\b(sentiment|vader|nlp)\b": (
                "📰 **NLP News Sentiment & Time-Decay Engine**:\n"
                "AlphaTrade scores news using VADER NLP augmented with domain financial rules.\n"
                "Scores are weighted by source credibility (Reuters 0.95 vs. blogs 0.50) and decayed exponentially: `Score * e^(-0.1 * Δt_hours)` with a ~7-hour half-life."
            ),
            r"\b(moving average|ma|ma20|sma)\b": (
                "📏 **20-Day Simple Moving Average (SMA 20)**:\n"
                "The 20-day SMA smooths short-term price action to establish baseline trend direction. Prices sustaining above SMA 20 reflect bullish regime bias."
            ),
            r"\b(event|earnings|merger|lawsuit)\b": (
                "⚡ **Corporate Event NER Detection**:\n"
                "Classifies breaking stories into corporate categories (Earnings, Merger & Acquisition, Regulatory/Legal, Dividends, Product Launch) to evaluate catalyst magnitude."
            ),
            r"\b(help|hi|hello|hey|start)\b": (
                "👋 Hello! I am your **AlphaTrade RAG-Augmented Financial Assistant**. You can ask me:\n"
                "• *'How does the system work?'* — For architectural overview\n"
                "• *'How do predictions work?'* — For LSTM & Focal Loss methodology\n"
                "• *'What is Value at Risk?'* — For quantitative risk metrics\n"
                "• *'What is RSI or MACD?'* — For technical indicators\n"
                "• Or ask for recent breaking news about any stock!"
            )
        }

    def get_response(self, user_query: str, current_stock_context: dict = None) -> str:
        """
        Generates a natural language response combining pattern-matching knowledge,
        quantitative market context, and RAG vector news retrieval.
        """
        if not user_query or not user_query.strip():
            return "Please enter a valid question about stock prediction, market indicators, or financial news."

        query_lower = user_query.strip().lower()
        response_parts = []

        # 1. Base Knowledge Match (if query matches technical or system concept)
        matched_concept = None
        for pattern, answer in self.knowledge_base.items():
            if re.search(pattern, query_lower):
                matched_concept = answer
                break

        if matched_concept:
            response_parts.append(matched_concept)

        # 2. RAG Semantic Vector News Retrieval
        # Trigger RAG if user specifically queries news/events or if no base concept matched
        news_keywords = ['news', 'headline', 'article', 'event', 'latest', 'today', 'ipo', 'war', 'earnings', 'legal', 'lawsuit', 'merger', 'scandal', 'probe', 'announcement']
        wants_news = any(kw in query_lower for kw in news_keywords)
        should_rag = wants_news or (matched_concept is None)

        retrieved_docs = []
        if should_rag and self.rag_engine and getattr(self.rag_engine, 'is_indexed', False):
            try:
                search_query = user_query
                if current_stock_context and 'stock' in current_stock_context:
                    search_query += f" {current_stock_context['stock']}"

                retrieved_docs = self.rag_engine.retrieve(search_query, top_k=3, min_similarity=0.03)
            except Exception as rag_err:
                print(f"[Chatbot Warning] RAG retrieval error: {rag_err}")

        # 3. Incorporate RAG Retrieved Context into Response
        if retrieved_docs:
            rag_text = "\n\n📰 **RAG Vector Search - Relevant Financial News Context**:\n"
            for i, doc in enumerate(retrieved_docs, 1):
                sent_score = doc.get('sentiment', 0.0)
                if sent_score > 0.05:
                    sent_badge = f"🟢 Positive ({sent_score:+.2f})"
                elif sent_score < -0.05:
                    sent_badge = f"🔴 Negative ({sent_score:+.2f})"
                else:
                    sent_badge = f"⚪ Neutral ({sent_score:+.2f})"

                rag_text += f"\n**{i}. {doc.get('title', 'Headline')}**\n"
                content = doc.get('content', '')
                if content:
                    snippet = content[:140] + "..." if len(content) > 140 else content
                    rag_text += f"   *\"{snippet}\"*\n"
                rag_text += f"   • **Event**: `{doc.get('event', 'General')}` | **Sentiment**: {sent_badge} | **Similarity Score**: `{doc.get('similarity', 0.0):.2f}`\n"

            response_parts.append(rag_text)

        # 4. Inject Stock Specific Context if available (and relevant)
        if current_stock_context and 'stock' in current_stock_context:
            stock = current_stock_context.get('stock', 'Selected Stock')
            close = current_stock_context.get('close', 0.0)
            rsi = current_stock_context.get('rsi', 50.0)
            macd = current_stock_context.get('macd', 0.0)
            conf = current_stock_context.get('confidence', None)

            if isinstance(close, (int, float)) and close > 0:
                ctx_str = f"\n\n📍 **Live Market Context for {stock}**:\n"
                ctx_str += f"• Close Price: ₹{close:.2f}\n"
                ctx_str += f"• RSI (14): {rsi:.2f}\n"
                ctx_str += f"• MACD: {macd:.2f}\n"
                if conf is not None:
                    dir_str = "UP 📈" if conf > 0.5 else "DOWN 📉"
                    ctx_str += f"• LSTM Model Forecast: **{dir_str}** (Confidence: {conf:.2%})\n"

                response_parts.append(ctx_str)

        # If nothing matched and RAG yielded no docs
        if not response_parts:
            return (
                "I'm your **AlphaTrade Financial Assistant**. I can help you understand how the system works, "
                "explain our LSTM prediction model and Focal Loss, breakdown technical indicators (RSI, MACD, MA_20), "
                "or perform RAG semantic news searches. Try asking: *'How does the system work?'* or *'What is RSI?'*"
            )

        return "".join(response_parts)
