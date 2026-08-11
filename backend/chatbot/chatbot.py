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
            r"\b(rsi|relative strength index)\b": (
                "RSI (Relative Strength Index) measures momentum on a scale of 0 to 100.\n"
                "• RSI > 70: Stock may be overbought (potential price drop/pullback).\n"
                "• RSI < 30: Stock may be oversold (potential buying opportunity)."
            ),
            r"\b(macd|moving average convergence divergence)\b": (
                "MACD tracks trend direction and momentum using two exponential moving averages (12-period and 26-period).\n"
                "• Bullish Crossover: MACD line crosses above signal line.\n"
                "• Bearish Crossover: MACD line crosses below signal line."
            ),
            r"\b(lstm|model|prediction|deep learning)\b": (
                "Our 64-unit LSTM neural network processes 5-step lookback sequences of 8 relative indicators "
                "(RSI, MACD, Return, Volatility, MA_20 ratio, Volume ratio) to classify price direction."
            ),
            r"\b(sentiment|vader)\b": (
                "News sentiment is analyzed using VADER NLP polarity scoring. "
                "Scores range from -1.0 (extremely negative) to +1.0 (extremely positive), reflecting overall market mood."
            ),
            r"\b(moving average|ma|ma20|sma)\b": (
                "The 20-day Simple Moving Average (MA_20) smoothes price data to identify prevailing short-term trends. "
                "Prices trading above MA_20 indicate an uptrend."
            ),
            r"\b(event|earnings|merger|lawsuit)\b": (
                "Event detection categorizes news stories into key corporate events (Earnings, Merger, Legal, General) "
                "to assess major volatility drivers."
            ),
            r"\b(help|hi|hello|hey|start)\b": (
                "Hello! I am your RAG-Augmented AI Market Assistant. I can help you with:\n"
                "1. Technical indicators (RSI, MACD, MA_20)\n"
                "2. LSTM Model & prediction logic\n"
                "3. RAG Semantic News Search & Event analysis\n"
                "Ask any question or search for financial news headlines!"
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

        # 1. Base Knowledge Match (if query matches technical concept)
        matched_concept = None
        for pattern, answer in self.knowledge_base.items():
            if re.search(pattern, query_lower):
                matched_concept = answer
                break

        if matched_concept:
            response_parts.append(matched_concept)

        # 2. RAG Semantic Vector News Retrieval
        # Check if user is asking about news, headlines, market events, stock ticker, or general topics
        news_keywords = ['news', 'headline', 'article', 'event', 'market', 'ipo', 'war', 'stock', 'earnings', 'legal', 'lawsuit', 'merger', 'sentiment', 'ai', 'tech', 'court', 'trade', 'ipo', 'price']
        should_rag = any(kw in query_lower for kw in news_keywords) or (matched_concept is None)

        retrieved_docs = []
        if should_rag and self.rag_engine.is_indexed:
            # Perform RAG retrieval
            search_query = user_query
            if current_stock_context and 'stock' in current_stock_context:
                search_query += f" {current_stock_context['stock']}"

            retrieved_docs = self.rag_engine.retrieve(search_query, top_k=3, min_similarity=0.03)

        # 3. Incorporate RAG Retrieved Context into Response
        if retrieved_docs:
            rag_text = "\n\n📰 **RAG Vector Search - Relevant Financial News Context**:\n"
            for i, doc in enumerate(retrieved_docs, 1):
                sent_score = doc['sentiment']
                if sent_score > 0.05:
                    sent_badge = f"🟢 Positive ({sent_score:+.2f})"
                elif sent_score < -0.05:
                    sent_badge = f"🔴 Negative ({sent_score:+.2f})"
                else:
                    sent_badge = f"⚪ Neutral ({sent_score:+.2f})"

                rag_text += f"\n**{i}. {doc['title']}**\n"
                if doc['content']:
                    snippet = doc['content'][:140] + "..." if len(doc['content']) > 140 else doc['content']
                    rag_text += f"   *\"{snippet}\"*\n"
                rag_text += f"   • **Event**: `{doc['event']}` | **Sentiment**: {sent_badge} | **Similarity Score**: `{doc['similarity']:.2f}`\n"

            response_parts.append(rag_text)

        # 4. Inject Stock Specific Context if available
        if current_stock_context and 'stock' in current_stock_context:
            stock = current_stock_context.get('stock', 'Selected Stock')
            close = current_stock_context.get('close', 'N/A')
            rsi = current_stock_context.get('rsi', 'N/A')
            macd = current_stock_context.get('macd', 'N/A')
            conf = current_stock_context.get('confidence', None)

            ctx_str = f"\n\n📍 **Live Market Context for {stock}**:\n"
            ctx_str += f"• Close Price: ₹{close:.2f}\n" if isinstance(close, (int, float)) else f"• Close Price: ₹{close}\n"
            ctx_str += f"• RSI (14): {rsi:.2f}\n" if isinstance(rsi, (int, float)) else f"• RSI: {rsi}\n"
            ctx_str += f"• MACD: {macd:.2f}\n" if isinstance(macd, (int, float)) else f"• MACD: {macd}\n"
            if conf is not None:
                dir_str = "UP 📈" if conf > 0.5 else "DOWN 📉"
                ctx_str += f"• LSTM Model Forecast: **{dir_str}** (Confidence: {conf:.2%})\n"

            response_parts.append(ctx_str)

        # If nothing matched and RAG yielded no docs
        if not response_parts:
            return (
                "I'm your RAG-Augmented Financial Assistant. I can help with technical analysis (RSI, MACD, MA_20), "
                "LSTM forecasts, or perform RAG vector searches on market news headlines. Try asking: 'Show recent market news about IPOs' or 'What is RSI?'"
            )

        return "".join(response_parts)
