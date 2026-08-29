import os
import sys
import json
import ssl
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import yfinance as yf

# Ensure project root in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nlp.event_ner_extractor import classify_corporate_event

# Source Credibility Registry
SOURCE_CREDIBILITY_SCORES = {
    "Economic Times": 0.90,
    "The Economic Times": 0.90,
    "Moneycontrol": 0.88,
    "Yahoo Finance": 0.85,
    "GlobeNewswire": 0.82,
    "Reuters": 0.95,
    "Bloomberg": 0.95,
    "AlphaVantage": 0.92,
    "CNBC": 0.85,
    "Business Standard": 0.88,
    "Mint": 0.86,
    "NewsAPI": 0.80,
    "Google News": 0.80,
    "General": 0.50
}

def get_source_credibility(source_name: str) -> float:
    """Returns quantitative credibility score for a given financial news outlet."""
    if not source_name:
        return 0.50
    for key, score in SOURCE_CREDIBILITY_SCORES.items():
        if key.lower() in str(source_name).lower():
            return score
    return 0.60

# VADER Sentiment Setup
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
except Exception:
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()

FINANCIAL_KEYWORD_ADJUSTMENTS = {
    "beat estimates": 0.35,
    "exceeded expectations": 0.35,
    "profit jumps": 0.30,
    "revenue up": 0.25,
    "debt restructured": 0.25,
    "guidance cut": -0.40,
    "profit warning": -0.45,
    "missed estimates": -0.35,
    "margin squeeze": -0.30,
    "regulatory fine": -0.30,
    "downgrade": -0.25,
    "upgrade": 0.25,
    "all-time high": 0.20,
    "record earnings": 0.30
}

def apply_financial_lexicon_adjustment(text: str, base_score: float) -> float:
    """Adjusts VADER base score using domain-specific financial keyword rules."""
    text_lower = text.lower()
    adjustment = 0.0
    for kw, boost in FINANCIAL_KEYWORD_ADJUSTMENTS.items():
        if kw in text_lower:
            adjustment += boost
    
    adjusted_score = np.clip(base_score + adjustment, -1.0, 1.0)
    return round(float(adjusted_score), 3)

def compute_weighted_composite_sentiment(df: pd.DataFrame, lambda_decay: float = 0.1) -> float:
    """
    Computes Source Credibility & Recency Decay Weighted Sentiment:
    S_composite = sum(W_source * exp(-lambda * delta_t) * S_i) / sum(W_source * exp(-lambda * delta_t))
    """
    if df.empty or 'sentiment' not in df.columns or 'credibility' not in df.columns:
        return 0.0
        
    now = datetime.now(timezone.utc)
    weights = []
    scores = []
    
    for idx, row in df.iterrows():
        credibility = float(row.get('credibility', 0.5))
        sentiment = float(row.get('sentiment', 0.0))
        
        # Parse date for recency decay
        raw_date = row.get('date', '')
        days_old = 0.0
        try:
            if isinstance(raw_date, str) and raw_date:
                dt = pd.to_datetime(raw_date, utc=True)
                delta = now - dt
                days_old = max(0.0, delta.total_seconds() / 86400.0)
        except Exception:
            days_old = 1.0
            
        time_decay = np.exp(-lambda_decay * days_old)
        composite_weight = credibility * time_decay
        
        weights.append(composite_weight)
        scores.append(sentiment)
        
    total_weight = sum(weights)
    if total_weight <= 1e-6:
        return 0.0
        
    weighted_sentiment = sum(w * s for w, s in zip(weights, scores)) / total_weight
    return round(float(weighted_sentiment), 3)

class LiveNewsFetcher:
    """
    Real-Time Multi-Source Live Financial News Data Fetcher.
    Dynamically fetches live breaking news across FOUR major sources:
    1. Alpha Vantage Real-Time Sentiment API (High speed institutional news)
    2. Google News RSS Search Feed (Real-time keyword search)
    3. yfinance Ticker News (Yahoo Finance real-time stream)
    4. NewsAPI Endpoint (Global news archive)
    """

    def __init__(self):
        self.newsapi_key = os.getenv("NEWS_API_KEY", "")
        self.alphavantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")

    def fetch_alphavantage_news(self, ticker_name: str) -> list:
        """Fetches real-time financial market news & institutional sentiment from Alpha Vantage."""
        articles = []
        api_key = self.alphavantage_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")
        if not api_key:
            return articles
            
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=financial_markets,earnings&apikey={api_key}"
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5, context=ctx) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            feed = data.get('feed', [])
            for item in feed[:10]:
                title = item.get('title', '')
                summary = item.get('summary', '')
                time_pub = item.get('time_published', '')
                source = item.get('source', 'AlphaVantage')
                url_link = item.get('url', '')
                
                if title:
                    articles.append({
                        "title": title,
                        "content": summary or title,
                        "date": time_pub or datetime.now(timezone.utc).isoformat(),
                        "source": f"{source} (AlphaVantage)",
                        "url": url_link
                    })
        except Exception as e:
            print(f"[LiveNewsFetcher Warning] AlphaVantage news fetch failed: {e}")
            
        return articles

    def fetch_yfinance_news(self, symbol: str) -> list:
        """Fetches live news directly from Yahoo Finance via yfinance."""
        articles = []
        if not symbol:
            return articles
            
        try:
            ticker_obj = yf.Ticker(symbol)
            news_items = ticker_obj.news or []
            
            for item in news_items:
                content_dict = item.get("content", item)
                title = content_dict.get("title", "")
                summary = content_dict.get("summary", "") or content_dict.get("description", "")
                pub_date = content_dict.get("pubDate", "") or content_dict.get("displayTime", "")
                
                provider = content_dict.get("provider", {})
                source_name = provider.get("displayName", "Yahoo Finance") if isinstance(provider, dict) else "Yahoo Finance"
                
                canonical = content_dict.get("canonicalUrl", {})
                click_url = content_dict.get("clickThroughUrl", {})
                url = (canonical.get("url") if isinstance(canonical, dict) else "") or \
                      (click_url.get("url") if isinstance(click_url, dict) else "") or \
                      item.get("link", "")
                
                if title:
                    articles.append({
                        "title": title,
                        "content": summary or title,
                        "date": pub_date or datetime.now(timezone.utc).isoformat(),
                        "source": source_name,
                        "url": url
                    })
        except Exception as e:
            print(f"[LiveNewsFetcher Warning] yfinance news fetch failed for {symbol}: {e}")
            
        return articles

    def fetch_google_news_rss(self, query: str) -> list:
        """Fetches real-time breaking news via Google News RSS Search Feed."""
        articles = []
        if not query:
            return articles
            
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        try:
            req = urllib.request.Request(
                rss_url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=4) as response:
                xml_data = response.read()
                
            root = ET.fromstring(xml_data)
            channel = root.find("channel")
            if channel is not None:
                for item in channel.findall("item")[:10]:
                    title = item.findtext("title", "")
                    link = item.findtext("link", "")
                    pub_date = item.findtext("pubDate", "")
                    description = item.findtext("description", "")
                    
                    source_elem = item.find("source")
                    source_name = source_elem.text if source_elem is not None else "Google News"
                    
                    if title:
                        articles.append({
                            "title": title,
                            "content": description or title,
                            "date": pub_date or datetime.now(timezone.utc).isoformat(),
                            "source": source_name,
                            "url": link
                        })
        except Exception as e:
            print(f"[LiveNewsFetcher Warning] Google News RSS fetch failed for '{query}': {e}")
            
        return articles

    def fetch_newsapi(self, query: str) -> list:
        """Fetches breaking global financial news via NewsAPI endpoint."""
        articles = []
        api_key = self.newsapi_key or os.getenv("NEWS_API_KEY", "")
        if not query or not api_key:
            return articles
            
        encoded_query = urllib.parse.quote(query)
        url = f"https://newsapi.org/v2/everything?q={encoded_query}&apiKey={api_key}&language=en&sortBy=publishedAt&pageSize=15"
        
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5, context=ctx) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            if data.get('status') == 'ok':
                for item in data.get('articles', []):
                    title = item.get('title', '')
                    description = item.get('description', '') or item.get('content', '')
                    pub_date = item.get('publishedAt', '')
                    source = item.get('source', {})
                    source_name = source.get('name', 'NewsAPI') if isinstance(source, dict) else 'NewsAPI'
                    url_link = item.get('url', '')
                    
                    if title and title != '[Removed]':
                        articles.append({
                            "title": title,
                            "content": description or title,
                            "date": pub_date or datetime.now(timezone.utc).isoformat(),
                            "source": source_name,
                            "url": url_link
                        })
        except Exception as e:
            print(f"[LiveNewsFetcher Warning] NewsAPI fetch failed for '{query}': {e}")
            
        return articles

    def process_and_score_articles(self, raw_articles: list) -> pd.DataFrame:
        """Applies VADER + Financial Lexicon sentiment, event classification, source credibility, and deduplication."""
        if not raw_articles:
            return pd.DataFrame(columns=['title', 'content', 'date', 'source', 'url', 'sentiment', 'event', 'credibility'])
            
        processed = []
        seen_titles = set()
        
        for art in raw_articles:
            title = art.get('title', '').strip()
            if not title:
                continue
                
            # Title Deduplication
            title_clean = "".join(e for e in title.lower() if e.isalnum())
            if title_clean in seen_titles:
                continue
            seen_titles.add(title_clean)
            
            content = art.get('content', title)
            text_to_score = f"{title}. {content}"
            
            # VADER Base Scoring + Financial Lexicon Adjustment
            vader_res = vader.polarity_scores(text_to_score)
            base_score = float(vader_res['compound'])
            sentiment_score = apply_financial_lexicon_adjustment(text_to_score, base_score)
            
            # Event Classification
            event_type = classify_corporate_event(text_to_score)
            
            # Credibility Weighting
            source_name = art.get('source', 'General')
            credibility = get_source_credibility(source_name)
            
            processed.append({
                "title": title,
                "content": content,
                "date": art.get('date', datetime.now(timezone.utc).isoformat()),
                "source": source_name,
                "url": art.get('url', ''),
                "sentiment": sentiment_score,
                "event": event_type,
                "credibility": credibility
            })
            
        df = pd.DataFrame(processed)
        return df

    def get_live_news_for_ticker(self, ticker_name: str, symbol: str = "") -> pd.DataFrame:
        """
        Main entrypoint: Fetches live news across FOUR major sources:
        1. Alpha Vantage Real-Time News & Sentiment API
        2. Google News RSS Feed
        3. yfinance Stream
        4. NewsAPI Feed
        Processes scores, deduplicates, and returns structured DataFrame.
        """
        raw_list = []
        symbol_val = symbol if isinstance(symbol, str) and symbol else ticker_name
        clean_name = ticker_name.replace(".NS", "").replace(".BO", "")
        
        # 1. Fetch Alpha Vantage Real-Time Market Sentiment News
        av_news = self.fetch_alphavantage_news(clean_name)
        raw_list.extend(av_news)
        
        # 2. Fetch yfinance news
        if symbol_val:
            yf_news = self.fetch_yfinance_news(symbol_val)
            raw_list.extend(yf_news)
            
        # 3. Fetch Google News RSS feed
        search_query = f"{clean_name} stock share price news"
        gn_news = self.fetch_google_news_rss(search_query)
        raw_list.extend(gn_news)
        
        # 4. Fetch NewsAPI feed
        newsapi_query = f"{clean_name} stock OR shares OR market"
        napi_news = self.fetch_newsapi(newsapi_query)
        raw_list.extend(napi_news)
        
        # 5. Sector Proxy Fallback if raw news is empty
        if not raw_list:
            print(f"[LiveNewsFetcher Info] Zero ticker-specific news found for {ticker_name}. Applying sector fallback search...")
            fallback_query = "Indian stock market market news earnings"
            gn_news_fallback = self.fetch_google_news_rss(fallback_query)
            raw_list.extend(gn_news_fallback)
            
        # 6. Process, score & deduplicate live articles
        df_live = self.process_and_score_articles(raw_list)
        return df_live

if __name__ == "__main__":
    fetcher = LiveNewsFetcher()
    df_test = fetcher.get_live_news_for_ticker("Reliance Industries", "RELIANCE.NS")
    print(f"Fetched {len(df_test)} live news articles for RELIANCE.NS across 4 sources:")
    if not df_test.empty:
        comp_sentiment = compute_weighted_composite_sentiment(df_test)
        print(f"Weighted Composite Sentiment: {comp_sentiment}")
        print(df_test[['title', 'source', 'sentiment', 'credibility']].head(10))
