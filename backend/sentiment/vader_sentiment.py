import ssl

# Bypass SSL context issues if downloading resources
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

analyzer = None

# Try importing vaderSentiment first
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
except ImportError:
    # Try importing NLTK SentimentIntensityAnalyzer
    try:
        import nltk
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
        except Exception:
            nltk.download('vader_lexicon', quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
    except Exception:
        analyzer = None

def get_sentiment(text):
    text_str = str(text)
    if analyzer is not None:
        try:
            score = analyzer.polarity_scores(text_str)
            return score['compound']
        except Exception:
            pass

    # Simple fallback keyword sentiment analysis if VADER is unavailable
    text_lower = text_str.lower()
    positive_words = {'profit', 'growth', 'gain', 'up', 'bullish', 'increase', 'rise', 'surge', 'record', 'high', 'strong', 'positive', 'win'}
    negative_words = {'loss', 'drop', 'decline', 'down', 'bearish', 'decrease', 'fall', 'plunge', 'low', 'weak', 'negative', 'lawsuit', 'delay', 'war'}

    words = text_lower.split()
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    total = pos_count + neg_count

    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total