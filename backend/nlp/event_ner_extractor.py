import re

# 7 Corporate Event Taxonomies
EVENT_KEYWORDS = {
    "Earnings": [r"\bearnings\b", r"\bprofit\b", r"\brevenue\b", r"\bquarterly\b", r"\bmargin\b", r"\bq[1-4]\b", r"\bdividend\b"],
    "Merger": [r"\bmerger\b", r"\bconsolidate\b", r"\bjoint venture\b", r"\bcombine\b"],
    "Acquisition": [r"\bacquire\b", r"\bacquisition\b", r"\bbuyout\b", r"\btakeover\b"],
    "Product Launch": [r"\blaunch\b", r"\bproduct\b", r"\bunveil\b", r"\brelease\b", r"\bexpansion\b", r"\b5g\b"],
    "Government Policy": [r"\bpolicy\b", r"\bgovernment\b", r"\brbi\b", r"\btax\b", r"\btariff\b", r"\bregulation\b", r"\bcentral bank\b"],
    "Legal Issue": [r"\blegal\b", r"\blawsuit\b", r"\bcourt\b", r"\bsupreme court\b", r"\bsebi\b", r"\bpenalty\b", r"\binvestigation\b", r"\bprobe\b"],
    "Market Crash": [r"\bcrash\b", r"\bplunge\b", r"\bwar\b", r"\bconflict\b", r"\bselloff\b", r"\bpanic\b", r"\brecession\b", r"\bvolatility\b"]
}

NSE_TICKER_MAP = {
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "hdfc": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "tata motors": "TATAMOTORS.NS",
    "kent ro": "KENT",
    "novartis": "NOVN",
    "huawei": "HUAWEI"
}

def extract_named_entities(text: str) -> dict:
    """
    Extracts corporate entities and maps to NSE stock tickers.
    """
    if not text:
        return {"tickers": [], "company": "General"}

    text_lower = text.lower()
    matched_tickers = []
    matched_company = "General"

    for name, ticker in NSE_TICKER_MAP.items():
        if name in text_lower:
            matched_tickers.append(ticker)
            if matched_company == "General":
                matched_company = name.title()

    return {
        "tickers": list(set(matched_tickers)),
        "company": matched_company
    }

def classify_corporate_event(text: str) -> str:
    """
    Classifies headline into one of 7 corporate event types:
    Earnings, Merger, Acquisition, Product Launch, Government Policy, Legal Issue, Market Crash.
    Default fallback: General
    """
    if not text:
        return "General"

    text_lower = text.lower()
    for event_type, patterns in EVENT_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return event_type
    return "General"
