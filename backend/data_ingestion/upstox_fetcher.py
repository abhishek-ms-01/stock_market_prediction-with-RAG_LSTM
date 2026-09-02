import os
import urllib.request
import urllib.parse
import json
import ssl
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pre-mapped Instrument Keys for all 54 top Indian NSE symbols
UPSTOX_ISIN_MAP = {
    'RELIANCE.NS': 'NSE_EQ|INE002A01018',
    'TCS.NS': 'NSE_EQ|INE467B01029',
    'INFY.NS': 'NSE_EQ|INE009A01021',
    'HDFCBANK.NS': 'NSE_EQ|INE040A01034',
    'ICICIBANK.NS': 'NSE_EQ|INE090A01021',
    'SBIN.NS': 'NSE_EQ|INE062A01020',
    'BHARTIARTL.NS': 'NSE_EQ|INE397D01024',
    'ITC.NS': 'NSE_EQ|INE154A01025',
    'LT.NS': 'NSE_EQ|INE018A01030',
    'TATASTEEL.NS': 'NSE_EQ|INE081A01020',
    'AXISBANK.NS': 'NSE_EQ|INE238A01034',
    'KOTAKBANK.NS': 'NSE_EQ|INE237A01036',
    'MARUTI.NS': 'NSE_EQ|INE585B01010',
    'SUNPHARMA.NS': 'NSE_EQ|INE044A01036',
    'TITAN.NS': 'NSE_EQ|INE280A01028',
    'ASIANPAINT.NS': 'NSE_EQ|INE021A01026',
    'BAJFINANCE.NS': 'NSE_EQ|INE296A01032',
    'HCLTECH.NS': 'NSE_EQ|INE860A01027',
    'NTPC.NS': 'NSE_EQ|INE733E01010',
    'POWERGRID.NS': 'NSE_EQ|INE752E01010',
    'ULTRACEMCO.NS': 'NSE_EQ|INE481G01011',
    'M&M.NS': 'NSE_EQ|INE101A01026',
    'ADANIPORTS.NS': 'NSE_EQ|INE742F01042',
    'ADANIENT.NS': 'NSE_EQ|INE423A01024',
    'COALINDIA.NS': 'NSE_EQ|INE522F01014',
    'HINDALCO.NS': 'NSE_EQ|INE038A01020',
    'GRASIM.NS': 'NSE_EQ|INE047A01021',
    'ONGC.NS': 'NSE_EQ|INE213A01029',
    'TECHM.NS': 'NSE_EQ|INE669C01036',
    'INDUSINDBK.NS': 'NSE_EQ|INE095A01012',
    'NESTLEIND.NS': 'NSE_EQ|INE239A01024',
    'CIPLA.NS': 'NSE_EQ|INE059A01026',
    'DRREDDY.NS': 'NSE_EQ|INE089A01031',
    'EICHERMOT.NS': 'NSE_EQ|INE066A01021',
    'WIPRO.NS': 'NSE_EQ|INE075A01022',
    'JSWSTEEL.NS': 'NSE_EQ|INE019A01038',
    'HEROMOTOCO.NS': 'NSE_EQ|INE158A01026',
    'BRITANNIA.NS': 'NSE_EQ|INE216A01030',
    'APOLLOHOSP.NS': 'NSE_EQ|INE437A01024',
    'SBILIFE.NS': 'NSE_EQ|INE123W01016',
    'HDFCLIFE.NS': 'NSE_EQ|INE795G01014',
    'TATACONSUM.NS': 'NSE_EQ|INE192A01025',
    'TRENT.NS': 'NSE_EQ|INE849A01020',
    'BEL.NS': 'NSE_EQ|INE263A01024',
    'SHRIRAMFIN.NS': 'NSE_EQ|INE721A01047',
    'DIVISLAB.NS': 'NSE_EQ|INE361B01024',
    'BAJAJ-AUTO.NS': 'NSE_EQ|INE917I01010',
    'HAL.NS': 'NSE_EQ|INE066F01020',
    'SUZLON.NS': 'NSE_EQ|INE040H01021',
    'IRFC.NS': 'NSE_EQ|INE053F01010',
    'TATAPOWER.NS': 'NSE_EQ|INE245A01021',
    'BHEL.NS': 'NSE_EQ|INE257A01026',
    'IREDA.NS': 'NSE_EQ|INE202E01016',
    'VEDL.NS': 'NSE_EQ|INE205A01025'
}

class UpstoxDataFetcher:
    """
    Upstox v2 API Provider for 0-delay real-time Indian stock market quotes & depth.
    """
    def __init__(self):
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN", "")

    def fetch_upstox_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches 0-delay real-time price quote for Indian stocks via Upstox API v2.
        """
        if not self.access_token:
            return {}

        instrument_key = UPSTOX_ISIN_MAP.get(symbol.upper(), "")
        if not instrument_key:
            # Try appending .NS if plain ticker name passed
            if not symbol.endswith(".NS"):
                instrument_key = UPSTOX_ISIN_MAP.get(f"{symbol.upper()}.NS", "")
        
        if not instrument_key:
            return {}

        encoded_key = urllib.parse.quote(instrument_key)
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_key}"
        
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        }

        try:
            req = urllib.request.Request(url, headers=headers)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            with urllib.request.urlopen(req, timeout=4, context=ctx) as response:
                payload = json.loads(response.read().decode('utf-8'))

            if payload.get("status") == "success" and "data" in payload:
                data_dict = payload["data"]
                # Find matching payload key (e.g. NSE_EQ:RELIANCE)
                for k, quote in data_dict.items():
                    ohlc = quote.get("ohlc", {})
                    return {
                        "provider": "Upstox API v2",
                        "symbol": quote.get("symbol", symbol),
                        "instrument_key": quote.get("instrument_token", instrument_key),
                        "current_price": float(quote.get("last_price", 0.0)),
                        "net_change": float(quote.get("net_change", 0.0)),
                        "open": float(ohlc.get("open", 0.0)),
                        "high": float(ohlc.get("high", 0.0)),
                        "low": float(ohlc.get("low", 0.0)),
                        "previous_close": float(ohlc.get("close", 0.0)),
                        "volume": int(quote.get("volume", 0)),
                        "average_price": float(quote.get("average_price", 0.0)),
                        "timestamp": quote.get("timestamp", "")
                    }
        except Exception as e:
            print(f"[UpstoxFetcher Warning] Quote fetch failed for {symbol}: {e}")

        return {}

if __name__ == "__main__":
    fetcher = UpstoxDataFetcher()
    test_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    for sym in test_symbols:
        q = fetcher.fetch_upstox_quote(sym)
        print(f"[{sym}] Upstox Real-Time Quote:")
        print(q)
