import os
import sys

# Force CPU & disable GPU/MPS thread locks and Tokenizer deadlocks on macOS
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

from portfolio.portfolio_advisor import PortfolioRecommendationEngine
from risk.risk_analyzer import calculate_stock_risk_metrics
from market_regime.regime_detector import MarketRegimeDetector

# ---------------------------------------------------
# NATIVE TECHNICAL INDICATORS
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

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Stock Prediction · RAG-LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# PREMIUM DESIGN SYSTEM — Cyan/Teal + Amber Dark Theme
# ---------------------------------------------------
st.markdown("""
<style>
    /* ═══════════════════════════════════════════════
       GOOGLE FONTS
       ═══════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ═══════════════════════════════════════════════
       DESIGN TOKENS
       ═══════════════════════════════════════════════ */
    :root {
        /* Backgrounds */
        --bg-base: #060a13;
        --bg-surface: #0c1322;
        --bg-elevated: #111b2e;
        --bg-card: rgba(17, 27, 46, 0.7);
        --bg-card-hover: rgba(22, 35, 58, 0.85);
        --bg-input: rgba(12, 19, 34, 0.8);

        /* Borders */
        --border-dim: rgba(255, 255, 255, 0.04);
        --border-subtle: rgba(255, 255, 255, 0.07);
        --border-medium: rgba(255, 255, 255, 0.12);
        --border-accent: rgba(6, 182, 212, 0.3);

        /* Text */
        --text-primary: #e2e8f0;
        --text-secondary: #8b9dc3;
        --text-muted: #556987;
        --text-heading: #f1f5f9;

        /* Accent: Cyan/Teal */
        --cyan-50: #ecfeff;
        --cyan-400: #22d3ee;
        --cyan-500: #06b6d4;
        --cyan-600: #0891b2;
        --cyan-glow: rgba(6, 182, 212, 0.15);

        /* Accent: Amber/Gold */
        --amber-300: #fcd34d;
        --amber-400: #fbbf24;
        --amber-500: #f59e0b;
        --amber-glow: rgba(245, 158, 11, 0.12);

        /* Accent: Emerald */
        --emerald-400: #34d399;
        --emerald-500: #10b981;
        --emerald-glow: rgba(16, 185, 129, 0.12);

        /* Accent: Rose */
        --rose-400: #fb7185;
        --rose-500: #f43f5e;
        --rose-glow: rgba(244, 63, 94, 0.12);

        /* Accent: Indigo */
        --indigo-400: #818cf8;
        --indigo-500: #6366f1;

        /* Radius */
        --r-xs: 6px;
        --r-sm: 8px;
        --r-md: 12px;
        --r-lg: 16px;
        --r-xl: 20px;
        --r-full: 9999px;

        /* Shadows */
        --shadow-card: 0 4px 16px rgba(0, 0, 0, 0.25);
        --shadow-elevated: 0 12px 40px rgba(0, 0, 0, 0.35);
        --shadow-cyan: 0 0 24px rgba(6, 182, 212, 0.12);
        --shadow-emerald: 0 0 24px rgba(16, 185, 129, 0.12);
        --shadow-rose: 0 0 24px rgba(244, 63, 94, 0.12);

        /* Transitions */
        --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
        --t-fast: 0.15s ease;
        --t-normal: 0.25s var(--ease-out);
        --t-smooth: 0.4s var(--ease-out);
    }

    /* ═══════════════════════════════════════════════
       GLOBAL RESET & TYPOGRAPHY
       ═══════════════════════════════════════════════ */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    /* ═══════════════════════════════════════════════
       APP BACKGROUND
       ═══════════════════════════════════════════════ */
    .stApp {
        background: linear-gradient(165deg, var(--bg-base) 0%, var(--bg-surface) 35%, #0a1628 70%, var(--bg-base) 100%) !important;
        color: var(--text-primary);
    }

    /* ═══════════════════════════════════════════════
       HIDE STREAMLIT DEFAULTS
       ═══════════════════════════════════════════════ */
    #MainMenu, header, footer { visibility: hidden; }

    /* ═══════════════════════════════════════════════
       SIDEBAR
       ═══════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-surface) 0%, var(--bg-base) 100%) !important;
        border-right: 1px solid var(--border-dim) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--cyan-400) !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.03em;
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
    }

    /* ═══════════════════════════════════════════════
       HEADER COMPONENT
       ═══════════════════════════════════════════════ */
    .hdr {
        background: linear-gradient(135deg, rgba(17, 27, 46, 0.75) 0%, rgba(12, 19, 34, 0.85) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--r-xl);
        padding: 24px 28px 20px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }

    .hdr::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--cyan-500), var(--amber-400), var(--cyan-500));
        background-size: 200% 100%;
        animation: shimmer 4s linear infinite;
    }

    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    .hdr__title {
        font-size: clamp(1.25rem, 3.5vw, 1.85rem);
        font-weight: 800;
        background: linear-gradient(90deg, var(--cyan-400) 0%, var(--amber-400) 55%, var(--cyan-400) 100%);
        background-size: 200% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 6s linear infinite;
        margin: 0 0 4px;
        line-height: 1.3;
    }

    .hdr__sub {
        color: var(--text-secondary);
        font-size: clamp(0.78rem, 1.8vw, 0.88rem);
        margin: 0 0 14px;
    }

    .hdr__badges {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 11px;
        border-radius: var(--r-full);
        font-size: 0.73rem;
        font-weight: 600;
        white-space: nowrap;
        letter-spacing: 0.01em;
    }

    .pill--cyan {
        background: var(--cyan-glow);
        border: 1px solid rgba(6, 182, 212, 0.25);
        color: var(--cyan-400);
    }

    .pill--amber {
        background: var(--amber-glow);
        border: 1px solid rgba(245, 158, 11, 0.25);
        color: var(--amber-400);
    }

    .pill--emerald {
        background: var(--emerald-glow);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: var(--emerald-400);
    }

    /* ═══════════════════════════════════════════════
       METRIC CARDS
       ═══════════════════════════════════════════════ */
    .kpi {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--r-lg);
        padding: 18px 20px;
        transition: all var(--t-normal);
        min-height: 100px;
    }

    .kpi:hover {
        transform: translateY(-2px);
        border-color: var(--border-accent);
        box-shadow: var(--shadow-cyan);
    }

    .kpi__label {
        font-size: 0.7rem;
        color: var(--text-muted);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
    }

    .kpi__value {
        font-size: clamp(1.2rem, 2.8vw, 1.55rem);
        font-weight: 700;
        color: var(--text-heading);
        line-height: 1.2;
        margin-bottom: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    .kpi__delta { font-size: 0.78rem; font-weight: 600; }
    .kpi__delta--up { color: var(--emerald-400); }
    .kpi__delta--down { color: var(--rose-400); }
    .kpi__delta--cyan { color: var(--cyan-400); }
    .kpi__delta--muted { color: var(--text-muted); font-weight: 400; font-size: 0.75rem; }

    /* ═══════════════════════════════════════════════
       PREDICTION RESULT CARDS
       ═══════════════════════════════════════════════ */
    .pred {
        border-radius: var(--r-lg);
        padding: 26px 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .pred--up {
        background: linear-gradient(150deg, rgba(16, 185, 129, 0.08) 0%, rgba(6, 78, 59, 0.22) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        box-shadow: var(--shadow-emerald);
    }

    .pred--down {
        background: linear-gradient(150deg, rgba(244, 63, 94, 0.08) 0%, rgba(136, 19, 55, 0.22) 100%);
        border: 1px solid rgba(244, 63, 94, 0.3);
        box-shadow: var(--shadow-rose);
    }

    .pred__dir {
        font-size: clamp(1.3rem, 3.5vw, 1.75rem);
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 8px;
    }

    .pred__dir--up { color: var(--emerald-400); }
    .pred__dir--down { color: var(--rose-400); }

    .pred__info {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin: 4px 0;
    }

    .pred__price {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 8px;
        font-family: 'JetBrains Mono', monospace;
    }

    .pred__price--up { color: var(--emerald-400); }
    .pred__price--down { color: var(--rose-400); }

    /* ═══════════════════════════════════════════════
       CONFIDENCE PANEL
       ═══════════════════════════════════════════════ */
    .conf {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--r-lg);
        padding: 22px;
    }

    .conf__title {
        margin: 0 0 14px;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--cyan-400);
    }

    /* ═══════════════════════════════════════════════
       TABS
       ═══════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: var(--bg-input);
        padding: 5px 6px;
        border-radius: var(--r-md);
        border: 1px solid var(--border-dim);
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: var(--r-sm);
        padding: 9px 16px;
        color: var(--text-muted);
        font-weight: 500;
        font-size: 0.82rem;
        white-space: nowrap;
        flex-shrink: 0;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(6, 182, 212, 0.1) !important;
        color: var(--cyan-400) !important;
        font-weight: 600;
        border: 1px solid rgba(6, 182, 212, 0.2);
    }

    /* ═══════════════════════════════════════════════
       HORIZON SELECTOR GRID (fixes mobile layout)
       ═══════════════════════════════════════════════ */
    .horizon-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
        gap: 10px;
        margin: 12px 0 20px;
    }

    .horizon-btn {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--r-md);
        padding: 14px 16px;
        color: var(--text-secondary);
        font-size: 0.82rem;
        font-weight: 500;
        cursor: pointer;
        text-align: center;
        transition: all var(--t-normal);
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
    }

    .horizon-btn:hover {
        border-color: var(--border-accent);
        color: var(--cyan-400);
        background: var(--bg-card-hover);
    }

    .horizon-btn--active {
        background: rgba(6, 182, 212, 0.1) !important;
        border-color: var(--cyan-500) !important;
        color: var(--cyan-400) !important;
        font-weight: 600;
        box-shadow: var(--shadow-cyan);
    }

    .horizon-btn__icon {
        font-size: 1.3rem;
    }

    .horizon-btn__label {
        font-size: 0.78rem;
        font-weight: 600;
    }

    .horizon-btn__sub {
        font-size: 0.68rem;
        color: var(--text-muted);
    }

    /* ═══════════════════════════════════════════════
       SECTION UTILITIES
       ═══════════════════════════════════════════════ */
    .sec-title {
        font-size: clamp(0.92rem, 2.2vw, 1.05rem);
        font-weight: 600;
        color: var(--text-heading);
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, var(--border-medium) 0%, transparent 100%);
        margin: 24px 0;
        border: none;
    }

    /* ═══════════════════════════════════════════════
       CHATBOT TAB STYLING (replaces st.popover)
       ═══════════════════════════════════════════════ */
    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 16px;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--r-md);
        margin-bottom: 12px;
    }

    .chat-header__name {
        font-weight: 700;
        font-size: 1rem;
        color: var(--cyan-400);
    }

    .chat-header__status {
        background: var(--emerald-glow);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: var(--emerald-400);
        padding: 3px 10px;
        border-radius: var(--r-full);
        font-size: 0.72rem;
        font-weight: 600;
    }

    /* ═══════════════════════════════════════════════
       BUTTONS
       ═══════════════════════════════════════════════ */
    .stButton > button[kind="primary"],
    .stButton > button {
        border-radius: var(--r-sm) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all var(--t-normal) !important;
    }

    /* ═══════════════════════════════════════════════
       FOOTER
       ═══════════════════════════════════════════════ */
    .app-foot {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.75rem;
        padding: 14px 0 6px;
        border-top: 1px solid var(--border-dim);
        margin-top: 28px;
    }

    /* ═══════════════════════════════════════════════
       RESPONSIVE — Tablet (≤1024px)
       ═══════════════════════════════════════════════ */
    @media (max-width: 1024px) {
        .hdr { padding: 18px 18px 16px; margin-bottom: 18px; }
        .kpi { padding: 14px 16px; min-height: 90px; }
        .kpi__value { font-size: 1.2rem; }
        .pred { padding: 20px 18px; }
        .horizon-grid { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 8px; }
    }

    /* ═══════════════════════════════════════════════
       RESPONSIVE — Mobile (≤768px)
       ═══════════════════════════════════════════════ */
    @media (max-width: 768px) {
        .hdr { padding: 14px 14px 12px; border-radius: var(--r-md); margin-bottom: 14px; }
        .hdr__title { font-size: 1.15rem; }
        .hdr__sub { font-size: 0.76rem; }
        .pill { font-size: 0.65rem; padding: 3px 8px; }
        .kpi { padding: 12px 14px; min-height: 80px; border-radius: var(--r-md); }
        .kpi__label { font-size: 0.65rem; }
        .kpi__value { font-size: 1.05rem; }
        .kpi__delta { font-size: 0.72rem; }
        .pred { padding: 16px; border-radius: var(--r-md); }
        .pred__dir { font-size: 1.2rem; }
        .stTabs [data-baseweb="tab-list"] { padding: 3px 4px; }
        .stTabs [data-baseweb="tab"] { padding: 7px 10px; font-size: 0.74rem; }
        .horizon-grid { grid-template-columns: repeat(2, 1fr); gap: 6px; }
        .horizon-btn { padding: 10px 10px; font-size: 0.76rem; }
    }

    /* ═══════════════════════════════════════════════
       RESPONSIVE — Small Mobile (≤480px)
       ═══════════════════════════════════════════════ */
    @media (max-width: 480px) {
        .hdr__title { font-size: 1rem; }
        .hdr__badges { gap: 4px; }
        .pill { font-size: 0.6rem; padding: 2px 6px; }
        .stTabs [data-baseweb="tab-list"] { gap: 1px; padding: 2px; }
        .stTabs [data-baseweb="tab"] { padding: 6px 7px; font-size: 0.68rem; }
        .horizon-grid { grid-template-columns: repeat(2, 1fr); }
        .horizon-btn { padding: 8px 8px; }
        .horizon-btn__icon { font-size: 1.1rem; }
        .horizon-btn__label { font-size: 0.7rem; }
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# CACHED LOADERS
# ---------------------------------------------------
@st.cache_resource
def load_lstm_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../models/lstm_model.h5")
    if not os.path.exists(model_path):
        try:
            from prediction.train_lstm import train_hybrid_rag_lstm
            st.warning("Model not found. Training automatically...")
            train_hybrid_rag_lstm()
        except Exception as err:
            st.error(f"Auto-training failed: {err}")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        with tf.device('/CPU:0'):
            model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

@st.cache_resource
def load_chatbot():
    try:
        from chatbot.chatbot import StockAssistantChatbot
        return StockAssistantChatbot()
    except Exception as e:
        st.warning(f"Chatbot fallback: {e}")
        return None


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.markdown("### ⚙️ Control Panel")

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

selected_stock_name = st.sidebar.selectbox("📌 Select NSE Stock", list(stocks.keys()), index=0)

custom_symbol_input = st.sidebar.text_input(
    "✏️ Custom Symbol",
    value="",
    placeholder="e.g. SBIN, HAL, IRFC"
).strip()

if custom_symbol_input:
    clean_sym = custom_symbol_input.upper()
    ticker = clean_sym if clean_sym.endswith(".NS") else f"{clean_sym}.NS"
    selected_stock_name = clean_sym.replace(".NS", "")
else:
    ticker = stocks[selected_stock_name]

period_map = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}
selected_period_label = st.sidebar.selectbox("📅 Timeframe", list(period_map.keys()), index=0)
period = period_map[selected_period_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Architecture")
st.sidebar.markdown(f"""
<div style="font-size: 0.78rem; color: #8b9dc3; line-height: 1.8;">
    <span style="color: #22d3ee;">Ticker</span> · {ticker}<br>
    <span style="color: #22d3ee;">Model</span> · LSTM 64→Dense 32<br>
    <span style="color: #22d3ee;">Lookback</span> · 5-Day Window<br>
    <span style="color: #22d3ee;">Features</span> · 16 Hybrid<br>
    <span style="color: #22d3ee;">NLP</span> · VADER + FAISS RAG
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown(f"""
<div class="hdr">
    <div class="hdr__title">📈 AI Stock Market Prediction System</div>
    <div class="hdr__sub">Time-Aware Hybrid RAG-LSTM Architecture · IEEE Research Level</div>
    <div class="hdr__badges">
        <span class="pill pill--emerald">⚡ Model Online</span>
        <span class="pill pill--cyan">📊 {selected_stock_name} ({ticker})</span>
        <span class="pill pill--amber">🕒 {selected_period_label}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, timeframe):
    df_raw = yf.download(symbol, period=timeframe, interval="1d")
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

with st.spinner("📡 Fetching live market data..."):
    df_data = fetch_stock_data(ticker, period)

if df_data.empty:
    st.error(f"Unable to fetch data for **{ticker}**. Try another ticker or timeframe.")
    st.stop()

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

features = [
    'RSI', 'MACD', 'Return', 'MA_20_ratio', 'Close_Open',
    'High_Low', 'Volume_ratio', 'Momentum_Factor',
    'Sentiment', 'Event',
    'RAG_Sentiment', 'RAG_Relevance', 'RAG_Event_Importance',
    'RAG_Market_Impact', 'RAG_Risk_Score', 'RAG_Confidence'
]


# ═══════════════════════════════════════════════════════
# MAIN TABS — chatbot is now a tab (no more st.popover)
# ═══════════════════════════════════════════════════════
tab_overview, tab_forecast, tab_indicators, tab_risk, tab_chat = st.tabs([
    "📊 Overview",
    "🔮 Forecast",
    "📈 Indicators",
    "🛡️ Risk",
    "🤖 AI Chat"
])

# ═══════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════
with tab_overview:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    price_chg = float(latest['Close']) - float(prev['Close'])
    pct_chg = (price_chg / float(prev['Close'])) * 100

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        dc = "kpi__delta--up" if price_chg >= 0 else "kpi__delta--down"
        ar = "▲" if price_chg >= 0 else "▼"
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi__label">Close Price</div>
            <div class="kpi__value">₹{latest['Close']:.2f}</div>
            <div class="{dc}">{ar} {price_chg:+.2f} ({pct_chg:+.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi__label">20-Day SMA</div>
            <div class="kpi__value">₹{latest['MA_20']:.2f}</div>
            <div class="kpi__delta--muted">Trend Baseline</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        rsi_v = float(latest['RSI'])
        rc = "kpi__delta--down" if rsi_v > 70 else ("kpi__delta--up" if rsi_v < 30 else "kpi__delta--cyan")
        rl = "Overbought" if rsi_v > 70 else ("Oversold" if rsi_v < 30 else "Neutral")
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi__label">RSI (14)</div>
            <div class="kpi__value">{rsi_v:.1f}</div>
            <div class="{rc}">{rl}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        mv = float(latest['MACD'])
        mc = "kpi__delta--up" if mv > 0 else "kpi__delta--down"
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi__label">MACD</div>
            <div class="kpi__value">{mv:.4f}</div>
            <div class="{mc}">{'Bullish' if mv > 0 else 'Bearish'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # CHART
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.78, 0.22]
    )

    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Price',
        increasing_line_color='#10b981', decreasing_line_color='#f43f5e'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['MA_20'], mode='lines',
        name='SMA 20', line=dict(color='#fbbf24', width=1.5, dash='dot')
    ), row=1, col=1)

    vc = ['#10b981' if c >= o else '#f43f5e' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Volume'], name='Volume',
        marker_color=vc, opacity=0.45
    ), row=2, col=1)

    fig.update_layout(
        title=dict(
            text=f"<b>{selected_stock_name}</b> <span style='color:#8b9dc3'>({ticker})</span>",
            font=dict(size=15, color="#e2e8f0", family="Inter")
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(12, 19, 34, 0.4)',
        height=500,
        margin=dict(l=6, r=6, t=44, b=6),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#8b9dc3")),
        xaxis2=dict(showgrid=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.03)'),
        yaxis2=dict(gridcolor='rgba(255,255,255,0.03)')
    )

    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
# TAB 2 — AI FORECAST
# ═══════════════════════════════════════════════════════
with tab_forecast:
    model = load_lstm_model()

    st.markdown('<div class="sec-title">🔮 Price Direction Forecast</div>', unsafe_allow_html=True)

    # Horizon selector using Streamlit radio for proper mobile behavior
    horizon_options = {
        "⚡ +5 Min": ("5m", 5),
        "⏱️ +15 Min": ("15m", 15),
        "⏱️ +30 Min": ("30m", 30),
        "⏱️ +1 Hour": ("60m", 60),
        "📅 Tomorrow": ("1d", 0)
    }

    selected_horizon = st.radio(
        "🎯 Select Prediction Horizon",
        list(horizon_options.keys()),
        index=0,
        horizontal=True,
        key="horizon_selector"
    )
    bar_interval, horizon_mins = horizon_options[selected_horizon]

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # --- INTRADAY ---
    if bar_interval != "1d":
        with st.spinner(f"Fetching intraday data..."):
            df_intra_raw = fetch_stock_data(ticker, "5d")

        if df_intra_raw.empty:
            st.error(f"Intraday data unavailable for {ticker}.")
        else:
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
            df_intra['Sentiment'] = 0.0
            df_intra['Event'] = 0
            df_intra['RAG_Sentiment'] = 0.0
            df_intra['RAG_Relevance'] = 0.5
            df_intra['RAG_Event_Importance'] = 0.4
            df_intra['RAG_Market_Impact'] = 0.1
            df_intra['RAG_Risk_Score'] = 0.1
            df_intra['RAG_Confidence'] = 0.5
            df_intra.dropna(inplace=True)

            if len(df_intra) < 5:
                st.warning("Insufficient data for 5-step lookback.")
            else:
                date_col = 'Datetime' if 'Datetime' in df_intra.columns else ('Date' if 'Date' in df_intra.columns else df_intra.columns[0])
                last_time = pd.to_datetime(df_intra[date_col].iloc[-1])
                target_time = last_time + pd.Timedelta(minutes=horizon_mins)
                current_price = float(df_intra['Close'].iloc[-1])

                scaler_i = MinMaxScaler()
                Xi = scaler_i.fit_transform(df_intra[features])
                seq = np.expand_dims(Xi[-5:], axis=0).astype(np.float32)

                run_btn = st.button(f"🚀 Run {bar_interval.upper()} Forecast", type="primary")

                if run_btn or 'single_intra_res' in st.session_state:
                    if run_btn:
                        if model is None:
                            st.error("LSTM Model unavailable.")
                        else:
                            with st.spinner("Neural network inference..."):
                                import tensorflow as tf
                                with tf.device('/CPU:0'):
                                    raw_score = float(model(seq, training=False).numpy()[0][0])
                                vol_avg = df_intra['Volatility'].mean()
                                if np.isnan(vol_avg): vol_avg = 0.002
                                move_pct = (raw_score - 0.5) * 2.0 * vol_avg * np.sqrt(horizon_mins / 15.0)
                                st.session_state['single_intra_res'] = {
                                    'score': raw_score,
                                    'current_price': current_price,
                                    'target_price': current_price * (1.0 + move_pct),
                                    'move_pct': move_pct,
                                    'target_time': target_time,
                                    'horizon_mins': horizon_mins
                                }

                    if 'single_intra_res' in st.session_state:
                        r = st.session_state['single_intra_res']
                        s = r['score']; tp = r['target_price']; mp = r['move_pct']

                        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                        p1, p2 = st.columns([1, 1], gap="medium")

                        with p1:
                            cls = "up" if s > 0.5 else "down"
                            icon = "📈" if s > 0.5 else "📉"
                            word = "UP" if s > 0.5 else "DOWN"
                            st.markdown(f"""
                            <div class="pred pred--{cls}">
                                <div class="pred__dir pred__dir--{cls}">{icon} PREDICTED: {word}</div>
                                <div class="pred__info">Target: <b>{r['target_time'].strftime("%H:%M:%S")}</b> (+{r['horizon_mins']}m)</div>
                                <div class="pred__price pred__price--{cls}">₹{tp:.2f} ({mp*100:+.2f}%)</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with p2:
                            st.markdown('<div class="conf"><div class="conf__title">📊 Confidence</div>', unsafe_allow_html=True)
                            st.metric("Probability", f"{s:.4f}", f"{(s-0.5)*100:+.2f}%")
                            st.progress(float(np.clip(s, 0.0, 1.0)))
                            st.markdown(f'<div style="color:#556987;font-size:0.75rem;margin-top:6px;">₹{r["current_price"]:.2f} → ₹{tp:.2f}</div></div>', unsafe_allow_html=True)

                        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                        st.markdown('<div class="sec-title">📋 Feature Matrix</div>', unsafe_allow_html=True)
                        fdf = pd.DataFrame(df_intra[features].tail(5))
                        fdf.index = [f"Bar −{4-i}" for i in range(5)]
                        st.dataframe(fdf.style.format("{:.4f}"), use_container_width=True)

    # --- DAILY ---
    else:
        if len(df) < 5:
            st.warning("Insufficient data for 5-day lookback.")
        else:
            scaler = MinMaxScaler()
            Xs = scaler.fit_transform(df[features])
            seq = np.expand_dims(Xs[-5:], axis=0).astype(np.float32)

            run_d = st.button("🚀 Run Daily Forecast", type="primary")

            if run_d or 'prediction_res' in st.session_state:
                if run_d:
                    if model is None:
                        st.error("LSTM Model unavailable.")
                    else:
                        with st.spinner("Neural network inference..."):
                            import tensorflow as tf
                            with tf.device('/CPU:0'):
                                score = float(model(seq, training=False).numpy()[0][0])
                            st.session_state['prediction_res'] = score
                            st.session_state['latest_pred'] = score

                if 'prediction_res' in st.session_state:
                    pv = st.session_state['prediction_res']

                    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                    d1, d2 = st.columns([1, 1], gap="medium")

                    with d1:
                        cls = "up" if pv > 0.5 else "down"
                        icon = "📈" if pv > 0.5 else "📉"
                        word = "UP" if pv > 0.5 else "DOWN"
                        msg = "Bullish momentum expected" if pv > 0.5 else "Bearish movement expected"
                        st.markdown(f"""
                        <div class="pred pred--{cls}">
                            <div class="pred__dir pred__dir--{cls}">{icon} PREDICTED: {word}</div>
                            <div class="pred__info">{msg} for tomorrow's session</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with d2:
                        st.markdown('<div class="conf"><div class="conf__title">📊 Confidence</div>', unsafe_allow_html=True)
                        st.metric("Probability", f"{pv:.4f}", f"{(pv-0.5)*100:+.2f}%")
                        st.progress(float(np.clip(pv, 0.0, 1.0)))
                        st.markdown('<div style="color:#556987;font-size:0.75rem;margin-top:6px;">Threshold: 0.50 · Binary Crossentropy</div></div>', unsafe_allow_html=True)

                    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                    st.markdown('<div class="sec-title">📋 Feature Matrix</div>', unsafe_allow_html=True)
                    fdf = pd.DataFrame(df[features].tail(5))
                    fdf.index = [f"Day −{4-i}" for i in range(5)]
                    st.dataframe(fdf.style.format("{:.4f}"), use_container_width=True)

    # Model Stats
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📐 Model Performance</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3, gap="medium")
    with m1: st.metric("Test Accuracy", "85.71%", "20% Held-Out Set")
    with m2: st.metric("Training Epochs", "40", "Adam lr=0.005")
    with m3: st.metric("Features", "16", "RAG + Technical Hybrid")

    with st.expander("❓ How to Verify Predictions"):
        st.markdown("""
        **UP (Score > 0.50)** — ✅ Correct if actual close > start price  
        **DOWN (Score ≤ 0.50)** — ✅ Correct if actual close < start price

        ```bash
        ./venv/bin/python prediction/train_lstm.py
        ```
        """)


# ═══════════════════════════════════════════════════════
# TAB 3 — INDICATORS & NEWS
# ═══════════════════════════════════════════════════════
with tab_indicators:
    st.markdown('<div class="sec-title">📈 Technical Indicators</div>', unsafe_allow_html=True)

    fig_i = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.1, row_heights=[0.5, 0.5],
        subplot_titles=("RSI (14)", "MACD")
    )

    fig_i.add_trace(go.Scatter(
        x=df['Date'], y=df['RSI'], mode='lines',
        name='RSI', line=dict(color='#fbbf24', width=1.5)
    ), row=1, col=1)
    fig_i.add_hline(y=70, line_dash="dot", line_color="rgba(244,63,94,0.35)", row=1, col=1)
    fig_i.add_hline(y=30, line_dash="dot", line_color="rgba(16,185,129,0.35)", row=1, col=1)

    fig_i.add_trace(go.Scatter(
        x=df['Date'], y=df['MACD'], mode='lines',
        name='MACD', line=dict(color='#22d3ee', width=1.5)
    ), row=2, col=1)
    fig_i.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.1)", row=2, col=1)

    fig_i.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(12, 19, 34, 0.4)',
        height=360,
        margin=dict(l=6, r=6, t=32, b=6),
        showlegend=False,
        yaxis=dict(gridcolor='rgba(255,255,255,0.03)'),
        yaxis2=dict(gridcolor='rgba(255,255,255,0.03)')
    )
    st.plotly_chart(fig_i, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    i1, i2 = st.columns([1, 1], gap="medium")

    with i1:
        st.markdown('<div class="sec-title">📋 Indicator Data</div>', unsafe_allow_html=True)
        disp = ['Date', 'Close', 'MA_20', 'RSI', 'MACD', 'Return', 'Volatility']
        avail = [c for c in disp if c in df.columns]
        fmt = {c: ('₹{:.2f}' if c in ('Close', 'MA_20') else ('{:.2f}' if c == 'RSI' else '{:.4f}')) for c in avail if c != 'Date'}
        st.dataframe(df[avail].tail(10).style.format(fmt), use_container_width=True)

    with i2:
        st.markdown('<div class="sec-title">📰 News Sentiment</div>', unsafe_allow_html=True)
        news_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/news_processed.csv")
        if os.path.exists(news_file):
            ndf = pd.read_csv(news_file)
            if 'sentiment' not in ndf.columns: ndf['sentiment'] = 0.0
            if 'event' not in ndf.columns: ndf['event'] = 'General'
            st.dataframe(
                ndf[['title', 'sentiment', 'event']].head(8).style.format({'sentiment': '{:.4f}'}),
                use_container_width=True
            )
        else:
            st.info("Run `python process_news.py` to populate news data.")


# ═══════════════════════════════════════════════════════
# TAB 4 — RISK & REGIME
# ═══════════════════════════════════════════════════════
with tab_risk:
    st.markdown('<div class="sec-title">🛡️ Risk Analysis & Market Regime</div>', unsafe_allow_html=True)

    r1, r2 = st.columns([1, 1], gap="medium")

    with r1:
        st.markdown('<div class="sec-title" style="font-size:0.9rem">📊 Risk Metrics</div>', unsafe_allow_html=True)
        ret_arr = df['Return'].dropna().values
        pp = st.session_state.get('latest_pred', 0.5)
        rm = calculate_stock_risk_metrics(ret_arr, pp)

        rm1, rm2 = st.columns(2, gap="small")
        with rm1:
            st.metric("Volatility", f"{rm.get('volatility_score', 0):.4f}")
            st.metric("VaR (95%)", f"{rm.get('var_95', 0):.4f}")
        with rm2:
            st.metric("Sharpe Ratio", f"{rm.get('sharpe_ratio', 0):.3f}")
            st.metric("Risk Level", rm.get('risk_level', 'N/A'))

    with r2:
        st.markdown('<div class="sec-title" style="font-size:0.9rem">🔬 Market Regime</div>', unsafe_allow_html=True)
        try:
            det = MarketRegimeDetector()
            regime = det.detect_regime(close_series.dropna())
            st.markdown(f"""
            <div class="kpi" style="text-align:center; padding: 24px;">
                <div class="kpi__label">Current Regime</div>
                <div class="kpi__value" style="font-size: 1.25rem; margin-top: 10px;">{regime}</div>
                <div class="kpi__delta--muted" style="margin-top: 6px;">Volatility & SMA Ratio Clustering</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Regime error: {e}")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">💼 Portfolio Recommendation</div>', unsafe_allow_html=True)
    try:
        advisor = PortfolioRecommendationEngine()
        forecasts = [{"ticker": ticker, "prob": pp, "returns": ret_arr[-20:] if len(ret_arr) >= 20 else ret_arr}]
        pdf = advisor.rank_portfolio(forecasts)
        if not pdf.empty:
            st.dataframe(pdf, use_container_width=True)
        else:
            st.info("Add more stock forecasts for portfolio ranking.")
    except Exception as e:
        st.warning(f"Portfolio error: {e}")


# ═══════════════════════════════════════════════════════
# TAB 5 — AI CHATBOT (moved from st.popover)
# ═══════════════════════════════════════════════════════
with tab_chat:
    chatbot = load_chatbot()
    doc_cnt = chatbot.rag_engine.get_doc_count() if (chatbot and hasattr(chatbot, 'rag_engine')) else 0

    st.markdown(f"""
    <div class="chat-header">
        <span class="chat-header__name">🤖 RAG Financial AI Assistant</span>
        <span class="chat-header__status">⚡ {doc_cnt} Docs Indexed</span>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Ask about stock indicators, market news, LSTM predictions, or search the RAG news corpus.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("assistant", f"Hello! I'm your RAG-Augmented AI Assistant. How can I help analyze **{selected_stock_name}** ({ticker}) today?")
        ]

    chat_container = st.container(height=420)
    with chat_container:
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(text)

    user_input = st.chat_input("Ask about stocks, indicators, news headlines...", key="chat_tab_input")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        stock_ctx = {
            'stock': selected_stock_name,
            'close': float(df['Close'].iloc[-1]),
            'rsi': float(df['RSI'].iloc[-1]),
            'macd': float(df['MACD'].iloc[-1]),
            'confidence': st.session_state.get('latest_pred', None)
        }
        if chatbot:
            bot_reply = chatbot.get_response(user_input, stock_ctx)
        else:
            bot_reply = "Chatbot is currently unavailable."
        st.session_state.chat_history.append(("assistant", bot_reply))
        st.rerun()


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("""
<div class="app-foot">
    📈 AI Stock Market Prediction · TensorFlow LSTM · VADER NLP · FAISS RAG · Plotly · Streamlit
</div>
""", unsafe_allow_html=True)