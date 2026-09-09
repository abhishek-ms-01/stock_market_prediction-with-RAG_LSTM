# 📘 AlphaTrade (v2.0): System Architecture, AI/Quant Models & Presentation Guide

> **Official Master Technical & Presentation Companion**  
> *An Event-Driven, Multi-Modal Stock Market Forecasting & Financial Intelligence Platform*

---

## 📑 Table of Contents
1. [Executive Summary & Presentation Elevator Pitch](#1-executive-summary--presentation-elevator-pitch)
2. [Ready-to-Present Slide Deck Blueprint (12-Slide Outline)](#2-ready-to-present-slide-deck-blueprint-12-slide-outline)
3. [The Core Problem & Novel Value Proposition](#3-the-core-problem--novel-value-proposition)
4. [Technology Stack & Architectural Justifications](#4-technology-stack--architectural-justifications)
5. [High-Level End-to-End System Architecture](#5-high-level-end-to-end-system-architecture)
6. [Multi-Source Real-Time Data Ingestion Layer](#6-multi-source-real-time-data-ingestion-layer)
7. [Feature Engineering & Technical Indicator Pipeline](#7-feature-engineering--technical-indicator-pipeline)
8. [Deep Dive: Dynamic Online LSTM & Binary Focal Loss](#8-deep-dive-dynamic-online-lstm--binary-focal-loss)
9. [Deep Dive: Event-Driven NLP Sentiment & Dense RAG Engine](#9-deep-dive-event-driven-nlp-sentiment--dense-rag-engine)
10. [Off-Market Hours & Tomorrow's Forecast Engine](#10-off-market-hours--tomorrows-forecast-engine)
11. [Risk Analysis, K-Means Market Regime & Portfolio Advisor](#11-risk-analysis-k-means-market-regime--portfolio-advisor)
12. [Frontend Architecture & Interactive Live Demo Script](#12-frontend-architecture--interactive-live-demo-script)
13. [Complete API Specification & Endpoints](#13-complete-api-specification--endpoints)
14. [Viva & Technical Q&A Defense Guide (Top 10 Questions)](#14-viva--technical-qa-defense-guide-top-10-questions)
15. [Financial Loopholes, Mitigations & Production Roadmap](#15-financial-loopholes-mitigations--production-roadmap)
16. [Quick-Start Execution Commands](#16-quick-start-execution-commands)

---

## 1. Executive Summary & Presentation Elevator Pitch

> **The 30-Second Elevator Pitch for Your Presentation:**  
> *"Traditional algorithmic trading models rely purely on historical numeric price candles (OHLCV), completely missing qualitative catalysts like breaking earnings reports, regulatory announcements, and sentiment shockwaves. Meanwhile, basic sentiment bots lack quantitative timing and risk discipline. **AlphaTrade** bridges this divide with a **Time-Aware Multi-Modal AI Architecture**. It ingests zero-delay market quotes via Upstox API v2, extracts live financial news via Google/Yahoo feeds, computes credibility-weighted time-decay sentiment, indexes events into a FAISS Dense Vector RAG store, and feeds an engineered 8-feature matrix into a dynamic **64-unit LSTM neural network trained on-the-fly with Binary Focal Loss**. Furthermore, it handles off-market hours seamlessly by predicting tomorrow's opening trajectory, and wraps everything in quantitative risk metrics (VaR 95%, Sharpe Ratio), K-Means market regime detection, and an interactive Next.js 14 institutional-grade dashboard."*

---

## 2. Ready-to-Present Slide Deck Blueprint (12-Slide Outline)

Use this slide structure for your academic presentation, viva, project demo, or conference talk:

| Slide # | Slide Title | Visual / Content Element | Key Talking Points (What to Say) |
| :--- | :--- | :--- | :--- |
| **Slide 1** | **Title & Introduction** | Project name, your name, institution, modern UI screenshot. | Introduce AlphaTrade: an event-driven, multi-modal stock forecasting and financial intelligence platform. |
| **Slide 2** | **The Problem in Algorithmic Trading** | Comparison graphic: Pure Technical vs. Pure Sentiment vs. Hybrid. | Pure technicals lag behind news; pure sentiment lacks quantitative risk management. Over 70% of intraday swings are catalyst-driven. |
| **Slide 3** | **Proposed Solution: Hybrid Architecture** | High-level 3-tier block diagram (Ingestion $\to$ AI Engine $\to$ Dashboard). | Combines quantitative price history, live streaming feeds (Upstox/Finnhub), NLP sentiment, and LSTM deep learning. |
| **Slide 4** | **System Architecture & Data Pipeline** | Full Mermaid architecture diagram. | Walk through the decoupled Next.js 14 frontend and Python FastAPI backend, detailing low-latency communication and modular design. |
| **Slide 5** | **Multi-Source Real-Time Data Ingestion** | Logos/diagram: Upstox API v2, Finnhub, yfinance, Google RSS, Yahoo News. | Explaining zero-delay quotes for Indian (NSE/BSE) equities, global quotes via Finnhub, and live news deduplication. |
| **Slide 6** | **NLP Sentiment & Event-Driven RAG** | Mathematical decay equation + FAISS vector search diagram. | Source credibility weighting (Reuters 0.95 vs. blogs), exponential recency decay ($e^{-\lambda \Delta t}$), and 384-dim semantic embeddings. |
| **Slide 7** | **The LSTM Neural Network & Focal Loss** | LSTM cell diagram + Focal Loss equation vs. Cross-Entropy. | Explain why LSTM overcomes vanishing gradients; explain how Binary Focal Loss solves market trend class imbalance ($\gamma=2.0, \alpha=0.25$). |
| **Slide 8** | **Off-Market Hours & Session Adaptation** | Flowchart: Market Open vs. Market Closed logic. | How the system avoids breaking on evenings/weekends by computing tomorrow's opening session forecast and target date. |
| **Slide 9** | **Quant Risk, Regime Clustering & Portfolio Engine** | Risk gauges + K-Means cluster breakdown + BUY/HOLD/SELL matrix. | VaR (95%), Sharpe Ratio, 5-cluster K-Means regime detection (Bull/Bear/Sideways/High Vol/Low Vol), and automated portfolio recommendations. |
| **Slide 10** | **Live Demonstration & UI Features** | Screenshots or live browser demo of Next.js 14 dashboard. | Walk through Candlestick chart, Forecast Card, Live Pipeline monitor, Risk dashboard, and RAG AI Chat assistant. |
| **Slide 11** | **Engineering Challenges & Mitigations** | Table of technical bottlenecks (overfitting, API limits, lookahead bias) and solutions. | Demonstrates intellectual maturity and engineering rigor (caching, focal loss, relative features, robust fallbacks). |
| **Slide 12** | **Conclusion & Future Roadmap** | Summary bullet points + future Level-2 order book & distributed caching. | Reiterate accuracy improvements, hybrid architecture advantage, and open the floor for Q&A. |

---

## 3. The Core Problem & Novel Value Proposition

### 3.1 The Limitations of Existing Approaches
1. **Technicals-Only Systems (ARIMA, Classic LSTM, MACD/RSI)**:
   - Price movements follow random walks until external catalysts appear.
   - Pure technical indicators are **lagging**; they only reflect what already occurred, making them vulnerable to sudden news shocks (earnings misses, geopolitical shocks).
2. **Sentiment-Only Bots (Twitter/Reddit Scrapers)**:
   - Noise-heavy and uncalibrated; treat unverified blog posts with the same weight as audited filings.
   - Ignore capital structure, momentum, and technical overbought/oversold levels.
3. **The Static Model Fallacy**:
   - Models trained offline on 2020 data fail when deployed in 2026 due to non-stationary market regimes.

### 3.2 AlphaTrade’s Novel Multi-Modal Innovations
```
             ┌─────────────────────────────────────────────────────────┐
             │                AlphaTrade Hybrid Engine                 │
             └────────────────────────────┬────────────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     ┌────────────────────────┐                      ┌────────────────────────┐
     │ Quantitative Subsystem │                      │ Qualitative Subsystem  │
     │ - Intraday / Daily OHLC│                      │ - Live News Scraping   │
     │ - Relative Indicators  │                      │ - Credibility Weighting│
     │ - Volatility Scaling   │                      │ - Recency Time-Decay   │
     └────────────┬───────────┘                      │ - FAISS Dense RAG      │
                  │                                  └────────────┬───────────┘
                  └───────────────────────┬───────────────────────┘
                                          ▼
                      ┌───────────────────────────────────────┐
                      │   Online Dynamic LSTM + Focal Loss    │
                      │  Direction, Probability & Target Price│
                      └───────────────────┬───────────────────┘
                                          ▼
                      ┌───────────────────────────────────────┐
                      │  Risk Metrics, Regimes & Tomorrow Fix │
                      └───────────────────────────────────────┘
```

---

## 4. Technology Stack & Architectural Justifications

| Component | Technology | Role | Why Chosen? (Defense Rationale) |
| :--- | :--- | :--- | :--- |
| **Frontend Framework** | **Next.js 14 (App Router)** | Web Client & Dashboard | React Server Components, high performance, automatic route splitting, and clean API integration. |
| **Styling & Motion** | **TailwindCSS + Framer Motion** | UI & Micro-animations | Fast design system, high aesthetic appeal, glassmorphism, responsive dark mode. |
| **Financial Charting** | **TradingView Lightweight Charts** | Interactive Candlesticks | High-performance HTML5 Canvas rendering designed specifically for high-frequency financial series. |
| **Backend API** | **FastAPI (Python 3.12)** | Microservice REST API | Asynchronous ASGI, high throughput, automatic OpenAPI documentation, native integration with NumPy/TensorFlow. |
| **Deep Learning** | **TensorFlow / Keras 2.x/3.x** | Neural Network Engine | Reliable LSTM implementations, custom loss function support (Focal Loss), CPU/GPU cross-compatibility. |
| **Vector Database** | **FAISS (Meta AI)** | Dense Vector Index | Highly optimized C++ vector similarity indexing with $O(1)$ nearest-neighbor search. |
| **Embeddings** | **SentenceTransformers (`all-MiniLM-L6-v2`)** | Text Vectorization | Compact 384-dimensional dense semantic vectors with fast inference latency (~15ms on CPU). |
| **NLP Sentiment** | **NLTK VADER + Custom Lexicon** | News Polarity Scoring | Rule-adjusted valence scoring optimized for financial headlines with domain keywords. |
| **Live Indian Quotes** | **Upstox API v2** | Real-Time NSE/BSE Quotes | Sub-second official market quote feed for Indian National Stock Exchange equities. |
| **Global Data** | **Finnhub + yfinance** | Global Quotes & Daily Data | Zero-delay US quotes via Finnhub; comprehensive 1-year historical daily and 5-day intraday data via yfinance. |
| **Machine Learning** | **Scikit-Learn** | Regime Detection & Scaling | K-Means clustering for regime detection and MinMaxScaler for tensor normalization. |

---

## 5. High-Level End-to-End System Architecture

```mermaid
flowchart TB
    classDef client fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#fff
    classDef server fill:#1e1b4b,stroke:#8b5cf6,stroke-width:2px,color:#fff
    classDef data fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#fff
    classDef ai fill:#701a75,stroke:#ec4899,stroke-width:2px,color:#fff

    subgraph PresentationLayer["🖥️ Presentation Layer (Next.js 14 Dashboard)"]
        UI_Home["/dashboard (Main Overview)"] :::client
        UI_Forecast["/dashboard/forecast (LSTM Forecasts)"] :::client
        UI_Indicators["/dashboard/indicators (Technical Charts)"] :::client
        UI_Live["/dashboard/live-pipeline (Streaming Feeds)"] :::client
        UI_Risk["/dashboard/risk (Risk & Regime Meters)"] :::client
        UI_Chat["/dashboard/ai-chat (Financial RAG Bot)"] :::client
    end

    subgraph GatewayLayer["⚡ API Gateway (FastAPI Python 3.12)"]
        Router["main.py Routing Engine"] :::server
        SessionMgr["Market Session Evaluator (IST/EST Timezone)"] :::server
        CacheMgr["In-Memory Model & Quote Cache (5-min TTL)"] :::server
    end

    subgraph IngestionLayer["📥 Multi-Feed Ingestion Engine"]
        UpstoxFeed["Upstox API v2 (NSE/BSE 0-Delay Quotes)"] :::data
        FinnhubFeed["Finnhub API (US Global Live Quotes)"] :::data
        YFHistorical["yfinance (Daily & 5-min Intraday OHLCV)"] :::data
        NewsScraper["Live News Scraper (Yahoo + Google RSS)"] :::data
    end

    subgraph IntelligenceLayer["🤖 AI Machine Learning & Quantitative Engine"]
        QuantPrep["Feature Engineer (RSI, MACD, MA20, Volatility)"] :::ai
        NLPFilter["VADER + Financial Rules + Recency Decay"] :::ai
        FAISSEngine["FAISS Vector DB (all-MiniLM-L6-v2)"] :::ai
        LSTMOnline["Online LSTM Network (64 Units + Binary Focal Loss)"] :::ai
        KMeansRegime["K-Means Clusterer (5 Market Regimes)"] :::ai
        RiskQuant["Quant Risk Analyzer (VaR 95%, Sharpe, Drawdown)"] :::ai
        PortfolioEngine["Portfolio Recommendation Engine (BUY/HOLD/SELL)"] :::ai
    end

    %% Wiring
    PresentationLayer <-->|Async REST API (JSON)| GatewayLayer
    GatewayLayer --> SessionMgr
    GatewayLayer --> CacheMgr
    GatewayLayer --> IngestionLayer
    
    IngestionLayer --> IntelligenceLayer
    QuantPrep & NLPFilter --> LSTMOnline
    QuantPrep --> KMeansRegime
    QuantPrep --> RiskQuant
    RiskQuant --> PortfolioEngine
    NewsScraper --> NLPFilter --> FAISSEngine
    
    IntelligenceLayer --> GatewayLayer
```

---

## 6. Multi-Source Real-Time Data Ingestion Layer

AlphaTrade implements a resilient, multi-tiered data acquisition pipeline:

```mermaid
flowchart LR
    subgraph IndianEquities["🇮🇳 Indian Equities (.NS/.BO)"]
        U1["Upstox API v2\nReal-Time Socket/REST"]
        U2["yfinance Fallback\n1d / 5m Historical"]
    end

    subgraph GlobalEquities["🇺🇸 US / Global Equities"]
        F1["Finnhub REST API\nZero-Delay Quotes"]
        F2["yfinance\nHistorical OHLCV"]
    end

    subgraph NewsFeeds["📰 Qualitative Feeds"]
        N1["Google News RSS Feed"]
        N2["Yahoo Finance News Scraper"]
    end

    IndianEquities --> Normalizer["Data Normalization & Cleaning Pipeline"]
    GlobalEquities --> Normalizer
    NewsFeeds --> Normalizer
```

### 6.1 Real-Time Quotes via Upstox API v2 & Finnhub
- **Indian Equities (`.NS`, `.BO`)**: Ingested via `UpstoxDataFetcher` (`/api/quote/realtime?symbol=RELIANCE.NS`). It queries Upstox's v2 market quote endpoint, delivering sub-second updates for NSE large-caps.
- **US Equities (`AAPL`, `TSLA`, etc.)**: Processed via Finnhub REST API using real-time tick aggregation.
- **Failover Guarantee**: If live API keys are absent or throttled, the engine automatically falls back to `yfinance` 1-minute/5-minute candle interpolation without raising server exceptions.

### 6.2 Live News Scraper & Deduplication
- **Multi-Source Scraping**: Pulls parallel RSS XML feeds from Google News and scrapes Yahoo Finance company endpoints.
- **Deduplication**: Hashes headline strings to purge duplicate syndications.
- **Event Extraction**: Classifies headlines into corporate categories (`EARNINGS`, `MERGER_ACQUISITION`, `REGULATORY`, `DIVIDEND`, `PRODUCT_LAUNCH`, `LITIGATION`).

---

## 7. Feature Engineering & Technical Indicator Pipeline

Rather than feeding raw price series (which are non-stationary and cause neural networks to overfit), AlphaTrade engineers **relative, stationary ratios**:

| Feature Name | Mathematical Formula | Purpose & Interpretation |
| :--- | :--- | :--- |
| **Log Returns ($R_t$)** | $\ln\left(\frac{P_t}{P_{t-1}}\right)$ | Stationary representation of daily price change. |
| **RSI(14)** | $100 - \frac{100}{1 + \frac{\text{EMA}_{14}(\text{Gain})}{\text{EMA}_{14}(\text{Loss})}}$ | Quantifies momentum and overbought ($>70$) vs. oversold ($<30$) boundaries. |
| **MACD** | $\text{EMA}_{12}(P) - \text{EMA}_{26}(P)$ | Measures convergence and divergence of short vs. long trend momentum. |
| **MA 20 Ratio** | $\frac{P_t}{\text{SMA}_{20}(P)} - 1.0$ | Price deviation relative to its 20-day trend mean. |
| **Intraday Body Ratio** | $\frac{\text{Close}_t}{\text{Open}_t} - 1.0$ | Captures daily bull/bear struggle and candlestick color. |
| **Daily Volatility Range** | $\frac{\text{High}_t}{\text{Low}_t} - 1.0$ | Intraday spread and market participant uncertainty. |
| **Volume Ratio** | $\frac{V_t}{\text{SMA}_{10}(V)} - 1.0$ | Volume surge or contraction relative to the 10-day baseline. |
| **Rolling Volatility** | $\sigma_{10}(R_t)$ | 10-period standard deviation of log returns. |

All features are normalized using `MinMaxScaler(feature_range=(0, 1))` fitted strictly on the lookback window to prevent forward-looking bias.

---

## 8. Deep Dive: Dynamic Online LSTM & Binary Focal Loss

### 8.1 Why Long Short-Term Memory (LSTM)?
Traditional Recurrent Neural Networks suffer from the **vanishing gradient problem** when backpropagating through time, losing memory of events that occurred multiple steps back. LSTMs solve this through internal **gates**:

```
                       ┌─────────────────────────┐
                       │   Previous Cell State   │
                       │         C_{t-1}         │
                       └────────────┬────────────┘
                                    │
                                    ▼
       ┌───────────┐         ┌─────────────┐
x_t ──►│Forget Gate│────────►│  Multiply   │
       │   f_t     │         └──────┬──────┘
       └───────────┘                │
                                    ▼
       ┌───────────┐         ┌─────────────┐
x_t ──►│Input Gate │────────►│     Add     │──────► C_t (Updated Cell State)
       │   i_t     │         └──────┬──────┘
       └───────────┘                │
                                    ▼
       ┌───────────┐         ┌─────────────┐
x_t ──►│Output Gate│────────►│  tanh & Mul │──────► h_t (Hidden State / Output)
       │   o_t     │         └─────────────┘
       └───────────┘
```

1. **Forget Gate ($f_t$)**: Decides what past trend momentum to discard:
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
2. **Input Gate ($i_t$)**: Selects which new technical/sentiment indicators to store in memory:
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
3. **Cell State Update ($C_t$)**:
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
4. **Output Gate ($o_t$)**: Computes directional probability:
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(C_t)$$

### 8.2 The Secret Weapon: Binary Focal Loss
In standard stock classification, models suffer from **class imbalance** during extended bull runs (mostly UP labels) or bear crashes (mostly DOWN labels). Standard Binary Cross Entropy (BCE) assigns equal weight to easily classifiable samples, drowning out the loss from critical trend inflection points.

AlphaTrade uses **Binary Focal Loss**:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where:
- $p_t$ is the model’s estimated probability for the correct class.
- $\alpha = 0.25$ balances positive vs. negative sample frequency.
- $\gamma = 2.0$ (focusing parameter): As the model becomes confident in an easy sample ($p_t \to 1$), the modulating factor $(1 - p_t)^\gamma \to 0$, reducing its weight. This forces the model to learn difficult market turnarounds and trend exhaustion patterns.

### 8.3 Volatility-Scaled Target Price Derivation
Instead of arbitrary price guesses, the expected price change is strictly tied to the stock's asset volatility:

$$\Delta P_{\text{expected}} = (P_{\text{pred}} - 0.5) \cdot 2.0 \cdot \sigma_{\text{asset}} \cdot \sqrt{\frac{\text{Horizon}}{15}}$$
$$P_{\text{target}} = P_{\text{current}} \cdot (1 + \Delta P_{\text{expected}})$$

---

## 9. Deep Dive: Event-Driven NLP Sentiment & Dense RAG Engine

### 9.1 Credibility & Time-Decay Sentiment Algorithm
Not all news is created equal. A breaking headline from *Reuters* or *Bloomberg* carries far more market impact than an unverified blog post. Furthermore, news sentiment decays exponentially over time.

$$\text{Score}_{\text{composite}} = \frac{\sum_{i=1}^N W_{\text{credibility}}(i) \cdot e^{-\lambda \Delta t_i} \cdot S_{\text{vader}}(i)}{\sum_{i=1}^N W_{\text{credibility}}(i) \cdot e^{-\lambda \Delta t_i}}$$

- **Credibility Weights ($W$)**:
  - Reuters / Bloomberg: `0.95`
  - AlphaVantage / Finnhub: `0.92`
  - Economic Times: `0.90`
  - Moneycontrol / Business Standard: `0.88`
  - General Blogs: `0.50`
- **Decay Factor ($\lambda = 0.1$)**: Halves headline weight every ~7 hours.

### 9.2 Custom Financial Lexicon Rules
Standard NLP packages frequently mistake financial terms (e.g., *"gross margin"*, *"liability"*, *"debt restructuring"*). AlphaTrade supplements VADER with rule-based financial modifiers:
- `"beats estimates"`: $+0.35$
- `"profit surge"` / `"all-time high"`: $+0.30$
- `"guidance cut"` / `"revenue miss"`: $-0.40$
- `"regulatory probe"` / `"fraud"`: $-0.50$
- `"dividend declared"`: $+0.20$

### 9.3 FAISS Dense Vector RAG Pipeline
```mermaid
flowchart TD
    News["Incoming Live News Article"] --> Embed["SentenceTransformers (all-MiniLM-L6-v2)"]
    Embed -->|384-dimensional Vector| Index[("FAISS IndexFlatIP Index")]
    
    Query["User Chat Query\ne.g., 'Why is Tata Motors dropping?'"] --> QEmbed["Vectorize Query"]
    QEmbed --> Search["Cosine Top-K Semantic Similarity Search"]
    Index --> Search
    Search --> RetContext["Retrieved 3 Most Relevant Context Articles"]
    RetContext --> Prompt["Augment Prompt with Stock Indicators & News Context"]
    Prompt --> Response["RAG Financial Response Generator"]
```

---

## 10. Off-Market Hours & Tomorrow's Forecast Engine

One of the biggest flaws in standard algorithmic trading prototypes is that **they crash or return misleading data outside trading hours** (e.g., evenings, nights, and weekends).

AlphaTrade implements a **Timezone-Aware Session Evaluator**:
- **Indian Market (NSE/BSE)**: `09:15` to `15:30` IST (`Asia/Kolkata`), Monday to Friday.
- **US Market (NYSE/NASDAQ)**: `09:30` to `16:00` EST (`America/New_York`), Monday to Friday.

```mermaid
flowchart TD
    Request["User Requests Forecast"] --> Eval{"Check Market Session Status"}
    
    Eval -->|During Trading Hours| OpenBranch["Market OPEN\nReturn Real-Time Intraday Momentum Forecast"]
    Eval -->|Evenings / Nights / Weekends| ClosedBranch["Market CLOSED\nGraceful Session Adaptation Engine"]
    
    ClosedBranch --> CalcTarget["Calculate Next Valid Trading Session\n(Tomorrow 09:15 IST or Next Monday)"]
    ClosedBranch --> DailyModel["Switch from 5m Intraday to Daily Online LSTM Model"]
    ClosedBranch --> Banner["Inject Notice Banner in API Response:\n'Showing workable prediction for Next Session Opening'"]
    
    OpenBranch --> JSON1["200 OK Live Response"]
    Banner --> JSON2["200 OK Adapted Response (Zero Tracebacks)"]
```

---

## 11. Risk Analysis, K-Means Market Regime & Portfolio Advisor

### 11.1 Quantitative Risk Metrics
- **Value at Risk (VaR 95%)**: Computes the 5th percentile historical daily return, representing the maximum expected loss with 95% statistical confidence.
- **Sharpe Ratio**:
  $$\text{Sharpe} = \frac{\mathbb{E}[R] - R_f}{\sigma_{\text{annual}}}$$
- **Maximum Drawdown**: Measures peak-to-trough capital decline over the 6-month historical period.

### 11.2 K-Means Unsupervised Market Regime Detection
Using Scikit-Learn’s `KMeans(n_clusters=5)`, AlphaTrade clusters the 2-dimensional feature space of **$\text{SMA}_{20}$ Ratio** and **10-day Return Volatility**:

| Cluster ID | Market Regime Label | Typical Market Condition | Recommended Action |
| :---: | :--- | :--- | :--- |
| `0` | **Bull Market 📈** | Price consistently above 20-day SMA, controlled volatility | Trend-following long positions |
| `1` | **Bear Market 📉** | Price consistently below 20-day SMA, high downside momentum | Defensive capital preservation / shorting |
| `2` | **Sideways Market ↔️** | Price fluctuating closely around 20-day SMA, low returns | Range-bound mean reversion strategies |
| `3` | **High Volatility ⚡** | Wide candlestick spreads, elevated standard deviation | Tight stop-losses, reduced position sizes |
| `4` | **Low Volatility 🟢** | Tight consolidation, low trading volume | Breakout preparation |

### 11.3 Automated Portfolio Recommendation Engine
Ranks stocks dynamically into **BUY 🟢**, **HOLD 🟡**, or **SELL 🔴**:
- **BUY**: Probability $P \ge 0.65$ **and** Expected Return $> +0.5\%$
- **SELL**: Probability $P \le 0.35$ **or** Expected Return $< -0.5\%$
- **HOLD**: Probability $0.35 < P < 0.65$ or balanced risk profile

---

## 12. Frontend Architecture & Interactive Live Demo Script

The frontend is an institutional dark-mode Next.js 14 dashboard engineered with TailwindCSS and Framer Motion.

### Step-by-Step Live Demo Presentation Script:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ALPHATRADE LIVE DEMO                            │
│                                                                        │
│ 1. [Overview Page]       --> Show live metrics, ticker selector,       │
│                              market status indicator (OPEN/CLOSED).    │
│                                                                        │
│ 2. [Forecast Tab]        --> Select RELIANCE.NS. Demonstrate dynamic   │
│                              online LSTM training, probability, and    │
│                              target price projection.                  │
│                                                                        │
│ 3. [Indicators Tab]      --> Display interactive TradingView candles,  │
│                              RSI momentum, and MACD crossovers.        │
│                                                                        │
│ 4. [Live Pipeline Tab]   --> Show real-time news scraping with source  │
│                              credibility scores and recency decay.     │
│                                                                        │
│ 5. [Risk Analysis Tab]   --> Walk through K-Means Market Regime card,  │
│                              VaR 95%, Sharpe Ratio, and BUY/SELL rank. │
│                                                                        │
│ 6. [AI Chat Tab]         --> Ask: "Why is this stock moving?"          │
│                              Demonstrate RAG retrieval from FAISS.     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Complete API Specification & Endpoints

Base URL: `http://127.0.0.1:8000`

| Method | Route | Parameters | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/stocks` | None | Lists all supported equities (50+ NSE large-caps & US stocks). |
| `GET` | `/api/quote/realtime` | `symbol` (e.g. `RELIANCE.NS`) | Returns zero-delay quote via Upstox API v2 / Finnhub. |
| `GET` | `/api/stock-data` | `ticker`, `period` (`6mo`, `1y`) | Returns historical OHLCV candles + engineered features in JSON. |
| `GET` | `/api/forecast` | `ticker`, `horizon` (`1d`, `15m`, `30m`) | Generates LSTM direction, confidence probability, target price, and off-market adaptation. |
| `POST` | `/api/train-model` | `ticker`, `force_retrain=True` | Triggers dynamic online LSTM training on the latest market data. |
| `GET` | `/api/live-news` | `ticker` | Scrapes breaking news and computes credibility/recency-decay sentiment. |
| `GET` | `/api/indicators` | `ticker`, `period` | Formats RSI, MACD, and moving averages for TradingView charting. |
| `GET` | `/api/risk` | `ticker`, `period`, `pp` (prob) | Returns VaR 95%, Sharpe ratio, K-Means regime, and portfolio ranking. |
| `POST` | `/api/chat` | `{"query": "...", "ticker": "..."}` | Queries the FAISS RAG chatbot for financial insights. |

---

## 14. Viva & Technical Q&A Defense Guide (Top 10 Questions)

Here are the most frequently asked questions by evaluators, professors, and technical interviewers:

#### Q1: "Why use an LSTM instead of a Transformer or Informer?"
> **Answer**: *"While Transformers excel in long natural language contexts, financial time series exhibit a low signal-to-noise ratio. Attention mechanisms across hundreds of noisy financial bars are prone to overfitting without massive parameter scale. An LSTM with 64 units, 20% dropout, and a 5-step lookback effectively captures temporal momentum without over-parameterization, training in under 2 seconds during real-time online updates."*

#### Q2: "How do you prevent Lookahead Bias (Data Leakage)?"
> **Answer**: *"First, all technical indicators (RSI, SMA, Returns) are computed strictly on past and current bars using rolling windows. Second, feature normalization via `MinMaxScaler` is fit strictly within the historical training window and never sees future test data. Third, target binary labels ($y_t = 1$ if $P_{t+1} > P_t$) are shifted forward by 1 step, ensuring that feature vector $X_t$ only predicts future step $t+1$."*

#### Q3: "Why choose Binary Focal Loss over Binary Cross-Entropy?"
> **Answer**: *"In trending financial markets, consecutive up-days cause extreme class imbalance. Standard Binary Cross-Entropy gets dominated by easy-to-classify trend-continuation days, causing the model to miss trend reversals. Focal loss dynamically reduces the weight of easy samples via the $(1 - p_t)^\gamma$ term, forcing the gradient updates to focus on difficult trend inflection points."*

#### Q4: "How does the system handle weekends and off-market hours?"
> **Answer**: *"The backend features a timezone-aware session evaluator (`Asia/Kolkata` for NSE, `America/New_York` for US). If a user queries the model outside trading hours, the system gracefully detects `market_status: CLOSED`, switches to daily momentum mode, identifies the next valid session date (e.g. tomorrow morning or next Monday), and labels the prediction as Tomorrow's Opening Forecast without throwing errors."*

#### Q5: "What happens if there is no news available for a specific ticker?"
> **Answer**: *"The ingestion engine implements a multi-tier fallback. If primary ticker news is sparse, it triggers a sector-proxy search (e.g., searching 'Indian IT sector' if Infosys has no news). If external news is completely offline, it falls back to a neutral sentiment score ($0.0$) while the quantitative indicators continue running uninterrupted."*

#### Q6: "Why combine VADER with rule-based adjustments instead of pure FinBERT?"
> **Answer**: *"FinBERT is a high-accuracy model, but its inference latency is ~150–300ms per article on CPU, creating significant lag when processing dozens of live RSS articles in real-time. VADER paired with domain-specific financial keyword modifiers and credibility/recency decay delivers high-speed inference (<5ms per article) while correcting for financial terminology blindspots."*

#### Q7: "How does the RAG system retrieve and answer questions?"
> **Answer**: *"Incoming news articles are encoded into 384-dimensional dense semantic vectors using `all-MiniLM-L6-v2` and indexed in a FAISS vector database. When a user asks a question, the query is vectorized, and FAISS retrieves the top-3 closest articles via Cosine Inner Product similarity. These articles, along with the latest stock RSI/MACD metrics, are injected into the chatbot's prompt context to generate grounded financial answers."*

#### Q8: "How is the target price calculated from a classification model?"
> **Answer**: *"The LSTM outputs a probability score $p \in [0, 1]$. We center this score around neutral ($p - 0.5$) and scale it by twice the asset's 10-day historical standard deviation (volatility). This ensures target price projections respect the statistical volatility profile of the stock rather than producing unrealistic linear extrapolations."*

#### Q9: "How is the Market Regime determined?"
> **Answer**: *"We run an unsupervised K-Means clustering algorithm ($k=5$) on a feature space consisting of the 20-day Simple Moving Average ratio and rolling 10-day return volatility. This groups historical price behavior into five distinct market dynamics: Bullish, Bearish, Sideways, High Volatility, and Low Volatility."*

#### Q10: "How do you handle model latency for concurrent users?"
> **Answer**: *"The backend implements an in-memory caching layer with a 5-minute Time-To-Live (TTL). Once an LSTM model or news sentiment payload is computed for a ticker like `RELIANCE.NS`, subsequent requests within 300 seconds are served instantly from memory in $<10\text{ms}$."*

---

## 15. Financial Loopholes, Mitigations & Production Roadmap

| Identified Limitation | Real-World Financial Impact | Implemented Mitigation / Roadmap Solution |
| :--- | :--- | :--- |
| **Transaction Fees & Slippage** | Theoretical profits get eroded by brokerage commissions and exchange fees. | Implemented net return thresholding requiring $|R_{\text{expected}}| > \text{Friction (0.1\%)}$ before triggering BUY signals. |
| **Order Book Microstructure Blindness** | Daily candles miss high-frequency bid-ask spread liquidity traps. | Roadmap: Integration with Level-2/3 WebSocket order book tick data feeds. |
| **Out-of-Distribution Market Crashes** | Black swan events (e.g., pandemic shocks) violate historical distributions. | Integrated VaR (95%) and automated regime classification to warn users of high volatility. |
| **Distributed Scaling** | In-memory cache is bound to a single server process. | Roadmap: Distributed Redis cache and Celery worker queues for parallel background model updates. |

---

## 16. Quick-Start Execution Commands

### Step 1: Launch FastAPI Backend Server
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```
- **Backend API URL**: `http://127.0.0.1:8000`
- **Interactive OpenAPI Documentation**: `http://127.0.0.1:8000/docs`

### Step 2: Launch Next.js Frontend Web Dashboard
```bash
cd frontend
npm install
npm run dev
```
- **Frontend Dashboard URL**: `http://localhost:3000`

---

*AlphaTrade (v2.0) — Developed with modern Full-Stack & Applied Deep Learning engineering principles.*
