export interface StockDataPoint {
  Date: string;
  Close: number;
  High: number;
  Low: number;
  Open: number;
  Volume: number;
  RSI: number;
  MACD: number;
  MA_20: number;
}

export interface StockDataResponse {
  symbol: string;
  data: StockDataPoint[];
}

export interface ForecastResponse {
  symbol: string;
  prediction_prob: number;
  signal: 'UP' | 'DOWN';
}

export interface RiskMetrics {
  volatility_score: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  max_drawdown?: number;
  sharpe_ratio?: number;
  var_95?: number;
}

export interface RiskResponse {
  symbol: string;
  risk_metrics: RiskMetrics;
  market_regime: string;
}

export interface ChatRequest {
  query: string;
  context?: Record<string, any>;
}

export interface ChatResponse {
  reply: string;
}

export interface PortfolioRecommendation {
  ticker: string;
  prob: number;
  rank: number;
  allocation_pct: number;
}

export interface PortfolioResponse {
  recommendations: PortfolioRecommendation[];
}
