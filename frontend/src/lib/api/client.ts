import {
  StockDataResponse,
  ForecastResponse,
  RiskResponse,
  ChatRequest,
  ChatResponse,
  PortfolioResponse,
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

/**
 * Generic fetch wrapper with error handling
 */
async function fetchAPI<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API Error: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`Error fetching ${url}:`, error);
    throw error;
  }
}

// --- API Service Methods ---

export async function fetchStockData(symbol: string): Promise<StockDataResponse> {
  return fetchAPI<StockDataResponse>(`/stock/${symbol}`);
}

export async function fetchForecast(symbol: string): Promise<ForecastResponse> {
  return fetchAPI<ForecastResponse>('/forecast', {
    method: 'POST',
    body: JSON.stringify({ symbol }),
  });
}

export async function fetchRiskMetrics(symbol: string): Promise<RiskResponse> {
  return fetchAPI<RiskResponse>(`/risk/${symbol}`);
}

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  return fetchAPI<ChatResponse>('/chat', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function fetchPortfolioRecommendations(): Promise<PortfolioResponse> {
  return fetchAPI<PortfolioResponse>('/portfolio');
}
