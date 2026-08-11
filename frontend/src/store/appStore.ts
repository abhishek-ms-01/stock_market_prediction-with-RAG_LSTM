import { create } from 'zustand';

interface AppState {
  ticker: string;
  stockName: string;
  period: string;
  periodLabel: string;
  stocks: Record<string, string>;
  setTicker: (ticker: string, stockName: string) => void;
  setPeriod: (period: string, periodLabel: string) => void;
  setStocks: (stocks: Record<string, string>) => void;
}

export const useAppStore = create<AppState>((set) => ({
  ticker: 'RELIANCE.NS',
  stockName: 'Reliance Industries',
  period: '6mo',
  periodLabel: '6 Months',
  stocks: {},
  setTicker: (ticker, stockName) => set({ ticker, stockName }),
  setPeriod: (period, periodLabel) => set({ period, periodLabel }),
  setStocks: (stocks) => set({ stocks }),
}));
