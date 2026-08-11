import { useState, useEffect } from "react";
import { useAppStore } from "@/store/appStore";

export interface StockDataPoint {
  Date: string;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
  MA_20?: number;
  RSI?: number;
  MACD?: number;
  Return?: number;
  Volatility?: number;
}

export function useStockData() {
  const { ticker, period } = useAppStore();
  const [data, setData] = useState<StockDataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    
    async function fetchData() {
      setLoading(true);
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/stock-data?ticker=${ticker}&period=${period}`);
        if (!res.ok) throw new Error("Backend not reachable or data unavailable");
        
        const json = await res.json();
        if (mounted) {
          setData(json);
          setError(null);
        }
      } catch (err: any) {
        if (mounted) {
          setError(err.message);
        }
      } finally {
        if (mounted) setLoading(false);
      }
    }
    
    fetchData();
    
    return () => { mounted = false; };
  }, [ticker, period]);

  return { data, loading, error };
}
