"use client";

import { useState } from "react";
import { ArrowUpRight, BarChart3, Activity, Clock, SlidersHorizontal } from "lucide-react";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { ChartPanel } from "@/components/charts/ChartPanel";
import { useStockData } from "@/hooks/useStockData";
import { motion } from "framer-motion";

export default function DashboardOverview() {
  const [showSMA, setShowSMA] = useState(true);
  const [showVolume, setShowVolume] = useState(true);
  
  const { data, loading, error } = useStockData();

  if (error) {
    return (
      <div className="flex items-center justify-center h-96 text-rose-400">
        Error loading data: {error}
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  const latest = data[data.length - 1];
  const previous = data[data.length - 2];
  
  const priceChange = latest.Close - previous.Close;
  const pctChange = (priceChange / previous.Close) * 100;
  
  const closeHistory = data.slice(-20).map(d => d.Close);
  const rsiHistory = data.slice(-20).map(d => d.RSI || 50);
  const macdHistory = data.slice(-20).map(d => d.MACD || 0);

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Overview</h1>
          <p className="text-secondary text-sm">Real-time market data & analysis</p>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard 
          title="Last Price" 
          value={`₹${latest.Close.toFixed(2)}`}
          icon={<ArrowUpRight className="w-5 h-5" />}
          trend={priceChange >= 0 ? "up" : "down"}
          trendValue={`${Math.abs(pctChange).toFixed(2)}%`}
          sparklineData={closeHistory}
        />
        <MetricCard 
          title="20-Day SMA" 
          value={`₹${latest.MA_20?.toFixed(2) || '---'}`}
          icon={<Activity className="w-5 h-5" />}
          sparklineData={data.slice(-20).map(d => d.MA_20 || latest.Close)}
        />
        <MetricCard 
          title="RSI (14)" 
          value={latest.RSI?.toFixed(2) || '---'}
          icon={<Clock className="w-5 h-5" />}
          trend={latest.RSI && latest.RSI > 70 ? "down" : latest.RSI && latest.RSI < 30 ? "up" : "neutral"}
          trendValue={latest.RSI && latest.RSI > 70 ? "Overbought" : latest.RSI && latest.RSI < 30 ? "Oversold" : "Neutral"}
          sparklineData={rsiHistory}
        />
        <MetricCard 
          title="MACD" 
          value={latest.MACD?.toFixed(2) || '---'}
          icon={<BarChart3 className="w-5 h-5" />}
          trend={latest.MACD && latest.MACD > 0 ? "up" : "down"}
          trendValue={latest.MACD && latest.MACD > 0 ? "Bullish" : "Bearish"}
          sparklineData={macdHistory}
        />
      </div>

      {/* Chart Section */}
      <div className="glass-card p-6 flex flex-col gap-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-primary" />
              Price Action
            </h2>
          </div>
          
          <div className="flex items-center gap-2">
            <button 
              onClick={() => setShowSMA(!showSMA)}
              className={`px-3 py-1.5 text-xs font-medium rounded-full border transition-colors flex items-center gap-2 ${showSMA ? 'bg-primary/10 border-primary text-primary' : 'bg-surface border-border text-secondary'}`}
            >
              <SlidersHorizontal className="w-3 h-3" />
              SMA 20
            </button>
            <button 
              onClick={() => setShowVolume(!showVolume)}
              className={`px-3 py-1.5 text-xs font-medium rounded-full border transition-colors flex items-center gap-2 ${showVolume ? 'bg-primary/10 border-primary text-primary' : 'bg-surface border-border text-secondary'}`}
            >
              <BarChart3 className="w-3 h-3" />
              Volume
            </button>
          </div>
        </div>
        
        <ChartPanel data={data} showSMA={showSMA} showVolume={showVolume} />
      </div>

    </div>
  );
}
