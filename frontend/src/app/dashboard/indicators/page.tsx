"use client";

import { Activity, LayoutDashboard, SlidersHorizontal, TrendingUp } from "lucide-react";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { useAppStore } from "@/store/appStore";
import { useEffect, useState } from "react";
import dynamic from "next/dynamic";

const DynamicResponsiveContainer = dynamic(() => import("recharts").then(mod => mod.ResponsiveContainer), { ssr: false });
const DynamicLineChart = dynamic(() => import("recharts").then(mod => mod.LineChart), { ssr: false });
const DynamicLine = dynamic(() => import("recharts").then(mod => mod.Line), { ssr: false });
const DynamicXAxis = dynamic(() => import("recharts").then(mod => mod.XAxis), { ssr: false });
const DynamicYAxis = dynamic(() => import("recharts").then(mod => mod.YAxis), { ssr: false });
const DynamicCartesianGrid = dynamic(() => import("recharts").then(mod => mod.CartesianGrid), { ssr: false });
const DynamicTooltip = dynamic(() => import("recharts").then(mod => mod.Tooltip), { ssr: false });

export default function IndicatorsPage() {
  const { ticker, period } = useAppStore();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function fetchIndicators() {
      setLoading(true);
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/indicators?ticker=${ticker}&period=${period}`);
        if (!res.ok) throw new Error("Failed to fetch indicators");
        const json = await res.json();
        if (mounted) {
          setData(json);
          setError(null);
        }
      } catch (err: any) {
        if (mounted) setError(err.message);
      } finally {
        if (mounted) setLoading(false);
      }
    }
    fetchIndicators();
    return () => { mounted = false; };
  }, [ticker, period]);

  if (loading) {
    return <div className="flex items-center justify-center h-96"><div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin"></div></div>;
  }
  
  if (error || !data) {
    return <div className="flex items-center justify-center h-96 text-rose-400">Error: {error}</div>;
  }

  const { indicators, news, chart_data } = data;

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
          <Activity className="w-6 h-6 text-primary" /> Technical Indicators
        </h1>
        <p className="text-secondary text-sm">Momentum, Volatility & Sentiment</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* RSI Chart */}
        <div className="glass-card p-6">
          <h3 className="text-lg font-semibold mb-4 text-primary">RSI (14)</h3>
          <div className="h-64">
            <DynamicResponsiveContainer width="100%" height="100%">
              <DynamicLineChart data={chart_data}>
                <DynamicCartesianGrid strokeDasharray="3 3" stroke="#202D45" vertical={false} />
                <DynamicXAxis dataKey="Date" stroke="#8B9DC3" tick={{fill: '#8B9DC3'}} />
                <DynamicYAxis domain={[0, 100]} stroke="#8B9DC3" tick={{fill: '#8B9DC3'}} />
                <DynamicTooltip contentStyle={{ backgroundColor: 'rgba(10, 14, 23, 0.9)', borderColor: '#202D45' }} />
                <DynamicLine type="step" dataKey={() => 70} stroke="#FF1744" strokeWidth={1} dot={false} strokeDasharray="5 5" />
                <DynamicLine type="step" dataKey={() => 30} stroke="#00E676" strokeWidth={1} dot={false} strokeDasharray="5 5" />
                <DynamicLine type="monotone" dataKey="RSI" stroke="#00F2FE" strokeWidth={2} dot={false} />
              </DynamicLineChart>
            </DynamicResponsiveContainer>
          </div>
        </div>

        {/* MACD Chart */}
        <div className="glass-card p-6">
          <h3 className="text-lg font-semibold mb-4 text-primary">MACD</h3>
          <div className="h-64">
            <DynamicResponsiveContainer width="100%" height="100%">
              <DynamicLineChart data={chart_data}>
                <DynamicCartesianGrid strokeDasharray="3 3" stroke="#202D45" vertical={false} />
                <DynamicXAxis dataKey="Date" stroke="#8B9DC3" tick={{fill: '#8B9DC3'}} />
                <DynamicYAxis stroke="#8B9DC3" tick={{fill: '#8B9DC3'}} />
                <DynamicTooltip contentStyle={{ backgroundColor: 'rgba(10, 14, 23, 0.9)', borderColor: '#202D45' }} />
                <DynamicLine type="step" dataKey={() => 0} stroke="rgba(255,255,255,0.1)" strokeWidth={1} dot={false} strokeDasharray="5 5" />
                <DynamicLine type="monotone" dataKey="MACD" stroke="#22d3ee" strokeWidth={2} dot={false} />
              </DynamicLineChart>
            </DynamicResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Indicators Table */}
        <div className="glass-card p-6">
          <h3 className="text-lg font-semibold mb-4">📋 Indicator Data</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-secondary uppercase bg-surface-raised border-b border-border">
                <tr>
                  <th className="px-4 py-3">Date</th>
                  <th className="px-4 py-3 text-right">Close</th>
                  <th className="px-4 py-3 text-right">MA 20</th>
                  <th className="px-4 py-3 text-right">RSI</th>
                  <th className="px-4 py-3 text-right">MACD</th>
                </tr>
              </thead>
              <tbody>
                {indicators.map((row: any, i: number) => (
                  <tr key={i} className="border-b border-border hover:bg-surface-raised/50">
                    <td className="px-4 py-3 font-mono">{row.Date}</td>
                    <td className="px-4 py-3 text-right font-mono">₹{row.Close?.toFixed(2)}</td>
                    <td className="px-4 py-3 text-right font-mono text-amber-400">₹{row.MA_20?.toFixed(2)}</td>
                    <td className="px-4 py-3 text-right font-mono text-primary">{row.RSI?.toFixed(2)}</td>
                    <td className={`px-4 py-3 text-right font-mono ${row.MACD > 0 ? "text-emerald-400" : "text-rose-400"}`}>{row.MACD?.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* News Table */}
        <div className="glass-card p-6">
          <h3 className="text-lg font-semibold mb-4">📰 News Sentiment</h3>
          {news && news.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-secondary uppercase bg-surface-raised border-b border-border">
                  <tr>
                    <th className="px-4 py-3">Event</th>
                    <th className="px-4 py-3">Sentiment</th>
                    <th className="px-4 py-3">Title</th>
                  </tr>
                </thead>
                <tbody>
                  {news.map((n: any, i: number) => (
                    <tr key={i} className="border-b border-border hover:bg-surface-raised/50">
                      <td className="px-4 py-3 font-medium whitespace-nowrap text-amber-400">{n.event}</td>
                      <td className={`px-4 py-3 font-mono ${n.sentiment > 0 ? "text-emerald-400" : n.sentiment < 0 ? "text-rose-400" : "text-secondary"}`}>
                        {n.sentiment?.toFixed(4)}
                      </td>
                      <td className="px-4 py-3 text-secondary truncate max-w-[200px]" title={n.title}>{n.title}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="p-4 bg-surface-raised border border-border text-secondary text-sm rounded-lg">
              No news data available. Run `python process_news.py` to populate news.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
