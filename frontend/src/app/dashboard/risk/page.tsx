"use client";

import { ShieldAlert, AlertTriangle, Scale, Activity, Briefcase } from "lucide-react";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { useAppStore } from "@/store/appStore";
import { useEffect, useState } from "react";

export default function RiskPage() {
  const { ticker, period } = useAppStore();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function fetchRisk() {
      setLoading(true);
      try {
        const res = await fetch(`http://localhost:8000/api/risk?ticker=${ticker}&period=${period}`);
        if (!res.ok) throw new Error("Failed to fetch risk data");
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
    fetchRisk();
    return () => { mounted = false; };
  }, [ticker, period]);

  if (loading) {
    return <div className="flex items-center justify-center h-96"><div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin"></div></div>;
  }
  
  if (error || !data) {
    return <div className="flex items-center justify-center h-96 text-rose-400">Error: {error}</div>;
  }

  const { risk_metrics, regime, portfolio } = data;

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
          <ShieldAlert className="w-6 h-6 text-primary" /> Risk Analysis & Market Regime
        </h1>
        <p className="text-secondary text-sm">Portfolio risk profiling & market regime detection</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h2 className="text-lg font-semibold flex items-center gap-2 text-primary">
            <Activity className="w-5 h-5" /> Risk Metrics
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <MetricCard 
              title="Volatility" 
              value={risk_metrics?.volatility_score?.toFixed(4) || '---'}
              icon={<Activity className="w-4 h-4" />}
            />
            <MetricCard 
              title="VaR (95%)" 
              value={risk_metrics?.var_95?.toFixed(4) || '---'}
              icon={<ShieldAlert className="w-4 h-4" />}
            />
            <MetricCard 
              title="Sharpe Ratio" 
              value={risk_metrics?.sharpe_ratio?.toFixed(3) || '---'}
              icon={<Scale className="w-4 h-4" />}
            />
            <MetricCard 
              title="Risk Level" 
              value={risk_metrics?.risk_level || '---'}
              icon={<AlertTriangle className="w-4 h-4" />}
            />
          </div>
        </div>

        <div className="space-y-4">
          <h2 className="text-lg font-semibold flex items-center gap-2 text-primary">
            <AlertTriangle className="w-5 h-5" /> Market Regime
          </h2>
          <div className="flex items-center gap-6 bg-surface border border-border rounded-xl p-6 h-full">
            <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
              <ShieldAlert className="w-8 h-8 text-primary" />
            </div>
            <div>
              <div className="text-sm font-mono text-secondary mb-1">Current Classified Regime</div>
              <div className="text-2xl font-bold text-foreground">{regime}</div>
              <p className="text-secondary text-sm mt-2 max-w-xl">
                Volatility & SMA Ratio Clustering
              </p>
            </div>
          </div>
        </div>
      </div>

      <hr className="border-border my-6" />

      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Briefcase className="w-5 h-5 text-primary" /> Portfolio Recommendation
        </h2>
        {portfolio && portfolio.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-secondary uppercase bg-surface-raised border-b border-border">
                <tr>
                  <th className="px-4 py-3">Ticker</th>
                  <th className="px-4 py-3 text-right">Pred Prob</th>
                  <th className="px-4 py-3 text-right">Volatility</th>
                  <th className="px-4 py-3 text-right">Signal</th>
                  <th className="px-4 py-3 text-right">Weight Allocation</th>
                </tr>
              </thead>
              <tbody>
                {portfolio.map((p: any, i: number) => (
                  <tr key={i} className="border-b border-border hover:bg-surface-raised/50">
                    <td className="px-4 py-3 font-semibold">{p.Ticker}</td>
                    <td className="px-4 py-3 text-right font-mono">{p['Pred Prob']?.toFixed(4)}</td>
                    <td className="px-4 py-3 text-right font-mono">{p.Volatility?.toFixed(4)}</td>
                    <td className={`px-4 py-3 text-right font-semibold ${p.Signal === "Buy" ? "text-emerald-400" : "text-rose-400"}`}>{p.Signal}</td>
                    <td className="px-4 py-3 text-right font-mono text-primary font-bold">{p['Weight Allocation']}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-4 bg-surface-raised border border-border text-secondary text-sm rounded-lg">
            Add more stock forecasts for portfolio ranking.
          </div>
        )}
      </div>
    </div>
  );
}
