"use client";

import { useState } from "react";
import { useAppStore } from "@/store/appStore";
import { BrainCircuit, LineChart, Target, Zap, TrendingUp, TrendingDown, Info } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const horizonOptions = [
  { label: "⚡ +5 Min", value: "5m" },
  { label: "⏱️ +15 Min", value: "15m" },
  { label: "⏱️ +30 Min", value: "30m" },
  { label: "⏱️ +1 Hour", value: "60m" },
  { label: "📅 Tomorrow", value: "1d" }
];

export default function ForecastPage() {
  const { ticker } = useAppStore();
  const [horizon, setHorizon] = useState("5m");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const runForecast = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/forecast?ticker=${ticker}&horizon=${horizon}`);
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Forecast failed");
      }
      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
          <BrainCircuit className="w-6 h-6 text-primary" /> Price Direction Forecast
        </h1>
        <p className="text-secondary text-sm mt-1">Time-Aware Hybrid RAG-LSTM prediction engine</p>
      </div>

      {/* Horizon Selector */}
      <div className="flex flex-wrap gap-2 bg-input p-2 rounded-xl border border-border w-fit">
        {horizonOptions.map(opt => (
          <button
            key={opt.value}
            onClick={() => { setHorizon(opt.value); setResult(null); setError(null); }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              horizon === opt.value 
                ? "bg-primary/20 text-primary border border-primary/30 shadow-[0_0_12px_rgba(0,242,254,0.15)]" 
                : "text-secondary hover:text-foreground hover:bg-surface"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Action Button */}
      <button 
        onClick={runForecast}
        disabled={loading}
        className="group relative inline-flex h-12 items-center justify-center overflow-hidden rounded-lg bg-primary px-8 font-medium text-background transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:hover:scale-100 shadow-[0_0_20px_rgba(0,242,254,0.3)]"
      >
        <span className="relative flex items-center gap-2 text-base font-bold">
          {loading ? (
            <div className="w-5 h-5 border-2 border-background border-t-transparent rounded-full animate-spin"></div>
          ) : (
            <Zap className="w-5 h-5" />
          )}
          {loading ? "Neural network inference..." : `Run ${horizon === '1d' ? 'Daily' : horizon.toUpperCase()} Forecast`}
        </span>
      </button>

      {error && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 text-rose-400 rounded-lg">
          {error}
        </div>
      )}

      {/* Result Panel */}
      <AnimatePresence>
        {result && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Direction Card */}
              <div className={`p-6 rounded-2xl border ${
                result.direction === "UP" 
                  ? "bg-gradient-to-br from-emerald-500/10 to-emerald-900/20 border-emerald-500/30 shadow-[0_0_24px_rgba(16,185,129,0.12)]" 
                  : "bg-gradient-to-br from-rose-500/10 to-rose-900/20 border-rose-500/30 shadow-[0_0_24px_rgba(244,63,94,0.12)]"
              }`}>
                <div className="text-center">
                  <div className={`text-2xl font-black mb-2 flex items-center justify-center gap-2 ${
                    result.direction === "UP" ? "text-emerald-400" : "text-rose-400"
                  }`}>
                    {result.direction === "UP" ? <TrendingUp className="w-6 h-6" /> : <TrendingDown className="w-6 h-6" />}
                    PREDICTED: {result.direction}
                  </div>
                  
                  {horizon !== "1d" ? (
                    <>
                      <div className="text-secondary text-sm">Target: <b className="text-foreground">{result.target_time}</b> (+{result.horizon_mins}m)</div>
                      <div className={`text-xl font-bold mt-2 font-mono ${result.direction === "UP" ? "text-emerald-400" : "text-rose-400"}`}>
                        ₹{result.target_price?.toFixed(2)} ({(result.move_pct * 100).toFixed(2)}%)
                      </div>
                    </>
                  ) : (
                    <div className="text-secondary text-sm">
                      {result.direction === "UP" ? "Bullish momentum expected" : "Bearish movement expected"} for tomorrow's session
                    </div>
                  )}
                </div>
              </div>

              {/* Confidence Card */}
              <div className="glass-card p-6 flex flex-col justify-center">
                <h3 className="text-primary font-semibold mb-4 flex items-center gap-2">
                  <Target className="w-4 h-4" /> Confidence
                </h3>
                <div className="flex justify-between items-end mb-2">
                  <span className="text-sm text-secondary">Probability</span>
                  <div className="text-right">
                    <span className="text-xl font-bold font-mono">{result.score.toFixed(4)}</span>
                    <span className="text-xs ml-2 text-primary font-bold">{(result.score - 0.5) > 0 ? '+' : ''}{((result.score - 0.5) * 100).toFixed(2)}%</span>
                  </div>
                </div>
                <div className="h-2 w-full bg-surface-raised rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-primary to-amber-400 transition-all duration-1000"
                    style={{ width: `${Math.max(0, Math.min(100, result.score * 100))}%` }}
                  />
                </div>
                <div className="text-xs text-secondary mt-3">
                  {horizon !== "1d" 
                    ? `₹${result.current_price?.toFixed(2)} → ₹${result.target_price?.toFixed(2)}`
                    : "Threshold: 0.50 · Binary Crossentropy"}
                </div>
              </div>
            </div>

            {/* Feature Matrix */}
            <div className="glass-card p-6">
              <h3 className="text-lg font-semibold mb-4">📋 Feature Matrix</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                  <thead className="text-xs text-secondary uppercase bg-surface-raised border-b border-border">
                    <tr>
                      <th className="px-4 py-3">Step</th>
                      {result.features && Object.keys(Object.values(result.features)[0] as any).map(col => (
                        <th key={col} className="px-4 py-3">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.features && Object.entries(result.features).map(([step, values]: [string, any]) => (
                      <tr key={step} className="border-b border-border hover:bg-surface-raised/50 transition-colors">
                        <td className="px-4 py-3 font-semibold text-primary whitespace-nowrap">{step}</td>
                        {Object.values(values).map((val: any, i) => (
                          <td key={i} className="px-4 py-3 font-mono">{typeof val === 'number' ? val.toFixed(4) : val}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <hr className="border-border my-8" />
      
      {/* Model Stats */}
      <div>
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <LineChart className="w-5 h-5 text-primary" /> Model Performance
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-surface p-4 rounded-xl border border-border">
            <div className="text-xs text-secondary uppercase tracking-wider mb-1">Test Accuracy</div>
            <div className="text-2xl font-bold font-mono">85.71%</div>
            <div className="text-xs text-muted mt-1">20% Held-Out Set</div>
          </div>
          <div className="bg-surface p-4 rounded-xl border border-border">
            <div className="text-xs text-secondary uppercase tracking-wider mb-1">Training Epochs</div>
            <div className="text-2xl font-bold font-mono">40</div>
            <div className="text-xs text-muted mt-1">Adam lr=0.005</div>
          </div>
          <div className="bg-surface p-4 rounded-xl border border-border">
            <div className="text-xs text-secondary uppercase tracking-wider mb-1">Features</div>
            <div className="text-2xl font-bold font-mono">16</div>
            <div className="text-xs text-muted mt-1">RAG + Technical Hybrid</div>
          </div>
        </div>
      </div>

      {/* Verification */}
      <details className="group glass-card p-4 rounded-xl [&_summary::-webkit-details-marker]:hidden">
        <summary className="flex items-center cursor-pointer list-none gap-2 font-semibold text-secondary group-open:text-foreground transition-colors">
          <Info className="w-4 h-4" /> ❓ How to Verify Predictions
        </summary>
        <div className="mt-4 pt-4 border-t border-border text-sm text-secondary space-y-2">
          <p><strong className="text-foreground">UP (Score &gt; 0.50)</strong> — ✅ Correct if actual close &gt; start price</p>
          <p><strong className="text-foreground">DOWN (Score ≤ 0.50)</strong> — ✅ Correct if actual close &lt; start price</p>
          <div className="bg-surface-raised p-3 rounded-lg font-mono text-xs text-primary mt-2">
            ./venv/bin/python prediction/train_lstm.py
          </div>
        </div>
      </details>
    </div>
  );
}
