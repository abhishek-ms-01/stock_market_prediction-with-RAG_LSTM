"use client";

import { useAppStore } from "@/store/appStore";
import { useState } from "react";
import { ChevronDown, Search } from "lucide-react";

const periodMap: Record<string, string> = {
  "6 Months": "6mo",
  "1 Year": "1y",
  "2 Years": "2y"
};

export function Header() {
  const { ticker, stockName, periodLabel, stocks, setTicker, setPeriod } = useAppStore();
  
  const [customSymbol, setCustomSymbol] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const filteredStocks = Object.keys(stocks).filter(s => s.toLowerCase().includes(searchQuery.toLowerCase()));

  const handleCustomSymbolChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setCustomSymbol(val);
    if (val.trim()) {
      const cleanSym = val.trim().toUpperCase();
      const newTicker = cleanSym.endsWith(".NS") ? cleanSym : `${cleanSym}.NS`;
      setTicker(newTicker, cleanSym.replace(".NS", ""));
    } else {
      const firstKey = Object.keys(stocks)[0];
      if (firstKey) {
        setTicker(stocks[firstKey], firstKey);
      }
    }
  };

  const handlePeriodSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const pLabel = e.target.value;
    const pVal = periodMap[pLabel];
    if (pVal) {
      setPeriod(pVal, pLabel);
    }
  };

  return (
    <header className="px-8 py-6 bg-surface/50 backdrop-blur-md border-b border-border sticky top-0 z-20 flex justify-between items-start gap-4 flex-wrap">
      {/* Title Section */}
      <div className="flex flex-col gap-2 shrink-0">
        <h1 className="text-3xl font-bold text-foreground">
          {stockName || ticker} <span className="text-xl text-secondary font-medium">({ticker})</span>
        </h1>
        <p className="text-sm text-secondary bg-gradient-to-r from-primary to-amber-400 bg-clip-text text-transparent font-semibold">
          AI Stock Market Prediction System · Time-Aware Hybrid RAG-LSTM Architecture
        </p>
        
        <div className="flex flex-wrap items-center gap-2 mt-2">
          {/* Emerald Pill */}
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            ⚡ Model Online
          </span>
          
          {/* Amber Pill */}
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
            🕒 {periodLabel}
          </span>
        </div>
      </div>

      {/* Control Panel Section */}
      <div className="flex flex-wrap items-end gap-4 shrink-0 bg-surface-raised/80 backdrop-blur-sm border border-border p-3 rounded-xl shadow-lg mt-2 lg:mt-0">
        {/* Select Stock */}
        <div className="relative w-48">
          <label className="text-secondary font-medium text-[10px] uppercase tracking-wide mb-1 block">📌 Select Stock</label>
          <button 
            onClick={() => setSearchOpen(!searchOpen)}
            className="w-full flex items-center justify-between bg-surface border border-border rounded-md px-3 py-2 text-sm text-foreground focus:border-primary outline-none hover:border-primary/50 transition-colors"
          >
            <span className="truncate font-medium">{customSymbol ? "Custom" : (stockName || "Select...")}</span>
            <ChevronDown className="w-4 h-4 text-primary shrink-0" />
          </button>
          
          {searchOpen && (
            <div className="absolute right-0 z-50 mt-1 w-56 bg-surface border border-border rounded-md shadow-xl overflow-hidden">
              <div className="p-2 border-b border-border flex items-center gap-2">
                <Search className="w-4 h-4 text-secondary" />
                <input 
                  type="text" 
                  autoFocus
                  className="bg-transparent border-none outline-none text-sm text-foreground w-full placeholder-secondary"
                  placeholder="Search..."
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                />
              </div>
              <div className="max-h-48 overflow-y-auto custom-scrollbar">
                {filteredStocks.length > 0 ? filteredStocks.map(s => (
                  <button 
                    key={s}
                    className="w-full text-left px-3 py-2 text-sm text-foreground hover:bg-primary/20 hover:text-primary transition-colors"
                    onClick={() => {
                      setTicker(stocks[s], s);
                      setCustomSymbol("");
                      setSearchOpen(false);
                      setSearchQuery("");
                    }}
                  >
                    {s}
                  </button>
                )) : (
                  <div className="px-3 py-2 text-sm text-secondary">No results found</div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Custom Symbol */}
        <div className="w-32">
          <label className="text-secondary font-medium text-[10px] uppercase tracking-wide mb-1 block">✏️ Custom</label>
          <input 
            type="text"
            placeholder="e.g. SBIN"
            className="w-full bg-surface border border-border rounded-md px-3 py-2 text-sm text-foreground focus:border-primary outline-none hover:border-primary/50 transition-colors placeholder-secondary font-medium"
            value={customSymbol}
            onChange={handleCustomSymbolChange}
          />
        </div>

        {/* Timeframe */}
        <div className="w-32">
          <label className="text-secondary font-medium text-[10px] uppercase tracking-wide mb-1 block">📅 Timeframe</label>
          <div className="relative">
            <select 
              className="w-full bg-surface border border-border rounded-md px-3 py-2 text-sm text-foreground focus:border-primary outline-none appearance-none hover:border-primary/50 transition-colors cursor-pointer font-medium"
              value={periodLabel}
              onChange={handlePeriodSelect}
            >
              {Object.keys(periodMap).map(p => (
                <option key={p} value={p} className="bg-surface text-foreground">{p}</option>
              ))}
            </select>
            <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none">
              <ChevronDown className="w-4 h-4 text-primary" />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
