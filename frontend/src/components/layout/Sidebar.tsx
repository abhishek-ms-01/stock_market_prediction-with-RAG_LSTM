"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, 
  LineChart, 
  Activity, 
  ShieldAlert, 
  MessageSquare,
  ChevronLeft,
  ChevronDown,
  Search
} from "lucide-react";
import clsx from "clsx";
import { useState, useEffect } from "react";
import { useAppStore } from "@/store/appStore";

const periodMap: Record<string, string> = {
  "6 Months": "6mo",
  "1 Year": "1y",
  "2 Years": "2y"
};

export function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  
  const { ticker, stockName, periodLabel, stocks, setTicker, setPeriod, setStocks } = useAppStore();
  const [customSymbol, setCustomSymbol] = useState("");

  useEffect(() => {
    fetch("http://localhost:8000/api/stocks")
      .then(res => res.json())
      .then(data => {
        setStocks(data);
      })
      .catch(console.error);
  }, [setStocks]);

  const navItems = [
    { name: "Overview", href: "/dashboard", icon: LayoutDashboard },
    { name: "Forecast", href: "/dashboard/forecast", icon: LineChart },
    { name: "Indicators", href: "/dashboard/indicators", icon: Activity },
    { name: "Risk", href: "/dashboard/risk", icon: ShieldAlert },
    { name: "AI Chat", href: "/dashboard/ai-chat", icon: MessageSquare },
  ];

  const handleCustomSymbolChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setCustomSymbol(val);
    if (val.trim()) {
      const cleanSym = val.trim().toUpperCase();
      const newTicker = cleanSym.endsWith(".NS") ? cleanSym : `${cleanSym}.NS`;
      setTicker(newTicker, cleanSym.replace(".NS", ""));
    } else {
      // Revert to dropdown selection if custom is cleared
      const firstKey = Object.keys(stocks)[0];
      if (firstKey) {
        setTicker(stocks[firstKey], firstKey);
      }
    }
  };

  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const filteredStocks = Object.keys(stocks).filter(s => s.toLowerCase().includes(searchQuery.toLowerCase()));

  const handlePeriodSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const pLabel = e.target.value;
    const pVal = periodMap[pLabel];
    if (pVal) {
      setPeriod(pVal, pLabel);
    }
  };

  return (
    <div className={clsx(
      "flex flex-col h-full bg-surface border-r border-border transition-all duration-300 relative",
      collapsed ? "w-20" : "w-72"
    )}>
      {/* Logo */}
      <Link href="/" className="h-16 flex items-center px-6 border-b border-border mb-4 shrink-0 hover:bg-surface-raised/50 transition-colors group text-decoration-none">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-blue-600 flex items-center justify-center shrink-0 shadow-[0_0_12px_rgba(0,242,254,0.4)] group-hover:scale-105 transition-transform">
          <Activity className="w-5 h-5 text-white" />
        </div>
        {!collapsed && (
          <span className="ml-3 font-bold text-xl whitespace-nowrap text-foreground tracking-tight flex items-center">
            Alpha<span className="text-primary">Trade</span>
          </span>
        )}
      </Link>

      <div className="flex-1 overflow-y-auto px-4 space-y-6 pb-6 custom-scrollbar">
        
        {/* Navigation */}
        <div className="space-y-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link 
                key={item.name} 
                href={item.href}
                className={clsx(
                  "flex items-center px-3 py-2.5 rounded-lg transition-colors group relative",
                  isActive 
                    ? "bg-primary/10 text-primary" 
                    : "text-secondary hover:bg-surface-raised hover:text-foreground"
                )}
              >
                <item.icon className={clsx("w-5 h-5 shrink-0", isActive ? "text-primary" : "text-secondary group-hover:text-foreground")} />
                {!collapsed && <span className="ml-3 font-medium whitespace-nowrap">{item.name}</span>}
                
                {isActive && (
                  <div className="absolute left-0 w-1 h-6 bg-primary rounded-r-full shadow-[0_0_8px_rgba(0,242,254,0.5)]" />
                )}
              </Link>
            );
          })}
        </div>

        {/* Architecture */}
        {!collapsed && (
          <div className="pt-4 border-t border-border mt-4">
            <h3 className="text-primary font-semibold text-sm tracking-wide mb-3">🧠 Architecture</h3>
            <div className="text-xs text-secondary leading-relaxed space-y-1 font-mono">
              <div><span className="text-primary">Ticker</span> · {ticker}</div>
              <div><span className="text-primary">Model</span> · LSTM 64→Dense 32</div>
              <div><span className="text-primary">Lookback</span> · 5-Day Window</div>
              <div><span className="text-primary">Features</span> · 16 Hybrid</div>
              <div><span className="text-primary">NLP</span> · VADER + FAISS RAG</div>
            </div>
          </div>
        )}
      </div>

      {/* Collapse toggle */}
      <button 
        onClick={() => setCollapsed(!collapsed)}
        className="absolute -right-3 top-20 w-6 h-6 rounded-full bg-surface border border-border flex items-center justify-center text-secondary hover:text-foreground hover:bg-surface-raised transition-colors z-10"
      >
        <ChevronLeft className={clsx("w-4 h-4 transition-transform", collapsed && "rotate-180")} />
      </button>
    </div>
  );
}
