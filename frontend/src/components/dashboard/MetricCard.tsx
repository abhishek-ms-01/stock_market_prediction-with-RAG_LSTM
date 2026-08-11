import clsx from "clsx";
import { ReactNode } from "react";

interface MetricCardProps {
  title: string;
  value: string | number;
  icon: ReactNode;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  sparklineData?: number[];
}

export function MetricCard({ title, value, icon, trend, trendValue, sparklineData }: MetricCardProps) {
  // Simple SVG sparkline generator
  const generateSparkline = (data: number[]) => {
    if (!data || data.length === 0) return null;
    
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    
    const width = 100;
    const height = 30;
    
    const points = data.map((val, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - ((val - min) / range) * height;
      return `${x},${y}`;
    }).join(" L ");
    
    const isUp = data[data.length - 1] >= data[0];
    const color = isUp ? "#00E676" : "#FF1744"; // bullish or bearish
    
    return (
      <svg width="100%" height="30" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        <path d={`M ${points}`} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
      </svg>
    );
  };

  return (
    <div className="glass-card p-6 group relative overflow-hidden transition-all duration-500 hover:border-primary/40">
      {/* Background Glows */}
      <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-bl from-primary/10 to-transparent rounded-bl-full pointer-events-none opacity-50 group-hover:opacity-100 transition-opacity" />
      <div className="absolute -bottom-10 -right-10 w-32 h-32 bg-blue-500/20 blur-3xl rounded-full pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
      
      <div className="flex justify-between items-start mb-4">
        <div className="text-sm font-medium text-secondary">{title}</div>
        <div className="text-secondary group-hover:text-primary transition-colors">
          {icon}
        </div>
      </div>
      
      <div className="flex items-end justify-between">
        <div>
          <div className="text-3xl font-sans font-semibold text-foreground tracking-tight tabular-nums">{value}</div>
          
          {trendValue && (
            <div className={clsx(
              "text-xs font-sans font-medium mt-1 flex items-center gap-1 tabular-nums",
              trend === "up" && "text-bullish",
              trend === "down" && "text-bearish",
              trend === "neutral" && "text-secondary"
            )}>
              {trend === "up" && "▲"}
              {trend === "down" && "▼"}
              {trend === "neutral" && "—"}
              {trendValue}
            </div>
          )}
        </div>
        
        {sparklineData && (
          <div className="w-20 opacity-70 group-hover:opacity-100 transition-opacity">
            {generateSparkline(sparklineData)}
          </div>
        )}
      </div>
    </div>
  );
}
