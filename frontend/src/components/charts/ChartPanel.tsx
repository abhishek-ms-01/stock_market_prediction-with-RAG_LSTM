"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType, CrosshairMode, IChartApi, ISeriesApi } from "lightweight-charts";

interface ChartProps {
  data: any[];
  showSMA?: boolean;
  showVolume?: boolean;
}

export function ChartPanel({ data, showSMA = true, showVolume = true }: ChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const smaSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#8B9DC3",
      },
      grid: {
        vertLines: { color: "rgba(32, 45, 69, 0.3)" },
        horzLines: { color: "rgba(32, 45, 69, 0.3)" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: "rgba(32, 45, 69, 0.5)",
      },
      timeScale: {
        borderColor: "rgba(32, 45, 69, 0.5)",
        timeVisible: true,
      },
      autoSize: true,
    });

    chartRef.current = chart;

    // Add Candlestick Series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: "#10b981", // green
      downColor: "#f43f5e", // red
      borderVisible: false,
      wickUpColor: "#10b981",
      wickDownColor: "#f43f5e",
    });
    candlestickSeriesRef.current = candlestickSeries;

    // Add Volume Series
    const volumeSeries = chart.addHistogramSeries({
      color: "#26a69a",
      priceFormat: {
        type: "volume",
      },
      priceScaleId: "", // Set as an overlay
    });
    
    // Scale volume overlay to bottom 20%
    chart.priceScale("").applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });
    volumeSeriesRef.current = volumeSeries;

    // Add SMA Series
    const smaSeries = chart.addLineSeries({
      color: "#f59e0b", // amber
      lineWidth: 2,
      lineStyle: 2, // Dotted (LineStyle.Dotted = 2, wait, 3 is Dashed? Lightweight charts has LineStyle.Dotted? Wait, we can just use integer 2 or 3)
      crosshairMarkerVisible: false,
    });
    smaSeriesRef.current = smaSeries;

    // We can't use enums directly if we didn't import them, but we imported `CrosshairMode` and `ColorType`. 
    // LineStyle.Dashed is 3, Dotted is 2. Let's use 2.

    return () => {
      chart.remove();
    };
  }, []);

  // Update data
  useEffect(() => {
    if (!candlestickSeriesRef.current || !volumeSeriesRef.current || !smaSeriesRef.current || !data.length) return;

    // Helper to format time for lightweight-charts
    const formatTime = (t: string) => {
      if (t.includes('T')) return t.split('T')[0];
      return t;
    };

    // Make sure data is sorted by date ascending
    const sortedData = [...data].sort((a, b) => new Date(a.Date).getTime() - new Date(b.Date).getTime());

    const candleData = sortedData.map(d => ({
      time: formatTime(d.Date) as any, // Lightweight charts typings
      open: d.Open,
      high: d.High,
      low: d.Low,
      close: d.Close,
    }));

    const volumeData = sortedData.map(d => ({
      time: formatTime(d.Date) as any,
      value: d.Volume,
      color: d.Close >= d.Open ? "rgba(16, 185, 129, 0.5)" : "rgba(244, 63, 94, 0.5)",
    }));

    const smaData = sortedData
      .filter(d => d.MA_20 !== null && d.MA_20 !== undefined && !isNaN(d.MA_20))
      .map(d => ({
        time: formatTime(d.Date) as any,
        value: d.MA_20,
      }));

    try {
      candlestickSeriesRef.current.setData(candleData);
      volumeSeriesRef.current.setData(volumeData);
      smaSeriesRef.current.setData(smaData);
    } catch (e) {
      console.warn("Chart data error:", e);
    }

    // Visibility toggles
    smaSeriesRef.current.applyOptions({ visible: showSMA });
    volumeSeriesRef.current.applyOptions({ visible: showVolume });

    // Fit content
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [data, showSMA, showVolume]);

  return (
    <div className="glass-panel p-1 w-full h-[450px]">
      <div ref={chartContainerRef} className="w-full h-full" />
    </div>
  );
}
