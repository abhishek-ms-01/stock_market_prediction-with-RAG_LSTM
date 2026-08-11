"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, BarChart3, BrainCircuit, ShieldAlert, MessagesSquare, Zap } from "lucide-react";

export default function LandingPage() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6 },
    },
  };

  const features = [
    {
      icon: <BrainCircuit className="w-6 h-6 text-primary" />,
      title: "Hybrid RAG-LSTM Model",
      description: "Combines deep learning sequence prediction with time-aware semantic news retrieval.",
    },
    {
      icon: <BarChart3 className="w-6 h-6 text-bullish" />,
      title: "Real-time NSE Data",
      description: "Live market data paired with advanced quantitative technical indicators.",
    },
    {
      icon: <ShieldAlert className="w-6 h-6 text-warning" />,
      title: "Risk Analysis",
      description: "Volatility clustering, Value at Risk (VaR), and market regime detection.",
    },
    {
      icon: <MessagesSquare className="w-6 h-6 text-primary" />,
      title: "AI Chat Assistant",
      description: "Conversational interface powered by financial context and RAG architecture.",
    },
  ];

  return (
    <div className="min-h-screen bg-background overflow-hidden selection:bg-primary/30 relative">
      {/* Background elements */}
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-primary/10 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-bullish/5 blur-[120px]" />
        
        {/* Grid pattern */}
        <div 
          className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"
          style={{ maskImage: "radial-gradient(ellipse 60% 60% at 50% 50%, #000 10%, transparent 100%)" }}
        />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 pt-24 pb-16 min-h-screen flex flex-col justify-center">
        {/* Header/Hero Section */}
        <motion.div 
          className="text-center max-w-4xl mx-auto mb-20"
          initial="hidden"
          animate="visible"
          variants={containerVariants}
        >
          <motion.div variants={itemVariants} className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-surface border border-border mb-6">
            <span className="flex h-2 w-2 rounded-full bg-bullish animate-pulse-slow"></span>
            <span className="text-xs font-mono text-secondary tracking-wide uppercase">Predictive Intelligence v2.0</span>
          </motion.div>
          
          <motion.h1 variants={itemVariants} className="font-bold tracking-tight mb-8">
            <span className="block text-xl md:text-2xl text-secondary tracking-[0.2em] uppercase mb-3 font-medium">Event-Driven</span>
            <span className="block text-4xl md:text-6xl text-foreground leading-[1.1] mb-2 tracking-tighter">Stock Market Prediction System</span>
            <span className="inline-block text-transparent bg-clip-text bg-gradient-to-r from-primary to-blue-400 text-3xl md:text-5xl mt-2 font-semibold tracking-tight">Using RAG-LSTM</span>
          </motion.h1>
          
          <motion.p variants={itemVariants} className="text-lg md:text-xl text-secondary mb-10 max-w-2xl mx-auto leading-relaxed">
            A state-of-the-art fintech terminal merging Time-Aware RAG with LSTM neural networks for high-fidelity short-term NSE forecasting.
          </motion.p>
          
          <motion.div variants={itemVariants} className="flex justify-center gap-4">
            <Link href="/dashboard" className="group relative inline-flex h-14 items-center justify-center overflow-hidden rounded-lg bg-primary px-8 font-medium text-background transition-all hover:scale-105 active:scale-95">
              <span className="absolute h-0 w-0 rounded-full bg-white/20 transition-all duration-300 ease-out group-hover:h-56 group-hover:w-56"></span>
              <span className="relative flex items-center gap-2 text-base font-bold">
                Initialize AI Engine <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </span>
            </Link>
          </motion.div>
        </motion.div>

        {/* Feature Grid */}
        <motion.div 
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
          initial="hidden"
          animate="visible"
          variants={containerVariants}
        >
          {features.map((feature, i) => (
            <motion.div 
              key={i}
              variants={itemVariants}
              className="glass-card p-6 hover:border-primary/50 transition-colors duration-300 group"
            >
              <div className="w-12 h-12 rounded-lg bg-surface flex items-center justify-center mb-4 border border-border group-hover:border-primary/30 transition-colors">
                {feature.icon}
              </div>
              <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
              <p className="text-secondary text-sm leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </motion.div>
        
        {/* Trusted By / Architecture Bar */}
        <motion.div 
          className="mt-24 pt-8 border-t border-border/50 flex flex-wrap justify-center gap-8 text-secondary/60 text-sm font-mono"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 1 }}
        >
          <div className="flex items-center gap-2"><Zap className="w-4 h-4" /> Real-time Inference</div>
          <div className="flex items-center gap-2"><Zap className="w-4 h-4" /> 5-Min Lookback Window</div>
          <div className="flex items-center gap-2"><Zap className="w-4 h-4" /> FAISS Vector DB</div>
          <div className="flex items-center gap-2"><Zap className="w-4 h-4" /> FastAPI + Next.js 14</div>
        </motion.div>
        {/* Floating Chatbot Button */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 1.5, duration: 0.5 }}
          className="fixed bottom-8 right-8 z-50"
        >
          <Link href="/dashboard/ai-chat" className="flex items-center justify-center w-14 h-14 bg-primary text-background rounded-full shadow-[0_0_20px_rgba(0,242,254,0.5)] hover:scale-110 hover:shadow-[0_0_30px_rgba(0,242,254,0.8)] transition-all duration-300">
            <MessagesSquare className="w-6 h-6" />
          </Link>
        </motion.div>
      </div>
    </div>
  );
}
