"use client";

import { useState, useRef, useEffect } from "react";
import { MessageSquare, Send, Bot, User, Sparkles } from "lucide-react";
import { useAppStore } from "@/store/appStore";

export default function AIChatPage() {
  const { ticker, stockName } = useAppStore();
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: `Hello! I am your RAG-powered financial assistant. I have access to real-time market data, technical indicators, and semantic news analysis. How can I help you analyze ${stockName} (${ticker}) today?`
    }
  ]);

  useEffect(() => {
    // Reset welcome message when ticker changes
    setMessages([
      {
        role: "assistant",
        content: `Hello! I am your RAG-powered financial assistant. I have access to real-time market data, technical indicators, and semantic news analysis. How can I help you analyze ${stockName} (${ticker}) today?`
      }
    ]);
  }, [ticker, stockName]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading) return;
    
    const userMsg = query.trim();
    setMessages(prev => [...prev, { role: "user", content: userMsg }]);
    setQuery("");
    setLoading(true);
    
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/chat?ticker=${ticker}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMsg })
      });
      
      if (!res.ok) throw new Error("Chat request failed");
      
      const data = await res.json();
      setMessages(prev => [...prev, {
        role: "assistant", 
        content: data.reply
      }]);
    } catch (err: any) {
      setMessages(prev => [...prev, {
        role: "assistant", 
        content: `Error: ${err.message}. Please try again.`
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-[calc(100vh-140px)] flex flex-col animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
          <MessageSquare className="w-6 h-6 text-primary" /> AI Chat Assistant
        </h1>
        <p className="text-secondary text-sm">Context-aware semantic query engine powered by FAISS</p>
      </div>

      <div className="glass-card flex-1 flex flex-col overflow-hidden relative">
        <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 blur-[100px] pointer-events-none" />
        
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
          {messages.map((msg, i) => (
            <div key={i} className={`flex gap-4 max-w-3xl ${msg.role === 'user' ? 'ml-auto flex-row-reverse' : ''}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.role === 'user' ? 'bg-surface-raised border border-border' : 'bg-primary/20 text-primary border border-primary/30'}`}>
                {msg.role === 'user' ? <User className="w-4 h-4 text-secondary" /> : <Bot className="w-5 h-5" />}
              </div>
              
              <div className={`p-4 rounded-2xl text-sm leading-relaxed ${
                msg.role === 'user' 
                  ? 'bg-primary text-background rounded-tr-sm' 
                  : 'bg-surface-raised border border-border text-foreground rounded-tl-sm shadow-md'
              }`}>
                {msg.content}
                
                {msg.role === 'assistant' && i === messages.length - 1 && messages.length > 1 && !loading && (
                  <div className="mt-3 pt-3 border-t border-border/50 flex items-center gap-2 text-xs text-primary/80">
                    <Sparkles className="w-3 h-3" />
                    Context: Technical Data + RAG Documents
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex gap-4 max-w-3xl">
              <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 bg-primary/20 text-primary border border-primary/30">
                <Bot className="w-5 h-5" />
              </div>
              <div className="p-4 rounded-2xl text-sm leading-relaxed bg-surface-raised border border-border text-foreground rounded-tl-sm shadow-md flex items-center gap-2">
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0.2s" }} />
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0.4s" }} />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        {/* Input Area */}
        <div className="p-4 bg-surface border-t border-border">
          <form onSubmit={handleSend} className="relative">
            <input 
              type="text" 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loading}
              placeholder={loading ? "Thinking..." : "Ask about market regimes, forecasts, or news impact..."}
              className="w-full bg-surface-raised border border-border rounded-xl pl-4 pr-12 py-4 text-sm text-foreground focus:outline-none focus:border-primary/50 transition-colors shadow-inner disabled:opacity-50"
            />
            <button 
              type="submit"
              disabled={!query.trim() || loading}
              className="absolute right-2 top-1/2 -translate-y-1/2 w-10 h-10 rounded-lg bg-primary text-background flex items-center justify-center hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
          <div className="text-center mt-2 text-xs text-secondary/70">
            Powered by Time-Aware Hybrid RAG-LSTM Architecture
          </div>
        </div>
      </div>
    </div>
  );
}
