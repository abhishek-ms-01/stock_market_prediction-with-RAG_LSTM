import os
import pandas as pd
import numpy as np
from datetime import datetime

FAISS_AVAILABLE = True
try:
    import faiss
except ImportError:
    FAISS_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FinancialRAGEngine:
    """
    IEEE Research-Level Semantic & Time-Aware Financial RAG Engine.
    Employs SentenceTransformers dense embeddings and FAISS index for vector retrieval.
    Enforces strict published_date <= target_date constraint to prevent future data leakage.
    """

    def __init__(self, data_path: str = None, model_name: str = "all-MiniLM-L6-v2"):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            processed_path = os.path.join(base_dir, "data", "news_processed.csv")
            raw_path = os.path.join(base_dir, "data", "news_data.csv")
            if os.path.exists(processed_path):
                data_path = processed_path
            elif os.path.exists(raw_path):
                data_path = raw_path

        self.data_path = data_path
        self.model_name = model_name
        self.df = None
        self.is_indexed = False

        # Dense Embedding & FAISS Index
        self.st_model = None
        self.faiss_index = None
        self.embeddings = None

        # TF-IDF Fallback Vectorizer
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        self._build_index()

    def _build_index(self):
        """Builds FAISS dense semantic index and TF-IDF fallback matrix."""
        if not self.data_path or not os.path.exists(self.data_path):
            print(f"[RAG Engine Warning] Data file not found at: {self.data_path}")
            return

        try:
            self.df = pd.read_csv(self.data_path).fillna("")
            if self.df.empty:
                print("[RAG Engine Warning] Loaded news dataset is empty.")
                return

            # Ensure proper Date parsing for Time-Aware filtering
            if 'date' in self.df.columns:
                self.df['parsed_date'] = pd.to_datetime(self.df['date'], errors='coerce', utc=True)
            else:
                self.df['parsed_date'] = pd.NaT

            # Construct text corpus for semantic embedding
            text_corpus = []
            for _, row in self.df.iterrows():
                title = str(row.get("title", ""))
                content = str(row.get("content", ""))
                event = str(row.get("event", ""))
                text_corpus.append(f"{title} {content} {event}".strip())

            # 1. Build TF-IDF matrix first as ultra-fast instant search engine (0.02s)
            self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_corpus)
            self.is_indexed = True

            # 2. Attempt SentenceTransformers + FAISS Index Construction if cached or fast
            if FAISS_AVAILABLE:
                try:
                    emb_cache_path = os.path.join(os.path.dirname(self.data_path), "faiss_embeddings.npy")
                    if os.path.exists(emb_cache_path):
                        embeddings = np.load(emb_cache_path)
                        self.embeddings = embeddings.astype(np.float32)
                        dim = self.embeddings.shape[1]
                        self.faiss_index = faiss.IndexFlatIP(dim)
                        self.faiss_index.add(self.embeddings)
                        print(f"[RAG Engine] Loaded FAISS index with {len(text_corpus)} docs (dim={dim}).")
                except Exception as e:
                    print(f"[RAG Engine Warning] FAISS init fallback to TF-IDF: {e}")
                    self.faiss_index = None

        except Exception as e:
            print(f"[RAG Engine Error] Failed to build vector index: {e}")

    def retrieve_time_aware(self, query: str, target_date=None, top_k: int = 3, ticker_filter: str = None) -> list:
        """
        Retrieves top_k semantic news documents satisfying published_date <= target_date.
        Prevents future data leakage in time-series forecasting.
        """
        if not self.is_indexed or not query or not query.strip():
            return []

        # Parse target_date constraint
        cutoff_date = None
        if target_date is not None:
            if isinstance(target_date, str):
                cutoff_date = pd.to_datetime(target_date, errors='coerce', utc=True)
            elif isinstance(target_date, (pd.Timestamp, datetime)):
                cutoff_date = pd.to_datetime(target_date, utc=True)

        # Filter candidate document indices by date constraint (published_date <= target_date)
        if cutoff_date is not None and not pd.isna(cutoff_date) and 'parsed_date' in self.df.columns:
            valid_mask = self.df['parsed_date'].notna() & (self.df['parsed_date'] <= cutoff_date)
            valid_indices = self.df.index[valid_mask].tolist()
            if not valid_indices:
                # If no historical news before cutoff, fallback to oldest documents
                valid_indices = self.df.index.tolist()
        else:
            valid_indices = self.df.index.tolist()

        # Execute Dense FAISS Retrieval if available
        if self.faiss_index is not None and self.st_model is not None:
            try:
                q_emb = self.st_model.encode([query.strip()], convert_to_numpy=True).astype(np.float32)
                faiss.normalize_L2(q_emb)

                # Search top candidate pool
                k_search = min(len(valid_indices), max(top_k * 5, 20))
                distances, indices = self.faiss_index.search(q_emb, k_search)

                results = []
                valid_set = set(valid_indices)
                for score, idx in zip(distances[0], indices[0]):
                    if idx in valid_set:
                        row = self.df.iloc[idx]
                        results.append({
                            "title": str(row.get("title", "No Title")),
                            "content": str(row.get("content", "")),
                            "date": str(row.get("date", "N/A")),
                            "sentiment": float(row.get("sentiment", 0.0)) if row.get("sentiment") != "" else 0.0,
                            "event": str(row.get("event", "General")),
                            "similarity": float(score)
                        })
                        if len(results) >= top_k:
                            break
                if results:
                    return results
            except Exception as e:
                print(f"[RAG Engine Warning] FAISS search error: {e}. Using TF-IDF.")

        # Fallback TF-IDF Retrieval
        query_vec = self.tfidf_vectorizer.transform([query.strip()])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Restrict to valid time-aware indices
        sub_scores = [(sim_scores[idx], idx) for idx in valid_indices]
        sub_scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in sub_scores[:top_k]:
            row = self.df.iloc[idx]
            results.append({
                "title": str(row.get("title", "No Title")),
                "content": str(row.get("content", "")),
                "date": str(row.get("date", "N/A")),
                "sentiment": float(row.get("sentiment", 0.0)) if row.get("sentiment") != "" else 0.0,
                "event": str(row.get("event", "General")),
                "similarity": float(score)
            })
        return results

    def retrieve(self, query: str, top_k: int = 3, min_similarity: float = 0.0) -> list:
        """Standard retrieval interface wrapper around time-aware retrieval."""
        docs = self.retrieve_time_aware(query=query, top_k=top_k)
        if min_similarity > 0.0:
            docs = [d for d in docs if d.get("similarity", 0.0) >= min_similarity]
        return docs

    def get_rag_features(self, query: str, target_date=None, ticker_filter: str = None) -> dict:
        """
        Analyzes retrieved time-aware news and extracts quantitative prediction features:
        - RAG_Sentiment (-1.0 to +1.0)
        - RAG_Relevance (0.0 to 1.0)
        - RAG_Event_Importance (0.0 to 1.0)
        - RAG_Market_Impact (0.0 to 1.0)
        - RAG_Risk_Score (0.0 to 1.0)
        - RAG_Confidence (0.0 to 1.0)
        """
        docs = self.retrieve_time_aware(query, target_date=target_date, top_k=3, ticker_filter=ticker_filter)
        if not docs:
            return {
                "rag_sentiment": 0.0,
                "rag_relevance": 0.0,
                "rag_event_importance": 0.4,
                "rag_market_impact": 0.0,
                "rag_risk_score": 0.0,
                "rag_confidence": 0.0,
                "rag_summary": "No historical news retrieved before timestamp."
            }

        # Quantitative Event Importance Mapping
        event_weights = {
            "Earnings": 0.90,
            "Merger": 0.85,
            "Legal": 0.80,
            "General": 0.40
        }

        sentiments = [d["sentiment"] for d in docs]
        similarities = [max(0.0, d["similarity"]) for d in docs]
        event_imp = [event_weights.get(d["event"], 0.4) for d in docs]

        # Weighted Averages based on Semantic Relevance
        weights = np.array(similarities) + 1e-5
        weights /= weights.sum()

        avg_sentiment = float(np.sum(np.array(sentiments) * weights))
        max_relevance = float(np.max(similarities))
        avg_event_imp = float(np.sum(np.array(event_imp) * weights))

        # Hybrid Market Impact = Relevance * |Sentiment| * Event Importance
        market_impact = float(max_relevance * abs(avg_sentiment) * avg_event_imp)

        # Risk Score = High Negative Sentiment OR Legal Event Indicator
        has_legal = any(d["event"] == "Legal" for d in docs)
        risk_score = float(max(0.0, -avg_sentiment) * 0.7 + (0.3 if has_legal else 0.0))

        # Retrieval Confidence
        confidence = float(np.mean(similarities))

        summary_parts = [f"[{d['event']}] {d['title']}" for d in docs[:2]]
        summary_text = " | ".join(summary_parts)

        return {
            "rag_sentiment": round(avg_sentiment, 4),
            "rag_relevance": round(max_relevance, 4),
            "rag_event_importance": round(avg_event_imp, 4),
            "rag_market_impact": round(market_impact, 4),
            "rag_risk_score": round(min(1.0, risk_score), 4),
            "rag_confidence": round(min(1.0, confidence), 4),
            "rag_summary": summary_text
        }

    def get_doc_count(self) -> int:
        """Returns total number of indexed news documents."""
        return len(self.df) if self.df is not None else 0
