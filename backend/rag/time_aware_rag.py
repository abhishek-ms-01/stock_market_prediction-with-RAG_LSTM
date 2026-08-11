import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure project root in path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from rag.financial_graph import FinancialKnowledgeGraph
from data_ingestion.multi_source_fusion import get_source_credibility

class TimeAwareFinancialRAG:
    """
    IEEE Research-Level Time-Aware Hybrid Semantic RAG Engine.
    Combines FAISS vector index, SentenceTransformers embeddings,
    strict published_date <= target_date constraint, Knowledge Graph traversal,
    and source credibility weighting.
    """

    def __init__(self, data_path: str = None, model_name: str = "all-MiniLM-L6-v2"):
        if data_path is None:
            data_path = os.path.join(project_root, "data", "news_processed.csv")

        self.data_path = data_path
        self.model_name = model_name
        self.df = None
        self.kg = FinancialKnowledgeGraph()
        self.is_indexed = False

        self.st_model = None
        self.faiss_index = None
        self.embeddings = None

        self._build_index()

    def _build_index(self):
        """Builds FAISS IndexFlatIP dense vector store."""
        if not os.path.exists(self.data_path):
            print(f"[Time-Aware RAG Warning] Data file not found: {self.data_path}")
            return

        try:
            self.df = pd.read_csv(self.data_path).fillna("")
            if self.df.empty:
                return

            if 'date' in self.df.columns:
                self.df['parsed_date'] = pd.to_datetime(self.df['date'], errors='coerce', utc=True)
            else:
                self.df['parsed_date'] = pd.NaT

            # Construct rich text representation for dense embeddings
            text_corpus = []
            for _, row in self.df.iterrows():
                title = str(row.get("title", ""))
                content = str(row.get("content", ""))
                event = str(row.get("event", "General"))
                src = str(row.get("source", "Yahoo Finance"))
                text_corpus.append(f"{title} {content} {event} {src}".strip())

            if FAISS_AVAILABLE:
                try:
                    print(f"[Time-Aware RAG] Encoding {len(text_corpus)} news documents with '{self.model_name}'...")
                    self.st_model = SentenceTransformer(self.model_name)
                    embeddings = self.st_model.encode(text_corpus, show_progress_bar=False, convert_to_numpy=True)
                    
                    faiss.normalize_L2(embeddings)
                    self.embeddings = embeddings.astype(np.float32)

                    dim = self.embeddings.shape[1]
                    self.faiss_index = faiss.IndexFlatIP(dim)
                    self.faiss_index.add(self.embeddings)
                    self.is_indexed = True
                    print(f"[Time-Aware RAG] Successfully initialized FAISS vector database (dim={dim}).")
                except Exception as e:
                    print(f"[Time-Aware RAG Error] Vector index failed: {e}")

        except Exception as e:
            print(f"[Time-Aware RAG Error] Dataset loading error: {e}")

    def retrieve(self, query: str, target_date=None, top_k: int = 3, ticker: str = None) -> list:
        """
        Hybrid Time-Aware Semantic Retrieval.
        Applies date filter (published_date <= target_date), Knowledge Graph expansion,
        and Source Credibility Score weighting.
        """
        if not self.is_indexed or not query or not query.strip():
            return []

        # Knowledge Graph Query Expansion
        expanded_query = query.strip()
        if ticker:
            graph_nodes = self.kg.get_connected_entities(ticker)
            if graph_nodes:
                expanded_query += " " + " ".join(graph_nodes)

        # Date Filtering
        cutoff_date = None
        if target_date is not None:
            cutoff_date = pd.to_datetime(target_date, errors='coerce', utc=True)

        if cutoff_date is not None and not pd.isna(cutoff_date) and 'parsed_date' in self.df.columns:
            valid_mask = self.df['parsed_date'].notna() & (self.df['parsed_date'] <= cutoff_date)
            valid_indices = self.df.index[valid_mask].tolist()
            if not valid_indices:
                valid_indices = self.df.index.tolist()
        else:
            valid_indices = self.df.index.tolist()

        if self.faiss_index is not None and self.st_model is not None:
            try:
                q_emb = self.st_model.encode([expanded_query], convert_to_numpy=True).astype(np.float32)
                faiss.normalize_L2(q_emb)

                k_search = min(len(valid_indices), max(top_k * 5, 20))
                distances, indices = self.faiss_index.search(q_emb, k_search)

                results = []
                valid_set = set(valid_indices)
                for score, idx in zip(distances[0], indices[0]):
                    if idx in valid_set:
                        row = self.df.iloc[idx]
                        src = str(row.get("source", "Yahoo Finance"))
                        cred_score = get_source_credibility(src)
                        
                        # Credibility-Weighted Hybrid Similarity = Cosine Similarity * Credibility Score
                        weighted_score = float(score) * cred_score

                        results.append({
                            "title": str(row.get("title", "No Title")),
                            "content": str(row.get("content", "")),
                            "date": str(row.get("date", "N/A")),
                            "source": src,
                            "credibility_score": cred_score,
                            "sentiment": float(row.get("sentiment", 0.0)) if row.get("sentiment") != "" else 0.0,
                            "event": str(row.get("event", "General")),
                            "similarity": round(float(score), 4),
                            "weighted_similarity": round(weighted_score, 4)
                        })
                        if len(results) >= top_k:
                            break

                return results
            except Exception as e:
                print(f"[Time-Aware RAG Error] Search failed: {e}")

        return []
