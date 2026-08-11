import os
import sys
import time
from datetime import datetime

# Ensure project root in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_ingestion.multi_source_fusion import MultiSourceNewsFusionPipeline
from rag.time_aware_rag import TimeAwareFinancialRAG

class AutonomousFinancialRAGAgent:
    """
    IEEE Research Component: Autonomous Financial AI Research Agent.
    Monitors data streams, fuses multi-source news, updates FAISS vector index,
    and maintains pipeline health.
    """

    def __init__(self):
        self.fusion_pipeline = MultiSourceNewsFusionPipeline()
        self.rag_engine = None

    def execute_agent_cycle(self) -> dict:
        """Runs a complete autonomous research iteration."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        print(f"[{timestamp}] [AI Research Agent] Initiating autonomous pipeline cycle...")

        # 1. Fuse Multi-Source News
        fused_df = self.fusion_pipeline.run_fusion_pipeline()

        # 2. Update Time-Aware RAG Engine
        self.rag_engine = TimeAwareFinancialRAG()
        doc_count = self.rag_engine.get_doc_count()

        status_report = {
            "timestamp": timestamp,
            "fused_articles_count": len(fused_df),
            "faiss_indexed_docs": doc_count,
            "status": "Healthy & Operational"
        }

        print(f"[AI Research Agent] Cycle complete. Indexed {doc_count} documents into FAISS vector store.")
        return status_report

if __name__ == "__main__":
    agent = AutonomousFinancialRAGAgent()
    report = agent.execute_agent_cycle()
    print(report)
