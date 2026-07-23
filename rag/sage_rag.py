# rag/sage_rag.py
"""
SAGE RAG Engine - Standalone SAGE Synthetic Data Protected RAG
Completely independent from Differential Privacy (DP) mechanism

Core Functions:
1. Use SAGEEngine to manage synthetic data generation and retrieval
2. Standard RAG answer generation (without DP mechanism)
3. Maintain full compatibility with BaseEngine interface

Inheritance:
    BaseEngine (ABC)
        ├── StandardRAGEngine (Original RAG)
        ├── DPRAGEngine (DP Protected RAG)
        └── SAGERAGEngine (This file - SAGE Protected RAG)

Responsibility Chain:
    User Query
        │
        ▼
    ┌──────────────────────┐
    │ Intent Detection     │
    │ (Safety Check)       │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ SAGE Synthetic Data  │
    │ Retrieval            │
    │ (SAGEEngine)         │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Standard LLM         │
    │ Answer Generation    │
    │ (No DP Mechanism)    │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Safety Detection     │
    │ (Leakage Check)      │
    └──────────┬───────────┘
               │
               ▼
    Final Answer
"""

import os
from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document

from .base_engine import BaseEngine
from . import prompts as pt
from .sage_engine import SAGEEngine


class SAGERAGEngine(BaseEngine):
    """
    Standalone SAGE RAG Engine
    
    Uses SAGE synthetic data to replace original data for retrieval, providing privacy protection.
    Does not contain any Differential Privacy (DP) mechanisms.

    Parameters:
        sage_engine: SAGEEngine instance (manages synthetic data)
        cache_dir: Synthetic data cache directory
    """

    def __init__(
        self,
        llm,
        embedding,
        reranker,
        top_p: int,
        top_k: int,
        knowledge_path: str,
        sage_engine: SAGEEngine = None,
        cache_dir: str = "./storage/synthetic_data",
    ):
        # Initialize base class
        super().__init__(llm, embedding, reranker, top_p, top_k, knowledge_path)

        # SAGE engine
        self.sage_engine = sage_engine
        if sage_engine is None:
            print("[SAGE-RAG] Initializing SAGEEngine...")
            self.sage_engine = SAGEEngine(
                llm=llm,
                embedding=embedding,
                original_data_path=knowledge_path,
                cache_dir=cache_dir,
            )

        # Cache directory
        self.cache_dir = cache_dir

        # Vector store (compatible with base class interface)
        self.vector_store = None
        self.collection_name = "sage_rag_collection"

        # Statistics
        self._stats = {
            "sage_retrieval_count": 0,
            "synthetic_mode": "sync",
        }

        # Ensure SAGE index is built
        self._ensure_sage_index()

    def _ensure_sage_index(self):
        """Ensure SAGE index is built"""
        if not self.sage_engine._is_index_built:
            print("[SAGE-RAG] Building SAGE synthetic index...")
            self.sage_engine.preprocess_and_build_index()

    # ─────────────────────────────────────────────────────────────────
    # Index Management (compatible with base class interface)
    # ─────────────────────────────────────────────────────────────────

    def _check_index_exists(self, persist_dir: str) -> bool:
        """Check if index exists"""
        if self.sage_engine and self.sage_engine._is_index_built:
            return True
        # Backup: check standard Chroma index
        sqlite_path = os.path.join(persist_dir, "chroma.sqlite3")
        return os.path.exists(sqlite_path)

    def _build_index(self, docs: List[Document], persist_dir: str):
        """Build index (use SAGE engine in SAGE mode)"""
        if self.sage_engine:
            print(f"[SAGE-RAG] Using SAGE synthetic index, skipping base build_index")
            return
        # Non-SAGE mode backup
        from langchain_chroma import Chroma
        self.vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding,
            persist_directory=persist_dir,
            collection_name=self.collection_name,
        )

    def _load_index(self, persist_dir: str):
        """Load index"""
        if self.sage_engine and self.sage_engine._is_index_built:
            print(f"[SAGE-RAG] Using SAGE synthetic index")
            return
        # Non-SAGE mode backup
        from langchain_chroma import Chroma
        self.vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding,
            collection_name=self.collection_name,
        )

    # ─────────────────────────────────────────────────────────────────
    # Retrieval (SAGE mode)
    # ─────────────────────────────────────────────────────────────────

    def search(self, query: str) -> List[Document]:
        """
        Retrieve from SAGE synthetic data
        
        Replace original document retrieval with synthetic data retrieval
        """
        docs = self.sage_engine.search(query, top_k=self.top_k, top_p=self.top_p)

        # Rerank if reranker exists
        if self.reranker and self.top_p > self.top_k:
            docs = self.reranker.rerank(query, docs, top_k=self.top_k)
        elif len(docs) > self.top_k:
            docs = docs[: self.top_k]

        self._stats["sage_retrieval_count"] += 1
        return docs

    # ─────────────────────────────────────────────────────────────────
    # End-to-end Answer
    # ─────────────────────────────────────────────────────────────────

    def answer(self, query: str) -> Tuple[str, Optional[List[Document]]]:
        """
        End-to-end RAG + SAGE Generation
        
        Flow:
        1. Intent detection
        2. SAGE synthetic data retrieval
        3. Standard LLM answer generation (no DP)
        4. Safety check
        """
        # 1. Intent detection
        if not self.safety_check_query(query):
            return "Unknown.Intent", None

        # 2. Retrieval (SAGE synthetic data)
        context_docs = self.search(query)

        if not context_docs:
            print("[SAGE-RAG] Warning: No documents retrieved from SAGE index")
            return "I don't have enough information to answer.", None

        # 3. Build prompt and generate answer (standard method, no DP)
        context_str = "\n\n".join(
            [f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(context_docs)]
        )

        # Select prompt template
        if "HP1_5ch" in self.knowledge_path:
            prompt_template = pt.RAG_PROMPT_TEMPLATE_HP
        elif "HealthCareMagic" in self.knowledge_path:
            prompt_template = pt.RAG_PROMPT_TEMPLATE_HC
        else:
            prompt_template = pt.RAG_PROMPT_TEMPLATE_DEFAULT

        # Build prompt
        prompt = prompt_template.format(context=context_str, question=query)
        
        # Generate answer (standard LLM call, no DP)
        try:
            response = self.llm.generate(prompt)
        except Exception as e:
            print(f"[SAGE-RAG] Generation failed: {e}")
            return "I encountered an error.", context_docs

        # 4. Safety check
        if not self.safety_check_response(response, context_str):
            return "Unknown.Copy", None

        return response, context_docs

    # ─────────────────────────────────────────────────────────────────
    # Statistics Interface
    # ─────────────────────────────────────────────────────────────────

    def get_dp_stats(self) -> dict:
        """Return statistics (compatible with DP_RAG interface)"""
        stats = self._stats.copy()
        if self.sage_engine:
            stats["sage_engine"] = self.sage_engine.get_dp_stats()
        return stats

    def get_stats(self) -> dict:
        """Return SAGE-specific statistics"""
        return self._stats.copy()

    def reset(self):
        """Reset engine state"""
        self._stats = {
            "sage_retrieval_count": 0,
            "synthetic_mode": "sync",
        }
        if self.sage_engine:
            self.sage_engine.reset()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_sage_rag_engine(
    llm,
    embedding,
    reranker,
    data_path: str,
    top_p: int = 10,
    top_k: int = 10,
    cache_dir: str = "./storage/synthetic_data",
) -> SAGERAGEngine:
    """
    Convenience function for creating SAGE RAG engine
    
    Usage Example:
        engine = create_sage_rag_engine(
            llm=llm,
            embedding=embedding,
            reranker=reranker,
            data_path="./datasets/mini_trec_covid.json",
        )
    """
    sage_engine = SAGEEngine(
        llm=llm,
        embedding=embedding,
        original_data_path=data_path,
        cache_dir=cache_dir,
    )

    engine = SAGERAGEngine(
        llm=llm,
        embedding=embedding,
        reranker=reranker,
        top_p=top_p,
        top_k=top_k,
        knowledge_path=data_path,
        sage_engine=sage_engine,
        cache_dir=cache_dir,
    )

    return engine


if __name__ == "__main__":
    print("[SAGE-RAG] Module loaded successfully")
    print("[SAGE-RAG] Usage: from rag.sage_rag import SAGERAGEngine, create_sage_rag_engine")