# rag/sage_rag.py
"""
SAGE RAG 引擎 - 独立的SAGE合成数据保护RAG
与差分隐私(DP)机制完全独立

核心功能:
1. 使用SAGEEngine管理合成数据生成和检索
2. 标准RAG回答生成 (不含DP机制)
3. 保持与BaseEngine接口完全兼容

继承关系:
    BaseEngine (ABC)
        ├── StandardRAGEngine (原始RAG)
        ├── DPRAGEngine (DP保护RAG)
        └── SAGERAGEngine (本文件 - SAGE保护RAG)

职责链:
    用户查询
        │
        ▼
    ┌──────────────────────┐
    │ 意图检测 (安全检查)  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ SAGE合成数据检索     │
    │ (SAGEEngine)         │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ 标准LLM生成回答      │
    │ (无DP机制)           │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ 安全检测 (泄露检查)  │
    └──────────┬───────────┘
               │
               ▼
    最终回答
"""

import os
from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document

from .base_engine import BaseEngine
from . import prompts as pt
from .sage_engine import SAGEEngine


class SAGERAGEngine(BaseEngine):
    """
    独立SAGE RAG引擎
    
    使用SAGE合成数据替代原始数据检索，提供隐私保护。
    不包含任何差分隐私(DP)机制。

    参数:
        sage_engine: SAGEEngine实例 (管理合成数据)
        cache_dir: 合成数据缓存目录
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
        # 初始化基类
        super().__init__(llm, embedding, reranker, top_p, top_k, knowledge_path)

        # SAGE引擎
        self.sage_engine = sage_engine
        if sage_engine is None:
            print("[SAGE-RAG] Initializing SAGEEngine...")
            self.sage_engine = SAGEEngine(
                llm=llm,
                embedding=embedding,
                original_data_path=knowledge_path,
                cache_dir=cache_dir,
            )

        # 缓存目录
        self.cache_dir = cache_dir

        # 向量存储 (兼容基类接口)
        self.vector_store = None
        self.collection_name = "sage_rag_collection"

        # 统计信息
        self._stats = {
            "sage_retrieval_count": 0,
            "synthetic_mode": "sync",
        }

        # 确保SAGE索引已构建
        self._ensure_sage_index()

    def _ensure_sage_index(self):
        """确保SAGE索引已构建"""
        if not self.sage_engine._is_index_built:
            print("[SAGE-RAG] Building SAGE synthetic index...")
            self.sage_engine.preprocess_and_build_index()

    # ─────────────────────────────────────────────────────────────────
    # 索引管理 (兼容基类接口)
    # ─────────────────────────────────────────────────────────────────

    def _check_index_exists(self, persist_dir: str) -> bool:
        """检查索引是否存在"""
        if self.sage_engine and self.sage_engine._is_index_built:
            return True
        # 备用: 检查标准Chroma索引
        sqlite_path = os.path.join(persist_dir, "chroma.sqlite3")
        return os.path.exists(sqlite_path)

    def _build_index(self, docs: List[Document], persist_dir: str):
        """构建索引 (SAGE模式下使用SAGE引擎)"""
        if self.sage_engine:
            print(f"[SAGE-RAG] Using SAGE synthetic index, skipping base build_index")
            return
        # 非SAGE模式备用
        from langchain_chroma import Chroma
        self.vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding,
            persist_directory=persist_dir,
            collection_name=self.collection_name,
        )

    def _load_index(self, persist_dir: str):
        """加载索引"""
        if self.sage_engine and self.sage_engine._is_index_built:
            print(f"[SAGE-RAG] Using SAGE synthetic index")
            return
        # 非SAGE模式备用
        from langchain_chroma import Chroma
        self.vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding,
            collection_name=self.collection_name,
        )

    # ─────────────────────────────────────────────────────────────────
    # 检索 (SAGE模式)
    # ─────────────────────────────────────────────────────────────────

    def search(self, query: str) -> List[Document]:
        """
        在SAGE合成数据中进行检索
        
        替换原始文档检索为合成数据检索
        """
        docs = self.sage_engine.search(query, top_k=self.top_k, top_p=self.top_p)

        # 如果有reranker则重排
        if self.reranker and self.top_p > self.top_k:
            docs = self.reranker.rerank(query, docs, top_k=self.top_k)
        elif len(docs) > self.top_k:
            docs = docs[: self.top_k]

        self._stats["sage_retrieval_count"] += 1
        return docs

    # ─────────────────────────────────────────────────────────────────
    # 端到端回答
    # ─────────────────────────────────────────────────────────────────

    def answer(self, query: str) -> Tuple[str, Optional[List[Document]]]:
        """
        端到端 RAG + SAGE 生成
        
        流程:
        1. 意图检测
        2. SAGE合成数据检索
        3. 标准LLM生成回答 (无DP)
        4. 安全检测
        """
        # 1. 意图检测
        if not self.safety_check_query(query):
            return "Unknown.Intent", None

        # 2. 检索 (SAGE合成数据)
        context_docs = self.search(query)

        if not context_docs:
            print("[SAGE-RAG] Warning: No documents retrieved from SAGE index")
            return "I don't have enough information to answer.", None

        # 3. 构建prompt并生成回答 (标准方式，无DP)
        context_str = "\n\n".join(
            [f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(context_docs)]
        )

        # 选择提示模板
        if "HP1_5ch" in self.knowledge_path:
            prompt_template = pt.RAG_PROMPT_TEMPLATE_HP
        elif "HealthCareMagic" in self.knowledge_path:
            prompt_template = pt.RAG_PROMPT_TEMPLATE_HC
        else:
            prompt_template = pt.RAG_PROMPT_TEMPLATE_DEFAULT

        # 构建prompt
        prompt = prompt_template.format(context=context_str, question=query)
        
        # 生成回答 (标准LLM调用，无DP)
        try:
            response = self.llm.generate(prompt)
        except Exception as e:
            print(f"[SAGE-RAG] Generation failed: {e}")
            return "I encountered an error.", context_docs

        # 4. 安全检测
        if not self.safety_check_response(response, context_str):
            return "Unknown.Copy", None

        return response, context_docs

    # ─────────────────────────────────────────────────────────────────
    # 统计接口
    # ─────────────────────────────────────────────────────────────────

    def get_dp_stats(self) -> dict:
        """返回统计信息 (兼容DP_RAG接口)"""
        stats = self._stats.copy()
        if self.sage_engine:
            stats["sage_engine"] = self.sage_engine.get_dp_stats()
        return stats

    def get_stats(self) -> dict:
        """返回SAGE特定统计信息"""
        return self._stats.copy()

    def reset(self):
        """重置引擎状态"""
        self._stats = {
            "sage_retrieval_count": 0,
            "synthetic_mode": "sync",
        }
        if self.sage_engine:
            self.sage_engine.reset()


# =============================================================================
# 便捷函数
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
    创建SAGE RAG引擎的便捷函数
    
    用法示例:
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