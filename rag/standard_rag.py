# rag/standard_rag.py
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from . import prompts as pt

# 导入基类
from .base_engine import BaseEngine

class StandardRAGEngine(BaseEngine):
    """
    标准的 Embedding RAG 引擎实现 (Two-step RAG)
    集成安全检测流程
    """

    def __init__(self, llm, embedding, reranker, top_p: int, top_k: int, knowledge_path: str):
        super().__init__(llm, embedding, reranker, top_p, top_k, knowledge_path)
        self.vector_store = None
        self.collection_name = "rag_collection"

    # 检查索引是否存在
    def _check_index_exists(self, persist_dir: str) -> bool:
        sqlite_path = os.path.join(persist_dir, "chroma.sqlite3")
        return os.path.exists(sqlite_path)

    # 构建索引
    def _build_index(self, docs: list[Document], persist_dir: str):
        self.vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding, 
            persist_directory=persist_dir,
            collection_name=self.collection_name
        )

    # 加载索引
    def _load_index(self, persist_dir: str):
        # print(f"[EmbeddingRAG] Loading VectorDB from {persist_dir}...")
        self.vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding,
            collection_name=self.collection_name
        )

    # 检索并重排文档
    def search(self, query: str) -> list[Document]:
        if not self.vector_store:
            raise ValueError("Index not loaded. Please call engine.index() first.")

        # 先进行向量检索，得到 top_p 个候选文档
        candidates = self.vector_store.similarity_search(query, k=self.top_p)

        if self.reranker:
            final_docs = self.reranker.rerank(query, candidates, top_k=self.top_k)
        else:
            final_docs = candidates[:self.top_k]

        return final_docs

    # 端到端回答
    def answer(self, query: str) -> tuple[str, list[Document]]:
        """
        端到端 RAG：检索 + 生成 + 防御
        """
        # 意图检测
        if not self.safety_check_query(query):
            return "Unknown.Intent", None

        # 获取并拼接检索块
        context_docs = self.search(query)
        context_str = "\n\n".join(
            [f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(context_docs)]
        )

        # 构建最终的提示词
        if "HP1_5ch" in self.knowledge_path:
            final_prompt = pt.RAG_PROMPT_TEMPLATE_HP.format(
                context=context_str,
                question=query
            )
        elif "HealthCareMagic" in self.knowledge_path:
            final_prompt = pt.RAG_PROMPT_TEMPLATE_HC.format(
                context=context_str,
                question=query
            )
        else:
            final_prompt = pt.RAG_PROMPT_TEMPLATE_DEFAULT.format(
                context=context_str,
                question=query
            )

        # 调用 LLM 生成答案
        response = self.llm.generate(final_prompt)

        # 回答安全检测 避免逐字输出上下文
        if not self.safety_check_response(response, context_str):
            return "Unknown.Copy", None

        return response, context_docs