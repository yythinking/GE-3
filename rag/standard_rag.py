import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from . import prompts as pt

# Import base class
from .base_engine import BaseEngine

class StandardRAGEngine(BaseEngine):
    """
    Standard Embedding RAG Engine Implementation (Two-step RAG)
    Integrated with security detection flow
    """

    def __init__(self, llm, embedding, reranker, top_p: int, top_k: int, knowledge_path: str):
        super().__init__(llm, embedding, reranker, top_p, top_k, knowledge_path)
        self.vector_store = None
        self.collection_name = "rag_collection"

    # Check if index exists
    def _check_index_exists(self, persist_dir: str) -> bool:
        sqlite_path = os.path.join(persist_dir, "chroma.sqlite3")
        return os.path.exists(sqlite_path)

    # Build index
    def _build_index(self, docs: list[Document], persist_dir: str):
        self.vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding, 
            persist_directory=persist_dir,
            collection_name=self.collection_name
        )

    # Load index
    def _load_index(self, persist_dir: str):
        # print(f"[EmbeddingRAG] Loading VectorDB from {persist_dir}...")
        self.vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding,
            collection_name=self.collection_name
        )

    # Retrieve and rerank documents
    def search(self, query: str) -> list[Document]:
        if not self.vector_store:
            raise ValueError("Index not loaded. Please call engine.index() first.")

        # First perform vector retrieval, get top_p candidate documents
        candidates = self.vector_store.similarity_search(query, k=self.top_p)

        if self.reranker:
            final_docs = self.reranker.rerank(query, candidates, top_k=self.top_k)
        else:
            final_docs = candidates[:self.top_k]

        return final_docs

    # End-to-end answer
    def answer(self, query: str) -> tuple[str, list[Document]]:
        """
        End-to-end RAG: Retrieval + Generation + Defense
        """
        # Intent detection
        if not self.safety_check_query(query):
            return "Unknown.Intent", None

        # Retrieve and concatenate retrieved chunks
        context_docs = self.search(query)
        context_str = "\n\n".join(
            [f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(context_docs)]
        )

        # Build final prompt
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

        # Call LLM to generate answer
        response = self.llm.generate(final_prompt)

        # Answer safety check to avoid verbatim output of context
        if not self.safety_check_response(response, context_str):
            return "Unknown.Copy", None

        return response, context_docs