# open_reranker.py
import requests
from typing import List, Dict, Any
from langchain_core.documents import Document
from ..interfaces.rerank_interface import BaseReranker

class OpenReranker(BaseReranker):
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://uni-api.cstcloud.cn/v1/rerank"):
        """
        Initialize cloud rerank model wrapper
        
        :param model_name: Model name, e.g., 'bge-reranker-v2-m3'
        :param api_key: Platform API Key
        :param base_url: Full Rerank API endpoint address
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        Call cloud API to rerank documents
        """
        if not documents:
            return []

        # 1. Prepare request data: extract text content from Documents
        texts = [doc.page_content for doc in documents]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": top_k
        }

        try:
            # 2. Make API call
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # 3. Parse results and reconstruct Document objects
            # Assume return format: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
            results = data.get("results", [])
            
            reranked_docs = []
            for item in results:
                original_idx = item["index"]
                score = item["relevance_score"]
                
                # Get original document object and inject relevance score
                doc = documents[original_idx]
                doc.metadata["relevance_score"] = score
                reranked_docs.append(doc)
                
            return reranked_docs[:top_k]

        except requests.exceptions.RequestException as e:
            print(f"Rerank API Error: {e}")
            raise

    def get_model_info(self) -> dict:
        """Return model metadata"""
        return {
            "provider": "open_reranker",
            "model": self.model_name,
            "endpoint": self.base_url
        }