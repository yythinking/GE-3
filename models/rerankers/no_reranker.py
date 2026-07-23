from typing import List
from langchain_core.documents import Document
from ..interfaces.rerank_interface import BaseReranker

class NoReranker(BaseReranker):
    """
    No-op Reranker (Pass-through)
    Does not perform any model inference, simply truncates retrieval results to Top-K.
    Used for:
    1. Debugging to exclude Reranker interference
    2. Scenarios requiring maximum speed without fine-grained ranking
    """
    def __init__(self):
        # No model loading needed, extremely fast startup
        pass

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        Directly return top K documents
        """
        # Python slicing is safe even if list length is less than top_k
        return documents[:top_k]

    def get_model_info(self) -> dict:
        return {
            "provider": "no_reranker",
            "model": "none",
            "status": "pass_through"
        }