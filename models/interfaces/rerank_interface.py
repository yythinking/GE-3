from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document

class BaseReranker(ABC):
    """
    Rerank Model Base Class
    """
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        Core method:
        1. Receive query and a set of documents
        2. Calculate relevance scores
        3. Sort in descending order by score
        4. Return top_k documents
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        pass