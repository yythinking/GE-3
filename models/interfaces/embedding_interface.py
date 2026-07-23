from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseEmbedding(ABC):
    """
    Embedding Model Abstract Base Class
    """

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Vectorize a single user query
        Note: Some models handle Query and Document differently (e.g., adding instruction prefixes)
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Batch vectorize document list
        Used for building knowledge base index
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get vector dimension (e.g., 768, 1024)
        This is crucial for creating vector database Collection
        """
        pass