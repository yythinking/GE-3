from typing import List, Dict, Any
from langchain_ollama import OllamaEmbeddings
from ..interfaces.embedding_interface import BaseEmbedding

class OllamaEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
        # Initialize LangChain's Ollama client
        self.client = OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )

    def embed_query(self, text: str) -> List[float]:
        """Call Ollama to generate single query vector"""
        return self.client.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call Ollama to generate batch document vectors"""
        return self.client.embed_documents(texts)

    def get_model_info(self) -> Dict[str, Any]:
        
        # Extract model name, return part before colon
        model_name_only = self.model_name.split(":")[0]

        # Simplify base_url to keep only domain part
        base_url_only = self.base_url.split("//")[-1].split("/")[0]
        # Remove port number
        base_url_only = base_url_only.split(':')[0]

        return {
            "provider": "ollama",
            "model": model_name_only,
            "base_url": base_url_only
        }

    def get_dimension(self) -> int:
        """
        Trick for getting dimension: run a trial
        Ollama API does not directly provide an endpoint for getting dimension, usually needs to be detected at runtime
        """
        # To avoid requesting every time, consider caching this value
        # Here for simplicity, make one actual call
        dummy_vec = self.client.embed_query("test")
        return len(dummy_vec)