# Wrapper for calling locally deployed HuggingFace Embedding model

from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from ..interfaces.embedding_interface import BaseEmbedding

class LocalHFEmbedding(BaseEmbedding):
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        :param model_path: Local path or HF Hub ID (e.g. "BAAI/bge-m3")
        :param device: "cpu", "cuda", "mps" (Mac)
        """
        self.model_path = model_path
        self.device = device

        print(f"Loading Embedding model {model_path} to {device}...")
        
        # model_kwargs controls model running device
        # encode_kwargs controls encoding process, e.g., whether to normalize vectors
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}        # Recommended, especially for cosine similarity
        )
        print("Embedding model loaded successfully.")

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.embed_documents(texts)

    def get_model_info(self) -> Dict[str, Any]:

        # Extract model name, return part after slash
        model_name_only = self.model_path.split("/")[-1]

        # Simplify base_url to keep only domain part
        base_url_only = self.device.split("//")[-1].split("/")[0]

        return {
            "provider": "local_huggingface",
            "model": model_name_only,
            "device": base_url_only
        }

    def get_dimension(self) -> int:
        # sentence-transformers models make it easy to get dimension
        # embedding_model._client points to the underlying sentence_transformers.SentenceTransformer object
        return self.embedding_model._client.get_sentence_embedding_dimension()