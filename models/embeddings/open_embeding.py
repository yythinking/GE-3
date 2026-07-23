# Wrapper for cloud Embedding model
# Compatible with platforms that support OpenAI Embedding API
# For platforms not compatible with OpenAI API, please refer to the API documentation and design accordingly

import requests
from typing import List, Dict, Any
from ..interfaces.embedding_interface import BaseEmbedding


class OpenEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://uni-api.cstcloud.cn/v1/embeddings"):
        """
        Initialize cloud Embedding wrapper
        
        :param model_name: Model name, e.g., 'text-embedding-v1'
        :param api_key: Platform API Key
        :param base_url: API base address, default is OpenAI compatible embeddings path
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self._dimension = None

    def embed_query(self, text: str) -> List[float]:
        """Vectorize a single query"""
        return self._call_embedding_api([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch vectorize documents"""
        return self._call_embedding_api(texts)

    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """Execute actual HTTP request"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "input": texts
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Parse OpenAI compatible format: data["data"] is a list of dicts containing embedding key
            # Sort to ensure return order matches input
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling embedding API: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model configuration information"""
        return {
            "provider": "open_embedding",
            "model": self.model_name,
            "base_url": self.base_url
        }

    def get_dimension(self) -> int:
        """Get vector dimension (cache result to reduce API calls)"""
        if self._dimension is None:
            # Use a very short text for detection
            dummy_vec = self.embed_query("test")
            self._dimension = len(dummy_vec)
        return self._dimension