from typing import List, Dict, Any
from langchain_ollama import OllamaEmbeddings
from ..interfaces.embedding_interface import BaseEmbedding

class OllamaEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
        # 初始化 LangChain 的 Ollama 客户端
        self.client = OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )

    def embed_query(self, text: str) -> List[float]:
        """调用 Ollama 生成单个查询向量"""
        return self.client.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """调用 Ollama 生成文档批处理向量"""
        return self.client.embed_documents(texts)

    def get_model_info(self) -> Dict[str, Any]:
        
        # 对模型名称进行提取，返回冒号前的部分
        model_name_only = self.model_name.split(":")[0]

        # 对 base_url 进行简化，只保留域名部分
        base_url_only = self.base_url.split("//")[-1].split("/")[0]
        # 去除端口号
        base_url_only = base_url_only.split(':')[0]

        return {
            "provider": "ollama",
            "model": model_name_only,
            "base_url": base_url_only
        }

    def get_dimension(self) -> int:
        """
        获取维度的技巧：试运行一次
        Ollama API 没有直接提供获取维度的端点，通常需要在运行时探测
        """
        # 为了避免每次都请求，可以考虑缓存这个值
        # 这里为了演示简单，做一次实际调用
        dummy_vec = self.client.embed_query("test")
        return len(dummy_vec)