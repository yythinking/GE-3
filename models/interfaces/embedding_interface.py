from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseEmbedding(ABC):
    """
    Embedding 模型抽象基类
    """

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        对用户的单个查询问题进行向量化
        注意：某些模型对 Query 和 Document 的处理方式不同（如添加指令前缀）
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        对文档列表进行批量向量化
        用于构建知识库索引
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型配置信息"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        获取向量维度 (如 768, 1024)
        这对于创建向量数据库 Collection 至关重要
        """
        pass