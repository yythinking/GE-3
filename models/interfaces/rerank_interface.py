from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document

class BaseReranker(ABC):
    """
    Rerank (重排序) 模型基类
    """
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        核心方法：
        1. 接收查询和一组文档
        2. 计算相关性得分
        3. 按得分降序排列
        4. 返回前 top_k 个文档
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        pass