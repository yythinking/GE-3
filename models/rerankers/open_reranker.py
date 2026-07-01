# open_reranker.py
import requests
from typing import List, Dict, Any
from langchain_core.documents import Document
from ..interfaces.rerank_interface import BaseReranker

class OpenReranker(BaseReranker):
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://uni-api.cstcloud.cn/v1/rerank"):
        """
        初始化云端重排序模型封装
        
        :param model_name: 模型名称，如 'bge-reranker-v2-m3'
        :param api_key: 平台的 API Key
        :param base_url: Rerank 接口完整地址
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        调用云端 API 对文档进行重排序
        """
        if not documents:
            return []

        # 1. 准备请求数据：提取 Document 中的文本内容
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
            # 2. 发起 API 调用
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # 3. 解析结果并重组 Document 对象
            # 假设返回格式为 {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
            results = data.get("results", [])
            
            reranked_docs = []
            for item in results:
                original_idx = item["index"]
                score = item["relevance_score"]
                
                # 获取原文档对象并注入相关性分数
                doc = documents[original_idx]
                doc.metadata["relevance_score"] = score
                reranked_docs.append(doc)
                
            return reranked_docs[:top_k]

        except requests.exceptions.RequestException as e:
            print(f"Rerank API Error: {e}")
            raise

    def get_model_info(self) -> dict:
        """返回模型元数据"""
        return {
            "provider": "open_reranker",
            "model": self.model_name,
            "endpoint": self.base_url
        }