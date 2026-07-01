# 使用云端 Embedding 模型的封装
# 兼容 OpenAI Embedding API 的平台也适用
# 不兼容 OpenAI API 的平台请参考 响应 API 手册 自行设计

import requests
from typing import List, Dict, Any
from ..interfaces.embedding_interface import BaseEmbedding


class OpenEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://uni-api.cstcloud.cn/v1/embeddings"):
        """
        初始化云端 Embedding 封装
        
        :param model_name: 模型名称，如 'text-embedding-v1'
        :param api_key: 平台的 API Key
        :param base_url: API 基础地址，默认为 OpenAI 兼容的 embeddings 路径
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self._dimension = None

    def embed_query(self, text: str) -> List[float]:
        """对单个查询进行向量化"""
        return self._call_embedding_api([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量对文档进行向量化"""
        return self._call_embedding_api(texts)

    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """执行实际的 HTTP 请求"""
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
            
            # 解析 OpenAI 兼容格式: data["data"] 是一个包含 embedding 键的字典列表
            # 排序确保返回顺序与输入一致
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling embedding API: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型配置信息"""
        return {
            "provider": "open_embedding",
            "model": self.model_name,
            "base_url": self.base_url
        }

    def get_dimension(self) -> int:
        """获取向量维度（缓存结果以减少 API 调用）"""
        if self._dimension is None:
            # 使用一个极短的文本进行探测
            dummy_vec = self.embed_query("test")
            self._dimension = len(dummy_vec)
        return self._dimension