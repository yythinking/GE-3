# 调用本地部署 HuggingFace Embedding 模型的封装

from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from ..interfaces.embedding_interface import BaseEmbedding

class LocalHFEmbedding(BaseEmbedding):
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        :param model_path: 本地路径 或 HF Hub ID (e.g. "BAAI/bge-m3")
        :param device: "cpu", "cuda", "mps" (Mac)
        """
        self.model_path = model_path
        self.device = device

        print(f"正在加载 Embedding 模型 {model_path} 到 {device}...")
        
        # model_kwargs 用于控制模型运行设备
        # encode_kwargs 用于控制编码过程，比如是否归一化向量
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}        # 推荐开启，特别是对于余弦相似度
        )
        print("Embedding 模型加载完成。")

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.embed_documents(texts)

    def get_model_info(self) -> Dict[str, Any]:

        # 对模型名称进行提取，返回斜杠后面的部分
        model_name_only = self.model_path.split("/")[-1]

        # 对 base_url 进行简化，只保留域名部分
        base_url_only = self.device.split("//")[-1].split("/")[0]

        return {
            "provider": "local_huggingface",
            "model": model_name_only,
            "device": base_url_only
        }

    def get_dimension(self) -> int:
        # sentence-transformers 的模型很容易获取维度
        # embedding_model._client 指向底层的 sentence_transformers.SentenceTransformer 对象
        return self.embedding_model._client.get_sentence_embedding_dimension()