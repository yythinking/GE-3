import torch
import numpy as np
from typing import List, Union
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from ..interfaces.rerank_interface import BaseReranker

class HFReranker(BaseReranker):
    def __init__(self, model_path: str, device: str = "cuda", max_length: int = 2048):
        """
        :param model_path: 本地路径或 HF Hub ID (e.g., "BAAI/bge-reranker-v2-m3")
        :param device: 'cuda', 'cpu', 'mps'
        """
        self.model_path = model_path
        self.device = device
        
        # print(f"正在加载 Rerank 模型 {model_path}...")

        self.model = CrossEncoder(
            model_name_or_path=model_path, 
            device=device,
            max_length=max_length 
        )
        # print("Rerank 模型加载完成。")

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        标准的 RAG Rerank 接口：接收 Document 对象列表
        """
        if not documents:
            return []

        # 1. 构造模型输入对: [[Query, Doc1], [Query, Doc2], ...]
        input_pairs = [[query, doc.page_content] for doc in documents]

        # 2. 推理打分
        scores = self.model.predict(input_pairs)

        # 3. 将分数绑定回文档，并排序
        results = sorted(
            zip(documents, scores), 
            key=lambda x: x[1], 
            reverse=True
        )

        # 4. 截取 Top K 并处理返回格式
        final_docs = []
        for doc, score in results[:top_k]:
            doc.metadata["relevance_score"] = float(score)
            final_docs.append(doc)

        return final_docs

    def compute_score(self, pairs: List[List[str]]) -> List[float]:
        """
        [新增方法] 直接计算文本对的相似度/相关性分数
        
        :param pairs: 二维列表，例如 [['Query', 'Text1'], ['Query', 'Text2']]
        :return: 分数列表 [score1, score2, ...]
        """
        if not pairs:
            return []
        
        # CrossEncoder.predict 返回 numpy array 或单个 float
        scores = self.model.predict(pairs)
        
        # 统一转换为 Python float list
        if isinstance(scores, np.ndarray):
            return scores.tolist()
        elif isinstance(scores, (float, int)):
            return [float(scores)]
        else:
            return [float(s) for s in scores]

    def get_model_info(self):
        return {
            "provider": "local_hf_cross_encoder",
            "model": self.model_path,
            "device": self.device
        }