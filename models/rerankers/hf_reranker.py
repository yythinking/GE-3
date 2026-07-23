import torch
import numpy as np
from typing import List, Union
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from ..interfaces.rerank_interface import BaseReranker

class HFReranker(BaseReranker):
    def __init__(self, model_path: str, device: str = "cuda", max_length: int = 2048):
        """
        :param model_path: Local path or HF Hub ID (e.g., "BAAI/bge-reranker-v2-m3")
        :param device: 'cuda', 'cpu', 'mps'
        """
        self.model_path = model_path
        self.device = device

        self.model = CrossEncoder(
            model_name_or_path=model_path, 
            device=device,
            max_length=max_length 
        )

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        Standard RAG Rerank interface: receives list of Document objects
        """
        if not documents:
            return []

        # 1. Construct model input pairs: [[Query, Doc1], [Query, Doc2], ...]
        input_pairs = [[query, doc.page_content] for doc in documents]

        # 2. Inference scoring
        scores = self.model.predict(input_pairs)

        # 3. Bind scores back to documents and sort
        results = sorted(
            zip(documents, scores), 
            key=lambda x: x[1], 
            reverse=True
        )

        # 4. Take Top K and handle return format
        final_docs = []
        for doc, score in results[:top_k]:
            doc.metadata["relevance_score"] = float(score)
            final_docs.append(doc)

        return final_docs

    def compute_score(self, pairs: List[List[str]]) -> List[float]:
        """
        [New method] Directly compute similarity/relevance scores for text pairs
        
        :param pairs: 2D list, e.g., [['Query', 'Text1'], ['Query', 'Text2']]
        :return: Score list [score1, score2, ...]
        """
        if not pairs:
            return []
        
        # CrossEncoder.predict returns numpy array or single float
        scores = self.model.predict(pairs)
        
        # Uniformly convert to Python float list
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