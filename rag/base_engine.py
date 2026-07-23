import os
import re
from typing import Any, Dict, Tuple
from abc import ABC, abstractmethod
from langchain_core.documents import Document

from models.interfaces.llm_interface import BaseLLM
from models.interfaces.embedding_interface import BaseEmbedding
from models.interfaces.rerank_interface import BaseReranker
from src.data_loader import DatasetLoader
from . import prompts as pt

class BaseEngine(ABC):
    """
    Abstract base class for all RAG engines (Standard RAG, Graph RAG, etc.)
    Integrated with Security Guardrails
    """

    def __init__(
        self, 
        llm: BaseLLM, 
        embedding: BaseEmbedding, 
        reranker: BaseReranker,
        top_p: int,   # Retrieval count
        top_k: int,   # Rerank count
        knowledge_path: str,
    ):
        self.llm = llm
        self.embedding = embedding
        self.reranker = reranker
        self.top_p = top_p
        self.top_k = top_k
        self.data_loader = DatasetLoader()
        self.knowledge_path = knowledge_path
        
        # Security configuration threshold
        self.rouge_threshold = 0.4  # Leakage threshold: block if generated answer exceeds 40% similarity with retrieved chunks
        
        # Define storage root directory
        self.storage_root = "storage"
        if not os.path.exists(self.storage_root):
            os.makedirs(self.storage_root)

    # Get index storage path
    # Construct unique path based on engine type, dataset name, and Embedding model info
    def get_index_path(self) -> str:
        engine_type = self.__class__.__name__.lower().replace("engine", "").replace("rag", "_rag")
        dataset_name = os.path.basename(self.knowledge_path).split('.')[0]
        embed_info = self.embedding.get_model_info()
        model_name = embed_info.get("model", "default_model")
        dir_name = f"{dataset_name}_{model_name}"
        full_path = os.path.join(self.storage_root, engine_type, dir_name)
        return full_path

    # Build or load index
    def index(self):
        persist_dir = self.get_index_path()
        print(f"Target Index Path: {persist_dir}")

        # Check if index exists
        # Load if exists, otherwise build
        if self._check_index_exists(persist_dir):
            print(f"Index found. Loading...")
            self._load_index(persist_dir)
        else:
            print(f"No index found. Building from {persist_dir}...")
            docs = self.data_loader.load_dataset(persist_dir)
            self._build_index(docs, persist_dir)
            print(f"[{self.__class__.__name__}] Index built and saved to {persist_dir}")

    # Input detection
    def safety_check_query(self, query: str) -> bool:
        """
        Input guardrail: detect malicious intent
        Returns: True (safe), False (unsafe)
        """
        # 1. Rule-based detection (illegal statements)
        if not self._detect_intent_keywords(query):
            print("[Security] Blocked by keyword filter.")
            return False

        return True

    # Output detection
    def safety_check_response(self, response: str, context_str: str) -> bool:
        """
        Output guardrail: detect data leakage
        Returns: True (safe), False (leakage detected)
        """
        # Character-by-character leakage detection (Rouge-L)
        f1_score = self._calculate_rouge_l_f1(response, context_str)
        if f1_score > self.rouge_threshold:
            print(f"[Security] Blocked by Leakage Detection. Rouge-L F1: {f1_score:.4f} > {self.rouge_threshold}")
            return False
        return True

    # Illegal statement detection
    def _detect_intent_keywords(self, query: str) -> bool:
        """Simple keyword blacklist matching"""
        blacklist = [
            "ignore previous instructions", "system prompt", "upstream instructions",
            "forget your rules", "repeat the following", "output your initialization"
        ]
        query_lower = query.lower()
        for term in blacklist:
            if term in query_lower:
                return False
        return True
    
    # LLM intent detection
    def _detect_intent_llm(self, query: str) -> bool:
        """Use LLM to determine if user attempts Prompt Injection"""

        prompt = pt.INTENT_DETECTION_PROMPT.format(query=query)
        
        try:
            # Use simple generate call
            judgment = self.llm.generate(prompt).strip().lower()
            is_MALICIOUS = "yes" in judgment
            return  not is_MALICIOUS  # Return True means safe
        except Exception as e:
            print(f"[Warning] Intent check failed: {e}. Defaulting to safe.")
            return True

    def _calculate_rouge_l_f1(self, prediction: str, target: str) -> float:
        """
        Calculate Rouge-L F1 score (Longest Common Subsequence)
        Manual implementation to avoid rouge-score dependency, ensure environment compatibility.
        """
        # Simple tokenization based on characters or words
        def tokenize(text):
            return re.findall(r'\w+', text.lower())

        pred_tokens = tokenize(prediction)
        target_tokens = tokenize(target)

        if not pred_tokens or not target_tokens:
            return 0.0

        # Dynamic programming to compute LCS length
        m, n = len(pred_tokens), len(target_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i - 1] == target_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        lcs_len = dp[m][n]

        # Calculate Precision, Recall, F1
        precision = lcs_len / m if m > 0 else 0
        recall = lcs_len / n if n > 0 else 0
        
        if precision + recall == 0:
            return 0.0
            
        f1 = 2 * precision * recall / (precision + recall)
        return f1


    @abstractmethod
    def _check_index_exists(self, persist_dir: str) -> bool:
        pass

    @abstractmethod
    def _build_index(self, docs: list[Any], persist_dir: str):
        pass

    @abstractmethod
    def _load_index(self, persist_dir: str):
        pass

    @abstractmethod
    def search(self, query: str) -> list[Any]:
        pass

    @abstractmethod
    def answer(self, query: str) -> tuple[str, list[Document]]:
        pass