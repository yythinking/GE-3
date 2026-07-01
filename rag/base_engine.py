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
    所有 RAG 引擎的抽象基类 (Standard RAG, Graph RAG, etc.)
    集成安全护栏 (Security Guardrails)
    """

    def __init__(
        self, 
        llm: BaseLLM, 
        embedding: BaseEmbedding, 
        reranker: BaseReranker,
        top_p: int,   # 检索数
        top_k: int,   # 重排数
        knowledge_path: str,
    ):
        self.llm = llm
        self.embedding = embedding
        self.reranker = reranker
        self.top_p = top_p
        self.top_k = top_k
        self.data_loader = DatasetLoader()
        self.knowledge_path = knowledge_path
        
        # 安全配置阈值
        self.rouge_threshold = 0.4  # 泄露阈值：如果生成的答案与检索块重复度超过 40% 则拦截
        
        # 定义存储根目录
        self.storage_root = "storage"
        if not os.path.exists(self.storage_root):
            os.makedirs(self.storage_root)

    # 获取索引存储路径
    # 根据引擎类型、数据集名称、Embedding 模型信息构建唯一路径
    def get_index_path(self) -> str:
        engine_type = self.__class__.__name__.lower().replace("engine", "").replace("rag", "_rag")
        dataset_name = os.path.basename(self.knowledge_path).split('.')[0]
        embed_info = self.embedding.get_model_info()
        model_name = embed_info.get("model", "default_model")
        dir_name = f"{dataset_name}_{model_name}"
        full_path = os.path.join(self.storage_root, engine_type, dir_name)
        return full_path

    # 构建或加载索引
    def index(self):
        persist_dir = self.get_index_path()
        print(f"Target Index Path: {persist_dir}")

        # 检查索引是否存在
        # 如果存在则加载，否则构建
        if self._check_index_exists(persist_dir):
            print(f"Index found. Loading...")
            self._load_index(persist_dir)
        else:
            print(f"No index found. Building from {persist_dir}...")
            docs = self.data_loader.load_dataset(persist_dir)
            self._build_index(docs, persist_dir)
            print(f"[{self.__class__.__name__}] Index built and saved to {persist_dir}")

    # 输入检测
    def safety_check_query(self, query: str) -> bool:
        """
        输入护栏：检测恶意意图
        返回: True (安全), False (不安全)
        """
        # 1. 规则检测 (非法语句)
        if not self._detect_intent_keywords(query):
            print("[Security] Blocked by keyword filter.")
            return False
            
        
        # # 2. LLM 意图判断 
        # # 待更新提示词
        # if not self._detect_intent_llm(query):
        #     print("[Security] Blocked by LLM intent detection.")
        #     return False
            
        return True

    # 输出检测
    def safety_check_response(self, response: str, context_str: str) -> bool:
        """
        输出护栏：检测数据泄露
        返回: True (安全), False (存在泄露)
        """
        # 逐字泄露检测 (Rouge-L)
        f1_score = self._calculate_rouge_l_f1(response, context_str)
        if f1_score > self.rouge_threshold:
            print(f"[Security] Blocked by Leakage Detection. Rouge-L F1: {f1_score:.4f} > {self.rouge_threshold}")
            return False
        return True

    # 非法语句检测
    def _detect_intent_keywords(self, query: str) -> bool:
        """简单的关键词黑名单匹配"""
        blacklist = [
            "ignore previous instructions", "system prompt", "upstream instructions",
            "forget your rules", "repeat the following", "output your initialization"
        ]
        query_lower = query.lower()
        for term in blacklist:
            if term in query_lower:
                return False
        return True
    
    # LLM 意图检测
    def _detect_intent_llm(self, query: str) -> bool:
        """利用 LLM 判断用户是否尝试进行 Prompt Injection"""

        prompt = pt.INTENT_DETECTION_PROMPT.format(query=query)
        
        try:
            # 使用简单的 generate 调用
            judgment = self.llm.generate(prompt).strip().lower()
            is_MALICIOUS = "yes" in judgment
            return  not is_MALICIOUS  # 返回 True 表示安全
        except Exception as e:
            print(f"[Warning] Intent check failed: {e}. Defaulting to safe.")
            return True

    def _calculate_rouge_l_f1(self, prediction: str, target: str) -> float:
        """
        计算 Rouge-L F1 分数 (最长公共子序列)
        手动实现以避免引入 rouge-score 依赖，确保环境兼容性。
        """
        # 简单的基于字符或单词的切分
        def tokenize(text):
            return re.findall(r'\w+', text.lower())

        pred_tokens = tokenize(prediction)
        target_tokens = tokenize(target)

        if not pred_tokens or not target_tokens:
            return 0.0

        # 动态规划计算 LCS 长度
        m, n = len(pred_tokens), len(target_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i - 1] == target_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        lcs_len = dp[m][n]

        # 计算 Precision, Recall, F1
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