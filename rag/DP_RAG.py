# 0501_XR_MtRAG/rag/DP_RAG.py
"""
差分隐私 RAG 引擎
基于 DPRAG 项目的 DP VoteRAG 机制实现

核心思路（对齐 DPRAG 原生实现）：
1. 检索文档后，将文档随机打乱，构建 n_split 个 prompt 变体
2. 每个 prompt 变体独立调用 LLM 生成回答
3. 对所有回答进行 tokenization 对齐
4. 在 token 级别使用 LDGumbel DP majority vote 选择最终 token
5. Detokenize 得到最终回答
6. 追踪 DP 隐私预算
"""

import os
import random
from typing import List, Tuple, Optional, Dict
from langchain_core.documents import Document
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .base_engine import BaseEngine
from . import prompts as pt
from .dp_mechanisms import (
    LDGumbelMechanism,
    majority_vote,
    NotFoundLDTop1,
    DPExpenseOverflow
)


class DPRAGEngine(BaseEngine):
    """
    差分隐私 RAG 引擎（对齐 DPRAG 原生 VoteRAG 机制）

    在答案生成阶段使用 DP 机制保护隐私：
    1. 检索文档后，通过打乱文档顺序构建 n_split 个 prompt 变体
    2. 每个 prompt 变体独立调用 LLM 生成回答（voter）
    3. 对所有 voter 回答进行 tokenization
    4. 在每个 token 位置使用 DP majority vote 选择最终 token
    5. Detokenize 得到最终回答
    6. 追踪 DP 隐私预算消耗

    参数:
        n_split: prompt 变体数量 / voter 数量 (默认 50)
        dp_eps: 每个 token 的 DP epsilon (默认 2.0)
        dp_delta: 每个 token 的 DP delta (默认 1e-5)
        target_eps: 总 epsilon 预算上限 (默认 1000.0)
        target_delta: 总 delta 预算上限 (默认 1.0)
        max_tokens: 生成的最大 token 数 (默认 100)
        fail_mode: DP 失败处理模式 (默认 'ld_pate')
        use_parallel: 是否使用并发执行 voter 生成 (默认 True)
        max_workers: 并发执行的最大 worker 数量 (默认 20)
    """

    def __init__(
        self,
        llm,
        embedding,
        reranker,
        top_p: int,
        top_k: int,
        knowledge_path: str,
        n_split: int = 50,
        dp_eps: float = 2.0,
        dp_delta: float = 1e-5,
        target_eps: float = 1000.0,
        target_delta: float = 1.0,
        max_tokens: int = 100,
        fail_mode: str = 'ld_pate',
        use_parallel: bool = True,
        max_workers: int = 20,
    ):
        super().__init__(llm, embedding, reranker, top_p, top_k, knowledge_path)

        self.n_split = n_split
        self.max_tokens = max_tokens
        self.fail_mode = fail_mode
        self.use_parallel = use_parallel
        self.max_workers = max_workers

        # DP 机制：k_bar 与 n_split 对齐（DPRAG 原生设置）
        self._dp_engine = LDGumbelMechanism(
            eps=dp_eps,
            delta=dp_delta,
            k_bar=n_split,
            target_eps=target_eps,
            target_delta=target_delta,
            fail_mode=fail_mode,
        )

        # 词表大小（从 LLM 获取）
        self._vocab_size = self.llm.get_vocab_size() if hasattr(self.llm, 'get_vocab_size') else 100256

        # 追踪 DP 使用情况
        self._dp_stats = {
            'total_tokens_generated': 0,
            'total_vote_calls': 0,
            'dp_failures': 0,
            'dp_budget_exceeded': 0,
        }

        self.vector_store = None
        self.collection_name = "dp_rag_collection"

    # ──────────────────────────────────────────────
    # 索引管理（与 StandardRAGEngine 一致）
    # ──────────────────────────────────────────────

    def _check_index_exists(self, persist_dir: str) -> bool:
        sqlite_path = os.path.join(persist_dir, "chroma.sqlite3")
        return os.path.exists(sqlite_path)

    def _build_index(self, docs: List[Document], persist_dir: str):
        from langchain_chroma import Chroma
        self.vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding,
            persist_directory=persist_dir,
            collection_name=self.collection_name
        )

    def _load_index(self, persist_dir: str):
        from langchain_chroma import Chroma
        self.vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding,
            collection_name=self.collection_name
        )

    # ──────────────────────────────────────────────
    # 检索（与 StandardRAGEngine 一致）
    # ──────────────────────────────────────────────

    def search(self, query: str) -> List[Document]:
        if not self.vector_store:
            raise ValueError("Index not loaded.")

        candidates = self.vector_store.similarity_search(query, k=self.top_p)

        if self.reranker:
            final_docs = self.reranker.rerank(query, candidates, top_k=self.top_k)
        else:
            final_docs = candidates[:self.top_k]

        return final_docs

    # ──────────────────────────────────────────────
    # 端到端回答
    # ──────────────────────────────────────────────

    def answer(self, query: str) -> Tuple[str, List[Document]]:
        """
        端到端 RAG + DP 生成

        流程：
        1. 意图检测
        2. 检索相关文档
        3. 构建 n_split 个 prompt 变体（打乱文档顺序）
        4. 使用 DP ensemble 生成答案
        5. 安全检测
        """
        # 1. 意图检测
        if not self.safety_check_query(query):
            return "Unknown.Intent", None

        # 2. 检索
        context_docs = self.search(query)

        # 3. 构建 n_split 个 prompt 变体
        prompt_list = self._build_prompt_variants(query, context_docs)

        # 4. DP ensemble 生成
        response = self._dp_ensemble_generate(prompt_list)

        # 5. 安全检测
        context_str = "\n\n".join([
            f"Document {i+1}: {doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])
        if not self.safety_check_response(response, context_str):
            return "Unknown.Copy", None

        return response, context_docs

    # ──────────────────────────────────────────────
    # Prompt 变体构建（DPRAG 原生方式）
    # ──────────────────────────────────────────────

    def _build_prompt_variants(self, query: str, docs: List[Document]) -> List[str]:
        """
        构建 n_split 个 prompt 变体

        对齐 DPRAG 原生实现：
        - 将检索到的文档内容提取出来
        - 随机打乱文档顺序
        - 将文档分成 n_split 组，每组包含若干文档
        - 为每组构建一个独立的 prompt

        当 n_split > 文档数时，每个 voter 仍然获得所有文档，
        但文档顺序不同（通过多次独立打乱实现）。
        """
        # 提取文档文本
        doc_texts = [doc.page_content for doc in docs]

        # 选择提示模板
        if "HP1_5ch" in self.knowledge_path:
            prompt_template = pt.RAG_PROMPT_TEMPLATE_HP
        elif "HealthCareMagic" in self.knowledge_path:
            prompt_template = pt.RAG_PROMPT_TEMPLATE_HC
        else:
            prompt_template = pt.RAG_PROMPT_TEMPLATE_DEFAULT

        prompt_list = []

        if len(doc_texts) == 0:
            # 无文档时，所有 voter 使用相同的无上下文 prompt
            for _ in range(self.n_split):
                prompt_list.append(prompt_template.format(context="", question=query))
            return prompt_list

        if self.n_split <= len(doc_texts):
            # voter 数 <= 文档数：每个 voter 获得不同的文档子集
            # 将文档分成 n_split 组
            shuffled_texts = doc_texts.copy()
            random.shuffle(shuffled_texts)

            docs_per_split = max(1, len(shuffled_texts) // self.n_split)
            for split_id in range(self.n_split):
                start = split_id * docs_per_split
                end = min(start + docs_per_split, len(shuffled_texts))
                # 最后一个 split 取剩余所有文档
                if split_id == self.n_split - 1:
                    end = len(shuffled_texts)

                subset = shuffled_texts[start:end]
                context_str = "\n\n".join([
                    f"Document {i+1}: {text}"
                    for i, text in enumerate(subset)
                ])
                prompt_list.append(
                    prompt_template.format(context=context_str, question=query)
                )
        else:
            # voter 数 > 文档数：每个 voter 获得所有文档，但顺序不同
            for _ in range(self.n_split):
                shuffled = doc_texts.copy()
                random.shuffle(shuffled)
                context_str = "\n\n".join([
                    f"Document {i+1}: {text}"
                    for i, text in enumerate(shuffled)
                ])
                prompt_list.append(
                    prompt_template.format(context=context_str, question=query)
                )

        return prompt_list

    # ──────────────────────────────────────────────
    # DP Ensemble 生成（核心机制）
    # ──────────────────────────────────────────────

    def _voter_generate(self, prompt: str, voter_id: int) -> Tuple[int, str]:
        """
        单个 voter 的 LLM 生成任务（用于并发执行）

        Args:
            prompt: 输入的 prompt
            voter_id: voter 编号

        Returns:
            (voter_id, 生成的回答)
        """
        try:
            response = self.llm.generate(prompt)
            return (voter_id, response)
        except Exception as e:
            print(f"[DP-RAG] Voter {voter_id} generation failed: {e}")
            return (voter_id, "")

    def _dp_ensemble_generate(self, prompt_list: List[str]) -> str:
        """
        使用 DP 机制进行 ensemble 生成

        对齐 DPRAG 原生实现流程：
        1. 每个 prompt 变体独立调用 LLM 生成回答（支持并发）
        2. Tokenize 所有回答
        3. 在每个 token 位置使用 DP majority vote
        4. Detokenize 得到最终回答

        注意：DPRAG 原生使用 vllm 进行 token-by-token 生成，
        我们的 LLM 接口只支持 generate(prompt) -> str，
        因此采用"先生成完整回答，后 DP 投票"的策略。
        """
        # 1. 每个 voter 独立生成回答（支持并发加速）
        candidates = [None] * len(prompt_list)

        if self.use_parallel:
            # 并发模式：使用 ThreadPoolExecutor 并行调用所有 voter
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._voter_generate, prompt, i): i
                    for i, prompt in enumerate(prompt_list)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        voter_id, response = future.result()
                        candidates[idx] = response
                    except Exception as e:
                        print(f"[DP-RAG] Voter {idx} failed: {e}")
                        candidates[idx] = ""
        else:
            # 串行模式：保持原有逻辑
            for i, prompt in enumerate(prompt_list):
                try:
                    response = self.llm.generate(prompt)
                    candidates[i] = response
                except Exception as e:
                    print(f"[DP-RAG] Voter {i} generation failed: {e}")
                    candidates[i] = ""

        # 过滤掉空候选
        non_empty_candidates = [c for c in candidates if c and c.strip()]
        if not non_empty_candidates:
            return ""
        empty_count = len([c for c in candidates if not c or not c.strip()])
        if empty_count > 0:
            print(f"[DP-RAG] {empty_count} voters produced empty responses (parallel={self.use_parallel})")

        # 2. Tokenize 所有候选
        tokenized_candidates = []
        for candidate in non_empty_candidates:
            try:
                tokens = self.llm.tokenize(candidate)
                # 截断到 max_tokens 长度
                tokens = tokens[:self.max_tokens]
                tokenized_candidates.append(tokens)
            except Exception as e:
                print(f"[DP-RAG] Tokenization failed for a candidate: {e}")
                continue

        if not tokenized_candidates:
            # Tokenization 全部失败，回退到简单投票
            print("[DP-RAG] All tokenization failed, using simple majority vote")
            return self._simple_majority_vote(non_empty_candidates)

        # 3. Token 级别 DP 投票
        final_tokens = self._token_level_dp_vote(tokenized_candidates)

        if not final_tokens:
            return self._simple_majority_vote(non_empty_candidates)

        # 4. Detokenize
        try:
            final_response = self.llm.detokenize(final_tokens)
        except Exception as e:
            print(f"[DP-RAG] Detokenization failed: {e}")
            return self._simple_majority_vote(non_empty_candidates)

        return final_response

    # ──────────────────────────────────────────────
    # Token 级别 DP 投票
    # ──────────────────────────────────────────────

    def _token_level_dp_vote(self, tokenized_candidates: List[List[int]]) -> List[int]:
        """
        在每个 token 位置使用 DP majority vote 选择最终 token

        对齐 DPRAG 原生的 get_majority_token_vote 逻辑：
        - 首先尝试 LDGumbel DP 投票
        - 如果失败且 fail_mode='ld_pate'，尝试 1-excluded LD（k_bar=dim-1）
        - 如果仍然失败，停止生成
        - 如果 DP 预算超支，停止生成
        """
        max_len = max(len(tokens) for tokens in tokenized_candidates)
        n_voters = len(tokenized_candidates)
        final_tokens = []

        # 为每次 answer 调用创建新的 DP engine 实例
        # （避免跨查询的隐私预算累积问题，每次查询独立计算）
        dp_engine = LDGumbelMechanism(
            eps=self._dp_engine.eps,
            delta=self._dp_engine.delta,
            k_bar=min(self.n_split, self._vocab_size),
            target_eps=self._dp_engine.target_eps,
            target_delta=self._dp_engine.target_delta,
            fail_mode=self.fail_mode,
        )

        for pos in range(max_len):
            # 收集该位置所有 voter 的 token
            tokens_at_pos = []
            for tokens in tokenized_candidates:
                if pos < len(tokens):
                    tokens_at_pos.append(tokens[pos])
                # 超出长度的 voter 不参与该位置投票（不 padding）

            if not tokens_at_pos:
                break

            tokens_tensor = torch.tensor(tokens_at_pos)

            # DP 投票（对齐 DPRAG 的 get_majority_token_vote）
            selected_token = self._get_majority_token_vote(
                tokens_tensor, dp_engine
            )

            if selected_token is None:
                # DP 失败或预算超支，停止生成
                print(f"[DP-RAG] DP vote failed at position {pos}, stopping generation")
                self._dp_stats['dp_failures'] += 1
                break

            final_tokens.append(selected_token)
            self._dp_stats['total_vote_calls'] += 1

        self._dp_stats['total_tokens_generated'] += len(final_tokens)

        # 记录 DP 预算消耗
        eps, delta = dp_engine.get_dp_expense()
        print(f"[DP-RAG] DP expense for this query: eps={eps:.4f}, delta={delta:.6g}")

        return final_tokens

    def _get_majority_token_vote(self, tokens_tensor: torch.Tensor, dp_engine: LDGumbelMechanism) -> Optional[int]:
        """
        对齐 DPRAG 原生的 get_majority_token_vote 逻辑

        处理流程：
        1. 尝试 DP majority vote
        2. 如果 NotFoundLDTop1：
           - ld_pate 模式：尝试 1-excluded LD (k_bar=dim-1)
           - rand 模式：随机选择
           - stop 模式：停止
        3. 如果 DPExpenseOverflow：停止
        """
        try:
            selected_token, _ = majority_vote(
                tokens_tensor,
                dim=self._vocab_size,
                dp_engine=dp_engine,
            )
            return selected_token
        except NotFoundLDTop1:
            if dp_engine.fail_mode == 'ld_pate':
                # 1-excluded LD：使用更小的 k_bar 提高成功率
                try:
                    selected_token, _ = majority_vote(
                        tokens_tensor,
                        dim=self._vocab_size,
                        dp_engine=dp_engine,
                        k_bar=self._vocab_size - 1,
                    )
                    return selected_token
                except NotFoundLDTop1:
                    print("[DP-RAG] LD failed even with 1-excluded mode")
                    return None
            elif dp_engine.fail_mode == 'rand':
                return torch.randint(self._vocab_size, (1,))[0].item()
            elif dp_engine.fail_mode == 'stop':
                return None
            else:
                raise NotFoundLDTop1()
        except DPExpenseOverflow:
            eps, delta = dp_engine.get_dp_expense()
            print(f"[DP-RAG] DP budget exceeded: eps={eps:.4f}, delta={delta:.4f}")
            self._dp_stats['dp_budget_exceeded'] += 1
            return None

    # ──────────────────────────────────────────────
    # 简单多数投票（回退方案）
    # ──────────────────────────────────────────────

    def _simple_majority_vote(self, candidates: List[str]) -> str:
        """
        简单多数投票作为 DP 生成失败时的回退

        返回出现频率最高的候选回答
        """
        if not candidates:
            return ""

        # 统计每个回答的出现频率
        from collections import Counter
        counter = Counter(candidates)
        most_common = counter.most_common(1)[0][0]
        return most_common

    # ──────────────────────────────────────────────
    # DP 统计与重置
    # ──────────────────────────────────────────────

    def get_dp_stats(self) -> dict:
        """返回 DP 使用统计"""
        self._dp_stats['dp_engine_state'] = {
            'total_queries': self._dp_engine.total_queries,
            'total_k': self._dp_engine.total_k,
        }
        return self._dp_stats.copy()

    def reset_dp_engine(self):
        """重置 DP 引擎状态"""
        self._dp_engine = LDGumbelMechanism(
            eps=self._dp_engine.eps,
            delta=self._dp_engine.delta,
            k_bar=self.n_split,
            target_eps=self._dp_engine.target_eps,
            target_delta=self._dp_engine.target_delta,
            fail_mode=self.fail_mode,
        )
        self._dp_stats = {
            'total_tokens_generated': 0,
            'total_vote_calls': 0,
            'dp_failures': 0,
            'dp_budget_exceeded': 0,
        }