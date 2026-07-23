# rag/DP_RAG.py
"""
Differential Privacy RAG Engine
Based on DPRAG project's DP VoteRAG mechanism implementation

Core Concept (aligned with DPRAG native implementation):
1. After retrieving documents, shuffle them randomly to construct n_split prompt variants
2. Each prompt variant independently calls LLM to generate answers (voters)
3. Align all answers via tokenization
4. Use LDGumbel DP majority vote at token level to select final token
5. Detokenize to get final answer
6. Track DP privacy budget
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
    Differential Privacy RAG Engine (aligned with DPRAG native VoteRAG mechanism)

    Uses DP mechanism to protect privacy during answer generation:
    1. After retrieval, construct n_split prompt variants by shuffling document order
    2. Each prompt variant independently calls LLM to generate answers (voters)
    3. Tokenize all voter answers
    4. Use DP majority vote at each token position to select final token
    5. Detokenize to get final answer
    6. Track DP privacy budget consumption

    Parameters:
        n_split: Number of prompt variants / voters (default 50)
        dp_eps: DP epsilon per token (default 2.0)
        dp_delta: DP delta per token (default 1e-5)
        target_eps: Total epsilon budget upper bound (default 1000.0)
        target_delta: Total delta budget upper bound (default 1.0)
        max_tokens: Maximum tokens to generate (default 100)
        fail_mode: DP failure handling mode (default 'ld_pate')
        use_parallel: Whether to use parallel execution for voter generation (default True)
        max_workers: Maximum number of parallel workers (default 20)
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

        # DP mechanism: k_bar aligned with n_split (DPRAG native setting)
        self._dp_engine = LDGumbelMechanism(
            eps=dp_eps,
            delta=dp_delta,
            k_bar=n_split,
            target_eps=target_eps,
            target_delta=target_delta,
            fail_mode=fail_mode,
        )

        # Vocabulary size (from LLM)
        self._vocab_size = self.llm.get_vocab_size() if hasattr(self.llm, 'get_vocab_size') else 100256

        # Track DP usage
        self._dp_stats = {
            'total_tokens_generated': 0,
            'total_vote_calls': 0,
            'dp_failures': 0,
            'dp_budget_exceeded': 0,
        }

        self.vector_store = None
        self.collection_name = "dp_rag_collection"

    # ──────────────────────────────────────────────
    # Index Management (consistent with StandardRAGEngine)
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
    # Retrieval (consistent with StandardRAGEngine)
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
    # End-to-end Answer
    # ──────────────────────────────────────────────

    def answer(self, query: str) -> Tuple[str, List[Document]]:
        """
        End-to-end RAG + DP Generation

        Flow:
        1. Intent detection
        2. Retrieve relevant documents
        3. Construct n_split prompt variants (shuffle document order)
        4. Use DP ensemble to generate answer
        5. Safety check
        """
        # 1. Intent detection
        if not self.safety_check_query(query):
            return "Unknown.Intent", None

        # 2. Retrieval
        context_docs = self.search(query)

        # 3. Construct n_split prompt variants
        prompt_list = self._build_prompt_variants(query, context_docs)

        # 4. DP ensemble generation
        response = self._dp_ensemble_generate(prompt_list)

        # 5. Safety check
        context_str = "\n\n".join([
            f"Document {i+1}: {doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])
        if not self.safety_check_response(response, context_str):
            return "Unknown.Copy", None

        return response, context_docs

    # ──────────────────────────────────────────────
    # Prompt Variant Construction (DPRAG native method)
    # ──────────────────────────────────────────────

    def _build_prompt_variants(self, query: str, docs: List[Document]) -> List[str]:
        """
        Construct n_split prompt variants

        Aligned with DPRAG native implementation:
        - Extract retrieved document content
        - Randomly shuffle document order
        - Split documents into n_split groups, each containing several documents
        - Construct independent prompt for each group

        When n_split > number of documents, each voter still gets all documents,
        but with different order (achieved through multiple independent shuffles).
        """
        # Extract document text
        doc_texts = [doc.page_content for doc in docs]

        # Select prompt template
        if "HP1_5ch" in self.knowledge_path:
            prompt_template = pt.RAG_PROMPT_TEMPLATE_HP
        elif "HealthCareMagic" in self.knowledge_path:
            prompt_template = pt.RAG_PROMPT_TEMPLATE_HC
        else:
            prompt_template = pt.RAG_PROMPT_TEMPLATE_DEFAULT

        prompt_list = []

        if len(doc_texts) == 0:
            # No documents: all voters use same prompt without context
            for _ in range(self.n_split):
                prompt_list.append(prompt_template.format(context="", question=query))
            return prompt_list

        if self.n_split <= len(doc_texts):
            # Number of voters <= number of documents: each voter gets different document subset
            # Split documents into n_split groups
            shuffled_texts = doc_texts.copy()
            random.shuffle(shuffled_texts)

            docs_per_split = max(1, len(shuffled_texts) // self.n_split)
            for split_id in range(self.n_split):
                start = split_id * docs_per_split
                end = min(start + docs_per_split, len(shuffled_texts))
                # Last split gets all remaining documents
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
            # Number of voters > number of documents: each voter gets all documents, but in different order
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
    # DP Ensemble Generation (Core Mechanism)
    # ──────────────────────────────────────────────

    def _voter_generate(self, prompt: str, voter_id: int) -> Tuple[int, str]:
        """
        Single voter LLM generation task (for parallel execution)

        Args:
            prompt: Input prompt
            voter_id: Voter ID

        Returns:
            (voter_id, generated answer)
        """
        try:
            response = self.llm.generate(prompt)
            return (voter_id, response)
        except Exception as e:
            print(f"[DP-RAG] Voter {voter_id} generation failed: {e}")
            return (voter_id, "")

    def _dp_ensemble_generate(self, prompt_list: List[str]) -> str:
        """
        Ensemble generation using DP mechanism

        Aligned with DPRAG native implementation flow:
        1. Each prompt variant independently calls LLM to generate answers (supports parallel)
        2. Tokenize all answers
        3. Use DP majority vote at each token position
        4. Detokenize to get final answer

        Note: DPRAG native uses vllm for token-by-token generation,
        our LLM interface only supports generate(prompt) -> str,
        so we use "first generate complete answer, then DP vote" strategy.
        """
        # 1. Each voter independently generates answer (supports parallel acceleration)
        candidates = [None] * len(prompt_list)

        if self.use_parallel:
            # Parallel mode: use ThreadPoolExecutor to parallelize all voter calls
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
            # Serial mode: keep original logic
            for i, prompt in enumerate(prompt_list):
                try:
                    response = self.llm.generate(prompt)
                    candidates[i] = response
                except Exception as e:
                    print(f"[DP-RAG] Voter {i} generation failed: {e}")
                    candidates[i] = ""

        # Filter out empty candidates
        non_empty_candidates = [c for c in candidates if c and c.strip()]
        if not non_empty_candidates:
            return ""
        empty_count = len([c for c in candidates if not c or not c.strip()])
        if empty_count > 0:
            print(f"[DP-RAG] {empty_count} voters produced empty responses (parallel={self.use_parallel})")

        # 2. Tokenize all candidates
        tokenized_candidates = []
        for candidate in non_empty_candidates:
            try:
                tokens = self.llm.tokenize(candidate)
                # Truncate to max_tokens length
                tokens = tokens[:self.max_tokens]
                tokenized_candidates.append(tokens)
            except Exception as e:
                print(f"[DP-RAG] Tokenization failed for a candidate: {e}")
                continue

        if not tokenized_candidates:
            # All tokenization failed, fallback to simple vote
            print("[DP-RAG] All tokenization failed, using simple majority vote")
            return self._simple_majority_vote(non_empty_candidates)

        # 3. Token-level DP voting
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
    # Token-level DP Voting
    # ──────────────────────────────────────────────

    def _token_level_dp_vote(self, tokenized_candidates: List[List[int]]) -> List[int]:
        """
        Use DP majority vote at each token position to select final token

        Aligned with DPRAG native get_majority_token_vote logic:
        - First try LDGumbel DP voting
        - If fails and fail_mode='ld_pate', try 1-excluded LD (k_bar=dim-1)
        - If still fails, stop generation
        - If DP budget exceeded, stop generation
        """
        max_len = max(len(tokens) for tokens in tokenized_candidates)
        n_voters = len(tokenized_candidates)
        final_tokens = []

        # Create new DP engine instance for each answer call
        # (Avoid cross-query privacy budget accumulation, calculate independently per query)
        dp_engine = LDGumbelMechanism(
            eps=self._dp_engine.eps,
            delta=self._dp_engine.delta,
            k_bar=min(self.n_split, self._vocab_size),
            target_eps=self._dp_engine.target_eps,
            target_delta=self._dp_engine.target_delta,
            fail_mode=self.fail_mode,
        )

        for pos in range(max_len):
            # Collect all voter tokens at this position
            tokens_at_pos = []
            for tokens in tokenized_candidates:
                if pos < len(tokens):
                    tokens_at_pos.append(tokens[pos])
                # Voters exceeding length don't participate in this position vote (no padding)

            if not tokens_at_pos:
                break

            tokens_tensor = torch.tensor(tokens_at_pos)

            # DP voting (aligned with DPRAG get_majority_token_vote)
            selected_token = self._get_majority_token_vote(
                tokens_tensor, dp_engine
            )

            if selected_token is None:
                # DP failure or budget exceeded, stop generation
                print(f"[DP-RAG] DP vote failed at position {pos}, stopping generation")
                self._dp_stats['dp_failures'] += 1
                break

            final_tokens.append(selected_token)
            self._dp_stats['total_vote_calls'] += 1

        self._dp_stats['total_tokens_generated'] += len(final_tokens)

        # Record DP budget consumption
        eps, delta = dp_engine.get_dp_expense()
        print(f"[DP-RAG] DP expense for this query: eps={eps:.4f}, delta={delta:.6g}")

        return final_tokens

    def _get_majority_token_vote(self, tokens_tensor: torch.Tensor, dp_engine: LDGumbelMechanism) -> Optional[int]:
        """
        Aligned with DPRAG native get_majority_token_vote logic

        Processing flow:
        1. Try DP majority vote
        2. If NotFoundLDTop1:
           - ld_pate mode: try 1-excluded LD (k_bar=dim-1)
           - rand mode: random selection
           - stop mode: stop
        3. If DPExpenseOverflow: stop
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
                # 1-excluded LD: use smaller k_bar to improve success rate
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
    # Simple Majority Vote (Fallback)
    # ──────────────────────────────────────────────

    def _simple_majority_vote(self, candidates: List[str]) -> str:
        """
        Simple majority vote as fallback when DP generation fails

        Returns the most frequent candidate answer
        """
        if not candidates:
            return ""

        # Count frequency of each answer
        from collections import Counter
        counter = Counter(candidates)
        most_common = counter.most_common(1)[0][0]
        return most_common

    # ──────────────────────────────────────────────
    # DP Statistics and Reset
    # ──────────────────────────────────────────────

    def get_dp_stats(self) -> dict:
        """Return DP usage statistics"""
        self._dp_stats['dp_engine_state'] = {
            'total_queries': self._dp_engine.total_queries,
            'total_k': self._dp_engine.total_k,
        }
        return self._dp_stats.copy()

    def reset_dp_engine(self):
        """Reset DP engine state"""
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