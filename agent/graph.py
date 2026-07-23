# agent/graph.py
import os
import random
import numpy as np
from langgraph.graph import StateGraph, END
from typing import Dict, List, Optional, Tuple, TypedDict, Literal

# Import interface definitions
from models.interfaces.llm_interface import BaseLLM
from models.interfaces.embedding_interface import BaseEmbedding
from rag.base_engine import BaseEngine

# Import Prompts and utility functions
from . import prompts as pt
from . import utils as ut

# =============================================================================
# Agent State Definition
# =============================================================================
class AgentState(TypedDict):
    """
    Agent State Definition (State Schema).
    active_pool structure update: List[Dict], each element contains {'text': str, 'score': float, 'source': str}
    """
    current_epoch: int       # Current attack epoch
    total_epochs: int        # Total planned attack epochs
    
    # --- Core Knowledge Base ---
    active_pool: List[Dict]  
    
    # --- History (for MMR calculation and deduplication) ---
    visited_query_vectors: List[List[float]]   # Vector list of historical questions

# =============================================================================
# RAG Attacker Class (Graph Definition)
# =============================================================================
class RAGAttacker:
    def __init__(
        self,
        attacker_llm: BaseLLM,
        target_rag: BaseEngine,
        embedding_model: BaseEmbedding,
        output_dir: str = "./output", 
        epochs: int = 32,
        thresholds: Dict[str, float] = None,
        suffix: str = "",
        use_maxmin: bool = True,
        seed: int = 42,
        ablation_mode: str = "full",
    ):
        self.attacker_llm = attacker_llm
        self.target_rag = target_rag
        self.embedder = embedding_model
        self.epochs = epochs
        self.suffix = suffix 
        self.use_maxmin = use_maxmin
        self.seed = seed
        self.ablation_mode = ablation_mode
        self.rng = random.Random(seed)
        
        # Threshold configuration
        self.thresholds = {
            "dedup_sim": 0.90,              
            "pool_limit": 150                
        }
        if thresholds:
            for k, v in thresholds.items():
                if k in self.thresholds:
                    self.thresholds[k] = v
        
        self._vector_cache: Dict[str, np.ndarray] = {}
        self.app = self._build_graph()

    # ================= Nodes (Node Logic) =================

    def init_node(self, state: AgentState) -> Dict:
        """Initialization node"""
        ut.dialogue_history.clear()
        return {
            'current_epoch': 0, 
            'total_epochs': self.epochs, 
            'active_pool': [], 
            'visited_query_vectors': []
        }

    def greet_node(self, state: AgentState) -> Dict:
        """Handshake node (Cold Start)"""
        seeds = [
            # "Who are you? What can you do for me?",
            "What areas are you familiar with?",
        ]
        
        greet_q = self.rng.choice(seeds)
        # Mark source of seed question as seed
        result = self._execute_unified_turn(greet_q, tag="greet|seed")
        
        q_vec = ut.get_normalized_embedding(greet_q, self.embedder, self._vector_cache).tolist()

        # Initialize pool, add unused seeds
        initial_pool = [{'text': s, 'score': 1.0, 'source': 'seed'} for s in seeds if s != greet_q]
        
        # Process extracted new questions
        if result['extracted_candidates']:
            valid_objs = ut.filter_candidates_by_similarity(
                candidates=result['extracted_candidates'], # Pass list of dicts with source
                history_vecs=[q_vec], 
                current_pool_texts=[x['text'] for x in initial_pool], 
                dedup_threshold=self.thresholds['dedup_sim'],
                embedder=self.embedder,
                vector_cache=self._vector_cache
            )
            initial_pool.extend(valid_objs)

        return {
            'current_epoch': 1,  
            'active_pool': initial_pool,
            'visited_query_vectors': [q_vec]
        }

    def feedback_loop_node(self, state: AgentState) -> Dict:
        """Main loop node"""
        active_pool = state['active_pool']
        history_vecs = state['visited_query_vectors']
        
        target_q = None
        attack_base_type = "drill"
        source_label = "unknown"
        
        # --- Strategy Selection ---
        if active_pool:
            if self.use_maxmin:
                best_idx = self._select_best_candidate_idx(active_pool, history_vecs)
            else:
                best_idx = self._select_random_candidate_idx(active_pool)
            target_obj = active_pool.pop(best_idx)
            target_q = target_obj['text']
            source_label = target_obj.get('source', 'unknown') # Get source (rag/gen/seed)
        else:
            # Fallback
            target_q = "Please provide more details about other key topics in the database."
            attack_base_type = "fallback"
            source_label = "system"

        # Compose Tag: action_type|source (e.g., drill|rag_returned)
        full_tag = f"{attack_base_type}|{source_label}"

        # --- Execute Attack ---
        result = self._execute_unified_turn(target_q, tag=full_tag)
        
        # --- State Update ---
        q_vec = ut.get_normalized_embedding(target_q, self.embedder, self._vector_cache).tolist()
        updated_history_vecs = history_vecs + [q_vec]
        
        # Maintain candidate pool
        new_candidates = result['extracted_candidates'] # list of dicts
        
        valid_objs = ut.filter_candidates_by_similarity(
            candidates=new_candidates,
            history_vecs=updated_history_vecs,
            current_pool_texts=[x['text'] for x in active_pool],
            dedup_threshold=self.thresholds['dedup_sim'],
            embedder=self.embedder,
            vector_cache=self._vector_cache
        )
        
        # Add to pool (score initialized to 0.0)
        for obj in valid_objs:
            obj['score'] = 0.0
            active_pool.append(obj)
            
        # Pruning
        if len(active_pool) > self.thresholds['pool_limit']:
            active_pool = ut.prune_pool_by_novelty(
                active_pool, updated_history_vecs, self.thresholds['pool_limit'],
                self.embedder, self._vector_cache
            )

        return {
            'current_epoch': state['current_epoch'] + 1,
            'active_pool': active_pool,
            'visited_query_vectors': updated_history_vecs
        }

    # ================= Core Execution Logic (Updated) =================

    def _execute_unified_turn(self, question: str, tag: str) -> Dict:
        """
        [Unified Interface] Dual-path Extraction Flow
        1. RAG Response -> Extract RAG native recommended questions (Source: rag_returned)
        2. Clean answer -> LLM generates questions based on content (Source: llm_generated)
        3. Merge and return
        """
        # 1. Construct Payload (inject suffix to induce RAG to recommend questions)
        payload = f"{question} {self.suffix}".strip() if self.suffix else question
        
        # 2. RAG Response
        response_raw, context_docs = self.target_rag.answer(payload)
        
        # 3. Rejection Detection
        if context_docs is None or response_raw.startswith("Unknown."):
            ut.log_interaction(question, response_raw, response_raw, [], False, tag)
            return {
                'is_success': False,
                'extracted_candidates': [],
                'response_vec': None
            }

        extracted_candidates = [] # Stores {'text': str, 'source': str}

        # --- Path A: Direct extraction from Raw Response (Source: rag_returned) ---
        # Ensure only extract questions from RAG original output, no LLM polishing, guarantee originality
        rag_suggested_qs = ut.extract_raw_rag_questions(response_raw)
        for q_text in rag_suggested_qs:
            extracted_candidates.append({'text': q_text, 'source': 'rag_returned'})

        # 4. Clean answer (for SS/SC calculation and LLM generation)
        clean_prompt = pt.GET_ANSWER_ONLY_PROMPT.format(answer=response_raw)
        answer_cleaned = self.attacker_llm.generate(clean_prompt).strip()
        
        # 5. Rejection Check
        is_rejected = ut.check_answer_rejection(answer_cleaned)
        is_success = not is_rejected
        response_vec = None

        if is_success:
            # --- Path B: Content-based generation (Source: llm_generated) ---
            q_prompt = pt.GENERATE_EVIDENCE_BASED_QUESTIONS_PROMPT.format(corpus=answer_cleaned)
            gen_qs_raw = self.attacker_llm.generate(q_prompt)
            gen_qs_list = ut.parse_line_based_output(gen_qs_raw)
            
            for q_text in gen_qs_list:
                extracted_candidates.append({'text': q_text, 'source': 'llm_generated'})
            
            # Compute response vector
            response_vec = ut.get_normalized_embedding(answer_cleaned, self.embedder, self._vector_cache).tolist()
        
        # 6. Log
        ut.log_interaction(question, answer_cleaned, response_raw, context_docs, is_success, tag)
        
        return {
            'is_success': is_success,
            'extracted_candidates': extracted_candidates, # Return list of dicts
            'response_vec': response_vec
        }

    # ================= Helper Strategy Algorithms =================

    def _select_best_candidate_idx(self, pool: List[Dict], history_vecs: List[List[float]]) -> int:
        if not history_vecs: return 0
        texts = [p['text'] for p in pool]
        cand_vecs = np.array([ut.get_normalized_embedding(t, self.embedder, self._vector_cache) for t in texts])
        hist_matrix = np.array(history_vecs) 
        sims_hist = np.dot(cand_vecs, hist_matrix.T)
        max_sim_hist = np.max(sims_hist, axis=1) 
        best_idx = int(np.argmin(max_sim_hist))
        return best_idx

    def _select_random_candidate_idx(self, pool: List[Dict]) -> int:
        if not pool:
            return 0
        return self.rng.randrange(len(pool))

    def _check_stop(self, state: AgentState) -> Literal["continue", "end"]:
        return "end" if state['current_epoch'] >= state['total_epochs'] else "continue"
    
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("init", self.init_node)
        workflow.add_node("greet", self.greet_node)
        workflow.add_node("feedback_loop", self.feedback_loop_node)
        
        workflow.set_entry_point("init")
        workflow.add_edge("init", "greet")
        workflow.add_edge("greet", "feedback_loop")
        
        workflow.add_conditional_edges(
            "feedback_loop", 
            self._check_stop, 
            {"continue": "feedback_loop", "end": END}
        )
        return workflow.compile()