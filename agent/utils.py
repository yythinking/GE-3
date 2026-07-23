# agent/utils.py
import re
import numpy as np
from typing import List, Dict, Any

# --- Global Log ---
dialogue_history: List[tuple] = [] 

def log_interaction(q: str, a_clean: str, a_raw: str, docs: list, success: bool, log_type: str):
    global dialogue_history
    dialogue_history.append((q, a_clean, a_raw, docs, success, log_type))

def get_normalized_embedding(text: str, embedder, cache: Dict[str, np.ndarray]) -> np.ndarray:
    if text not in cache:
        if not text or not text.strip():
            return np.zeros(1024) 
        v = np.array(embedder.embed_query(text))
        n = np.linalg.norm(v)
        cache[text] = v / n if n > 0 else v
    return cache[text]

# --- Parsers ---
def parse_line_based_output(text: str, min_len: int = 5) -> List[str]:
    """Parse LLM-generated text list"""
    if not text: return []
    lines = text.split('\n')
    results = []
    for line in lines:
        clean_line = re.sub(r'^[\d\-\*\.]+\s*', '', line.strip())
        clean_line = re.sub(r'^(Q:|Question:|Entity:)\s*', '', clean_line, flags=re.IGNORECASE)
        if len(clean_line) >= min_len:
            results.append(clean_line)
    return list(set(results)) 

def extract_raw_rag_questions(raw_text: str) -> List[str]:
    """
    [New] Targeted extraction of RAG-returned original questions.
    Strategy:
    1. Search for common leading phrases (induced by Suffix).
    2. If no explicit leading phrase found, scan from the end for lines matching list format and ending with question marks.
    3. Strictly preserve original text, no LLM rewriting, ensure 'RAG Original'.
    """
    if not raw_text: return []
    
    questions = []
    lines = raw_text.split('\n')
    
    # Reverse scan because suffix requires "at the very end"
    # We assume question list is within last 10 lines, or until non-question large text
    scan_limit = 15
    scanned_count = 0
    
    # Simple heuristic: line starts with number/symbol, ends with question mark
    list_pattern = re.compile(r'^[\d\-\*\•]+\.?\s*')
    
    candidates = []
    for line in reversed(lines):
        line = line.strip()
        if not line: continue
        
        scanned_count += 1
        
        # Check if it's a question
        is_question = line.endswith('?')
        has_list_prefix = list_pattern.match(line)
        
        if is_question:
            # Remove sequence number
            clean_q = list_pattern.sub('', line).strip()
            if len(clean_q) > 5:
                candidates.append(clean_q)
        
        # Stop when encountering obvious separator or leading phrase, e.g., "Here are some questions:"
        if "questions" in line.lower() and ":" in line:
            break
            
        if scanned_count > scan_limit:
            break
            
    # Candidate list is reversed, flip back
    return candidates[::-1]

# --- Rejection Detection ---
def check_answer_rejection(answer: str, llm_check_func=None) -> bool:
    if not answer or len(answer) < 5: return True
    refusal_keywords = [
        "i don't know", "i cannot answer", "no information", "not provided",
        "context does not contain", "sorry", "cannot assist", "unrelated to",
        "not mentioned", "don't have enough info"
    ]
    ans_lower = answer.lower()
    if any(k in ans_lower for k in refusal_keywords):
        return True
    return False

# --- Candidate Pool Maintenance Logic (Updated to support dict objects) ---
def filter_candidates_by_similarity(
    candidates: List[Dict], # List[{'text': str, 'source': str, ...}]
    history_vecs: List[List[float]], 
    current_pool_texts: List[str],
    dedup_threshold: float,
    embedder,
    vector_cache: Dict
) -> List[Dict]:
    """
    [Deduplication] Filter out candidates too similar to historical questions or current pool questions.
    Input and output are Dict objects, deduplication based on text field.
    """
    valid_objs = []
    hist_matrix = np.array(history_vecs[-600:]) if history_vecs else np.empty((0,0))
    pool_set = set(current_pool_texts)

    META_KEYWORDS = [
        "in the context", "in the document", "provided text", "according to", "document",
        "based on the", "mentioned", "in this section", "the user wants", "context","verification question"
    ]  
    
    for obj in candidates:
        q = obj['text'].strip()
        
        if len(q) < 15 or len(q) > 300: continue 
        if not q.endswith('?'): continue
        if q in pool_set: continue 
        q_lower = q.lower()
        if any(k in q_lower for k in META_KEYWORDS): continue
        
        # 1. Compute vector
        q_vec = get_normalized_embedding(q, embedder, vector_cache)
        
        # 2. Historical similarity check
        if hist_matrix.shape[0] > 0:
            sims = np.dot(hist_matrix, q_vec)
            if np.max(sims) > dedup_threshold:
                continue 
        
        # Update cleaned text
        obj['text'] = q
        valid_objs.append(obj)
        
    return valid_objs

def prune_pool_by_novelty(pool: List[Dict], history_vecs, keep_limit, embedder, vector_cache):
    """
    [Pruning] Calculate novelty based on text field.
    """
    if len(pool) <= keep_limit: return pool
    
    texts = [p['text'] for p in pool]
    cand_vecs = np.array([get_normalized_embedding(t, embedder, vector_cache) for t in texts])
    
    if not history_vecs: return pool[:keep_limit]
    
    hist_matrix = np.array(history_vecs[-50:])
    max_sims = np.max(np.dot(cand_vecs, hist_matrix.T), axis=1)
    
    # Lower similarity is better (argsort ascending)
    keep_indices = np.argsort(max_sims)[:keep_limit]
    
    return [pool[i] for i in keep_indices]