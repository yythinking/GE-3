# agent_alg/utils.py
import re
import numpy as np
from typing import List, Dict, Any

# --- 全局日志 ---
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

# --- 解析器 ---
def parse_line_based_output(text: str, min_len: int = 5) -> List[str]:
    """解析 LLM 生成的纯文本列表"""
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
    [新增] 针对性提取 RAG 返回的原始问题。
    策略：
    1. 寻找常见的引导词（根据 Suffix 诱导产生）。
    2. 如果找不到明确引导词，从尾部扫描，寻找符合列表格式且以问号结尾的行。
    3. 严格保留原始文本，不做 LLM 改写，确保是 'RAG Original'。
    """
    if not raw_text: return []
    
    questions = []
    lines = raw_text.split('\n')
    
    # 倒序扫描，因为 suffix 要求 "at the very end"
    # 我们假设问题列表在最后 10 行以内，或者直到遇到非问题的大段文本
    scan_limit = 15
    scanned_count = 0
    
    # 简单的 heuristic: 行首是数字/符号，行尾是问号
    list_pattern = re.compile(r'^[\d\-\*\•]+\.?\s*')
    
    candidates = []
    for line in reversed(lines):
        line = line.strip()
        if not line: continue
        
        scanned_count += 1
        
        # 检查是否是问题
        is_question = line.endswith('?')
        has_list_prefix = list_pattern.match(line)
        
        if is_question:
            # 去除序号
            clean_q = list_pattern.sub('', line).strip()
            if len(clean_q) > 5:
                candidates.append(clean_q)
        
        # 遇到明显的分割线或引导语时停止，例如 "Here are some questions:"
        if "questions" in line.lower() and ":" in line:
            break
            
        if scanned_count > scan_limit:
            break
            
    # 候选列表是倒序的，翻转回来
    return candidates[::-1]

# --- 拒绝检测 ---
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

# --- 候选池维护逻辑 (更新支持字典对象) ---
def filter_candidates_by_similarity(
    candidates: List[Dict], # List[{'text': str, 'source': str, ...}]
    history_vecs: List[List[float]], 
    current_pool_texts: List[str],
    dedup_threshold: float,
    embedder,
    vector_cache: Dict
) -> List[Dict]:
    """
    [去重] 过滤掉与历史问题或当前池中问题过于相似的候选。
    输入输出均为 Dict 对象，去重基于 text 字段。
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
        
        # 1. 计算向量
        q_vec = get_normalized_embedding(q, embedder, vector_cache)
        
        # 2. 历史相似度检查
        if hist_matrix.shape[0] > 0:
            sims = np.dot(hist_matrix, q_vec)
            if np.max(sims) > dedup_threshold:
                continue 
        
        # 更新清洗后的文本
        obj['text'] = q
        valid_objs.append(obj)
        
    return valid_objs

def prune_pool_by_novelty(pool: List[Dict], history_vecs, keep_limit, embedder, vector_cache):
    """
    [剪枝] 基于 text 字段计算新颖性。
    """
    if len(pool) <= keep_limit: return pool
    
    texts = [p['text'] for p in pool]
    cand_vecs = np.array([get_normalized_embedding(t, embedder, vector_cache) for t in texts])
    
    if not history_vecs: return pool[:keep_limit]
    
    hist_matrix = np.array(history_vecs[-50:])
    max_sims = np.max(np.dot(cand_vecs, hist_matrix.T), axis=1)
    
    # 相似度越低越好 (argsort 升序)
    keep_indices = np.argsort(max_sims)[:keep_limit]
    
    return [pool[i] for i in keep_indices]