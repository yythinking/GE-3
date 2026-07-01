# rag/sage_engine.py
"""
SAGE合成数据生成引擎
基于 "Mitigating the Privacy Issues in Retrieval-Augmented Generation (RAG) via Pure Synthetic Data"

核心功能:
1. Stage 1: 属性提取与合成数据生成
2. Stage 2 (可选): Agent迭代精炼增强隐私保护
3. 构建合成数据向量索引用于RAG检索
"""

import os
import json
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_chroma import Chroma

from models.interfaces.llm_interface import BaseLLM
from models.interfaces.embedding_interface import BaseEmbedding
from .base_engine import BaseEngine
from . import prompts as pt


# =============================================================================
# SAGE Prompt 模板 (对齐SAGE原始实现)
# =============================================================================

def get_attributes_prompt(input_context: str, dataset_type: str = "chat") -> str:
    """
    Stage 1 - Step 1: 属性提取Prompt
    
    根据数据集类型选择对应模板:
    - chat: 医疗对话数据集 (Patient-Doctor)
    - wiki/doc: 通用文本数据集
    """
    if dataset_type in ["chat", "chatdoctor", "healthcaremagic"]:
        return f"""Please summarize the key points from the following Doctor-Patient conversation:
    
    
        {input_context}
    
        Provide a summary for the Patient's information, including:
        [Attribute 1: Clear Symptom Description]
        [Attribute 2: Medical History]
        [Attribute 3: Current Concerns]  
        [Attribute 4: Recent Events]
        [Attribute 5: Specific Questions]
    
        Then, provide a summary for the Doctor's information, including:
        [Attribute 1: Clear Diagnosis or Assessment]
        [Attribute 2: Reassurance and Empathy]
        [Attribute 3: Treatment Options and Explanations]
        [Attribute 4: Follow-up and Next Steps]
        [Attribute 5: Education and Prevention]
    
        Please format your response as follows:
    
        Patient:
        - [Attribute 1: Clear Symptom Description]: 
        - [Attribute 2: Medical History]:
        - [Attribute 3: Current Concerns]:
        - [Attribute 4: Recent Events]:
        - [Attribute 5: Specific Questions]:
    
        Doctor:
        - [Attribute 1: Clear Diagnosis or Assessment]:
        - [Attribute 2: Reassurance and Empathy]:
        - [Attribute 3: Treatment Options and Explanations]:
        - [Attribute 4: Follow-up and Next Steps]:
        - [Attribute 5: Education and Prevention]:
    
        Please provide a concise summary for each attribute, capturing the most important information related to that attribute from the conversation.
        """
    elif dataset_type in ["wiki", "doc", "trec", "scidocs"]:
        return f"""Please summarize the key points from the following text:


        {input_context}

        Provide a summary the knowledge from the text, including:
        [Attribute 1: Clear TOPIC or CENTRAL IDEA of the text]
        [Attribute 2: Main details of the TOPIC or CENTRAL IDEA]
        [Attribute 3: Important facts, data, events, or viewpoints]

        Please format your response as follows:

        - [Attribute 1: Clear TOPIC or CENTRAL IDEA of the text]:
        - [Attribute 2: Main details of the TOPIC or CENTRAL IDEA]:
        - [Attribute 3: Important facts, data, events, or viewpoints]:

        Please provide a concise summary for each attribute, capturing the most important information related to that attribute. And remember to maintain logical order and accuracy.
        """
    else:
        # 默认通用模板
        return f"""Please summarize the key points from the following text:


        {input_context}

        Provide a concise summary covering:
        - Main topic or central idea
        - Key details and information
        - Important facts or viewpoints

        Format your response as:
        [Summary]: """


def get_synthetic_prompt(input_attributes: str, dataset_type: str = "chat") -> str:
    """
    Stage 1 - Step 2: 合成数据生成Prompt
    
    根据数据集类型选择对应模板生成合成数据
    """
    if dataset_type in ["wiki", "doc", "trec", "scidocs"]:
        return f"""Here is a summary of the key points:
    
        {input_attributes}
    
        Please generate a text using the ALL key points provided. 
        The text should read like a real-world document.
        """
    elif dataset_type in ["chat", "chatdoctor", "healthcaremagic"]:
        return f"""Here is a summary of the key points:

        {input_attributes}

        Please generate a SINGLE-ROUND patient-doctor medical dialog using the ALL key points provided. 
        The conversation should like a real-word medical conversation and contain ONLY ONE question 
        from the patient and ONE response from the doctor. The format should be as follows:

        Patient:[Patient's question contains ALL Patient's key points provided] 
        Doctor:[Doctor's response contains ALL Doctor's key points provided]

        Do not generate any additional rounds of dialog beyond the single question and response specified above."""
    else:
        return f"""Here is a summary of the key points:

        {input_attributes}

        Please generate a text using the ALL key points provided."""


def get_paraphrase_prompt(input_context: str, input_query: str) -> str:
    """
    改写Prompt (对比基线方法)
    """
    return f"""Given the following context, extract the useful or important part of the Context.
    
    Remember, *DO NOT* edit the extracted parts of the context.
    
    > Context:
    > > >
    {input_context}
    > > >
    Extracted relevant parts:
    """


# =============================================================================
# LLM客户端封装
# =============================================================================

class SAGEClientWrapper:
    """
    SAGE LLM客户端封装
    支持多种LLM类型: Ollama, OpenAI兼容, Gemini等
    """
    
    def __init__(self, llm: BaseLLM, model_name: str = "default"):
        self.llm = llm
        self.model_name = model_name
        self._error_count = 0
    
    def generate(self, prompt: str, system_content: str = "You are a helpful assistant.",
                 max_tokens: int = 256, temperature: float = 0.6) -> str:
        """
        调用LLM生成内容
        
        对齐项目现有接口：generate() 只接受单个 prompt 参数
        系统提示通过拼接方式注入
        
        Args:
            prompt: 用户输入prompt
            system_content: 系统提示 (通过拼接方式注入)
            max_tokens: 最大生成token数 (保留参数，实际不传递)
            temperature: 生成温度 (保留参数，实际不传递)
            
        Returns:
            生成的文本
        """
        try:
            # 项目中的generate()只接受单个prompt参数
            # 将系统提示拼接到prompt前面
            full_prompt = f"{system_content}\n\n{prompt}"
            
            response = self.llm.generate(full_prompt)
            
            if response and response.strip():
                return response.strip()
            else:
                self._error_count += 1
                print(f"[SAGE Client] Warning: Empty response received")
                return ""
        except Exception as e:
            self._error_count += 1
            print(f"[SAGE Client] Error: {e}")
            return ""
    
    @property
    def error_count(self) -> int:
        return self._error_count
    
    def reset_error_count(self):
        self._error_count = 0


# =============================================================================
# SAGE主引擎类
# =============================================================================

class SAGEEngine:
    """
    SAGE合成数据生成引擎
    
    提供两种使用模式:
    1. sync模式: Stage 1 (属性提取 + 合成生成)
    2. agent2模式: Stage 1 + Stage 2 (Agent迭代精炼)
    
    用法示例:
        sage = SAGEEngine(
            llm=llm,
            embedding=embedding,
            original_data_path="./datasets/mini_HealthCareMagic.json"
        )
        
        # 一次性预处理 (可选, 也可以lazy生成)
        sage.preprocess_and_build_index()
        
        # 检索时使用
        docs = sage.search("What are the symptoms of diabetes?")
    """
    
    # 数据集类型自动识别映射
    DATASET_TYPE_KEYWORDS = {
        "chat": ["chat", "healthcaremagic", "healthcare", "medical", "doctor", "patient"],
        "wiki": ["wiki", "trec", "covid", "scidocs", "nfcopurs"],
    }
    
    def __init__(
        self,
        llm: BaseLLM,
        embedding: BaseEmbedding,
        original_data_path: str,
        cache_dir: str = "./storage/synthetic_data",
        attr_llm: BaseLLM = None,  # 属性提取LLM (可与synth_llm相同)
        synth_llm: BaseLLM = None,  # 合成生成LLM
        attr_model_name: str = "attributes-extractor",
        synth_model_name: str = "synthetic-generator",
    ):
        """
        初始化SAGE引擎
        
        Args:
            llm: 默认LLM实例
            embedding: Embedding模型
            original_data_path: 原始数据集路径
            cache_dir: 合成数据缓存目录
            attr_llm: 属性提取专用LLM (默认为llm)
            synth_llm: 合成生成专用LLM (默认为llm)
            attr_model_name: 属性提取模型名称
            synth_model_name: 合成生成模型名称
        """
        self.llm = llm
        self.embedding = embedding
        self.original_data_path = original_data_path
        
        # LLM客户端
        self.attr_llm = attr_llm if attr_llm else llm
        self.synth_llm = synth_llm if synth_llm else llm
        
        self.attr_client = SAGEClientWrapper(self.attr_llm, attr_model_name)
        self.synth_client = SAGEClientWrapper(self.synth_llm, synth_model_name)
        
        # 缓存配置
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
        
        # 数据集信息
        self.dataset_name = os.path.basename(original_data_path).split('.')[0]
        self.dataset_type = self._detect_dataset_type()
        self._hash_id = self._compute_data_hash()
        
        # 缓存路径 (包含synthetic_mode确保sync/agent2数据隔离)
        self.synthetic_mode = "sync"  # 默认值
        self.synthetic_data_cache = os.path.join(
            cache_dir, f"{self.dataset_name}_{self._hash_id}_{self.synthetic_mode}_synthetic.json"
        )
        self.attributes_cache = os.path.join(
            cache_dir, f"{self.dataset_name}_{self._hash_id}_{self.synthetic_mode}_attributes.json"
        )
        self.index_dir = os.path.join(
            cache_dir, f"{self.dataset_name}_{self._hash_id}_{self.synthetic_mode}_index"
        )
        
        # 原始数据
        self._original_docs: List[Document] = []
        self._synthetic_docs: List[Document] = []
        self._vector_store: Optional[Chroma] = None
        
        # 状态
        self._is_preprocessed = False
        self._is_index_built = False
    
    def set_synthetic_mode(self, mode: str = "sync"):
        """
        设置合成模式并更新缓存路径
        
        确保sync和agent2模式使用独立的缓存文件
        """
        if mode not in ["sync", "agent2"]:
            print(f"[SAGE] Warning: Unknown synthetic_mode '{mode}', using 'sync'")
            mode = "sync"
        
        if self.synthetic_mode != mode:
            print(f"[SAGE] Switching synthetic_mode from '{self.synthetic_mode}' to '{mode}'")
            self.synthetic_mode = mode
            # 更新缓存路径
            self.synthetic_data_cache = os.path.join(
                self.cache_dir, f"{self.dataset_name}_{self._hash_id}_{mode}_synthetic.json"
            )
            self.attributes_cache = os.path.join(
                self.cache_dir, f"{self.dataset_name}_{self._hash_id}_{mode}_attributes.json"
            )
            self.index_dir = os.path.join(
                self.cache_dir, f"{self.dataset_name}_{self._hash_id}_{mode}_index"
            )
            # 重置状态，强制重新加载/生成
            self._is_preprocessed = False
            self._is_index_built = False
            self._vector_store = None
    
    def get_document_count(self) -> int:
        """获取合成数据文档数量"""
        if self._vector_store:
            try:
                return self._vector_store._collection.count()
            except:
                pass
        return len(self._synthetic_docs) if self._synthetic_docs else 0
    
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _detect_dataset_type(self) -> str:
        """自动检测数据集类型"""
        path_lower = self.original_data_path.lower()
        
        for dtype, keywords in self.DATASET_TYPE_KEYWORDS.items():
            if any(kw in path_lower for kw in keywords):
                return dtype
        
        # 默认返回chat类型 (与SAGE原实现一致)
        return "chat"
    
    def _compute_data_hash(self) -> str:
        """计算数据集哈希用于缓存标识"""
        if not os.path.exists(self.original_data_path):
            return "unknown"
        
        with open(self.original_data_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        return file_hash
    
    def _load_original_docs(self) -> List[Document]:
        """加载原始数据集"""
        if self._original_docs:
            return self._original_docs
        
        # 使用现有的data_loader
        from src.data_loader import DatasetLoader
        loader = DatasetLoader()
        self._original_docs = loader.load_dataset(self.original_data_path)
        
        print(f"[SAGE] Loaded {len(self._original_docs)} original documents")
        return self._original_docs
    
    # =========================================================================
    # Stage 1: 属性提取与合成数据生成
    # =========================================================================
    
    def get_synthetic_context(
        self, 
        ori_contexts: List[str], 
        dataset_type: str = None,
        use_cache: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        生成合成上下文 (Stage 1)
        
        对每个原始上下文:
        1. 属性提取
        2. 合成数据生成
        
        Args:
            ori_contexts: 原始上下文列表 [[ctx1, ctx2, ...], [ctx1, ctx2, ...], ...]
            dataset_type: 数据集类型 (默认自动检测)
            use_cache: 是否使用缓存
            
        Returns:
            (属性列表, 合成上下文列表)
        """
        dtype = dataset_type or self.dataset_type
        
        all_attributes_con = []
        all_synthetic_con = []
        
        for ori_context in tqdm(ori_contexts, desc="[SAGE] Generating synthetic context"):
            attributes_con = []
            synthetic_con = []
            
            for ori_con in ori_context:
                # Step 1: 属性提取
                attributes_prompt = get_attributes_prompt(ori_con, dtype)
                attributes_context = self.attr_client.generate(
                    attributes_prompt,
                    system_content="You are a helpful assistant."
                )
                
                # Step 2: 合成数据生成
                synthetic_prompt = get_synthetic_prompt(attributes_context, dtype)
                synthetic_context = self.synth_client.generate(
                    synthetic_prompt,
                    system_content="You are a helpful assistant."
                )
                
                attributes_con.append(attributes_context)
                synthetic_con.append(synthetic_context)
            
            all_attributes_con.append(attributes_con)
            all_synthetic_con.append(synthetic_con)
        
        return all_attributes_con, all_synthetic_con
    
    def get_single_synthetic(self, original_text: str, dataset_type: str = None) -> str:
        """
        为单个文本生成合成数据 (用于按需生成)
        
        Args:
            original_text: 原始文本
            dataset_type: 数据集类型
            
        Returns:
            合成后的文本
        """
        dtype = dataset_type or self.dataset_type
        
        # Step 1: 属性提取
        attributes_prompt = get_attributes_prompt(original_text, dtype)
        attributes_context = self.attr_client.generate(
            attributes_prompt,
            system_content="You are a helpful assistant."
        )
        
        if not attributes_context:
            print("[SAGE] Warning: Attribute extraction returned empty")
            return original_text
        
        # Step 2: 合成数据生成
        synthetic_prompt = get_synthetic_prompt(attributes_context, dtype)
        synthetic_context = self.synth_client.generate(
            synthetic_prompt,
            system_content="You are a helpful assistant."
        )
        
        return synthetic_context if synthetic_context else original_text
    
    # =========================================================================
    # Stage 2: Agent迭代精炼 (可选)
    # =========================================================================
    
    def agent_refinement(self, original_text: str, synthetic_text: str) -> str:
        """
        Agent迭代精炼 (Stage 2)
        根据评估结果决定是否需要隐私增强
        """
        evaluator_prompt = f"""You are a privacy evaluation agent. Analyze the synthetic data and original data:

ORIGINAL DATA:
{original_text[:500] if original_text else "N/A"}

SYNTHETIC DATA:
{synthetic_text[:500] if synthetic_text else "N/A"}

Evaluate the synthetic data for privacy risks:
1. Does it contain PII from the original (names, emails, phones, addresses)?
2. Does it expose sensitive attributes (health, financial, personal)?
3. Could it allow recovery of the original data?

If the synthetic data is SAFE (no privacy leaks), respond with exactly:
[SAFE]

If there are privacy concerns, respond with:
[UNSAFE] - {{brief reason}}
"""

        evaluation = self.synth_client.generate(evaluator_prompt)

        if "[SAFE]" in evaluation.upper():
            return synthetic_text
        else:
            # 不安全，添加隐私噪声
            noise_prompt = f"""Rewrite the following text to remove any potential privacy risks while preserving utility.
Remove or generalize: names, specific dates, locations, contact info, health/financial details.
Keep the general knowledge and structure.

Text:
{synthetic_text}

Rewritten (privacy-preserving version):
"""
            result = self.synth_client.generate(noise_prompt)
            return result if result else synthetic_text
    
    # =========================================================================
    # 数据预处理与索引构建
    # =========================================================================
    
    def preprocess_and_build_index(
        self, 
        rebuild: bool = False,
        batch_size: int = 100
    ) -> bool:
        """
        预处理数据集并构建合成数据索引
        
        Args:
            rebuild: 是否强制重建 (忽略缓存)
            batch_size: 批处理大小
            
        Returns:
            是否成功
        """
        if self._is_preprocessed and self._is_index_built and not rebuild:
            print(f"[SAGE] Already preprocessed, skipping (use rebuild=True to force)")
            return True
        
        # Step 1: 加载原始数据
        print("[SAGE] Step 0: Loading original dataset...")
        original_docs = self._load_original_docs()
        
        if not original_docs:
            print("[SAGE] Error: No documents loaded")
            return False
        
        # Step 2: 检查缓存
        if not rebuild and os.path.exists(self.synthetic_data_cache):
            print(f"[SAGE] Loading synthetic data from cache: {self.synthetic_data_cache}")
            with open(self.synthetic_data_cache, 'r', encoding='utf-8') as f:
                synthetic_data = json.load(f)
            
            # 转换为Document
            self._synthetic_docs = []
            for i, item in enumerate(synthetic_data):
                if not item.get("content"):
                    continue
                chunk_id = f"synth_{i:04d}"  # 与原始数据格式保持一致
                doc = Document(
                    page_content=item["content"],
                    metadata={
                        "id": chunk_id,
                        "source": f"synthetic_{i}",
                        "original_content": item.get("original", ""),  # 保存原始内容用于追踪
                        "is_synthetic": True
                    }
                )
                self._synthetic_docs.append(doc)
        else:
            # Step 3: 生成合成数据 (Stage 1) - 并行处理
            print("[SAGE] Stage 1: Generating synthetic data (parallel processing)...")
            
            # 判断总文档数量
            total_docs = len(original_docs)
            print(f"[SAGE] Total documents to process: {total_docs}")
            
            # 并行处理配置
            max_workers = min(8, os.cpu_count() or 4)  # 根据CPU核心数设置并行度
            print(f"[SAGE] Using {max_workers} parallel workers")
            
            all_synthetic = [None] * total_docs  # 预分配列表保持顺序
            
            # 单个文档处理函数 (用于并行执行)
            def process_single_doc(args):
                idx, text = args
                try:
                    synthetic_text = self.get_single_synthetic(text)

                    # 如果是 agent2 模式，执行 Stage 2
                    if self.synthetic_mode == "agent2":
                        synthetic_text = self.agent_refinement(text, synthetic_text)

                    return idx, {
                        "content": synthetic_text,
                        "original": text,
                        "metadata": {"source": "synthetic", "idx": idx}
                    }
                except Exception as e:
                    print(f"[SAGE] Error processing doc {idx}: {e}")
                    return idx, {
                        "content": text,  # 回退到原始文本
                        "original": text,
                        "metadata": {"source": "fallback", "idx": idx}
                    }
            
            # 准备任务列表
            tasks = [(i, doc.page_content) for i, doc in enumerate(original_docs)]
            
            # 使用ThreadPoolExecutor并行处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 使用imap保持顺序，每处理完一个就更新进度条
                results = list(tqdm(
                    executor.map(process_single_doc, tasks),
                    total=total_docs,
                    desc="[SAGE] Generating synthetic data",
                    unit="doc",
                    ncols=80  # 限制进度条宽度
                ))
            
            # 收集结果 (保持顺序)
            for idx, result in results:
                all_synthetic[idx] = result
            
            # 保存缓存
            print(f"[SAGE] Saving synthetic data cache to: {self.synthetic_data_cache}")
            with open(self.synthetic_data_cache, 'w', encoding='utf-8') as f:
                json.dump(all_synthetic, f, ensure_ascii=False, indent=2)
            
            # 转换为Document (与原始数据格式保持一致)
            self._synthetic_docs = []
            for i, item in enumerate(all_synthetic):
                if not item.get("content"):
                    continue
                chunk_id = f"synth_{i:04d}"
                doc = Document(
                    page_content=item["content"],
                    metadata={
                        "id": chunk_id,
                        "source": f"synthetic_{i}",
                        "original_content": item.get("original", ""),
                        "is_synthetic": True
                    }
                )
                self._synthetic_docs.append(doc)
        
        if not self._synthetic_docs:
            print("[SAGE] Error: No synthetic documents generated")
            return False
        
        print(f"[SAGE] Generated {len(self._synthetic_docs)} synthetic documents")
        
        # Step 4: 构建向量索引
        print("[SAGE] Building vector index...")
        self._build_synthetic_index()
        
        self._is_preprocessed = True
        print("[SAGE] Preprocessing completed successfully")
        return True
    
    def _build_synthetic_index(self):
        """构建合成数据的向量索引"""
        if self._is_index_built:
            return
        
        if not self._synthetic_docs:
            print("[SAGE] Warning: No synthetic docs to index")
            return
        
        # 如果索引已存在则加载
        if os.path.exists(os.path.join(self.index_dir, "chroma.sqlite3")):
            print(f"[SAGE] Loading existing index from: {self.index_dir}")
            self._vector_store = Chroma(
                persist_directory=self.index_dir,
                embedding_function=self.embedding
            )
        else:
            print(f"[SAGE] Creating new index at: {self.index_dir}")
            self._vector_store = Chroma.from_documents(
                documents=self._synthetic_docs,
                embedding=self.embedding,
                persist_directory=self.index_dir
            )
        
        self._is_index_built = True
        count = self._vector_store._collection.count()
        print(f"[SAGE] Index built with {count} documents")
    
    # =========================================================================
    # 检索接口 (供RAG使用)
    # =========================================================================
    
    def search(self, query: str, top_k: int = 4, top_p: int = None) -> List[Document]:
        """
        在合成数据中进行检索
        
        对齐 BaseEngine.search 接口
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            top_p: 候选文档数量 (可选, 与top_k相同则不重排)
            
        Returns:
            检索到的文档列表
        """
        # 确保索引已构建
        if not self._is_index_built:
            print("[SAGE] Index not built, triggering lazy preprocessing...")
            self.preprocess_and_build_index()
        
        if not self._vector_store:
            raise ValueError("[SAGE] Vector store not initialized")
        
        # 执行检索
        candidates = self._vector_store.similarity_search(query, k=top_k)
        return candidates
    
    def get_index_info(self) -> Dict[str, Any]:
        """获取索引信息"""
        info = {
            "dataset_type": self.dataset_type,
            "original_path": self.original_data_path,
            "original_count": len(self._original_docs),
            "synthetic_count": len(self._synthetic_docs),
            "index_built": self._is_index_built,
            "index_dir": self.index_dir,
            "cache_dir": self.cache_dir,
        }
        
        if self._vector_store:
            try:
                info["document_count"] = self._vector_store._collection.count()
            except:
                info["document_count"] = len(self._synthetic_docs)
        
        return info
    
    def get_dp_stats(self) -> Dict[str, Any]:
        """获取SAGE统计信息 (兼容DP_RAG接口)"""
        return {
            "synthetic_mode": "stage1",
            "attr_llm_errors": self.attr_client.error_count,
            "synth_llm_errors": self.synth_client.error_count,
            "synthetic_count": len(self._synthetic_docs),
            "is_preprocessed": self._is_preprocessed,
            "is_index_built": self._is_index_built,
        }
    
    def reset(self):
        """重置SAGE引擎状态"""
        self._original_docs = []
        self._synthetic_docs = []
        self._vector_store = None
        self._is_preprocessed = False
        self._is_index_built = False
        self.attr_client.reset_error_count()
        self.synth_client.reset_error_count()


# =============================================================================
# 便捷函数
# =============================================================================

def build_sage_index(
    data_path: str,
    llm: BaseLLM,
    embedding: BaseEmbedding,
    cache_dir: str = "./storage/synthetic_data",
    rebuild: bool = False
) -> SAGEEngine:
    """
    快速构建SAGE索引的便捷函数
    
    Args:
        data_path: 数据集路径
        llm: LLM实例
        embedding: Embedding实例
        cache_dir: 缓存目录
        rebuild: 是否强制重建
        
    Returns:
        配置好的SAGEEngine实例
    """
    sage = SAGEEngine(
        llm=llm,
        embedding=embedding,
        original_data_path=data_path,
        cache_dir=cache_dir,
    )
    
    sage.preprocess_and_build_index(rebuild=rebuild)
    
    return sage


if __name__ == "__main__":
    # 测试代码
    print("[SAGE] Module loaded successfully")
    print("[SAGE] Usage: from rag.sage_engine import SAGEEngine")