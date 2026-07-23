# rag/sage_engine.py
"""
SAGE Synthetic Data Generation Engine
Based on "Mitigating the Privacy Issues in Retrieval-Augmented Generation (RAG) via Pure Synthetic Data"

Core Functions:
1. Stage 1: Attribute Extraction and Synthetic Data Generation
2. Stage 2 (Optional): Agent Iterative Refinement for Enhanced Privacy Protection
3. Build synthetic data vector index for RAG Retrieval
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
# SAGE Prompt Templates (aligned with SAGE original implementation)
# =============================================================================

def get_attributes_prompt(input_context: str, dataset_type: str = "chat") -> str:
    """
    Stage 1 - Step 1: Attribute Extraction Prompt
    
    Select corresponding template based on dataset type:
    - chat: Medical dialogue dataset (Patient-Doctor)
    - wiki/doc: General text dataset
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
        # Default general template
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
    Stage 1 - Step 2: Synthetic Data Generation Prompt
    
    Select corresponding template based on dataset type to generate synthetic data
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
    Paraphrase Prompt (for baseline comparison)
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
# LLM Client Wrapper
# =============================================================================

class SAGEClientWrapper:
    """
    SAGE LLM Client Wrapper
    Supports multiple LLM types: Ollama, OpenAI-compatible, Gemini, etc.
    """
    
    def __init__(self, llm: BaseLLM, model_name: str = "default"):
        self.llm = llm
        self.model_name = model_name
        self._error_count = 0
    
    def generate(self, prompt: str, system_content: str = "You are a helpful assistant.",
                 max_tokens: int = 256, temperature: float = 0.6) -> str:
        """
        Call LLM to generate content
        
        Aligned with project existing interface: generate() only accepts single prompt parameter
        System prompt injected via concatenation
        
        Args:
            prompt: User input prompt
            system_content: System prompt (injected via concatenation)
            max_tokens: Maximum generated tokens (reserved parameter, not actually passed)
            temperature: Generation temperature (reserved parameter, not actually passed)
            
        Returns:
            Generated text
        """
        try:
            # generate() in project only accepts single prompt parameter
            # Concatenate system prompt to front of prompt
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
# SAGE Main Engine Class
# =============================================================================

class SAGEEngine:
    """
    SAGE Synthetic Data Generation Engine
    
    Provides two usage modes:
    1. sync mode: Stage 1 (Attribute Extraction + Synthetic Generation)
    2. agent2 mode: Stage 1 + Stage 2 (Agent Iterative Refinement)
    
    Usage Example:
        sage = SAGEEngine(
            llm=llm,
            embedding=embedding,
            original_data_path="./datasets/mini_HealthCareMagic.json"
        )
        
        # One-time preprocessing (optional, can also generate on-demand)
        sage.preprocess_and_build_index()
        
        # Use at retrieval time
        docs = sage.search("What are the symptoms of diabetes?")
    """
    
    # Dataset type auto-detection mapping
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
        attr_llm: BaseLLM = None,  # Attribute extraction LLM (can be same as synth_llm)
        synth_llm: BaseLLM = None,  # Synthetic generation LLM
        attr_model_name: str = "attributes-extractor",
        synth_model_name: str = "synthetic-generator",
    ):
        """
        Initialize SAGE engine
        
        Args:
            llm: Default LLM instance
            embedding: Embedding model
            original_data_path: Original dataset path
            cache_dir: Synthetic data cache directory
            attr_llm: Attribute extraction dedicated LLM (default: llm)
            synth_llm: Synthetic generation dedicated LLM (default: llm)
            attr_model_name: Attribute extraction model name
            synth_model_name: Synthetic generation model name
        """
        self.llm = llm
        self.embedding = embedding
        self.original_data_path = original_data_path
        
        # LLM clients
        self.attr_llm = attr_llm if attr_llm else llm
        self.synth_llm = synth_llm if synth_llm else llm
        
        self.attr_client = SAGEClientWrapper(self.attr_llm, attr_model_name)
        self.synth_client = SAGEClientWrapper(self.synth_llm, synth_model_name)
        
        # Cache configuration
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
        
        # Dataset information
        self.dataset_name = os.path.basename(original_data_path).split('.')[0]
        self.dataset_type = self._detect_dataset_type()
        self._hash_id = self._compute_data_hash()
        
        # Cache paths (include synthetic_mode to ensure sync/agent2 data isolation)
        self.synthetic_mode = "sync"  # Default value
        self.synthetic_data_cache = os.path.join(
            cache_dir, f"{self.dataset_name}_{self._hash_id}_{self.synthetic_mode}_synthetic.json"
        )
        self.attributes_cache = os.path.join(
            cache_dir, f"{self.dataset_name}_{self._hash_id}_{self.synthetic_mode}_attributes.json"
        )
        self.index_dir = os.path.join(
            cache_dir, f"{self.dataset_name}_{self._hash_id}_{self.synthetic_mode}_index"
        )
        
        # Original data
        self._original_docs: List[Document] = []
        self._synthetic_docs: List[Document] = []
        self._vector_store: Optional[Chroma] = None
        
        # State
        self._is_preprocessed = False
        self._is_index_built = False
    
    def set_synthetic_mode(self, mode: str = "sync"):
        """
        Set synthetic mode and update cache paths
        
        Ensure sync and agent2 modes use independent cache files
        """
        if mode not in ["sync", "agent2"]:
            print(f"[SAGE] Warning: Unknown synthetic_mode '{mode}', using 'sync'")
            mode = "sync"
        
        if self.synthetic_mode != mode:
            print(f"[SAGE] Switching synthetic_mode from '{self.synthetic_mode}' to '{mode}'")
            self.synthetic_mode = mode
            # Update cache paths
            self.synthetic_data_cache = os.path.join(
                self.cache_dir, f"{self.dataset_name}_{self._hash_id}_{mode}_synthetic.json"
            )
            self.attributes_cache = os.path.join(
                self.cache_dir, f"{self.dataset_name}_{self._hash_id}_{mode}_attributes.json"
            )
            self.index_dir = os.path.join(
                self.cache_dir, f"{self.dataset_name}_{self._hash_id}_{mode}_index"
            )
            # Reset state, force reload/regenerate
            self._is_preprocessed = False
            self._is_index_built = False
            self._vector_store = None
    
    def get_document_count(self) -> int:
        """Get synthetic data document count"""
        if self._vector_store:
            try:
                return self._vector_store._collection.count()
            except:
                pass
        return len(self._synthetic_docs) if self._synthetic_docs else 0
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _detect_dataset_type(self) -> str:
        """Auto-detect dataset type"""
        path_lower = self.original_data_path.lower()
        
        for dtype, keywords in self.DATASET_TYPE_KEYWORDS.items():
            if any(kw in path_lower for kw in keywords):
                return dtype
        
        # Default to chat type (aligned with SAGE original implementation)
        return "chat"
    
    def _compute_data_hash(self) -> str:
        """Compute dataset hash for cache identification"""
        if not os.path.exists(self.original_data_path):
            return "unknown"
        
        with open(self.original_data_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        return file_hash
    
    def _load_original_docs(self) -> List[Document]:
        """Load original dataset"""
        if self._original_docs:
            return self._original_docs
        
        # Use existing data_loader
        from src.data_loader import DatasetLoader
        loader = DatasetLoader()
        self._original_docs = loader.load_dataset(self.original_data_path)
        
        print(f"[SAGE] Loaded {len(self._original_docs)} original documents")
        return self._original_docs
    
    # =========================================================================
    # Stage 1: Attribute Extraction and Synthetic Data Generation
    # =========================================================================
    
    def get_synthetic_context(
        self, 
        ori_contexts: List[str], 
        dataset_type: str = None,
        use_cache: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        Generate synthetic context (Stage 1)
        
        For each original context:
        1. Attribute extraction
        2. Synthetic data generation
        
        Args:
            ori_contexts: Original context list [[ctx1, ctx2, ...], [ctx1, ctx2, ...], ...]
            dataset_type: Dataset type (auto-detect by default)
            use_cache: Whether to use cache
            
        Returns:
            (attribute list, synthetic context list)
        """
        dtype = dataset_type or self.dataset_type
        
        all_attributes_con = []
        all_synthetic_con = []
        
        for ori_context in tqdm(ori_contexts, desc="[SAGE] Generating synthetic context"):
            attributes_con = []
            synthetic_con = []
            
            for ori_con in ori_context:
                # Step 1: Attribute extraction
                attributes_prompt = get_attributes_prompt(ori_con, dtype)
                attributes_context = self.attr_client.generate(
                    attributes_prompt,
                    system_content="You are a helpful assistant."
                )
                
                # Step 2: Synthetic data generation
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
        Generate synthetic data for single text (for on-demand generation)
        
        Args:
            original_text: Original text
            dataset_type: Dataset type
            
        Returns:
            Synthetic text
        """
        dtype = dataset_type or self.dataset_type
        
        # Step 1: Attribute extraction
        attributes_prompt = get_attributes_prompt(original_text, dtype)
        attributes_context = self.attr_client.generate(
            attributes_prompt,
            system_content="You are a helpful assistant."
        )
        
        if not attributes_context:
            print("[SAGE] Warning: Attribute extraction returned empty")
            return original_text
        
        # Step 2: Synthetic data generation
        synthetic_prompt = get_synthetic_prompt(attributes_context, dtype)
        synthetic_context = self.synth_client.generate(
            synthetic_prompt,
            system_content="You are a helpful assistant."
        )
        
        return synthetic_context if synthetic_context else original_text
    
    # =========================================================================
    # Stage 2: Agent Iterative Refinement (Optional)
    # =========================================================================
    
    def agent_refinement(self, original_text: str, synthetic_text: str) -> str:
        """
        Agent iterative refinement (Stage 2)
        Decide whether privacy enhancement is needed based on evaluation results
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
            # Unsafe, add privacy noise
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
    # Data Preprocessing and Index Building
    # =========================================================================
    
    def preprocess_and_build_index(
        self, 
        rebuild: bool = False,
        batch_size: int = 100
    ) -> bool:
        """
        Preprocess dataset and build synthetic data index
        
        Args:
            rebuild: Whether to force rebuild (ignore cache)
            batch_size: Batch processing size
            
        Returns:
            Whether successful
        """
        if self._is_preprocessed and self._is_index_built and not rebuild:
            print(f"[SAGE] Already preprocessed, skipping (use rebuild=True to force)")
            return True
        
        # Step 1: Load original data
        print("[SAGE] Step 0: Loading original dataset...")
        original_docs = self._load_original_docs()
        
        if not original_docs:
            print("[SAGE] Error: No documents loaded")
            return False
        
        # Step 2: Check cache
        if not rebuild and os.path.exists(self.synthetic_data_cache):
            print(f"[SAGE] Loading synthetic data from cache: {self.synthetic_data_cache}")
            with open(self.synthetic_data_cache, 'r', encoding='utf-8') as f:
                synthetic_data = json.load(f)
            
            # Convert to Document
            self._synthetic_docs = []
            for i, item in enumerate(synthetic_data):
                if not item.get("content"):
                    continue
                chunk_id = f"synth_{i:04d}"  # Keep consistent with original data format
                doc = Document(
                    page_content=item["content"],
                    metadata={
                        "id": chunk_id,
                        "source": f"synthetic_{i}",
                        "original_content": item.get("original", ""),  # Save original content for tracking
                        "is_synthetic": True
                    }
                )
                self._synthetic_docs.append(doc)
        else:
            # Step 3: Generate synthetic data (Stage 1) - parallel processing
            print("[SAGE] Stage 1: Generating synthetic data (parallel processing)...")
            
            # Determine total document count
            total_docs = len(original_docs)
            print(f"[SAGE] Total documents to process: {total_docs}")
            
            # Parallel processing configuration
            max_workers = min(8, os.cpu_count() or 4)  # Set parallelism based on CPU cores
            print(f"[SAGE] Using {max_workers} parallel workers")
            
            all_synthetic = [None] * total_docs  # Pre-allocate list to maintain order
            
            # Single document processing function (for parallel execution)
            def process_single_doc(args):
                idx, text = args
                try:
                    synthetic_text = self.get_single_synthetic(text)

                    # If agent2 mode, execute Stage 2
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
                        "content": text,  # Fallback to original text
                        "original": text,
                        "metadata": {"source": "fallback", "idx": idx}
                    }
            
            # Prepare task list
            tasks = [(i, doc.page_content) for i, doc in enumerate(original_docs)]
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Use imap to maintain order, update progress bar after each is processed
                results = list(tqdm(
                    executor.map(process_single_doc, tasks),
                    total=total_docs,
                    desc="[SAGE] Generating synthetic data",
                    unit="doc",
                    ncols=80  # Limit progress bar width
                ))
            
            # Collect results (maintain order)
            for idx, result in results:
                all_synthetic[idx] = result
            
            # Save cache
            print(f"[SAGE] Saving synthetic data cache to: {self.synthetic_data_cache}")
            with open(self.synthetic_data_cache, 'w', encoding='utf-8') as f:
                json.dump(all_synthetic, f, ensure_ascii=False, indent=2)
            
            # Convert to Document (keep consistent with original data format)
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
        
        # Step 4: Build vector index
        print("[SAGE] Building vector index...")
        self._build_synthetic_index()
        
        self._is_preprocessed = True
        print("[SAGE] Preprocessing completed successfully")
        return True
    
    def _build_synthetic_index(self):
        """Build vector index for synthetic data"""
        if self._is_index_built:
            return
        
        if not self._synthetic_docs:
            print("[SAGE] Warning: No synthetic docs to index")
            return
        
        # If index already exists, load it
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
    # Retrieval Interface (for RAG use)
    # =========================================================================
    
    def search(self, query: str, top_k: int = 4, top_p: int = None) -> List[Document]:
        """
        Retrieve from synthetic data
        
        Aligned with BaseEngine.search interface
        
        Args:
            query: Query text
            top_k: Number of documents to return
            top_p: Candidate document count (optional, same as top_k means no rerank)
            
        Returns:
            List of retrieved documents
        """
        # Ensure index is built
        if not self._is_index_built:
            print("[SAGE] Index not built, triggering lazy preprocessing...")
            self.preprocess_and_build_index()
        
        if not self._vector_store:
            raise ValueError("[SAGE] Vector store not initialized")
        
        # Execute retrieval
        candidates = self._vector_store.similarity_search(query, k=top_k)
        return candidates
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get index information"""
        info = {
            "dataset_type": self.dataset_type,
            "original_path": self.original_data_path,
            "original_count": len(self._original_docs),
            "synthetic_count": len(self._synthetic_docs),
            "is_index_built": self._is_index_built,
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
        """Get SAGE statistics (compatible with DP_RAG interface)"""
        return {
            "synthetic_mode": "stage1",
            "attr_llm_errors": self.attr_client.error_count,
            "synth_llm_errors": self.synth_client.error_count,
            "synthetic_count": len(self._synthetic_docs),
            "is_preprocessed": self._is_preprocessed,
            "is_index_built": self._is_index_built,
        }
    
    def reset(self):
        """Reset SAGE engine state"""
        self._original_docs = []
        self._synthetic_docs = []
        self._vector_store = None
        self._is_preprocessed = False
        self._is_index_built = False
        self.attr_client.reset_error_count()
        self.synth_client.reset_error_count()


# =============================================================================
# Convenience Functions
# =============================================================================

def build_sage_index(
    data_path: str,
    llm: BaseLLM,
    embedding: BaseEmbedding,
    cache_dir: str = "./storage/synthetic_data",
    rebuild: bool = False
) -> SAGEEngine:
    """
    Convenience function for quickly building SAGE index
    
    Args:
        data_path: Dataset path
        llm: LLM instance
        embedding: Embedding instance
        cache_dir: Cache directory
        rebuild: Whether to force rebuild
        
    Returns:
        Configured SAGEEngine instance
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
    # Test code
    print("[SAGE] Module loaded successfully")
    print("[SAGE] Usage: from rag.sage_engine import SAGEEngine")