# utils/data_loader.py
import os
import json
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DatasetLoader:
    """
    针对特定数据集实例的加载器。
    用于隐私研究中的特定切片策略。
    """
    
    def load_dataset(self, file_path: str) -> List[Document]:
        """
        根据文件名模式自动选择加载策略
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        filename = os.path.basename(file_path)

        # 策略 1: HP1_5ch (哈利波特) - 动态计算 chunk_size 以获得约 100 个切片
        if "HP1_5ch" in filename:
            return self._load_hp1_dynamic(file_path)
        
        # 策略 2: HealthCareMagic (医患问答) 或 问答对格式的 JSON 列表处理
        elif "HealthCareMagic" in filename or "QA" in filename or "PokemonInfo" in filename:
            return self._load_qa_json(file_path)

        elif "trec" in filename or "sci" in filename or "nfc" in filename:
            return self._load_qa_json(file_path)
        
        # 策略 3: 默认处理 (简单的文本加载)
        elif file_path.endswith(".txt"):
            return self._load_generic_txt(file_path)
        
        else:
            raise ValueError(f"No loading strategy defined for dataset: {filename}")

    def _load_hp1_dynamic(self, file_path: str) -> List[Document]:
        """
        针对 HP1_5ch 的特殊策略：
        读取全文 -> 计算 chunk_size -> 分割为约 100 个 chunks
        """
        print(f"[DataLoader] Applying 'txt Strategy' for {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        # 计算目标 chunk_size
        target_chunks = 100
        text_length = len(text_content)
        # 避免除以零或过小的 chunk
        if text_length < target_chunks:
            chunk_size = text_length
        else:
            chunk_size = text_length // target_chunks
        
        chunk_overlap = chunk_size // 10  # 10% overlap
        
        print(f"[DataLoader] Text Length: {text_length}, Calculated Chunk Size: {chunk_size}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 原始文档对象
        raw_doc = Document(page_content=text_content, metadata={"source": file_path})
        
        # 执行切分
        chunks = splitter.split_documents([raw_doc])
        
        # 格式化输出 (添加ID，简化Metadata)
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{i:04d}"
            new_doc = Document(
                page_content=chunk.page_content,
                metadata={
                    "id": chunk_id,
                    "source": os.path.basename(file_path),
                    "original_index": i
                }
            )
            processed_chunks.append(new_doc)
            
        print(f"[DataLoader] Generated {len(processed_chunks)} chunks.")
        return processed_chunks

    def _load_qa_json(self, file_path: str) -> List[Document]:
        """
        针对 HealthCareMagic 的特殊策略：
        JSON List -> 每个 Item 作为一个 Chunk (Q + A)
        """
        print(f"[DataLoader] Applying 'json QA Strategy' for {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        chunks = []
        for i, item in enumerate(data):
            # 兼容不同的 JSON 键名，根据实际情况调整
            # 假设结构是 input/output 或者 instruction/output
            question = item.get('input') or item.get('instruction') or item.get('question') or ""
            answer = item.get('output') or item.get('response') or item.get('answer') or ""
            
            # 构建文本内容
            qa_text = f"Q: {question}\nA: {answer}"
            
            chunk_id = f"{i:04d}"
            doc = Document(
                page_content=qa_text,
                metadata={
                    "id": chunk_id,
                    "source": os.path.basename(file_path)
                }
            )
            chunks.append(doc)
            
        print(f"[DataLoader] Generated {len(chunks)} chunks from QA pairs.")
        return chunks

    def _load_generic_txt(self, file_path: str) -> List[Document]:
        """默认的 TXT 加载策略"""
        print(f"[DataLoader] Applying 'Generic TXT Strategy' for {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 默认使用固定大小切分
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.create_documents([text], metadatas=[{"source": os.path.basename(file_path)}])