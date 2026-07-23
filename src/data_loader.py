# utils/data_loader.py
import os
import json
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DatasetLoader:
    """
    Dataset-specific loader instance.
    Used for specific slicing strategies in privacy research.
    """
    
    def load_dataset(self, file_path: str) -> List[Document]:
        """
        Automatically select loading strategy based on filename pattern
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        filename = os.path.basename(file_path)

        # Strategy 1: HP1_5ch (Harry Potter) - dynamically compute chunk_size to get approximately 100 slices
        if "HP1_5ch" in filename:
            return self._load_hp1_dynamic(file_path)
        
        # Strategy 2: HealthCareMagic (Medical QA) or QA pair format JSON list processing
        elif "HealthCareMagic" in filename or "QA" in filename or "PokemonInfo" in filename:
            return self._load_qa_json(file_path)

        elif "trec" in filename or "sci" in filename or "nfc" in filename:
            return self._load_qa_json(file_path)
        
        # Strategy 3: Default processing (simple text loading)
        elif file_path.endswith(".txt"):
            return self._load_generic_txt(file_path)
        
        else:
            raise ValueError(f"No loading strategy defined for dataset: {filename}")

    def _load_hp1_dynamic(self, file_path: str) -> List[Document]:
        """
        Special strategy for HP1_5ch:
        Read full text -> compute chunk_size -> split into approximately 100 chunks
        """
        print(f"[DataLoader] Applying 'txt Strategy' for {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        # Calculate target chunk_size
        target_chunks = 100
        text_length = len(text_content)
        # Avoid division by zero or too small chunk
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

        # Raw document object
        raw_doc = Document(page_content=text_content, metadata={"source": file_path})
        
        # Execute splitting
        chunks = splitter.split_documents([raw_doc])
        
        # Format output (add ID, simplify metadata)
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
        Special strategy for HealthCareMagic:
        JSON List -> each Item as a Chunk (Q + A)
        """
        print(f"[DataLoader] Applying 'json QA Strategy' for {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        chunks = []
        for i, item in enumerate(data):
            # Compatible with different JSON key names, adjust based on actual situation
            # Assume structure is input/output or instruction/output
            question = item.get('input') or item.get('instruction') or item.get('question') or ""
            answer = item.get('output') or item.get('response') or item.get('answer') or ""
            
            # Build text content
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
        """Default TXT loading strategy"""
        print(f"[DataLoader] Applying 'Generic TXT Strategy' for {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Default to fixed-size splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.create_documents([text], metadatas=[{"source": os.path.basename(file_path)}])