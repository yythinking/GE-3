# eval/evaluator.py

import json
import re
import os
import math
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_core.documents import Document

# Fix import paths
from data_loader import DatasetLoader
from models.interfaces.llm_interface import BaseLLM

GENERATE_EVAL_QA_PROMPT = """
You are an expert data annotator specializing in creating high-quality, extraction-based Question-Answering datasets. Your task is to analyze the provided Text Chunk or Dialogue and generate valid Question-Answer (QA) pairs based **strictly** on the content.

### Guidelines:
1.  **Strict Extraction Only:** Questions must be answerable solely using the information present in the input text. Do not use external knowledge, common sense, or make inferences/extensions beyond what is explicitly written.
2.  **Source-Grounded Answers:** The Answer must be derived directly from the text. Do not paraphrase if it changes the meaning, and do not add context that isn't there.
3.  **Comprehensive Coverage:** A single text chunk often contains multiple distinct information points. Generate as many QA pairs as necessary to cover all factual details in the text.
4.  **Format:** Output a raw JSON array containing objects with `Question` and `Answer` fields. Do not include markdown code blocks (like ```json), explanations, or any text other than the JSON structure.

### Output Format:
[
  {{
    "Question": "The specific question based on the text",
    "Answer": "The direct answer derived from the text"
  }},
  {{
    "Question": "Another specific question...",
    "Answer": "Another direct answer..."
  }}
]

### Input Text:
{text}
"""

class Evaluator:
    def __init__(self, dataset_loader: DatasetLoader, power_llms: List[BaseLLM], dataset_path: str):
        """
        :param power_llms: Pass in a list of LLM instances (corresponding to multiple API KEYs)
        """
        self.dataset_loader = dataset_loader
        # Ensure it's a list
        self.power_llms = power_llms if isinstance(power_llms, list) else [power_llms]
        self.dataset_path = dataset_path

    def _extract_json(self, text: str):
        """Clean and extract JSON"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    return []
            return []

    def _process_chunk_worker(self, documents: List[Document], llm: BaseLLM, worker_id: int) -> List[dict]:
        """
        Worker logic: process the document list assigned to it
        """
        local_qa_pairs = []
        # Use position parameter to avoid progress bar overlap
        desc = f"Gen-Worker-{worker_id}"
        
        for doc in tqdm(documents, desc=desc, position=worker_id, leave=False):
            context = doc.page_content
            prompt = GENERATE_EVAL_QA_PROMPT.format(text=context)
            
            try:
                # Use the specific LLM instance bound to this Worker (corresponding to specific Key)
                response = llm.generate(prompt)
                extracted_pairs = self._extract_json(response)
                
                if isinstance(extracted_pairs, list):
                    valid_pairs = [
                        item for item in extracted_pairs 
                        if "Question" in item and "Answer" in item
                    ]
                    local_qa_pairs.extend(valid_pairs)
            except Exception as e:
                print(f"[{desc}] Error: {e}")
                
        return local_qa_pairs

    def generate_qa_pairs(self, output_path: str):
        """Parallel generation entry point"""
        print(f"Loading dataset from {self.dataset_path}...")
        documents: List[Document] = self.dataset_loader.load_dataset(self.dataset_path)
        
        total_docs = len(documents)
        num_workers = len(self.power_llms)
        
        if total_docs == 0:
            print("No documents found.")
            return

        print(f"Generating QA pairs for {total_docs} chunks using {num_workers} parallel LLM Keys...")

        # 1. Task distribution: evenly distribute documents to N LLMs
        # Each Key handles a portion, without interference
        chunk_size = math.ceil(total_docs / num_workers)
        doc_chunks = [documents[i:i + chunk_size] for i in range(0, total_docs, chunk_size)]
        
        all_qa_pairs = []

        # 2. Thread pool parallel execution
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            # Submit tasks, map document chunks to LLM instances one-to-one
            for i in range(len(doc_chunks)):
                # Prevent chunks count being less than workers
                if i < len(self.power_llms):
                    future = executor.submit(
                        self._process_chunk_worker, 
                        doc_chunks[i], 
                        self.power_llms[i], 
                        i
                    )
                    futures.append(future)

            # 3. Collect results
            for future in as_completed(futures):
                try:
                    all_qa_pairs.extend(future.result())
                except Exception as e:
                    print(f"Worker execution failed: {e}")

        print(f"\nTotal QA pairs generated: {len(all_qa_pairs)}")
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_qa_pairs, f, ensure_ascii=False, indent=4)
        print(f"QA pairs saved to {output_path}")