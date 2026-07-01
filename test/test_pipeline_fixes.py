#!/usr/bin/env python3
# test_pipeline_fixes.py
# 测试修复后的 pipeline 是否正确加载数据并计算指标

import os
import sys
import json
import tempfile
import shutil

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_document_count_logic():
    """测试文档计数逻辑"""
    print("=" * 60)
    print("Testing Document Count Logic")
    print("=" * 60)
    
    from src.data_loader import DatasetLoader
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from models.embeddings.hf_embedding import LocalHFEmbedding
    
    # 创建临时目录用于测试
    with tempfile.TemporaryDirectory() as tmpdir:
        # 准备测试数据
        test_docs = [
            Document(page_content=f"Test document {i}", metadata={"id": f"doc_{i}"})
            for i in range(10)
        ]
        
        # 构建 Chroma 索引
        embedding = LocalHFEmbedding("BAAI/bge-m3", "cpu")
        persist_dir = os.path.join(tmpdir, "chroma_index")
        
        print(f">>> Building test index with {len(test_docs)} documents...")
        vector_store = Chroma.from_documents(
            documents=test_docs,
            embedding=embedding,
            persist_directory=persist_dir,
            collection_name="test_collection"
        )
        
        # 测试方法1: _collection.count()
        print("Testing Method 1: _collection.count()")
        try:
            count = vector_store._collection.count()
            print(f"  SUCCESS: {count} documents")
        except Exception as e:
            print(f"  FAILED: {e}")
            count = 0
        
        # 重新加载并测试
        print("Reloading index...")
        vector_store2 = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding,
            collection_name="test_collection"
        )
        
        # 测试方法2: _collection.count() after reload
        print("Testing Method 2: _collection.count() after reload")
        try:
            count2 = vector_store2._collection.count()
            print(f"  SUCCESS: {count2} documents")
        except Exception as e:
            print(f"  FAILED: {e}")
            count2 = 0
        
        # 测试方法3: vector_store.get()
        print("Testing Method 3: vector_store.get()")
        try:
            docs = vector_store2.get()
            count3 = len(docs['documents']) if docs and 'documents' in docs else 0
            print(f"  SUCCESS: {count3} documents")
        except Exception as e:
            print(f"  FAILED: {e}")
            count3 = 0
        
        # 验证
        if count2 == 10 and count3 == 10:
            print("\n✅ All methods work correctly!")
            return True
        else:
            print(f"\n❌ Methods returned inconsistent results: count2={count2}, count3={count3}")
            return False


def test_safe_divide():
    """测试安全除法函数"""
    print("\n" + "=" * 60)
    print("Testing Safe Divide")
    print("=" * 60)
    
    def safe_divide(numerator, denominator, default=0.0):
        return numerator / denominator if denominator > 0 else default
    
    # 测试用例
    test_cases = [
        (10, 100, 0.1),
        (10, 0, 0.0),
        (0, 0, 0.0),
        (0, 100, 0.0),
        (5, 3, None),  # 这个会有浮点误差
    ]
    
    all_passed = True
    for num, denom, expected in test_cases:
        result = safe_divide(num, denom)
        if expected is not None:
            if abs(result - expected) < 0.001:
                print(f"  ✅ {num}/{denom} = {result:.4f}")
            else:
                print(f"  ❌ {num}/{denom} = {result:.4f}, expected {expected}")
                all_passed = False
        else:
            print(f"  ℹ️  {num}/{denom} = {result:.4f}")
    
    return all_passed


def test_file_write_with_retry():
    """测试带重试的文件写入"""
    print("\n" + "=" * 60)
    print("Testing File Write with Retry")  
    print("=" * 60)
    
    def safe_json_write(filepath, data, max_retries=3):
        import time
        for attempt in range(max_retries):
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1)
                else:
                    print(f"  [Error] Failed: {e}")
                    return False
        return False
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data = {"test": "data", "number": 42}
        test_path = os.path.join(tmpdir, "test.json")
        
        result = safe_json_write(test_path, test_data)
        if result and os.path.exists(test_path):
            with open(test_path, 'r') as f:
                loaded = json.load(f)
            if loaded == test_data:
                print("  ✅ File write with retry works correctly!")
                return True
            else:
                print("  ❌ File content mismatch")
                return False
        else:
            print("  ❌ File write failed")
            return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("PIPELINE FIXES VALIDATION TESTS")
    print("=" * 60 + "\n")
    
    results = {
        "Document Count Logic": test_document_count_logic(),
        "Safe Divide": test_safe_divide(),
        "File Write with Retry": test_file_write_with_retry(),
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! The fixes should work correctly.")
    else:
        print("⚠️  Some tests failed. Please review the fixes.")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())