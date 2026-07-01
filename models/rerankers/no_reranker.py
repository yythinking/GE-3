from typing import List
from langchain_core.documents import Document
from ..interfaces.rerank_interface import BaseReranker

class NoReranker(BaseReranker):
    """
    空重排序器 (Pass-through)
    不做任何模型推理，仅仅是对检索结果进行 Top-K 截断。
    用于：
    1. 调试时排除 Reranker 干扰
    2. 追求极致速度、不需要精排的场景
    """
    def __init__(self):
        # 不需要加载模型，启动速度极快
        pass

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        直接返回前 Top K 个文档
        """
        # Python 的切片操作很安全，即使列表长度小于 top_k 也不会报错
        return documents[:top_k]

    def get_model_info(self) -> dict:
        return {
            "provider": "no_reranker",
            "model": "none",
            "status": "pass_through"
        }