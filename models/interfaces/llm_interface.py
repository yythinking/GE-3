# 定义 LLM 抽象基类
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLLM(ABC):
    """
    所有 LLM 实现必须继承的基类
    """
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        核心方法：输入 prompt，直接返回字符串文本
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型元数据
        """
        pass


