# Define LLM Abstract Base Class
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLLM(ABC):
    """
    Base class that all LLM implementations must inherit from
    """
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Core method: Input prompt, directly return string text
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata
        """
        pass