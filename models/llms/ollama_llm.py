from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from ..interfaces.llm_interface import BaseLLM
from langchain_ollama import ChatOllama
import tiktoken
from typing import List

class OllamaLLM(BaseLLM):  
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", temperature: float = 0.7):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature

        # 初始化聊天模型
        self.chat_model = self._initialize_model()
    
    def _initialize_model(self):
        """初始化 Ollama 聊天模型"""
        # return init_chat_model(
        #     model=self.model_name,
        #     model_provider="ollama",
        #     base_url=self.base_url,
        #     temperature=self.temperature,
        #     model_kwargs={          # ollama 模型的特殊参数
        #             "options": {
        #                 "reasoning": False
        #             }
        #         }
        # )
        return ChatOllama(
            reasoning=False, # 取消thinking
            model=self.model_name,
            temperature=self.temperature,
            base_url=self.base_url
    )

    def generate(self, prompt: str) -> str:
        """
        生成响应 - 实现父类定义的抽象方法
        """
        try:
            # 调用 Ollama 模型生成响应
            # 直接返回内容
            if "qwen3:" in self.model_name:
                prompt = prompt + " /no_think"
            response = self.chat_model.invoke(prompt)
            return response.content 
            
        except Exception as e:
            # 可以在这里加日志
            print(f"Ollama {self.model_name} 调用失败: {e}")
            return f"Error generating response: {str(e)}"
    
    def _get_tokenizer(self):
        """获取 tiktoken tokenizer 实例（懒加载）"""
        if not hasattr(self, '_tokenizer'):
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer
    
    def tokenize(self, text: str) -> List[int]:
        """
        使用 tiktoken 对文本进行 tokenize
        
        参数:
            text: 输入文本
            
        返回:
            token ID 列表
        """
        try:
            encoding = self._get_tokenizer()
            return encoding.encode(text)
        except Exception as e:
            print(f"[OllamaLLM] Tokenization failed: {e}")
            # 回退：简单按字符拆分
            return [ord(c) for c in text]
    
    def detokenize(self, tokens: List[int]) -> str:
        """
        使用 tiktoken 对 token 列表进行 detokenize
        
        参数:
            tokens: token ID 列表
            
        返回:
            解码后的文本
        """
        try:
            encoding = self._get_tokenizer()
            return encoding.decode(tokens)
        except Exception as e:
            print(f"[OllamaLLM] Detokenization failed: {e}")
            # 回退：简单按字符解码
            return ''.join([chr(t) if 0 < t < 128 else ' ' for t in tokens])
    
    def get_vocab_size(self) -> int:
        """
        返回词表大小
        
        动态从 tokenizer 获取，避免硬编码不一致
        """
        return self._get_tokenizer().n_vocab
    
    def get_model_info(self):
        """返回模型信息"""
        
        # 对 base_url 进行简化，只保留域名部分
        base_url_only = self.base_url.split('//')[-1].split('/')[0]
        # 去除端口号
        base_url_only = base_url_only.split(':')[0]

        return {
            "model_name": self.model_name,
            "base_url": base_url_only,
            "temperature": self.temperature,
            "provider": "ollama",
        }