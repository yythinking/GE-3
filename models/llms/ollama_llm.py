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

        # Initialize chat model
        self.chat_model = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Ollama chat model"""
        return ChatOllama(
            reasoning=False, # Disable thinking
            model=self.model_name,
            temperature=self.temperature,
            base_url=self.base_url
    )

    def generate(self, prompt: str) -> str:
        """
        Generate response - implement abstract method defined by parent class
        """
        try:
            # Call Ollama model to generate response
            # Directly return content
            if "qwen3:" in self.model_name:
                prompt = prompt + " /no_think"
            response = self.chat_model.invoke(prompt)
            return response.content 
            
        except Exception as e:
            # Can add logging here
            print(f"Ollama {self.model_name} call failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def _get_tokenizer(self):
        """Get tiktoken tokenizer instance (lazy loading)"""
        if not hasattr(self, '_tokenizer'):
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using tiktoken
        
        Parameters:
            text: Input text
            
        Returns:
            Token ID list
        """
        try:
            encoding = self._get_tokenizer()
            return encoding.encode(text)
        except Exception as e:
            print(f"[OllamaLLM] Tokenization failed: {e}")
            # Fallback: simple character splitting
            return [ord(c) for c in text]
    
    def detokenize(self, tokens: List[int]) -> str:
        """
        Detokenize token list using tiktoken
        
        Parameters:
            tokens: Token ID list
            
        Returns:
            Decoded text
        """
        try:
            encoding = self._get_tokenizer()
            return encoding.decode(tokens)
        except Exception as e:
            print(f"[OllamaLLM] Detokenization failed: {e}")
            # Fallback: simple character decoding
            return ''.join([chr(t) if 0 < t < 128 else ' ' for t in tokens])
    
    def get_vocab_size(self) -> int:
        """
        Return vocabulary size
        
        Dynamically obtained from tokenizer to avoid hardcoded inconsistencies
        """
        return self._get_tokenizer().n_vocab
    
    def get_model_info(self):
        """Return model information"""
        
        # Simplify base_url to keep only domain part
        base_url_only = self.base_url.split('//')[-1].split('/')[0]
        # Remove port number
        base_url_only = base_url_only.split(':')[0]

        return {
            "model_name": self.model_name,
            "base_url": base_url_only,
            "temperature": self.temperature,
            "provider": "ollama",
        }