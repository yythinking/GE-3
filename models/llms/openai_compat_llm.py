from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from ..interfaces.llm_interface import BaseLLM
from langchain_openai import ChatOpenAI
import tiktoken
from langchain_core.rate_limiters import InMemoryRateLimiter
from typing import List

class OpenLLM(BaseLLM):  
    def __init__(self, model_name: str, base_url: str, api_key: str, temperature: float = 0.7):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        
        # Base configuration parameters
        base_config = {
            "model": self.model_name,
            "model_provider": "openai",
            "base_url": self.base_url,
            "api_key": api_key,
            "temperature": self.temperature
        }
        
        # Set extra configuration parameters based on model name
        model_name_lower = model_name.lower()
        extra_body = None
        
        if "glm" in model_name or "minimax" in model_name_lower or "qwen" in model_name_lower:
            extra_body = {"enable_thinking": False}
        elif "doubao" in model_name:
            extra_body = {"thinking": {"type": "disabled"}}
        elif "gemini" in model_name_lower:
            if "pro" in model_name_lower: # Gemini Pro series models don't support minimal level thinking config, can only set to low
                extra_body={
                    'extra_body': {
                        "google": {
                            "thinking_config": {
                                "thinking_level": "low",
                            }
                        }
                    }
                }
            else:      # Gemini series models support minimal level thinking config
                extra_body={
                    'extra_body': {
                        "google": {
                            "thinking_config": {
                                "thinking_level": "minimal",
                            }
                        }
                    }
                }
        # Build final configuration
        if extra_body:
            config = {**base_config, "extra_body": extra_body}
        else:
            config = base_config
        
        # Add rate limiting for Qwen models
        if "qwen" in model_name_lower and "235b" in model_name_lower:
            # Set stricter rate limiting for Qwen3-235B-A22B-Instruct-2507 model
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=0.116667,    # 0.1 requests per second (once per 10 seconds)
                check_every_n_seconds=0.1,   # Check every 100ms
                max_bucket_size=3            # Strictly limit to max 3 requests per minute
            )
            # Add rate limiter to config
            config["rate_limiter"] = rate_limiter
        
        # Initialize chat model
        self.chat_model = init_chat_model(**config)


    def generate(self, prompt: str) -> str:
        """
        Generate response - implement abstract method defined by parent class
        """
        try:
            # Call OpenAI compatible model to generate response
            # Directly return content
            response = self.chat_model.invoke(prompt)
            return response.content 
            
        except Exception as e:
            # Can add logging here
            print(f"Third-party compatible model {self.model_name} call failed: {e}")
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
            print(f"[OpenLLM] Tokenization failed: {e}")
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
            print(f"[OpenLLM] Detokenization failed: {e}")
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



        return {
            "model_name": self.model_name,
            "base_url": base_url_only,
            "temperature": self.temperature,
            "provider": "openai_compat",
        }