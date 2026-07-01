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
        
        # 基础配置参数
        base_config = {
            "model": self.model_name,
            "model_provider": "openai",
            "base_url": self.base_url,
            "api_key": api_key,
            "temperature": self.temperature
        }
        
        # 根据模型名称设置额外的配置参数
        model_name_lower = model_name.lower()
        extra_body = None
        
        if "glm" in model_name or "minimax" in model_name_lower or "qwen" in model_name_lower:
            extra_body = {"enable_thinking": False}
        elif "doubao" in model_name:
            extra_body = {"thinking": {"type": "disabled"}}
        elif "gemini" in model_name_lower:
            if "pro" in model_name_lower: # Gemini Pro 系列模型不支持 minimal 级别的思考配置, 只能设置为 low
                extra_body={
                    'extra_body': {
                        "google": {
                            "thinking_config": {
                                "thinking_level": "low",
                            }
                        }
                    }
                }
            else:      # Gemini 系列模型支持 minimal 级别的思考配置
                extra_body={
                    'extra_body': {
                        "google": {
                            "thinking_config": {
                                "thinking_level": "minimal",
                            }
                        }
                    }
                }
        # 构建最终配置
        if extra_body:
            config = {**base_config, "extra_body": extra_body}
        else:
            config = base_config
        
        # 为Qwen模型添加限速设置
        if "qwen" in model_name_lower and "235b" in model_name_lower:
            # 为Qwen3-235B-A22B-Instruct-2507模型设置更严格的限速
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=0.116667,    # 每秒0.1个请求（每10秒一次）
                check_every_n_seconds=0.1,   # 每100ms检查一次
                max_bucket_size=3            # 严格按照每分钟最多三次请求
            )
            # 将限速器添加到配置中
            config["rate_limiter"] = rate_limiter
        # else:
        #     # 其他模型使用默认限速
        #     rate_limiter = InMemoryRateLimiter(
        #         requests_per_second=0.1,     # 每秒0.1个请求（每10秒一次）
        #         check_every_n_seconds=0.1,   # 每100ms检查一次
        #         max_bucket_size=10           # 最大突发请求量10
        #     )
        #     config["rate_limiter"] = rate_limiter
        
        # 初始化聊天模型
        self.chat_model = init_chat_model(**config)


    def generate(self, prompt: str) -> str:
        """
        生成响应 - 实现父类定义的抽象方法
        """
        try:
            # 调用 OpenAI 兼容模型生成响应
            # 直接返回内容
            response = self.chat_model.invoke(prompt)

            # # 统计一次交互消耗的 token 数(计算 prompt 和 response 的 token 数), 并输出.需要处理第三方模型的情况,计算大概的 token 数即可
            # prompt_tokens = 0
            # completion_tokens = 0
            # total_tokens = 0

            # # 策略 1: 优先尝试从 API 返回的元数据中获取准确数值
            # if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            #     usage = response.response_metadata['token_usage']
            #     prompt_tokens = usage.get('prompt_tokens', 0)
            #     completion_tokens = usage.get('completion_tokens', 0)
            #     total_tokens = usage.get('total_tokens', 0)
            
            # # 策略 2: 如果 API 没有返回，进行粗略估算 (Fallback)
            # if total_tokens == 0:
            #     try:
            #         # 使用 OpenAI 的标准编码器估算
            #         # 注意：第三方模型(如 Kimi/GLM)的分词器不同，这里仅为粗略估算
            #         encoding = tiktoken.get_encoding("cl100k_base")
            #         prompt_tokens = len(encoding.encode(prompt))
            #         completion_tokens = len(encoding.encode(response.content))
            #         total_tokens = prompt_tokens + completion_tokens
            #     except Exception as e:
            #         # 策略 3: 最差情况下的启发式估算 (适用于 tiktoken 加载失败等情况)
            #         # 假设中文语境下，1 token 约等于 1.5 - 2 个字符
            #         prompt_tokens = int(len(prompt) / 1.5)
            #         completion_tokens = int(len(response.content) / 1.5)
            #         total_tokens = prompt_tokens + completion_tokens
            #         print(f"[{self.model_name}] Tiktoken 估算失败，降级为字符长度估算: {e}")

            # # 输出 token 消耗信息（你可以将这部分记录到日志或者累加到限速器中）
            # # \033[46m: 青色背景 | \033[30m: 黑色文字 | \033[1m: 加粗 | \033[0m: 重置格式
            # highlight_start = "\033[46;30;1m"
            # highlight_end = "\033[0m"
            
            # log_message = f"[{self.model_name}] Token 消耗 - 输入: {prompt_tokens}, 输出: {completion_tokens}, 总计: {total_tokens}"
            
            # # 使用空行 \n 确保与上下文有间隔
            # print(f"\n{highlight_start} {log_message} {highlight_end}\n")
            # # ----------------------------------------------

            return response.content 
            
        except Exception as e:
            # 可以在这里加日志
            print(f"第三方兼容模型 {self.model_name} 调用失败: {e}")
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
            print(f"[OpenLLM] Tokenization failed: {e}")
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
            print(f"[OpenLLM] Detokenization failed: {e}")
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



        return {
            "model_name": self.model_name,
            "base_url": base_url_only,
            "temperature": self.temperature,
            "provider": "openai_compat",
        }