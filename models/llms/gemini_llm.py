from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from ..interfaces.llm_interface import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.7):
        """
        官方 Gemini 初始化：不需要 base_url。
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key

        # 使用官方 Google Generative AI 集成
        self.chat_model = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature
        )

    def generate(self, prompt: str) -> str:
        """
        生成响应 - 实现父类定义的抽象方法
        """
        try:
            # 官方集成支持直接传入字符串或 Message 对象
            response = self.chat_model.invoke(prompt)
            content = response.content

            # --- gemini 返回列表 ---
            if isinstance(content, str):
                return content
            
            if isinstance(content, list):
                # 针对多模态或复杂输出，提取所有文本内容并合并
                extracted_text = []
                for item in content:
                    if isinstance(item, str):
                        extracted_text.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        extracted_text.append(item["text"])
                    elif hasattr(item, "text"): # 兼容某些特定的对象格式
                        extracted_text.append(item.text)
                
                return "".join(extracted_text).strip()
            # --- 核心修复逻辑结束 ---

            return str(content) # 保底处理
            
        except Exception as e:
            print(f"Gemini 模型 {self.model_name} 调用失败: {e}")
            return f"Error generating response: {str(e)}"
    
    def get_model_info(self):
        """返回模型信息"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "provider": "google_genai",
        }