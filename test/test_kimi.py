from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import tiktoken
from langchain_core.rate_limiters import InMemoryRateLimiter

class OpenLLM():  
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

            return response.content 
            
        except Exception as e:
            # 可以在这里加日志
            print(f"第三方兼容模型 {self.model_name} 调用失败: {e}")
            return f"Error generating response: {str(e)}"
    
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
    


def test_kimi():

    import os
    from dotenv import load_dotenv
    load_dotenv()  # 从 .env 文件加载环境变量

    # kimi_api_keys = [
    # os.getenv("sf_api_key"), os.getenv("sf_api_key_1"), os.getenv("sf_api_key_2"), os.getenv("sf_api_key_3"), 
    # os.getenv("sf_api_key_4"), os.getenv("sf_api_key_5"), os.getenv("sf_api_key_6"), 
    # os.getenv("sf_api_key_7"), os.getenv("sf_api_key_8"), os.getenv("sf_api_key_9"), 
    # os.getenv("sf_api_key_10"), os.getenv("sf_api_key_11"), 
    # ]

    # # 遍历 API KEY 池，测试每个 KEY 是否可用
    # for api_key in kimi_api_keys:
    #     print(f"测试 API KEY 序号: {kimi_api_keys.index(api_key) + 1}, KEY: {api_key[:4]}...{api_key[-4:]}")
    #     # 初始化模型
    #     model_name = "moonshotai/Kimi-K2-Instruct-0905"
    #     base_url = os.getenv("sf_url")
    #     llm = OpenLLM(model_name, base_url, api_key)

    #     prompt = "connect testing"
    #     response = llm.generate(prompt)
    #     print(f"模型响应: {response}\n")

    test = 10
    name = "qwen3-235b-a22b-instruct"

    import time
    # 记录平均调用时间
    start_time = time.time()
    for i in range(test):
        # 记录每次调用的时间
        call_start_time = time.time()
        # print(f"第 {i+1} 次测试")
        llm = OpenLLM(name, os.getenv("iflow_url"), os.getenv("iflow_api_key"))
        # print(llm.get_model_info())
        print(f"\n开始测试 {name} 模型生成响应...\n")
        prompt = "connect testing"
        response = llm.generate(prompt)
        print(f"模型响应: {response}\n")
        # 计算每次调用的时间
        call_end_time = time.time()
        print(f"第 {i+1} 次调用时间: {call_end_time - call_start_time} 秒\n")
    end_time = time.time()
    print(f"平均调用时间: {(end_time - start_time) / test} 秒")
if __name__ == "__main__":
    test_kimi()



