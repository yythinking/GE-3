from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from langchain_openai import ChatOpenAI
import time

class OpenLLM:  
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
        
        if "kimi" in model_name or "glm" in model_name or "minimax" in model_name_lower or "qwen3" in model_name_lower:
            extra_body = {"enable_thinking": False}
        elif "doubao" in model_name:
            extra_body = {"thinking": {"type": "disabled"}}
        
        # 构建最终配置
        if extra_body:
            config = {**base_config, "extra_body": extra_body}
        else:
            config = base_config
        
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


def test_model_with_timeout(model_name: str, base_url: str, api_key: str, prompt: str, timeout_seconds: int = 30):
    """测试单个模型，超时后返回 timeout 状态。"""
    start_time = time.time()
    llm_attacker = OpenLLM(model_name, base_url, api_key)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(llm_attacker.generate, prompt)
        try:
            response = future.result(timeout=timeout_seconds)
            elapsed = time.time() - start_time
            return {
                "model": model_name,
                "status": "success",
                "elapsed": elapsed,
                "response": response,
            }
        except FutureTimeoutError:
            elapsed = time.time() - start_time
            return {
                "model": model_name,
                "status": "timeout",
                "elapsed": elapsed,
                "response": None,
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "model": model_name,
                "status": "error",
                "elapsed": elapsed,
                "response": f"{type(e).__name__}: {e}",
            }


def main():
    import os
    from dotenv import load_dotenv
    load_dotenv()

############################   更换模型名称  ######################################
    name_list = [
                #  "qwen3.5-122b-a10b",
                #  "qwen3-max-2026-01-23", 
                 "Qwen/Qwen3.5-397B-A17B",
                 "Qwen/Qwen3.5-397B-A17B",
                 "Qwen/Qwen3.5-397B-A17B",
                 "Qwen/Qwen3.5-397B-A17B",
                #  "MiniMax/MiniMax-M2.5",
                #  "qwen3-coder-next", 
                #  "kimi/kimi-k2.5",
                #  "glm-5"
                 ]
##############################################################################
    base_url = os.getenv("sf_url")
    api_key = os.getenv("sf_api_key")

    if not base_url or not api_key:
        print("环境变量缺失：请检查 sf_url 和 sf_api_key")
        return

    prompt = "你好，你是谁？你可以帮助我什么？你擅长什么事情？"
    timeout_seconds = 300

    all_results = []
    suite_start_time = time.time()

    for model_name in name_list:
        print(f"\n===== 开始测试模型: {model_name} =====")
        result = test_model_with_timeout(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
        )
        all_results.append(result)

        if result["status"] == "success":
            print(result["response"])
            print(f"模型调用+回答耗时: {result['elapsed']:.3f} 秒")
        elif result["status"] == "timeout":
            print(f"超过 {timeout_seconds} 秒无回复，判定超时，进入下一个模型。")
            print(f"本次耗时: {result['elapsed']:.3f} 秒")
        else:
            print(f"调用失败: {result['response']}")
            print(f"本次耗时: {result['elapsed']:.3f} 秒")

    suite_elapsed = time.time() - suite_start_time
    print("\n===== 测试汇总 =====")
    for item in all_results:
        print(f"{item['model']}: {item['status']}, {item['elapsed']:.3f} 秒")
    print(f"全部模型总耗时: {suite_elapsed:.3f} 秒")


if __name__ == "__main__":
    main()
