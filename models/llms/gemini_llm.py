from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from ..interfaces.llm_interface import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.7):
        """
        Official Gemini initialization: no base_url required.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key

        # Use official Google Generative AI integration
        self.chat_model = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature
        )

    def generate(self, prompt: str) -> str:
        """
        Generate response - implement abstract method defined by parent class
        """
        try:
            # Official integration supports direct string or Message object input
            response = self.chat_model.invoke(prompt)
            content = response.content

            # --- Gemini returns list ---
            if isinstance(content, str):
                return content
            
            if isinstance(content, list):
                # For multimodal or complex output, extract all text content and merge
                extracted_text = []
                for item in content:
                    if isinstance(item, str):
                        extracted_text.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        extracted_text.append(item["text"])
                    elif hasattr(item, "text"): # Compatible with certain specific object formats
                        extracted_text.append(item.text)
                
                return "".join(extracted_text).strip()
            # --- Core fix logic ends ---

            return str(content) # Fallback handling
            
        except Exception as e:
            print(f"Gemini model {self.model_name} call failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def get_model_info(self):
        """Return model information"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "provider": "google_genai",
        }