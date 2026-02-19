import json
import requests

class BrainAgent:
    """
    Kết nối với Ollama/Llama: Luồng xử lý tư duy.
    """
    def __init__(self, model_name="llama3", api_url="http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.api_url = api_url

    def get_decision(self, prompt):
        """
        Gửi yêu cầu LLM để đưa ra quyết định hành động.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                return response.json().get("response", "")
            return f"Lỗi Brain: {response.status_code}"
        except Exception as e:
            return f"Lỗi Kết nối Brain: {str(e)}"
