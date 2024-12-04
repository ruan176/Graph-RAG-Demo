from typing import List, Optional
from langchain.schema.runnable import Runnable
import requests

class LocalLlamaLLM():
    """
    Local LLM class defines LLM to interact with using Ollama.
    
    This enables demonstrator to operate with offline LLM, so no internet
    connection is required.

    Parameters below may need to be altered depending on specific set up.
    """
    def __init__(self, endpoint: str = "http://localhost:11434/api/generate", model: str = "llama3", max_tokens: int = 100, temperature: float = 0.0):
        self.endpoint = endpoint
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @property
    def _llm_type(self) -> str:
        return "local_llama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Method to interact with LLM.
        TODO: Use this instead of query.py?
        """
        prompt = str(prompt)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

        # Use try except block to catch an API request exception.
        try:
            response = requests.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()
            response_json = response.json()
            return response_json.get("response", "")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error communicating with Llama3 model: {e}")


class RunnableLocalLlama(Runnable):
    """
    Class required to wrap LocalLlamaLLM instance with Runnable to create usable chain object.
    """
    def __init__(self, llama_llm: LocalLlamaLLM):
        self.llama_llm = llama_llm

    def invoke(self, *args, **kwargs) -> str:
        # Extract the first positional argument as input
        input_str = args[0] if args else kwargs.get("input", "")
        # Pass 'stop' if provided in kwargs
        stop = kwargs.get("stop", None)
        return self.llama_llm._call(input_str, stop=stop)