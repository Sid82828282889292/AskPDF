from langchain_core.language_models import LLM
from typing import List, Optional
from huggingface_hub import InferenceClient
from pydantic import PrivateAttr
import os

class HuggingFaceLLM(LLM):
    model: str = "HuggingFaceH4/zephyr-7b-beta"
    temperature: float = 0.7
    max_new_tokens: int = 512

    # Declare the private attribute for the client
    _client: InferenceClient = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = InferenceClient(
            model=self.model,
            token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._client.text_generation(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            stop_sequences=stop
        )
        return response

    @property
    def _llm_type(self) -> str:
        return "custom-huggingface"
