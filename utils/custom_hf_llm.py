# from langchain_core.language_models import LLM
# from typing import List, Optional
# from huggingface_hub import InferenceClient
# from pydantic import PrivateAttr
# import os

# class HuggingFaceLLM(LLM):
#     model: str = "EleutherAI/gpt-neo-1.3B" 
#     temperature: float = 0.7
#     max_new_tokens: int = 512

#     _client: InferenceClient = PrivateAttr()

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self._client = InferenceClient(
#             model=self.model,
#             token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
#         )

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         if "zephyr" in self.model.lower():
#             response = self._client.conversational(
#                 prompt=prompt,
#                 temperature=self.temperature,
#                 max_new_tokens=self.max_new_tokens,
#             )
#         else:
#             response = self._client.text_generation(
#                 prompt=prompt,
#                 temperature=self.temperature,
#                 max_new_tokens=self.max_new_tokens,
#                 stop_sequences=stop or [],
#                 do_sample=True
#             )
#         return response.generated_text


#     @property
#     def _llm_type(self) -> str:
#         return "custom-huggingface"

# utils/custom_hf_llm.py
from langchain_core.language_models import LLM
from typing import List, Optional
from huggingface_hub import InferenceClient
from pydantic import PrivateAttr
import os

class HuggingFaceLLM(LLM):
    model: str = "google/flan-t5-large"
    temperature: float = 0.7
    max_new_tokens: int = 512

    _client: InferenceClient = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = InferenceClient(
            model=self.model,
            token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Construct chat prompt manually
        formatted_prompt = (
            "<|system|>\nYou are a helpful assistant.\n"
            "<|user|>\n" + prompt + "\n"
            "<|assistant|>"
        )

        response = self._client.text_generation(
            prompt=formatted_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stop_sequences=stop or ["<|user|>", "<|system|>"]
        )
        return response.generated_text.strip()

    @property
    def _llm_type(self) -> str:
        return "custom-huggingface"
