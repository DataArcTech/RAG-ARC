from framework.config import AbstractConfig
from encapsulation.llm.openai import OpenAILLM
from typing import Literal

class OpenAIChatConfig(AbstractConfig):
    """Configuration for OpenAI Chat LLM"""
    type: Literal["openai"] = "openai"
    model_name: str = "gpt-4o-mini"
    task_types: list = ["chat"]
    base_url: str = "https://api.gptsapi.net/v1"
    api_key: str = "sk-xxx"
    max_tokens: int = 100
    temperature: float = 0.2

    def build(self) -> OpenAILLM:
        return OpenAILLM(self)