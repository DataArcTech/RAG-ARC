"""Configuration for Listwise Rerank LLM"""

from framework.config import AbstractConfig
from encapsulation.llm.rerank.listwise import ListwiseRerankLLM
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from typing import Literal, Optional


class ListwiseRerankConfig(AbstractConfig):
    """Configuration for Listwise Rerank LLM"""
    # Discriminator for config type identification
    type: Literal["listwise_rerank"] = "listwise_rerank"

    # Chat LLM configuration - uses dependency injection
    chat_llm_config: OpenAIChatConfig  # Any ChatLLMBase config (OpenAI, etc.)

    # Prompt template configuration
    prompt_template: Optional[str] = None  # Custom prompt template (optional)

    def build(self) -> ListwiseRerankLLM:
        return ListwiseRerankLLM(self)

