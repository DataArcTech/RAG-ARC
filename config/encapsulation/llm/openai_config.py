from typing import Literal, Optional
from pydantic import Field
from framework.config import AbstractConfig
from encapsulation.llm.openai import OpenAILLM

class OpenAIConfig(AbstractConfig):
    """
    OpenAI LLM configuration
    """
    type: Literal["openai"] = "openai"
    
    chat_model_name: str = Field(default="gpt-4o-mini", description="Chat Model name")
    embedding_model_name: str = Field(default="text-embedding-ada-002", description="Embedding Model name")
    task_types: Literal["chat", "embedding"] = Field(default="chat", description="Supported task types")
    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    base_url: Optional[str] = Field(default=None, description="API base URL")
    organization: Optional[str] = Field(default=None, description="Organization ID")
    max_retries: int = Field(default=3, description="Max retry attempts")
    timeout: float = Field(default=60.0, description="Request timeout")
    default_max_tokens: Optional[int] = Field(default=None, description="Default max tokens for chat")
    default_temperature: float = Field(default=0.7, description="Default temperature for chat")
    kwargs: dict = Field(default_factory=dict, description="Additional configuration parameters")
    chat_max_tokens: Optional[int] = Field(default=2000, description="Max tokens for chat")
    chat_temperature: Optional[float] = Field(default=0.7, description="Temperature for chat")
    chat_return_token_count: bool = Field(default=False, description="Return token count for chat")
    chat_system_prompt: Optional[str] = Field(
        default=(
            "Please answer the user's question based on the following background information. Please ensure:\n"
            "1. The answer is based on the provided evidence\n"
            "2. Cite relevant evidence in your answer (e.g., [Document1], [Event1], etc.)\n"
            "3. If information is insufficient, please state so clearly\n"
            "4. Distinguish between document information and event information sources\n\n"
            "{context}\n\n"
            "[User Question]\n"
            "{query}\n\n"
            "[Your Answer]\n"
        ),
        description="System prompt"
    )
    
    def build(self):
        return OpenAILLM(config=self)
