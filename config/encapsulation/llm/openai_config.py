from typing import Literal, Optional
from pydantic import Field
from framework.config import AbstractConfig
from encapsulation.llm.openai import OpenAILLM

class OpenAIConfig(AbstractConfig):
    """
    OpenAI LLM configuration
    """
    type: Literal["openai"] = "openai"
    
    model_name: str = Field(description="Model name")
    task_types: Literal["chat", "embedding"] = Field(default="chat", description="Supported task types")
    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    base_url: Optional[str] = Field(default=None, description="API base URL")
    organization: Optional[str] = Field(default=None, description="Organization ID")
    max_retries: int = Field(default=3, description="Max retry attempts")
    timeout: float = Field(default=60.0, description="Request timeout")
    default_max_tokens: Optional[int] = Field(default=None, description="Default max tokens for chat")
    default_temperature: float = Field(default=0.7, description="Default temperature for chat")
    kwargs: dict = Field(default_factory=dict, description="Additional configuration parameters")
    
    def build(self):
        return OpenAILLM(config=self)
