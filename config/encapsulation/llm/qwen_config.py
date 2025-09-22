from typing import Literal, Optional, Dict, Any
from pydantic import Field
from framework.config import AbstractConfig
from encapsulation.llm.qwen3 import QwenLLM

class QwenConfig(AbstractConfig):
    """
    Qwen reranker model configuration
    """
    type: Literal["qwen_reranker"] = "qwen_reranker"
    
    model_name: str = Field(description="Model name")
    task_types: Literal["rerank"] = Field(default="rerank", description="Supported task types")
    device: str = Field(default="cpu", description="Device to use for model")
    cache_folder: Optional[str] = Field(default=None, description="Cache folder for model")
    model_kwargs: Optional[Dict[str, Any]] = Field(default=None, description="Model kwargs")
    instruction: str = Field(default="Given the user query, retrieve the relevant passages", description="Default instruction for reranking")
    kwargs: dict = Field(default_factory=dict, description="Additional configuration parameters")
    
    def build(self):
        return QwenLLM(config=self)
