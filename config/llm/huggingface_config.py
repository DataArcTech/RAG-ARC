from typing import Literal, Optional, Dict, Any
from pydantic import Field
from framework.config import AbstractConfig
from encapsulation.llm.huggingface import HuggingFaceLLM

class HuggingFaceEmbedConfig(AbstractConfig):
    """
    HuggingFace embedding model configuration
    """
    type: Literal["huggingface_embedding"] = "huggingface_embedding"
    
    model_name: str = Field(description="Model name")
    task_types: Literal["embedding"] = Field(default="embedding", description="Supported task types")
    device: str = Field(default="cpu", description="Device to use for embedding")
    cache_folder: Optional[str] = Field(default=None, description="Cache folder for embedding model")
    model_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model kwargs for embedding model")
    encode_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Encode kwargs for embedding model")
    kwargs: dict = Field(default_factory=dict, description="Additional configuration parameters")
    
    def build(self):
        return HuggingFaceLLM(self)
