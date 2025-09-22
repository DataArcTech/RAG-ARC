from framework.config import AbstractConfig
from encapsulation.llm.huggingface import HuggingFaceLLM
from typing import Literal

class HuggingFaceEmbeddingConfig(AbstractConfig):
    """Configuration for HuggingFace Embedding"""
    type: Literal["huggingface_embedding"] = "huggingface_embedding"
    model_name: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B"
    device: str = "cuda:0"
    task_types: list = ["embedding"]

    def build(self) -> HuggingFaceLLM:
        return HuggingFaceLLM(self)