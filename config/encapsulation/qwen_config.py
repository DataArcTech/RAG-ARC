from framework.config import AbstractConfig
from encapsulation.llm.qwen3 import QwenLLM
from typing import Literal

class QwenLLMConfig(AbstractConfig):
    """Configuration for Qwen LLM"""
    type: Literal["qwen"] = "qwen"
    model_name: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_reranker_0.6B"
    device: str = "cuda:0"
    cache_folder: str = None
    task_types: list = ["rerank"]
    instruction: str = "Given the user query, retrieve the relevant passages"
    batch_size: int = 4

    def build(self) -> QwenLLM:
        return QwenLLM(self)