"""Configuration for DashScope Rerank API client."""

import os
from typing import Literal, Optional

from pydantic import Field

from encapsulation.llm.rerank.dashscope_rerank import DashScopeRerankLLM
from framework.config import AbstractConfig


class DashScopeRerankConfig(AbstractConfig):
    """Configuration for DashScope API-based reranker (qwen3-rerank).

    Environment variable defaults:
        RERANK_API_KEY      - DashScope API key
        RERANK_BASE_URL     - defaults to https://dashscope.aliyuncs.com/compatible-api/v1
        RERANK_MODEL_NAME   - defaults to qwen3-rerank
    """

    type: Literal["dashscope_rerank"] = "dashscope_rerank"

    api_key: str = Field(
        default_factory=lambda: os.getenv("RERANK_API_KEY", ""),
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv(
            "RERANK_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-api/v1",
        ),
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("RERANK_MODEL_NAME", "qwen3-rerank"),
    )
    instruct: str = Field(
        default="Given a web search query, retrieve relevant passages that answer the query.",
        description="Instruction prefix sent to the rerank model.",
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="HTTP request timeout in seconds.",
    )
    top_n: Optional[int] = Field(
        default=None,
        description="Default top_n to return (None = return all scored).",
    )

    def build(self) -> DashScopeRerankLLM:
        return DashScopeRerankLLM(self)
