from typing import Literal, Optional

from pydantic import BaseModel, Field

from framework.config import AbstractConfig
from application.rag_inference.module import RAGInference
from config.core.query_rewrite_config import LLMQueryRewriterConfig
from config.core.retrieval.multipath_config import MultiPathRetrieverConfig
from config.core.rerank_config import LLMRerankerConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig


class TavilyWebSearchConfig(BaseModel):
    """Runtime configuration for Tavily-backed web search (used by HippoRAG Q&A)."""

    enabled: bool = Field(True, description="Enable Tavily web search augmentation for HippoRAG Q&A.")
    endpoint_url: str = Field(
        "https://api.tavily.com/search",
        description="Tavily REST endpoint.",
    )
    api_key: Optional[str] = Field(None, description="Tavily API key (defaults to env TAVILY_API_KEY via JSON interpolation).")
    search_depth: Literal["basic", "advanced"] = Field("advanced", description="Tavily search depth.")
    timeout_seconds: float = Field(20.0, gt=0.0, le=120.0, description="HTTP timeout in seconds.")
    timeout_grace_seconds: float = Field(
        2.0,
        ge=0.0,
        le=30.0,
        description="Extra time budget for thread scheduling/serialization around Tavily calls.",
    )
    max_results: int = Field(5, ge=1, le=20, description="Maximum Tavily results to return.")
    aggregate_enabled: bool = Field(
        True,
        description=(
            "Apply source-level web result aggregation before downstream rerank/LLM steps "
            "(FileScore = sum(score) / sqrt(n+1))."
        ),
    )
    aggregate_group_by: Literal["domain", "url", "provider"] = Field(
        "domain",
        description="Grouping key used by Tavily aggregation.",
    )
    aggregate_max_groups: int = Field(
        3,
        ge=1,
        le=20,
        description="Maximum source groups kept after aggregation.",
    )
    aggregate_max_results_per_group: int = Field(
        2,
        ge=1,
        le=20,
        description="Maximum Tavily results retained per source group.",
    )


class RAGCandidateSelectionConfig(BaseModel):
    """Candidate selection knobs for HippoRAG Q&A."""

    graph_candidates_k: int = Field(30, ge=1, le=200, description="Number of graph candidates before web augmentation.")
    web_candidates_k: int = Field(5, ge=0, le=20, description="Number of web candidates to add.")
    rerank_keep_k: int = Field(5, ge=1, le=50, description="Number of chunks kept after LLM reranking.")

    file_prune_enabled: bool = Field(
        default=True,
        description=(
            "Whether to prune retrieved chunks by file coverage before reranking (file-first candidate selection). "
            "This improves precision and reduces reranker context cost."
        ),
    )
    file_prune_max_files: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Max number of files to keep after file-level aggregation (before reranking).",
    )
    file_prune_max_chunks_per_file: int = Field(
        default=6,
        ge=1,
        le=200,
        description="Max number of chunks to keep per selected file (before reranking).",
    )


class RAGInferenceConfig(AbstractConfig):
    type: Literal["rag_inference"] = "rag_inference"
    query_rewrite_config: LLMQueryRewriterConfig
    retrieval_config: MultiPathRetrieverConfig
    reranker_config: LLMRerankerConfig
    llm_config: OpenAIChatConfig
    web_search: TavilyWebSearchConfig = Field(default_factory=TavilyWebSearchConfig)
    candidate_selection: RAGCandidateSelectionConfig = Field(default_factory=RAGCandidateSelectionConfig)
    
    def build(self):
        return RAGInference(self)
