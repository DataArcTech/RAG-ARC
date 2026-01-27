import os

from framework.config import AbstractConfig
from typing import List, Optional, Literal
from pydantic import Field
from core.file_management.extractor.graphextractor import GraphExtractor
from config.encapsulation.llm.chat.openai import OpenAIChatConfig


class GraphExtractorConfig(AbstractConfig):
    """GraphExtractor configuration (JSON-only)"""
    type: Literal['graph_extractor'] = 'graph_extractor'

    entity_types: Optional[List[str]] = Field(
        default=None,
        description="Optional entity types filter (soft constraint for the model).",
    )

    kg_schema_path: Optional[str] = Field(
        default_factory=lambda: os.getenv("KG_SCHEMA_PATH", "").strip() or None,
        description="Optional KG schema YAML path (relation aliases/allowlist/unknown policy).",
    )

    schema_prompt_domain: Optional[str] = Field(
        default=None,
        description="Optional KG schema domain key (defaults to schema.default_domain).",
    )

    schema_prompt_max_allowed_relations: int = Field(
        default=80,
        ge=0,
        le=500,
        description="Max allowed relations to include in schema hint.",
    )

    schema_prompt_max_relation_aliases: int = Field(
        default=120,
        ge=0,
        le=800,
        description="Max alias entries to include in schema hint.",
    )

    edge_reference_time_override: Optional[str] = Field(
        default_factory=lambda: os.getenv("KG_EDGE_REFERENCE_TIME", "").strip() or None,
        description="Optional ISO-8601 UTC reference time for resolving relative time mentions.",
    )
    language_detection_chinese_ratio_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Detect zh when Chinese chars ratio exceeds this threshold.",
    )
    language_detection_default_language: Literal["zh", "en"] = Field(
        default="zh",
        description="Default language when input is empty/whitespace.",
    )

    max_concurrent: int = Field(default=100, description="Maximum number of concurrent operations", ge=1)
    error_policy: Literal["attach", "raise", "empty"] = Field(
        default="attach",
        description="How to handle extraction errors: attach=return empty graph with error metadata; raise=propagate; empty=legacy silent empty graph",
    )
    llm_config: OpenAIChatConfig = Field(default=None, description="Configuration for the LLM to be used")

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization"""
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than 0")
        if self.llm_config is None:
            raise ValueError("llm_config is required")

    def build(self):
        return GraphExtractor(self)
