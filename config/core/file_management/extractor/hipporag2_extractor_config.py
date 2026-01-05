"""
HippoRAG2 Graph Extractor Configuration

Optimized for minimal token usage with TSV format
"""

from framework.config import AbstractConfig
from typing import Literal, Optional, List
from pydantic import Field
from core.file_management.extractor.hipporag2_extractor import HippoRAG2Extractor
from config.encapsulation.llm.chat.openai import OpenAIChatConfig


class HippoRAG2ExtractorConfig(AbstractConfig):
    """HippoRAG2 Graph Extractor Configuration"""
    type: Literal['hipporag2_extractor'] = 'hipporag2_extractor'

    # LLM configuration
    llm_config: OpenAIChatConfig = Field(
        default=None,
        description="Configuration for the LLM to be used"
    )

    # Entity type specification
    entity_types: Optional[List[str]] = Field(
        default=None,
        description="List of entity types to extract. If None, LLM will freely determine entity types"
    )

    # Concurrency control
    max_concurrent: int = Field(
        default=100,
        description="Maximum number of concurrent operations",
        ge=1
    )

    batch_size: Optional[int] = Field(
        default=None,
        description="Optional batch size for extractor concurrency scheduling (limits in-flight tasks for very large corpora).",
        ge=1,
    )

    retry_attempts: Optional[int] = Field(
        default=None,
        description="Optional override for LLM retry attempts (mapped to llm_config.max_retries).",
        ge=0,
    )

    timeout: Optional[float] = Field(
        default=None,
        description="Optional override for LLM timeout seconds (mapped to llm_config.timeout).",
        gt=0,
    )

    error_policy: Literal["attach", "raise", "empty"] = Field(
        default="attach",
        description="How to handle extraction errors: attach=return empty graph with error metadata; raise=propagate; empty=legacy silent empty graph",
    )

    enable_temporal_extraction: bool = Field(
        default=True,
        description="Whether to extract business-time (effective_date/valid_from/valid_to) via LLM for temporal tools (e.g. latest_truth).",
    )

    enable_sdf_extraction: bool = Field(
        default=False,
        description=(
            "Whether to extract hierarchical process schema (HS) and convert it into an SDF contract "
            "stored in chunk metadata. Disabled by default for general-domain deployments."
        ),
    )

    sdf_store_raw_hs: bool = Field(
        default=False,
        description="Whether to store the raw HS text in chunk metadata (`sdf_hs`) in addition to the derived SDF JSON (`sdf`).",
    )

    sdf_max_events_per_chunk: int = Field(
        default=40,
        ge=1,
        le=300,
        description="Max HS event blocks to accept per chunk when SDF extraction is enabled.",
    )

    # Optional custom prompts (if user wants to override defaults)
    ner_prompt: Optional[str] = Field(
        default=None,
        description="Custom NER prompt (overrides default)"
    )

    triple_prompt: Optional[str] = Field(
        default=None,
        description="Custom triple extraction prompt (overrides default)"
    )

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization"""
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than 0")
        if self.llm_config is None:
            raise ValueError("llm_config is required for HippoRAG2 extraction")

        if self.retry_attempts is not None:
            self.llm_config.max_retries = int(self.retry_attempts)
        if self.timeout is not None:
            self.llm_config.timeout = float(self.timeout)

    def build(self):
        return HippoRAG2Extractor(self)
