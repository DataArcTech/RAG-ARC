"""
HippoRAG2 Graph Extractor Configuration

Optimized for minimal token usage with TSV format
"""

from framework.config import AbstractConfig
from typing import Literal, Optional, List
from pydantic import Field
from pathlib import Path
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

    enable_mindmap_extraction: bool = Field(
        default=False,
        description=(
            "Whether to extract mindmap TSV blocks into chunk.metadata['mindmap']. "
            "Disabled by default because HippoRAG/DeepSearch core flows do not require mindmaps; "
            "enable only when users need schema-layer scaffolding or mindmap UX."
        ),
    )

    temporal_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Optional custom temporal/business-time extraction prompt template. "
            "Use placeholders like {passage} and {language}. When empty, built-in EN/ZH prompts are used."
        ),
    )

    temporal_prompt_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional file path for the temporal/business-time extraction prompt template. "
            "When set, the file contents are used as the template."
        ),
    )

    enable_sdf_extraction: bool = Field(
        default=False,
        description=(
            "Whether to extract hierarchical process schema (HS) and convert it into an SDF contract "
            "stored in chunk metadata. Disabled by default for general-domain deployments."
        ),
    )

    sdf_hs_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Optional custom HS (hierarchical structure) extraction prompt template used by SDF extraction. "
            "Use placeholders like {passage} and {language}. When empty, built-in prompts are used."
        ),
    )

    sdf_hs_prompt_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional file path for the HS (hierarchical structure) extraction prompt template used by SDF extraction. "
            "When set, the file contents are used as the template."
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

    ner_prompt_path: Optional[str] = Field(
        default=None,
        description="Optional file path for the custom NER prompt template (overrides default).",
    )

    triple_prompt: Optional[str] = Field(
        default=None,
        description="Custom triple extraction prompt (overrides default)"
    )

    triple_prompt_path: Optional[str] = Field(
        default=None,
        description="Optional file path for the custom triple extraction prompt template (overrides default).",
    )

    kg_schema_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional KG schema YAML path used for schema-aware triple extraction prompting. "
            "When unset, the extractor will fall back to env KG_SCHEMA_PATH."
        ),
    )

    schema_aware_triple_extraction: Optional[bool] = Field(
        default=None,
        description=(
            "Whether to inject KG schema predicate allow-list + alias hints into the triple extraction prompt. "
            "When null, defaults to enabled when a KG schema path is configured (kg_schema_path or env KG_SCHEMA_PATH)."
        ),
    )

    schema_prompt_domain: Optional[str] = Field(
        default=None,
        description="Optional KG schema domain key to use when building prompt hints (defaults to schema.default_domain).",
    )

    schema_prompt_max_allowed_relations: int = Field(
        default=80,
        ge=0,
        le=500,
        description="Max number of allowed predicate tokens to include in the schema hint prompt.",
    )

    schema_prompt_max_relation_aliases: int = Field(
        default=120,
        ge=0,
        le=800,
        description="Max number of predicate alias entries to include in the schema hint prompt.",
    )

    @staticmethod
    def _validate_optional_path(value: Optional[str], *, field_name: str) -> None:
        token = str(value or "").strip()
        if not token:
            return
        path = Path(token)
        if not path.exists():
            raise ValueError(f"{field_name} points to a missing file: {token}")
        if not path.is_file():
            raise ValueError(f"{field_name} must be a file path, got: {token}")

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization"""
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than 0")
        if self.llm_config is None:
            raise ValueError("llm_config is required for HippoRAG2 extraction")

        self._validate_optional_path(self.temporal_prompt_path, field_name="temporal_prompt_path")
        self._validate_optional_path(self.sdf_hs_prompt_path, field_name="sdf_hs_prompt_path")
        self._validate_optional_path(self.ner_prompt_path, field_name="ner_prompt_path")
        self._validate_optional_path(self.triple_prompt_path, field_name="triple_prompt_path")
        self._validate_optional_path(self.kg_schema_path, field_name="kg_schema_path")

        if self.retry_attempts is not None:
            self.llm_config.max_retries = int(self.retry_attempts)
        if self.timeout is not None:
            self.llm_config.timeout = float(self.timeout)

    def build(self):
        return HippoRAG2Extractor(self)
