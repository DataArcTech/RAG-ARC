"""HippoRAG2 Graph Extractor Configuration (JSON-only)."""

import os

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

    edge_reference_time_override: Optional[str] = Field(
        default_factory=lambda: os.getenv("KG_EDGE_REFERENCE_TIME", "").strip() or None,
        description=(
            "Optional ISO-8601 UTC timestamp used as REFERENCE_TIME when extracting edges (for resolving relative time). "
            "If unset, the extractor uses the current UTC time and records it in graph metadata."
        ),
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

    dedup_by_content: bool = Field(
        default=True,
        description=(
            "Whether to de-duplicate identical chunk.content within a single extraction batch. "
            "This can significantly reduce LLM calls for PDFs with repeated headers/footers."
        ),
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

    extract_chunk_roles: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional filter for which chunk roles should run LLM extraction. "
            "When set, only chunks whose `chunk.metadata['chunk_role']` is in this list are extracted; "
            "other chunks are still ingested but get an empty graph (with an observable skip marker). "
            "This is useful for speed when semantic_unit chunking produces many table 'slice' chunks."
        ),
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

    # ==================== EXTRACTION CACHE (REDIS) ====================
    extraction_cache_enabled: bool = Field(
        default=True,
        description=(
            "Enable Redis-backed caching for HippoRAG2 extraction outputs (GraphData). "
            "This reduces repeated LLM calls during re-indexing or when chunks repeat across documents. "
            "Cache keys are tenant-scoped by default (share across users within a tenant; isolates across tenants)."
        ),
    )

    extraction_cache_ttl_seconds: int = Field(
        default=30 * 24 * 3600,
        ge=60,
        description="TTL for cached extraction outputs.",
    )

    extraction_cache_namespace: str = Field(
        default="ragarc:extract:hipporag2",
        description="Redis key namespace/prefix for extraction cache entries.",
    )

    extraction_cache_compress: bool = Field(
        default=True,
        description="Whether to compress cached values (zlib+base64) to reduce Redis memory/network usage.",
    )

    extraction_cache_mget_batch_size: int = Field(
        default=256,
        ge=1,
        le=5000,
        description="Max keys per Redis MGET batch when prefetching extraction cache.",
    )

    extraction_cache_mset_batch_size: int = Field(
        default=128,
        ge=1,
        le=5000,
        description="Max SETEX ops per pipeline execute when writing extraction cache.",
    )

    extraction_cache_key_scope: Literal["tenant", "owner"] = Field(
        default="tenant",
        description=(
            "Cache key scoping strategy. "
            "tenant=share extraction outputs across users within the same tenant (recommended); "
            "owner=isolate per owner_id (stricter, less reuse). "
            "In multi-tenant deployments, set tenant_id from JWT/middleware."
        ),
    )

    extraction_cache_max_unique_keys_per_call: int = Field(
        default=10000,
        ge=1,
        le=200000,
        description=(
            "Safety cap for Redis cache IO per extractor call. "
            "If unique chunk contents exceed this value, Redis cache is skipped for that call "
            "(dedup_by_content still applies). This prevents Redis overload on very large corpora."
        ),
    )

    extraction_cache_max_value_bytes: int = Field(
        default=96 * 1024,
        ge=1024,
        le=5 * 1024 * 1024,
        description=(
            "Max uncompressed JSON byte size for a single cached value. "
            "Oversized entries are skipped to reduce Redis memory/IO pressure."
        ),
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
        self._validate_optional_path(self.kg_schema_path, field_name="kg_schema_path")

        if self.retry_attempts is not None:
            self.llm_config.max_retries = int(self.retry_attempts)
        if self.timeout is not None:
            self.llm_config.timeout = float(self.timeout)

    def build(self):
        return HippoRAG2Extractor(self)
