from typing import Literal, Optional, List

from pydantic import Field

from framework.config import AbstractConfig
from core.file_management.extractor.heuristic_cooccurrence_extractor import HeuristicCooccurrenceExtractor


class HeuristicCooccurrenceExtractorConfig(AbstractConfig):
    """LLM-free extractor config for benchmark/offline mode."""

    type: Literal["heuristic_cooccurrence_extractor"] = "heuristic_cooccurrence_extractor"

    stopwords_language: str = Field(default="en", description="Stopwords language code (best-effort).")
    extra_stopwords: Optional[List[str]] = Field(default=None, description="Extra stopwords for entity filtering.")

    max_entity_words: int = Field(default=4, ge=1, description="Max words per extracted entity phrase.")
    min_entity_len_chars: int = Field(default=3, ge=1, description="Min chars for an entity phrase (after normalize).")
    max_entities_per_chunk: int = Field(default=32, ge=1, description="Max entities to keep per chunk.")
    default_entity_type: str = Field(default="Entity", description="Entity type to assign for heuristic entities.")

    enable_cooccurrence_relations: bool = Field(default=True, description="Whether to emit co-occurrence relations.")
    cooccurrence_relation_type: str = Field(default="CO_OCCUR", description="Relation type for co-occurrence edges.")
    max_cooccurrence_pairs_per_chunk: int = Field(default=256, ge=0, description="Max co-occurrence pairs per chunk.")

    max_concurrent: int = Field(default=256, ge=1, description="Maximum number of concurrent operations.")
    error_policy: Literal["attach", "raise", "empty"] = Field(
        default="attach",
        description="How to handle extraction errors: attach=return empty graph with error metadata; raise=propagate; empty=legacy silent empty graph",
    )

    def build(self):
        return HeuristicCooccurrenceExtractor(self)
