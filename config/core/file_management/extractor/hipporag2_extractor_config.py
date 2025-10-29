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

    def build(self):
        return HippoRAG2Extractor(self)

