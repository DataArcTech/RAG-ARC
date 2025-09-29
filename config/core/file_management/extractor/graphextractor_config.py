from framework.config import AbstractConfig
from typing import List, Dict, Optional, Literal
from pydantic import Field
from core.file_management.extractor.graphextractor import GraphExtractor
from config.encapsulation.llm.chat.openai import OpenAIChatConfig


class GraphExtractorConfig(AbstractConfig):
    """GraphExtractor configuration"""
    type: Literal['graph_extractor'] = 'graph_extractor'

    entity_types: Optional[List[str]] = Field(default=None, description="entity_types")
    relation_types: Optional[List[str]] = Field(default=None, description="relation_types")
    extraction_prompt: Optional[str] = Field(default=None, description="extraction_prompt")
    cleaning_prompt: Optional[str] = Field(default=None, description="cleaning_prompt")
    entity_examples: Optional[List[Dict]] = Field(default=None, description="entity_examples")
    relation_examples: Optional[List[List]] = Field(default=None, description="relation_examples")

    enable_cleaning: bool = Field(default=True, description="enable_cleaning")
    enable_llm_cleaning: bool = Field(default=False, description="enable_llm_cleaning")
    max_rounds: int = Field(default=3, description="max_rounds", ge=1)

    max_concurrent: int = Field(default=100, description="Maximum number of concurrent operations", ge=1)
    llm_config: OpenAIChatConfig = Field(default=None, description="Configuration for the LLM to be used")

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization"""
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than 0")
        if self.llm_config is None:
            raise ValueError("llm_config is required")

    def build(self):
        return GraphExtractor(self)