"""
Base extractor with simple concurrent extraction functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Literal, Callable, TypeVar, Generic, Awaitable, Union
from pydantic import Field
import asyncio
import logging

from encapsulation.llm.base import LLMBase, LLMBaseConfig
from core.utils.data_model import Document
from framework.module import AbstractModule
from framework.config import AbstractConfig

# Set up logger
logger = logging.getLogger(__name__)

ConfigType = TypeVar("ConfigType", bound="ExtractorBaseConfig")

class ExtractorBaseConfig(AbstractConfig):
    """
    Abstract base class for all Extractor configurations.
    
    This class defines the common configuration parameters for all extractors.
    Subclasses must:
    1. Define a unique `type` field with a Literal value
    2. Implement the `build()` method to create the corresponding Extractor instance
    
    Attributes:
        type (Literal): The type of the extractor, must be overridden by subclasses
        max_concurrent (int): Maximum number of concurrent operations (default: 100, must be > 0)
        llm_config (LLMBaseConfig): Configuration for the LLM to be used (required)
    
    Validation:
        - max_concurrent must be greater than 0
        - llm_config is required and cannot be None
    """
    type: Literal["base_extractor"] = "base_extractor"

    max_concurrent: int = Field(default=100, description="Maximum number of concurrent operations", ge=1)
    llm_config: LLMBaseConfig = Field(default=None, description="Configuration for the LLM to be used")

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization"""
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than 0")
        if self.llm_config is None:
            raise ValueError("llm_config is required")

    @abstractmethod
    def build(self) -> "ExtractorBase":
        """
        Build the Extractor instance.
        
        This method must be implemented by subclasses to create and return 
        the corresponding Extractor instance.
        
        Returns:
            ExtractorBase: The constructed extractor instance
        """
        raise NotImplementedError("Subclasses must implement build() method")


class ExtractorBase(AbstractModule, Generic[ConfigType]):
    """
    Base class for extractors, defining simple concurrent extraction functionality.
    抽取结果统一放在metadata['triples']中
    
    Core features:
    - Single-round extraction with semaphore concurrency control
    - Synchronous call interface with async/await support
    - Comprehensive error handling and logging
    
    Subclasses need to implement:
    - _aextract: Asynchronously extract graph structure from a single document
    
    Usage:
    - Call extractor(documents) for synchronous processing
    - The method automatically handles async/sync context switching
    """
    config: ConfigType
    llm: LLMBase

    # ==================== Abstract Methods ====================
    @abstractmethod
    async def _aextract(
        self, 
        document: Document
    ) -> Document:
        """
        Asynchronously extract graph structure from a document (abstract method, must be implemented by subclasses)
        
        Args:
            document: Document to be processed
            
        Returns:
            Document: Processed document, metadata contains the extracted graph structure
        """
        pass

    # ==================== Core Processing Logic ====================

    async def _run_with_semaphore(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Simple semaphore-controlled concurrent extraction.
        
        Args:
            documents: List of documents to be processed.
            
        Returns:
            List of processed documents.
        """
        if not documents:
            return []
            
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def process_document(doc: Document) -> Document:
            async with semaphore:
                try:
                    return await self._aextract(doc)
                except Exception as e:
                    doc_id = getattr(doc, 'id', 'unknown')
                    logger.error(f"Error processing document {doc_id}: {e}", exc_info=True)
                    return doc  # Fallback to original

        # Process all documents concurrently
        return await asyncio.gather(*[process_document(doc) for doc in documents])
        

    def __call__(
        self, 
        documents: List[Document]
    ) -> List[Document]:
        """同步接口：提取图结构"""
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self._run_with_semaphore(documents))
        except RuntimeError:
            return asyncio.run(self._run_with_semaphore(documents))

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__