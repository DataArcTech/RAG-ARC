"""
Base extractor with simple concurrent extraction functionality.
"""

from abc import abstractmethod
from typing import List, Literal
from pydantic import Field
import asyncio
import logging

from encapsulation.llm.base import LLMBaseConfig
from encapsulation.data_model.data_model import Document, GraphData
from framework.module import AbstractModule
from framework.config import AbstractConfig

# Set up logger
logger = logging.getLogger(__name__)


class ExtractorBaseConfig(AbstractConfig):
    """Extractor基础配置"""
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
        """Build the Extractor instance"""
        raise NotImplementedError("Subclasses must implement build() method")


class ExtractorBase(AbstractModule):
    """BaseExtractor，只负责单轮抽取和并发控制"""

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.llm_config.build()

    @abstractmethod
    async def extract(self, document: Document) -> GraphData:
        """单轮抽取：从文档中抽取图数据

        Args:
            document: 待处理的文档

        Returns:
            GraphData: 抽取的图数据
        """
        pass

    async def process_document(self, document: Document) -> Document:
        """处理单个文档"""
        try:
            graph_data = await self.extract(document)
            document.graph = graph_data
            return document
        except Exception as e:
            logger.error(f"Error processing document {document.id}: {e}", exc_info=True)
            document.graph = GraphData()  # 返回空图数据
            return document

    async def extract_concurrent(self, documents: List[Document]) -> List[Document]:
        """批量异步抽取"""
        if not documents:
            return []

        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def process_with_semaphore(doc: Document) -> Document:
            async with semaphore:
                return await self.process_document(doc)

        return await asyncio.gather(*[process_with_semaphore(doc) for doc in documents])

    def __call__(self, documents: List[Document]) -> List[Document]:
        """同步接口"""
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.extract_concurrent(documents))
        except RuntimeError:
            return asyncio.run(self.extract_concurrent(documents))