from typing import List, Union, Annotated, Literal
from pydantic import Field

from framework.config import AbstractConfig
from config.core.file_management.parser_combinator_config import ParserCombinatorConfig
from config.core.file_management.chunker.chunker_config import (
    TokenChunkerConfig,
    RecursiveChunkerConfig,
    MarkdownHeaderChunkerConfig,
    SemanticChunkerConfig
)
from config.core.file_management.indexing.faiss_indexing_config import FaissIndexerConfig
from config.core.file_management.indexing.bm25_indexing_config import BM25IndexerConfig
from config.core.file_management.indexing.graph_indexing.networkx_indexing_config import NetworkXGraphIndexerConfig
from config.core.file_management.indexing.graph_indexing.pruned_hipporag_indexing_config import PrunedHippoRAGIndexerConfig
from config.core.file_management.indexing.graph_indexing.pruned_hipporag_neo4j_indexing_config import PrunedHippoRAGNeo4jIndexerConfig
from config.core.file_management.storage.file_storage import FileStorageConfig
from config.core.file_management.storage.parsed_content_storage import ParsedContentStorageConfig
from config.core.file_management.storage.chunk_storage import ChunkStorageConfig

from core.file_management.index_manager import IndexManager

class IndexManagerConfig(AbstractConfig):
    """Configuration for IndexManager"""
    type: Literal["index_manager"] = "index_manager"

    # Storage configurations
    file_storage_config: FileStorageConfig
    parsed_content_storage_config: ParsedContentStorageConfig
    chunk_storage_config: ChunkStorageConfig

    # Parser configuration
    parser_config: ParserCombinatorConfig = Field(
        default_factory=lambda: ParserCombinatorConfig(),
        description="Parser configuration for content parsing"
    )

    # Chunker configuration
    chunker_config: Annotated[
        Union[TokenChunkerConfig, RecursiveChunkerConfig, MarkdownHeaderChunkerConfig, SemanticChunkerConfig],
        Field(discriminator="type")
    ] = Field(
        default_factory=lambda: TokenChunkerConfig(),
        description="Chunker configuration for text chunking"
    )

    # Indexer configurations (optional, can be empty list if no indexing needed)
    indexer_configs: List[Annotated[
        Union[FaissIndexerConfig, BM25IndexerConfig, NetworkXGraphIndexerConfig, PrunedHippoRAGIndexerConfig, PrunedHippoRAGNeo4jIndexerConfig],
        Field(discriminator="type")
    ]] = Field(
        default_factory=list,
        description="List of indexer configurations for building indexes"
    )

    def build(self):
        return IndexManager(self)