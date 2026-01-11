from typing import Literal, Union, Annotated
from pydantic import Field

from framework.config import AbstractConfig
from config.core.file_management.extractor.graphextractor_config import GraphExtractorConfig
from config.core.file_management.extractor.heuristic_cooccurrence_extractor_config import (
    HeuristicCooccurrenceExtractorConfig,
)
from config.encapsulation.database.graph_db.networkx_with_embedding_config import NetworkXVectorConfig
from config.encapsulation.database.graph_db.networkx_config import NetworkXConfig
from core.file_management.indexing.graph_indexing.networkx_indexing import NetworkXGraphIndexer


class NetworkXGraphIndexerConfig(AbstractConfig):
    """
    Configuration for NetworkX Graph Indexer.
    
    This indexer combines:
    - GraphExtractor: Extracts entities and relations from chunk content
    - NetworkX Graph Store: Stores chunks and their graph data with optional embeddings
    """
    type: Literal["networkx_graph_indexer"] = "networkx_graph_indexer"
    
    extractor_config: Annotated[
        Union[GraphExtractorConfig, HeuristicCooccurrenceExtractorConfig],
        Field(discriminator="type"),
    ] = Field(description="Extractor configuration for graph data.")
    
    graph_store_config: Annotated[
        Union[NetworkXVectorConfig, NetworkXConfig],
        Field(discriminator="type")
    ] = Field(
        description="Configuration for the NetworkX graph store (with or without embeddings)"
    )

    index_empty_graph_chunks: bool = Field(
        default=True,
        description="Whether to still add chunk nodes when graph extraction returns an empty graph.",
    )

    def build(self):
        return NetworkXGraphIndexer(self)
