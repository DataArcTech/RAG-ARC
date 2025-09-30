from typing import Literal, Union, Annotated
from pydantic import Field

from framework.config import AbstractConfig
from config.core.file_management.extractor.graphextractor_config import GraphExtractorConfig
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
    
    extractor_config: GraphExtractorConfig = Field(
        description="Configuration for the GraphExtractor to extract graph data from chunks"
    )
    
    graph_store_config: Annotated[
        Union[NetworkXVectorConfig, NetworkXConfig],
        Field(discriminator="type")
    ] = Field(
        description="Configuration for the NetworkX graph store (with or without embeddings)"
    )

    def build(self):
        return NetworkXGraphIndexer(self)

