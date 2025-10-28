from typing import Literal, Union
from pydantic import Field

from framework.config import AbstractConfig
from config.core.file_management.extractor.hipporag2_extractor_config import HippoRAG2ExtractorConfig
from core.file_management.indexing.graph_indexing.pruned_hipporag_indexing import PrunedHippoRAGIndexer

# Import both graph store configs
try:
    from config.encapsulation.database.graph_db.pruned_hipporag_igraph_config import PrunedHippoRAGIGraphConfig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    PrunedHippoRAGIGraphConfig = None

from config.encapsulation.database.graph_db.pruned_hipporag_neo4j_config import PrunedHippoRAGNeo4jConfig


class PrunedHippoRAGIndexerConfig(AbstractConfig):
    """
    Configuration for Pruned HippoRAG Graph Indexer.
    
    This indexer combines:
    - HippoRAG2Extractor: Extracts entities and relations from chunk content using TSV format
    - Pruned HippoRAG Optimized Store: Stores chunks and their graph data with FAISS + igraph + SQLite
    """
    type: Literal["pruned_hipporag_indexer"] = "pruned_hipporag_indexer"
    
    extractor_config: HippoRAG2ExtractorConfig = Field(
        description="Configuration for the HippoRAG2Extractor to extract graph data from chunks"
    )
    
    graph_store_config: Union[PrunedHippoRAGIGraphConfig, PrunedHippoRAGNeo4jConfig] = Field(
        description="Configuration for the Pruned HippoRAG graph store (igraph or Neo4j)"
    )

    def build(self):
        return PrunedHippoRAGIndexer(self)

