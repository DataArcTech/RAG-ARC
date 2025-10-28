from typing import Literal
from pydantic import Field

from framework.config import AbstractConfig
from config.core.file_management.extractor.hipporag2_extractor_config import HippoRAG2ExtractorConfig
from config.encapsulation.database.graph_db.pruned_hipporag_neo4j_config import PrunedHippoRAGNeo4jConfig
from core.file_management.indexing.graph_indexing.pruned_hipporag_indexing import PrunedHippoRAGIndexer


class PrunedHippoRAGNeo4jIndexerConfig(AbstractConfig):
    """
    Configuration for Pruned HippoRAG Graph Indexer with Neo4j backend.
    
    This indexer combines:
    - HippoRAG2Extractor: Extracts entities and relations from chunk content using TSV format
    - Pruned HippoRAG Neo4j Store: Stores chunks and their graph data with FAISS + Neo4j
    """
    type: Literal["pruned_hipporag_neo4j_indexer"] = "pruned_hipporag_neo4j_indexer"
    
    extractor_config: HippoRAG2ExtractorConfig = Field(
        description="Configuration for the HippoRAG2Extractor to extract graph data from chunks"
    )
    
    graph_store_config: PrunedHippoRAGNeo4jConfig = Field(
        description="Configuration for the Pruned HippoRAG Neo4j graph store"
    )

    def build(self):
        return PrunedHippoRAGIndexer(self)

