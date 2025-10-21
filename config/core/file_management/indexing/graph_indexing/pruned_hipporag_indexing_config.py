from typing import Literal
from pydantic import Field

from framework.config import AbstractConfig
from config.core.file_management.extractor.hipporag2_extractor_config import HippoRAG2ExtractorConfig
from config.encapsulation.database.graph_db.pruned_hipporag_igraph_config import PrunedHippoRAGIGraphConfig
from core.file_management.indexing.graph_indexing.pruned_hipporag_indexing import PrunedHippoRAGIndexer


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
    
    graph_store_config: PrunedHippoRAGIGraphConfig = Field(
        description="Configuration for the Pruned HippoRAG graph store"
    )

    def build(self):
        return PrunedHippoRAGIndexer(self)

