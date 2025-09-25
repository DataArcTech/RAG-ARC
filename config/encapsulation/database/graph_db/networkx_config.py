from framework.config import AbstractConfig
from encapsulation.database.graph_db.networkx_graph import NetworkXGraphStore
from typing import Literal, Optional
from pydantic import Field


class NetworkXConfig(AbstractConfig):
    """NetworkX Graph Store Configuration Class"""
    type: Literal["networkx"] = "networkx"

    # Optional storage configuration for persistence
    storage_path: Optional[str] = Field(
        default=None,
        description="Optional path for saving/loading graph data. If None, graph is memory-only"
    )
    
    # Optional index name for persistence
    index_name: str = Field(
        default="networkx_index",
        description="Name for the saved index files"
    )
    
    # Performance tuning options
    auto_save: bool = Field(
        default=False,
        description="Whether to automatically save changes to disk"
    )
    
    # Graph configuration
    allow_self_loops: bool = Field(
        default=True,
        description="Whether to allow self-loops in the graph"
    )
    
    allow_parallel_edges: bool = Field(
        default=True,
        description="Whether to allow parallel edges between nodes"
    )

    def build(self):
        return NetworkXGraphStore(self)