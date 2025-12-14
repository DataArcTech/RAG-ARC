"""Lightweight datamodels shared by application + core presentation layers."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from encapsulation.data_model.schema import Chunk


@dataclass
class PipelineArtifacts:
    """Intermediate outputs captured from the RAG pipeline."""

    original_query: str
    rewritten_query: str
    retrieved_chunks: List[Chunk]
    reranked_chunks: List[Chunk]
    messages: List[Dict[str, str]]
    subgraph_data: Optional[Dict[str, Any]]
    subgraph_info: Optional[Dict[str, Any]]
    llm_response: Optional[str]

