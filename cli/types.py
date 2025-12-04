from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from encapsulation.data_model.schema import Chunk
from application.rag_inference.cli_module import PipelineArtifacts


@dataclass
class ChunkPreview:
    """Lightweight chunk summary for CLI output."""

    index: int
    chunk_id: Optional[str]
    owner_id: Optional[str]
    preview: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chunk(cls, index: int, chunk: Chunk, max_chars: int = 160) -> "ChunkPreview":
        content = (chunk.content or "").strip().replace("\n", " ")
        if len(content) > max_chars:
            content = f"{content[:max_chars].rstrip()}..."
        return cls(
            index=index,
            chunk_id=getattr(chunk, "id", None),
            owner_id=getattr(chunk, "owner_id", None),
            preview=content,
            metadata=getattr(chunk, "metadata", {}) or {},
        )


@dataclass
class PipelineSummary:
    """Structured payload used for serialization/printing."""

    owner_id: str
    original_query: str
    rewritten_query: str
    llm_response: Optional[str]
    chunk_previews: List[ChunkPreview]
    subgraph: Optional[Dict[str, Any]]
    raw_chunks: List[Dict[str, Any]]

    @classmethod
    def from_artifacts(
        cls,
        owner_id: str,
        artifacts: PipelineArtifacts,
        max_chars: int = 160,
    ) -> "PipelineSummary":
        previews = [
            ChunkPreview.from_chunk(idx + 1, chunk, max_chars=max_chars)
            for idx, chunk in enumerate(artifacts.reranked_chunks)
        ]
        raw_chunks = [
            {
                "id": getattr(chunk, "id", None),
                "owner_id": getattr(chunk, "owner_id", None),
                "content": chunk.content,
                "metadata": getattr(chunk, "metadata", {}) or {},
            }
            for chunk in artifacts.reranked_chunks
        ]
        return cls(
            owner_id=owner_id,
            original_query=artifacts.original_query,
            rewritten_query=artifacts.rewritten_query,
            llm_response=artifacts.llm_response,
            chunk_previews=previews,
            subgraph=artifacts.subgraph_data,
            raw_chunks=raw_chunks,
        )
