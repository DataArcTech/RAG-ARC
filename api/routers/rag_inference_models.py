import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from encapsulation.data_model.orm_models import ChatMessage
from encapsulation.data_model.schema import Chunk


class ChatRequest(BaseModel):
    query: str
    return_subgraph: bool = False
    target_owner_id: uuid.UUID | None = None
    include_all_owners: bool = False
    include_evidence: bool = False
    enable_web_search: bool = False


class StreamChatRequest(BaseModel):
    """Request model for POST SSE stream chat endpoint."""

    query: str
    return_subgraph: bool = True  # 默认启用，用于生成和存储 subgraph_data 到会话详情
    target_owner_id: Optional[uuid.UUID] = None
    include_all_owners: bool = False
    include_evidence: bool = False
    enable_web_search: bool = False
    enable_deepsearch: bool = False


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str
    chunks: list | None = None
    subgraph: dict | None = None
    evidence: Dict[str, Any] | None = None


class GraphOverviewResponse(BaseModel):
    """Response payload for the admin graph overview endpoint."""

    chunks: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]


def build_stream_chat_payload(
    message: ChatMessage,
    chunks: list[Chunk],
    *,
    subgraph: dict | None = None,
    evidence: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], dict | None]:
    """
    Build stream chat payload, returning (payload_dict, subgraph) tuple.
    Subgraph is extracted from arguments and should be placed at outer level.
    
    Returns:
        Tuple of (payload_dict without subgraph, subgraph_data)
    """
    message_dict = {
        "id": str(message.id),
        "session_id": str(message.session_id),
        "content": message.content,
        "created_at": (message.created_at.isoformat() if message.created_at else None),
    }
    chunks_dict = []
    for chunk in chunks:
        # Remove mindmap from metadata since subgraph is now stored at outer level
        original_metadata = dict(chunk.metadata or {})
        metadata = {}
        
        # Keep essential fields for frontend
        essential_fields = [
            "source_file_id",
            "filename",
            "score",
            "chunk_id",
            "chunk_index",
            "start_idx",
            "end_idx",
        ]
        
        for field in essential_fields:
            if field in original_metadata:
                metadata[field] = original_metadata[field]
        
        # Remove internal/technical fields that frontend doesn't need
        # These fields are removed: token_count, strategy, segment_type, 
        # is_fenced_code_block, parsed_content_id, parser_type, owner_id,
        # rerank_score, rerank_method, mindmap
        
        # Only include graph if it has actual data (non-empty entities or relations)
        graph_dict = chunk.graph.to_dict() if chunk.graph else {}
        has_graph_data = False
        if graph_dict:
            entities = graph_dict.get("entities", [])
            relations = graph_dict.get("relations", [])
            # Only include graph if it has entities or relations
            has_graph_data = bool(entities or relations)
        
        chunk_dict = {
            "id": str(chunk.id),
            "content": chunk.content,
        }
        # Only add metadata if it has any fields
        if metadata:
            chunk_dict["metadata"] = metadata
        # Only add graph if it has actual data
        if has_graph_data:
            chunk_dict["graph"] = graph_dict
        
        chunks_dict.append(chunk_dict)
    response_dict: Dict[str, Any] = {"message": message_dict, "chunks": chunks_dict}
    # Note: subgraph is NOT included in response_dict, it will be returned separately
    if evidence is not None:
        response_dict["evidence"] = evidence
    return response_dict, subgraph
