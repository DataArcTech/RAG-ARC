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
    return_subgraph: bool = False
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
) -> Dict[str, Any]:
    message_dict = {
        "id": str(message.id),
        "session_id": str(message.session_id),
        "content": message.content,
        "created_at": (message.created_at.isoformat() if message.created_at else None),
    }
    chunks_dict = [
        {
            "id": str(chunk.id),
            "content": chunk.content,
            "metadata": chunk.metadata,
            "graph": chunk.graph.to_dict(),
        }
        for chunk in chunks
    ]
    response_dict: Dict[str, Any] = {"message": message_dict, "chunks": chunks_dict}
    if subgraph is not None:
        response_dict["subgraph"] = subgraph
    if evidence is not None:
        response_dict["evidence"] = evidence
    return response_dict

