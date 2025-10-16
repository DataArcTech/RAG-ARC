from fastapi import (
    APIRouter,
    status,
)
from pydantic import BaseModel
from typing import Optional
from framework.register import Register


router = APIRouter(prefix="/rag_inference", tags=["rag_inference"])

registrator = Register()


class ChatRequest(BaseModel):
    query: str
    owner_id: Optional[str] = None  # Optional for now, will be required after adding authentication


# This currently only supports one round of chat, will support multiple rounds once user login is supported.
@router.post("/chat", response_model=str, status_code=status.HTTP_200_OK)
def chat(request: ChatRequest):
    """
    Chat endpoint with optional user isolation

    Args:
        request: ChatRequest containing query and optional owner_id

    Returns:
        LLM response

    Note:
        owner_id is optional for backward compatibility.
        After adding JWT authentication, it will be extracted from the token.
    """
    rag_inference = registrator.get_object("rag_inference")
    return rag_inference.chat(request.query, owner_id=request.owner_id)