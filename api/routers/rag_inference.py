from fastapi import (
    APIRouter,
    status,
)
from pydantic import BaseModel
from framework.register import Register


router = APIRouter(prefix="/rag_inference", tags=["rag_inference"])

registrator = Register()


class ChatRequest(BaseModel):
    query: str


# This currently only supports one round of chat, will support multiple rounds once user login is supported.
@router.post("/chat", response_model=str, status_code=status.HTTP_200_OK)
def chat(request: ChatRequest):
    rag_inference = registrator.get_object("rag_inference")
    return rag_inference.chat(request.query)