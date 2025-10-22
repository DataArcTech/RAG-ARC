from encapsulation.data_model.schema import Chunk
from fastapi import WebSocket, status
from encapsulation.data_model.orm_models import ChatMessage
from framework.singleton_decorator import singleton

@singleton
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(
        self, websocket: WebSocket, code: int = status.WS_1000_NORMAL_CLOSURE
    ):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            try:
                await websocket.close(code=code)
            except RuntimeError:
                # WebSocket already closed
                pass


    async def send_response(self, message: ChatMessage, chunks: list[Chunk], websocket: WebSocket, subgraph: dict | None = None):
        message_dict = {
            "id": str(message.id),
            "session_id": str(message.session_id),
            "content": message.content,
            "created_at": (
                message.created_at.isoformat() if message.created_at else None
            ),
        }
        chunks_dict = [{
            "id": str(chunk.id),
            "content": chunk.content,
            "metadata": chunk.metadata,
            "graph": chunk.graph.to_dict(),
        } for chunk in chunks]
        response_dict = {
            "message": message_dict,
            "chunks": chunks_dict,
        }
        # Add subgraph data if provided
        if subgraph is not None:
            response_dict["subgraph"] = subgraph
        await websocket.send_json(response_dict)