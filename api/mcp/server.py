"""
MCP Server implementation using FastMCP
"""
import datetime
import json
import uuid
from fastmcp import Context, FastMCP
from typing import Any, Dict, List, Literal, Optional
from fastapi import HTTPException
from encapsulation.data_model.schema import Chunk, GraphData
from framework.register import Register
from api.routers.auth import get_current_user_from_token, validate_user_session
from encapsulation.data_model.orm_models import ChatMessage, User
from core.utils.owner_guard import is_admin_owner
from core.presentation.graph_chain import build_graph_chain
from core.presentation.evidence import build_chat_evidence
from core.presentation.deepsearch_payload import trim_deepsearch_payload
from core.utils.json_safe import json_safe
from config.output_limits import CHAT_TOP_CHUNKS
from application.rag_inference.module import RAGInference
import logging

logger = logging.getLogger(__name__)

mcp = FastMCP("RAG-ARC MCP Server")
registrator = Register()

def _safe_get_current_user_from_token(token: str) -> Optional[User]:
    try:
        return get_current_user_from_token(token)
    except HTTPException as exc:
        logger.warning("Authentication failed: %s", exc.detail)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Error getting user from token: %s", exc, exc_info=True)
        return None

@mcp.tool(name="hello_world", description="test")
async def hello_world_tool() -> Dict[str, Any]:
    """
    A simple Hello World tool for MCP.
    """
    return {"message": "Hello, world!"}

@mcp.tool(name="create_chat", description="Create a new chat session")
async def create_chat(auth_token: str) -> Dict[str, Any]:
    """
    Create a new chat session for authenticated user

    Args:
        auth_token: JWT authentication token

    Returns:
        dict: Response containing session_id
    """
    try:
        # Authenticate user from token (use get_user_from_token to avoid HTTPException)
        current_user = _safe_get_current_user_from_token(auth_token)
        if not current_user:
            return {"isError": True, "message": "Authentication failed"}
        
        # Create session using session handler (use thread pool to avoid blocking)
        from framework.thread_pool import get_thread_pool
        session_handler = registrator.get_object("chat_session")
        chat_name = f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        session_id = await get_thread_pool().run_blocking(
            session_handler.create_session,
            current_user.id,
            chat_name
        )
        
        return {"session_id": str(session_id)}
        
    except Exception as e:
        logger.error(f"Error in create_chat function: {str(e)}")
        return {"isError": True, "message": f"Internal server error: {str(e)}"}


async def run_stdio_async() -> None:
    """Run the MCP server via stdio transport."""

    await mcp.run_stdio_async()


async def run_sse_async(*, host: str = "127.0.0.1", port: int = 8785, path: str = "mcp/chat") -> None:
    """Run the MCP server via SSE transport."""

    await mcp.run_http_async(transport="sse", host=host, port=port, path=path)


async def run_streamable_http_async(
    *, host: str = "127.0.0.1", port: int = 8785, path: str = "mcp/chat"
) -> None:
    """Run the MCP server via streamable-http transport."""

    await mcp.run_http_async(transport="streamable-http", host=host, port=port, path=path)


def http_app(*, path: str = "/mcp/chat", transport: str = "sse"):
    """Expose FastMCP ASGI application for embedding."""

    return mcp.http_app(path=path, transport=transport)


async def list_tools() -> List[str]:
    """Return registered tool names for diagnostics."""

    tools = await mcp.get_tools()
    if isinstance(tools, dict):
        return [tool.name for tool in tools.values()]
    return [tool.name for tool in tools]


@mcp.tool(name="chat", description="Streamable chat interface")
async def chat(
    session_id: str,
    query: str,
    auth_token: str,
    ctx: Context = None,
) -> dict:
    """
    Chat with RAG system using authenticated user and session validation

    Args:
        session_id: UUID of the chat session
        query: User's question/query
        auth_token: JWT authentication token
        ctx: MCP context for progress reporting

    Returns:
        dict: Response containing session_id and reply
    """
    try:
        # Authenticate user from token
        current_user = _safe_get_current_user_from_token(auth_token)
        if not current_user:
            return {"isError": True, "message": "Authentication failed"}

        # Validate session_id format
        try:
            session_uuid = uuid.UUID(session_id)
        except ValueError:
            return {"isError": True, "message": "Invalid session_id format"}
        
        # Get session handler and validate session ownership (use thread pool to avoid blocking)
        from framework.thread_pool import get_thread_pool
        session_handler = registrator.get_object("chat_session")
        session = await get_thread_pool().run_blocking(
            session_handler.get_session,
            session_uuid
        )
        
        if not session or not validate_user_session(session, current_user):
            return {"isError": True, "message": "Session not found or unauthorized access"}

        progress_events: list[dict[str, Any]] = []

        def _on_progress(payload: dict[str, Any]) -> None:
            progress_events.append(json_safe(dict(payload or {})))
            if ctx is None:
                return
            stage = str(payload.get("stage") or "")
            status = str(payload.get("status") or "")
            percent_map = {
                "rewrite": 10,
                "retrieve": 40,
                "rerank": 60,
                "subgraph_export": 80,
                "generate": 90,
            }
            pct = percent_map.get(stage, 0)
            if status == "end" and stage in percent_map:
                pct = min(percent_map[stage] + 5, 95)
            try:
                import asyncio

                loop = asyncio.get_running_loop()
                loop.create_task(ctx.report_progress(pct, 100, f"{stage}:{status}"))
            except Exception:  # noqa: BLE001
                return

        if ctx is not None:
            await ctx.report_progress(0, 100, "generating")

        # Get RAG inference and chat with user isolation
        # Use async version to avoid blocking the event loop
        rag_inference = registrator.get_object("rag_inference")
        response_text: str = ""
        chunks: list[Chunk] = []
        subgraph_data: GraphData = None
        response_text, chunks, subgraph_data, subgraph_info = await rag_inference.chat_async(
            query,
            owner_id=current_user.id,
            return_subgraph=True,
            progress_callback=_on_progress,
        )

        # Create message in the session (use thread pool to avoid blocking)
        message_handler = registrator.get_object("chat_message")
        await get_thread_pool().run_blocking(
            message_handler.create_message,
            ChatMessage(
                session_id=session_uuid, 
                source_file_ids=[chunk.id for chunk in chunks] if chunks else None,
                content={"role": "user", "content": query}, 
                created_at=datetime.datetime.now()
            )
        )
        await get_thread_pool().run_blocking(
            message_handler.create_message,
            ChatMessage(
                session_id=session_uuid,
                source_file_ids=[chunk.id for chunk in chunks] if chunks else None,
                content={"role": "assistant", "content": response_text},
                subgraph_data=subgraph_data if subgraph_data else None,
                created_at=datetime.datetime.now()
            )
        )
        
        if ctx is not None:
            await ctx.report_progress(100, 100, "done")

        graph_store = None
        try:
            if isinstance(rag_inference, RAGInference):
                graph_store = rag_inference.get_graph_store()
        except Exception:  # noqa: BLE001
            graph_store = None
        
        evidence = build_chat_evidence(
            chunks,
            subgraph_data=subgraph_data,
            subgraph_info=subgraph_info,
            max_chunks=CHAT_TOP_CHUNKS,
            graph_store=graph_store,
        )
        return {
            "session_id": session_id,
            "response": response_text,
            "chunks": json_safe(evidence.get("chunks")),
            "subgraph": json_safe(subgraph_data),
            "evidence": json_safe(evidence),
            "progress": progress_events,
        }
        
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}")
        return {"isError": True, "message": f"Internal server error: {str(e)}"}


@mcp.tool(name="rag-arc.deepsearch.run", description="Execute DeepSearch pipeline with the authenticated user scope")
async def deepsearch_run(
    question: str,
    auth_token: str,
    owner_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
    response_format: Literal["concise", "detailed"] = "detailed",
) -> dict:
    """Expose DeepSearchService via MCP for upstream orchestrators."""

    current_user = _safe_get_current_user_from_token(auth_token)
    if not current_user:
        return {"isError": True, "message": "Authentication failed"}

    try:
        service = registrator.get_object("deepsearch_service")
    except KeyError:
        return {"isError": True, "message": "DeepSearch service not initialized"}

    effective_owner = current_user.id
    if owner_id:
        try:
            requested_owner = uuid.UUID(owner_id)
        except ValueError:
            return {"isError": True, "message": "owner_id must be a valid UUID string"}
        if not is_admin_owner(current_user.id):
            return {"isError": True, "message": "Only administrators may override owner scope"}
        effective_owner = requested_owner

    try:
        result = await service.run(
            question,
            owner_id=str(effective_owner),
            metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("DeepSearch MCP run failed: %s", exc)
        return {"isError": True, "message": f"DeepSearch run failed: {exc}"}

    include_evidence = response_format != "concise"
    graph_store = None
    try:
        rag = registrator.get_object("rag_inference")
        if isinstance(rag, RAGInference):
            graph_store = rag.get_graph_store()
    except Exception:  # noqa: BLE001
        graph_store = None

    payload = trim_deepsearch_payload(result, include_evidence=include_evidence, graph_store=graph_store)
    if response_format == "concise":
        report_block = payload.get("report") or {}
        reasoning_block = payload.get("reasoning") or {}
        return {
            "question": payload.get("question"),
            "answer": report_block.get("answer"),
            "highlights": report_block.get("highlights"),
            "coverage_metrics": reasoning_block.get("coverage_metrics"),
        }
    return payload
