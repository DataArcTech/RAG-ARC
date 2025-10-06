"""
MCP Server implementation using FastMCP
"""
import uuid
from fastmcp import Context, FastMCP
from typing import Dict, Any
from framework.register import Register
from config.application.rag_inference_config import RAGInferenceConfig

SESSIONS = {} # mock session table

mcp = FastMCP("RAG-ARC MCP Server")
registrator = Register() # singleton

@mcp.tool(name="hello_world", description="test")
async def hello_world_tool() -> Dict[str, Any]:
    """
    A simple Hello World tool for MCP.
    """
    return {"message": "Hello, world!"}

@mcp.tool(name="create_chat", description="Create a new chat session")
async def create_chat() -> Dict[str, Any]:
    """
    Create a new chat session
    """
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = [] # mock session
    return {"session_id": session_id}


@mcp.tool(name="chat", description="Streamable chat interface")
async def chat(
    session_id: str,
    query: str,
    ctx: Context = None,
) -> dict:
    # if session_id not in SESSIONS:
    #     return {"isError": True, "message": "unknown session_id"}

    # SESSIONS[session_id].append({"role": "user", "text": query})

    await ctx.report_progress(0, 100, "generating")
    rag_inference = registrator.get_object("rag_inference")
    reply = rag_inference.chat(query)
    # SESSIONS[session_id].append({"role": "assistant", "text": reply})
    await ctx.report_progress(100, 100, "done")
    # return {"session_id": session_id, "reply": reply}
    return {"session_id": "fake_session_id", "reply": reply}

