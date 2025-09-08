"""
MCP Server implementation using FastMCP
"""
import uuid
from fastmcp import Context, FastMCP
from typing import Dict, Any

SESSIONS = {} # mock session table

async def mock_app_chat(messages: str):
    """
    App chat
    """
    return "mocked chat result"

mcp = FastMCP("RAG-ARC MCP Server")

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
    message: str,
    ctx: Context = None,
) -> dict:
    if session_id not in SESSIONS:
        return {"isError": True, "message": "unknown session_id"}

    SESSIONS[session_id].append({"role": "user", "text": message})
    prompt = "\n".join([f"{m['role']}: {m['text']}" for m in SESSIONS[session_id]]) + "\nassistant:"

    await ctx.report_progress(0, 100, "generating")
    resp = await mock_app_chat(messages=prompt)
    reply = resp.strip()
    SESSIONS[session_id].append({"role": "assistant", "text": reply})
    await ctx.report_progress(100, 100, "done")
    return {"session_id": session_id, "reply": reply}
