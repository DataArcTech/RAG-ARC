"""
MCP Server implementation using FastMCP
"""
from fastmcp import FastMCP
from typing import Dict, Any

mcp = FastMCP("RAG-ARC MCP Server")

@mcp.tool(name="hello_world", description="test")
async def hello_world_tool() -> Dict[str, Any]:
    """
    A simple Hello World tool for MCP.
    """
    return {"message": "Hello, world!"}



