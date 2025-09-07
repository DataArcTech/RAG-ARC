"""
MCP Router for FastAPI integration with FastMCP
"""
from fastapi import APIRouter
from fastapi.routing import Mount

from ..mcp.server import mcp

mcp_app = mcp.http_app(path="/")