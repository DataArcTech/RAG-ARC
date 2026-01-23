import asyncio
import socket
from contextlib import asynccontextmanager

import pytest
import uvicorn

from application.deepsearch.tool_mcp_server import build_tool_mcp_server
from encapsulation.deepsearch.tooling import DeepSearchToolManager
from core.deepsearch.tools import ToolResult, ToolRunRequest, get_tool_descriptor
from core.graph_adapter.base import GraphAccessScope, GraphAdapterCapability, GraphAdapterMetadata
from encapsulation.mcp.client import MCPToolClient


class _StubAdapter:
    def metadata(self):
        capability = GraphAdapterCapability(name="stub_capability", modes=("think",))
        return GraphAdapterMetadata(
            adapter_name="stub_adapter",
            graph_type="stub_graph",
            version="v1",
            capabilities=(capability,),
        )


class _ScopeEchoTool:
    def __init__(self, descriptor):
        self.descriptor = descriptor

    async def run(self, request: ToolRunRequest) -> ToolResult:
        scope_id = request.access_scope.scope_id if request.access_scope else None
        return ToolResult(summary=f"scope::{scope_id}", diagnostics={"observed_scope_id": scope_id})


@asynccontextmanager
async def _serve_mcp_app(app):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
        sock.close()
    except PermissionError:
        pytest.skip("Socket operations are not permitted in this environment.")

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.01)
        if not server.started:
            raise RuntimeError("uvicorn server failed to start")
        yield host, port
    finally:
        server.should_exit = True
        await task


@pytest.mark.asyncio
async def test_mcp_tool_naming_and_scope_override_default_ignored(tmp_path):
    descriptor = get_tool_descriptor("think")
    assert descriptor is not None
    server_scope = GraphAccessScope(scope_id="server-owner", scope_type="owner")

    server = build_tool_mcp_server(
        llm_connector=object(),
        enabled_tools=["think"],
        instructions="test",
        adapter=_StubAdapter(),
        default_scope=server_scope,
        tool_manager_config={
            "enable_builtin_tools": False,
            "enabled_tools": {"think": {"enabled": True}},
            "artifact_dir": str(tmp_path),
        },
        local_tools={"think": _ScopeEchoTool(descriptor)},
    )

    app = server.http_app(path="/mcp", transport="sse")
    async with _serve_mcp_app(app) as (host, port):
        server_uri = f"http://{host}:{port}/mcp"
        mcp_client = MCPToolClient(server_uri=server_uri, transport="sse", persistent_session=False)

        tools = await mcp_client.list_tools()
        assert any(tool.name == descriptor.namespace for tool in tools)

        manager = DeepSearchToolManager(
            tool_configs={
                "enable_builtin_tools": False,
                "enabled_tools": {"think": {"mcp_only": True}},
                "artifact_dir": str(tmp_path),
                "max_remote_evidences": 32,
                "max_remote_context_chars": 4096,
            },
            telemetry_client=None,
            mcp_client=mcp_client,
        )
        result = await manager.invoke(
            "think",
            payload={
                "question": "scope test",
                "context_evidences": [],
                "access_scope": GraphAccessScope(scope_id="attacker", scope_type="owner"),
            },
        )
        assert result.summary == "scope::server-owner"


@pytest.mark.asyncio
async def test_mcp_tool_scope_override_trusted_token_allows_override(tmp_path):
    descriptor = get_tool_descriptor("think")
    assert descriptor is not None
    server_scope = GraphAccessScope(scope_id="server-owner", scope_type="owner")

    server = build_tool_mcp_server(
        llm_connector=object(),
        enabled_tools=["think"],
        instructions="test",
        adapter=_StubAdapter(),
        default_scope=server_scope,
        tool_manager_config={
            "enable_builtin_tools": False,
            "enabled_tools": {"think": {"enabled": True}},
            "artifact_dir": str(tmp_path),
        },
        local_tools={"think": _ScopeEchoTool(descriptor)},
        scope_override_policy="allow_trusted",
        scope_override_token="test-secret",
    )

    app = server.http_app(path="/mcp", transport="sse")
    async with _serve_mcp_app(app) as (host, port):
        server_uri = f"http://{host}:{port}/mcp"
        mcp_client = MCPToolClient(server_uri=server_uri, transport="sse", persistent_session=False)

        manager = DeepSearchToolManager(
            tool_configs={
                "enable_builtin_tools": False,
                "enabled_tools": {"think": {"mcp_only": True}},
                "artifact_dir": str(tmp_path),
                "max_remote_evidences": 32,
                "max_remote_context_chars": 4096,
            },
            telemetry_client=None,
            mcp_client=mcp_client,
        )
        result = await manager.invoke(
            "think",
            payload={
                "question": "scope test",
                "context_evidences": [],
                "access_scope": GraphAccessScope(scope_id="attacker", scope_type="owner"),
                "extra": {"scope_override_token": "test-secret"},
            },
        )
        assert result.summary == "scope::attacker"
