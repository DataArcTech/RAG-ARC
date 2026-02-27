import pytest

from encapsulation.data_model.deepsearch import EvidenceChunk
from encapsulation.data_model.deepsearch import ToolExecutionLog
from encapsulation.deepsearch.tooling import DeepSearchToolManager, MCPToolRouter
from core.deepsearch.tools import ToolDescriptor, ToolResult, ToolRunRequest
from core.deepsearch.tooling.errors import ToolErrorKind, ToolInvocationError

def _tool_manager_configs(
    tmp_path,
    *,
    enable_builtin_tools: bool,
    llm_connector=None,
    enabled_tools: dict | None = None,
    remote_tools: dict | None = None,
) -> dict:
    return {
        "enable_builtin_tools": bool(enable_builtin_tools),
        "enabled_tools": dict(enabled_tools or {}),
        "remote_tools": dict(remote_tools or {}),
        "artifact_dir": f"io://deepsearch_artifacts/{tmp_path.name}",
        "max_remote_evidences": 32,
        "max_remote_context_chars": 4096,
        "llm_connector": llm_connector,
    }


@pytest.mark.asyncio
async def test_tool_manager_wraps_schema_errors(tmp_path) -> None:
    class _BadTool:
        descriptor = ToolDescriptor(
            name="graph.bad_schema",
            channel="graph",
            description="bad tool",
            profile="F",
            determinism="deterministic",
        )

        async def run(self, request: ToolRunRequest) -> ToolResult:  # noqa: ARG002
            raise ValueError("missing required field")

    configs = _tool_manager_configs(tmp_path, enable_builtin_tools=False, enabled_tools={"graph.bad_schema": {"enabled": True}})
    manager = DeepSearchToolManager(tool_configs=configs, telemetry_client=None, local_tools={"graph.bad_schema": _BadTool()})
    payload = {
        "question": "q",
        "plan_step": "plan_01",
        "context_evidences": [],
        "adapter": None,
        "access_scope": None,
        "extra": {},
        "graph_context": {"adapter_name": "hipporag", "metadata": {}},
    }
    with pytest.raises(ToolInvocationError) as excinfo:
        await manager.invoke("graph.bad_schema", payload=payload)
    assert excinfo.value.kind is ToolErrorKind.SCHEMA_ERROR
    assert excinfo.value.tool_name == "graph.bad_schema"


@pytest.mark.asyncio
async def test_tool_manager_marks_empty_hits(tmp_path) -> None:
    class _EmptyTool:
        descriptor = ToolDescriptor(
            name="graph.empty",
            channel="graph",
            description="empty tool",
            profile="F",
            determinism="deterministic",
        )

        async def run(self, request: ToolRunRequest) -> ToolResult:  # noqa: ARG002
            return ToolResult(summary="no match", evidences=[], diagnostics={})

    configs = _tool_manager_configs(tmp_path, enable_builtin_tools=False, enabled_tools={"graph.empty": {"enabled": True}})
    manager = DeepSearchToolManager(tool_configs=configs, telemetry_client=None, local_tools={"graph.empty": _EmptyTool()})
    payload = {
        "question": "q",
        "plan_step": "plan_01",
        "context_evidences": [],
        "adapter": None,
        "access_scope": None,
        "extra": {},
        "graph_context": {"adapter_name": "hipporag", "metadata": {}},
    }
    result = await manager.invoke("graph.empty", payload=payload)
    assert result.diagnostics.get("result_kind") == "empty_hit"


@pytest.mark.asyncio
async def test_tool_manager_marks_non_empty_hits(tmp_path) -> None:
    class _NonEmptyTool:
        descriptor = ToolDescriptor(
            name="graph.nonempty",
            channel="graph",
            description="non-empty tool",
            profile="F",
            determinism="deterministic",
        )

        async def run(self, request: ToolRunRequest) -> ToolResult:  # noqa: ARG002
            return ToolResult(
                summary="ok",
                evidences=[EvidenceChunk(chunk_id="c1", source="x", content="y")],
                diagnostics={},
            )

    configs = _tool_manager_configs(tmp_path, enable_builtin_tools=False, enabled_tools={"graph.nonempty": {"enabled": True}})
    manager = DeepSearchToolManager(tool_configs=configs, telemetry_client=None, local_tools={"graph.nonempty": _NonEmptyTool()})
    payload = {
        "question": "q",
        "plan_step": "plan_01",
        "context_evidences": [],
        "adapter": None,
        "access_scope": None,
        "extra": {},
        "graph_context": {"adapter_name": "hipporag", "metadata": {}},
    }
    result = await manager.invoke("graph.nonempty", payload=payload)
    assert result.diagnostics.get("result_kind") != "empty_hit"


@pytest.mark.asyncio
async def test_mcp_router_raises_provider_error_on_is_error() -> None:
    class _FakeResult:
        isError = True
        content = []

    class _FakeMCPClient:
        async def call_tool(self, tool_name: str, arguments=None, graph_context=None, server_name=None):  # noqa: ANN001,ARG002
            return type(
                "_Outcome",
                (),
                {
                    "result": _FakeResult(),
                    "log": ToolExecutionLog(
                        tool_name=tool_name,
                        server_name=server_name,
                        arguments_snapshot=dict(arguments or {}),
                        response_excerpt="boom",
                        latency_ms=1,
                        graph_context=graph_context,
                        extra={},
                    ),
                },
            )()

    router = MCPToolRouter(mcp_client=_FakeMCPClient(), default_server_name="default")
    descriptor = ToolDescriptor(
        name="graph.remote",
        channel="graph",
        description="remote tool",
        namespace="graph.remote",
        mcp_callable=True,
    )
    with pytest.raises(ToolInvocationError) as excinfo:
        await router.invoke(descriptor, payload={"arguments": {}, "graph_context": None})
    assert excinfo.value.kind is ToolErrorKind.PROVIDER_ERROR
