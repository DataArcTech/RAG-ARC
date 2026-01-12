import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphQueryContext, ToolExecutionLog
from core.graph_adapter.base import GraphAdapterCapability, GraphAdapterMetadata
from core.deepsearch.tools import (
    BeamSearchTool,
    ContextRewriterTool,
    CrossAdapterPlannerTool,
    GraphTool,
    ParallelThinkTool,
    ToolDescriptor,
    ToolResult,
    ToolRunRequest,
)
from encapsulation.deepsearch.tooling import DeepSearchToolManager


def _tool_manager_configs(
    tmp_path: Path,
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
        "artifact_dir": str(tmp_path),
        "max_remote_evidences": 32,
        "max_remote_context_chars": 4096,
        "llm_connector": llm_connector,
    }


class _StubAdapter:
    def __init__(self):
        capability = GraphAdapterCapability(name="test")
        self._metadata = GraphAdapterMetadata(
            adapter_name="hipporag",
            graph_type="hipporag",
            version="test",
            capabilities=(capability,),
        )

    def metadata(self):
        return self._metadata

    async def chain_traverse(self, strategy, *, access_scope=None):  # noqa: ARG002
        mode = (strategy or {}).get("strategy") or "unknown"
        if mode != "beam_search":
            return {"strategy": mode, "paths": []}
        return {
            "strategy": mode,
            "paths": [
                {
                    "path_id": "beam-0",
                    "nodes": ["OpenAI", "Microsoft"],
                    "triples": [{"head": "OpenAI", "relation": "partnered_with", "tail": "Microsoft"}],
                    "score": 0.6,
                    "summary": "OpenAI partnered with Microsoft; Microsoft provides cloud + investment support.",
                },
                {
                    "path_id": "beam-1",
                    "nodes": ["OpenAI", "Azure"],
                    "triples": [{"head": "OpenAI", "relation": "runs_on", "tail": "Azure"}],
                    "score": 0.4,
                    "summary": "OpenAI workloads run on Azure as part of the partnership.",
                },
            ],
        }


class _StubLLM:
    def __init__(self, response: str):
        self.response = response

    def chat(self, messages, **kwargs):
        return self.response


@pytest.mark.asyncio
async def test_tool_manager_routes_to_mcp_with_arguments(tmp_path):
    pytest.importorskip("mcp")
    from encapsulation.mcp.client import MCPToolCallOutcome

    remote_payload = {
        "summary": "remote summary",
        "evidences": [
            {"chunk_id": "remote-1", "source": "mcp", "content": "remote evidence"},
        ],
        "think_notes": [
            {"plan_step_id": "plan_01", "reasoning": "remote think"},
        ],
        "diagnostics": {"remote": True},
    }
    call_result = SimpleNamespace(content=[SimpleNamespace(text=json.dumps(remote_payload))])
    log = ToolExecutionLog(
        tool_name="graph.think",
        server_name="stub",
        arguments_snapshot={},
        response_excerpt="remote summary",
        latency_ms=10,
        extra={"transport": "stdio"},
    )

    class _StubMCPClient:
        def __init__(self):
            self.calls = []

        async def call_tool(self, tool_name, arguments=None, graph_context=None, server_name=None):
            self.calls.append({"tool_name": tool_name, "arguments": arguments, "graph_context": graph_context})
            return MCPToolCallOutcome(result=call_result, log=log)

    mcp_client = _StubMCPClient()
    manager = DeepSearchToolManager(
        tool_configs={
            **_tool_manager_configs(tmp_path, enable_builtin_tools=True),
            "enabled_tools": {"graph.think": {"enabled": False}},
            "remote_argument_templates": {"graph.think": {"custom_field": "$question"}},
        },
        telemetry_client=None,
        mcp_client=mcp_client,
    )
    graph_context = GraphQueryContext(adapter_name="hipporag", question="hello")
    result = await manager.invoke(
        "graph.think",
        payload={"question": "hello", "context_evidences": [], "graph_context": graph_context},
    )
    assert result.summary == "remote summary"
    assert result.evidences and result.evidences[0].chunk_id == "remote-1"
    assert result.think_notes
    assert mcp_client.calls[0]["arguments"]["question"] == "hello"
    assert mcp_client.calls[0]["arguments"]["custom_field"] == "hello"


@pytest.mark.asyncio
async def test_tool_manager_serializes_complex_payloads_for_remote(tmp_path):
    pytest.importorskip("mcp")
    from encapsulation.mcp.client import MCPToolCallOutcome

    remote_payload = {"summary": "complex summary"}
    call_result = SimpleNamespace(content=[SimpleNamespace(text=json.dumps(remote_payload))])
    log = ToolExecutionLog(
        tool_name="graph.think",
        server_name="stub",
        arguments_snapshot={},
        response_excerpt="complex summary",
        latency_ms=15,
        extra={},
    )

    class _StubMCPClient:
        def __init__(self):
            self.calls = []

        async def call_tool(self, tool_name, arguments=None, graph_context=None, server_name=None):
            self.calls.append({"arguments": arguments})
            return MCPToolCallOutcome(result=call_result, log=log)

    meta = GraphAdapterMetadata(
        adapter_name="hipporag",
        graph_type="hipporag",
        version="1.0",
        capabilities=(GraphAdapterCapability(name="test"),),
    )
    context_evidences = [
        EvidenceChunk(chunk_id="local-1", source="graph", content="cached evidence"),
        {"chunk_id": "dict-2", "source": "graph", "content": "dict style"},
    ]

    manager = DeepSearchToolManager(
        tool_configs={
            **_tool_manager_configs(tmp_path, enable_builtin_tools=False),
            "remote_tools": {
                "graph.think": {
                    "description": "remote think",
                    "namespace": "rag-arc.deepsearch.remote.think",
                    "profile": "H",
                    "determinism": "llm_heavy",
                    "speed": "slow",
                    "cost": "high",
                    "strategy_tags": ["think"],
                }
            },
            "max_remote_context_chars": 16,
        },
        telemetry_client=None,
        mcp_client=_StubMCPClient(),
    )

    adapter = _StubAdapter()
    adapter._metadata = meta

    await manager.invoke(
        "graph.think",
        payload={
            "question": "serialize payload",
            "context_evidences": context_evidences,
            "adapter": adapter,
            "extra": {"alternate_adapters": [meta]},
        },
    )

    arguments = manager.mcp_router.mcp_client.calls[0]["arguments"]
    context_window = arguments["context_evidences"]
    assert context_window[0]["chunk_id"] == "local-1"
    assert context_window[1]["chunk_id"] == "dict-2"
    assert set(context_window[0].keys()) == {"chunk_id", "source", "score", "content"}
    assert len(context_window[0]["content"]) <= 16
    alt_adapters = arguments["extra"]["alternate_adapters"]
    assert isinstance(alt_adapters[0], dict)


@pytest.mark.asyncio
async def test_tool_manager_routes_remote_only_descriptor(tmp_path):
    pytest.importorskip("mcp")
    from encapsulation.mcp.client import MCPToolCallOutcome

    remote_payload = {
        "summary": "remote-only summary",
        "evidences": [],
        "diagnostics": {},
    }
    call_result = SimpleNamespace(content=[SimpleNamespace(text=json.dumps(remote_payload))])
    log = ToolExecutionLog(
        tool_name="custom.remote",
        server_name="stub",
        arguments_snapshot={},
        response_excerpt="remote-only summary",
        latency_ms=12,
        extra={"transport": "stdio"},
    )

    class _StubMCPClient:
        def __init__(self):
            self.calls = []

        async def call_tool(self, tool_name, arguments=None, graph_context=None, server_name=None):
            self.calls.append({"tool_name": tool_name, "arguments": arguments})
            return MCPToolCallOutcome(result=call_result, log=log)

    mcp_client = _StubMCPClient()
    manager = DeepSearchToolManager(
        tool_configs={
            **_tool_manager_configs(tmp_path, enable_builtin_tools=False),
            "remote_tools": {
                "custom.remote": {
                    "description": "remote only",
                    "namespace": "rag-arc.remote.custom",
                    "profile": "H",
                    "determinism": "llm_heavy",
                    "speed": "slow",
                    "cost": "high",
                    "strategy_tags": ["remote"],
                }
            },
        },
        telemetry_client=None,
        mcp_client=mcp_client,
    )
    result = await manager.invoke(
        "custom.remote",
        payload={"question": "remote q", "context_evidences": []},
    )
    assert result.summary == "remote-only summary"
    assert mcp_client.calls


@pytest.mark.asyncio
async def test_tool_manager_raises_when_remote_errors_even_if_local_tool_exists(tmp_path):
    class _LocalTool(GraphTool):
        descriptor = ToolDescriptor(
            name="custom.echo",
            channel="graph",
            description="echo",
            profile="F",
            determinism="deterministic",
            namespace="rag-arc.deepsearch.tools.fast.custom_echo",
        )

        async def run(self, request: ToolRunRequest) -> ToolResult:
            return ToolResult(summary=f"local::{request.question}")

    class _FailingMCPClient:
        async def call_tool(self, *args, **kwargs):
            raise RuntimeError("mcp-down")

    class _Telemetry:
        def __init__(self):
            self.remote_logs = []

        def log_tool_invocation(self, **kwargs):
            return None

        def log_remote_tool(self, *, tool_name, log):
            self.remote_logs.append(tool_name)

    telemetry = _Telemetry()
    manager = DeepSearchToolManager(
        tool_configs={
            **_tool_manager_configs(tmp_path, enable_builtin_tools=False),
            "enabled_tools": {"custom.echo": {"mcp_only": True}},
        },
        telemetry_client=telemetry,
        mcp_client=_FailingMCPClient(),
        local_tools={"custom.echo": _LocalTool()},
    )

    with pytest.raises(RuntimeError):
        await manager.invoke("custom.echo", payload={"question": "fallback", "context_evidences": []})


@pytest.mark.asyncio
async def test_cross_adapter_planner_handles_multiple_adapters():
    llm = _StubLLM(json.dumps({"summary": "comparison done", "actions": ["align depth"]}))
    tool = CrossAdapterPlannerTool(llm)
    adapter = _StubAdapter()

    request = ToolRunRequest(
        question="Compare adapters",
        plan_step="plan_01",
        context_evidences=[],
        adapter=adapter,
        access_scope=None,
        extra={
            "alternate_adapters": [
                {"adapter_name": "lightrag", "graph_type": "lightrag", "version": "1.0"},
            ]
        },
    )

    result = await tool.run(request)
    assert "comparison" in result.summary.lower()
    assert result.evidences
    assert result.diagnostics["adapter_count"] == 2


@pytest.mark.asyncio
async def test_parallel_think_generates_think_notes():
    llm = _StubLLM(json.dumps([{"thought": "branch one", "action": "probe"}]))
    tool = ParallelThinkTool(llm, branches=1)
    request = ToolRunRequest(
        question="Need more options",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=None,
        extra={},
    )

    result = await tool.run(request)
    assert "parallel think" in result.summary.lower()
    assert result.think_notes
    assert result.diagnostics["branches"] == 1
    assert result.diagnostics.get("thought_log")


@pytest.mark.asyncio
async def test_context_rewriter_emits_rewritten_chunk():
    llm = _StubLLM("rewritten context")
    tool = ContextRewriterTool(llm, window_size=2)
    evidences = [
        EvidenceChunk(chunk_id="c1", source="test", content="original text 1"),
        EvidenceChunk(chunk_id="c2", source="test", content="original text 2"),
    ]
    request = ToolRunRequest(
        question="Summarize",
        plan_step="plan_01",
        context_evidences=evidences,
        adapter=None,
        access_scope=None,
        extra={},
    )

    result = await tool.run(request)
    assert result.evidences
    assert result.evidences[0].content == "rewritten context"


@pytest.mark.asyncio
async def test_beam_search_tool_ranks_paths_with_llm():
    llm = _StubLLM(json.dumps([{"path_id": "beam-0", "score": 0.9}, {"path_id": "beam-1", "score": 0.2}]))
    adapter = _StubAdapter()
    tool = BeamSearchTool(llm, beam_size=2, max_depth=2)
    request = ToolRunRequest(
        question="Explain OpenAI partnerships",
        plan_step="plan_beam",
        context_evidences=[],
        adapter=adapter,
        access_scope=None,
        extra={},
    )
    result = await tool.run(request)
    assert result.evidences
    assert "Beam search" in result.summary
    assert result.diagnostics["selected_paths"] == len(result.evidences)
    assert result.think_notes


def test_llm_required_tools_not_created_when_connector_missing_attrs(tmp_path):
    manager = DeepSearchToolManager(
        tool_configs=_tool_manager_configs(tmp_path, enable_builtin_tools=True, llm_connector="fake-string"),
        telemetry_client=None,
    )
    assert manager.local_registry.resolve("graph.llm_chain_explorer") is None


@pytest.mark.asyncio
async def test_tool_manager_records_local_latency_in_diagnostics(tmp_path):
    class _EchoTool:
        descriptor = ToolDescriptor(
            name="graph.echo",
            channel="graph",
            description="Echo tool for telemetry tests.",
            profile="F",
            determinism="deterministic",
            namespace="rag-arc.deepsearch.tools.fast.custom_echo",
            mcp_callable=False,
        )

        async def run(self, request: ToolRunRequest) -> ToolResult:
            await asyncio.sleep(0.01)
            chunk = EvidenceChunk(chunk_id="echo-1", source="echo", content="ok")
            return ToolResult(summary="ok", evidences=[chunk], diagnostics={})

    manager = DeepSearchToolManager(
        tool_configs=_tool_manager_configs(tmp_path, enable_builtin_tools=False),
        telemetry_client=None,
        local_tools={"graph.echo": _EchoTool()},
    )
    graph_context = GraphQueryContext(adapter_name="hipporag", question="hello")
    result = await manager.invoke(
        "graph.echo",
        payload={"question": "hello", "context_evidences": [], "graph_context": graph_context},
    )
    assert isinstance(result.diagnostics.get("latency_ms"), int)
    assert result.diagnostics.get("evidence_count") == 1
