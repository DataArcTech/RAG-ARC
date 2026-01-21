import json

import pytest

from application.deepsearch.tool_mcp_server import build_tool_mcp_server
from config.application.deepsearch_tool_server_config import load_tool_server_config
from core.deepsearch.tools import get_tool_descriptor
from core.graph_adapter.base import GraphAccessScope, GraphAdapterCapability, GraphAdapterMetadata
from core.graph_adapter import registry


class _StubLLM:
    def chat(self, messages, **kwargs):
        return json.dumps(
            {
                "reasoning": "stub reasoning",
                "tool_calls": [],
            }
        )


class _StubAdapter:
    def __init__(self):
        capability = GraphAdapterCapability(name="chain_of_exploration", modes=("beam_search",))
        self._metadata = GraphAdapterMetadata(
            adapter_name="stub_adapter",
            graph_type="stub_graph",
            version="v1",
            capabilities=(capability,),
        )

    async def prepare(self, question: str, *, access_scope=None):
        return None

    async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope=None, query_options=None):
        return {"nodes": [], "edges": [], "metadata": {}}

    async def context_filter(self, data, *, filter_type: str = "semantic", access_scope=None):
        return data

    async def summarize(self, channel: str, data, *, access_scope=None):
        return "summary"

    async def chain_traverse(self, strategy, *, access_scope=None):
        return {
            "strategy": strategy.get("strategy", "beam_search") if isinstance(strategy, dict) else "beam_search",
            "paths": [
                {
                    "path_id": "beam-0",
                    "nodes": ["OpenAI", "Anthropic"],
                    "triples": [{"head": "OpenAI", "relation": "partners_with", "tail": "Anthropic"}],
                    "score": 0.7,
                    "summary": "OpenAI partners with Anthropic.",
                }
            ],
        }

    def metadata(self):
        return self._metadata


@pytest.mark.asyncio
async def test_tool_mcp_server_invokes_registered_tool_with_adapter_injection(tmp_path):
    server = build_tool_mcp_server(
        llm_connector=_StubLLM(),
        enabled_tools=["graph.beam_search"],
        instructions="test",
        adapter=_StubAdapter(),
        default_scope=GraphAccessScope(scope_id="stub-owner"),
        tool_manager_config={"artifact_dir": str(tmp_path)},
    )

    descriptor = get_tool_descriptor("graph.beam_search")
    assert descriptor is not None
    tool = await server.fastmcp._tool_manager.get_tool(descriptor.namespace)
    assert tool is not None

    result = await tool.fn(
        None,
        question="Test MCP bridge lookup",
        plan_step="plan_test",
        extra={"seed_entities": ["OpenAI", "Anthropic"]},
        context_evidences=[],
    )

    assert result["tool_name"] == "graph.beam_search"
    assert result["evidences"]
    assert result["summary"]
    assert await server.list_registered_mcp_tool_names() == server.expected_mcp_tool_names()


@pytest.mark.asyncio
async def test_tool_server_config_loader_builds_server(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_SCOPE_ID", "owner-xyz")
    monkeypatch.setenv("TEST_TOOL_ARTIFACT_DIR", str(tmp_path / "tool_artifacts"))
    monkeypatch.delenv("DEEPSEARCH_DEFAULT_ADAPTER", raising=False)
    payload = {
        "type": "deepsearch_tool_mcp_server",
        "instructions": "Test scope ${TEST_SCOPE_ID}",
        "enabled_tools": ["graph.beam_search"],
        "llm_config": {
            "type": "openai_chat",
            "model_name": "gpt-4o-mini",
            "openai_api_key": "test-key",
            "openai_base_url": "https://example.com/v1"
        },
        "graph_adapter": {
            "type": "graph_adapter",
            "adapter_name": "config_stub_adapter",
            "parameters": {}
        },
        "tool_manager": {
            "enable_builtin_tools": True,
            "enabled_tools": {},
            "audit_label": "config-test",
            "artifact_dir": "${TEST_TOOL_ARTIFACT_DIR}"
        },
        "scope": {
            "scope_id": "${TEST_SCOPE_ID}",
            "scope_type": "owner",
            "labels": [],
            "attributes": {}
        }
    }
    config_path = tmp_path / "tool_server.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    registry.override_adapter("config_stub_adapter", lambda **kwargs: _StubAdapter())

    config = load_tool_server_config(config_path)
    server = config.build()

    assert server.default_scope is not None
    assert server.default_scope.scope_id == "owner-xyz"
    assert server.enabled_tools == {"graph.beam_search"}
