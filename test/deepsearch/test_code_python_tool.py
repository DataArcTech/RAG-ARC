import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from core.deepsearch.trace.context import TraceEvent, reset_trace_emitter, set_trace_emitter
from encapsulation.deepsearch.tooling import DeepSearchToolManager
from core.deepsearch.tools.base import ToolDescriptor, ToolResult, ToolRunRequest
from api.routers.deepsearch_weaver_render import render_trace_payload
from core.deepsearch.reasoning import GraphReasoningLoop
from encapsulation.data_model.deepsearch import GraphQueryContext
from core.graph_adapter.base import GraphAccessScope


class _CaptureEmitter:
    def __init__(self) -> None:
        self.events: List[TraceEvent] = []

    async def emit(self, event: TraceEvent) -> None:
        self.events.append(event)


def _tool_manager_configs(tmp_path, *, enabled_tools: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "enable_builtin_tools": True,
        "enabled_tools": dict(enabled_tools or {}),
        "remote_tools": {},
        "artifact_dir": f"io://deepsearch_artifacts/{tmp_path.name}",
        "max_remote_evidences": 32,
        "max_remote_context_chars": 4096,
        "llm_connector": None,
    }


class _StubAdapter:
    async def prepare(self, question: str, *, access_scope=None) -> None:  # pragma: no cover
        return None

    async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope=None, query_options=None):
        return {
            "chunks": [
                {"content": f"chunk::{query}", "metadata": {"chunk_id": "c1"}},
            ],
            "nodes": [{"id": "n1"}],
            "edges": [{"id": "e1"}],
            "metadata": {"adapter": "hipporag"},
        }

    async def context_filter(self, data, *, filter_type: str = "semantic", access_scope=None):
        return data

    async def summarize(self, channel: str, data, *, access_scope=None):
        return f"summary::{channel}"

    async def chain_traverse(self, strategy, *, access_scope=None):
        return {"strategy": strategy.get("strategy"), "hops": 1}

    def metadata(self):
        return type(
            "_Meta",
            (),
            {
                "adapter_name": "hipporag",
                "graph_type": "hipporag",
                "version": "test",
                "capabilities": (),
                "domain_tags": (),
                "config_fingerprint": None,
            },
        )()


class _SequencedLLM:
    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.seen_user_payloads: List[Dict[str, Any]] = []

    def chat(self, messages, **kwargs):
        try:
            user = messages[-1]["content"]
            self.seen_user_payloads.append(json.loads(user))
        except Exception:
            self.seen_user_payloads.append({})
        if not self._responses:
            raise RuntimeError("No more stub responses")
        return self._responses.pop(0)


def _strategy_config(*, think_overrides: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "strategy_name": "ppr_chain",
        "allow_semantic_channel": True,
        "chain_depth": 2,
        "parallel_branches": 1,
        "max_parallel_branches": 4,
        "step_summary_max_chars": 2000,
        "tool_context_max_evidences": 5,
        "tool_context_max_chars": 800,
        "coverage_expected_min_chunks": 3,
        "trace_reflection_enabled": False,
        "trace_reflection_max": 0,
        "tool_timeout_seconds": 10.0,
        "think": {
            "tool_name": "think",
            "every_n_steps": 0,
            "min_coverage": 0.0,
            "enable_tool_calls": False,
            "max_tool_calls": 0,
            "tool_call_concurrency": 0,
            "tool_catalog_max_items": 0,
            "include_llm_tools": True,
            "max_rounds_per_checkpoint": 1,
        },
    }
    merged = dict(base["think"])
    merged.update(dict(think_overrides))
    base["think"] = merged
    return base


@pytest.mark.asyncio
async def test_code_python_tool_executes_and_emits_trace(tmp_path):
    manager = DeepSearchToolManager(_tool_manager_configs(tmp_path), telemetry_client=None)
    emitter = _CaptureEmitter()
    token = set_trace_emitter(emitter)
    try:
        payload = {
            "question": "calc sqrt(9)",
            "plan_step": "plan_01",
            "context_evidences": [],
            "adapter": None,
            "access_scope": None,
            "extra": {
                "purpose": "unit-test",
                "code": "import math\nresult = math.sqrt(9)\nprint('sqrt', result)\n",
            },
        }
        result = await manager.invoke("code.python", payload=payload)
    finally:
        reset_trace_emitter(token)

    assert result.tool_name == "code.python"
    assert result.diagnostics.get("exec_status") == "ok"
    assert "```python" in (result.diagnostics.get("code_block") or "")
    assert result.evidences
    assert result.evidences[0].source == "code.python"
    assert "result:" in result.evidences[0].content

    # Trace should include tool_call + tool_response with runtime hints.
    tags = [e.tag for e in emitter.events]
    assert "tool_call" in tags
    assert "tool_response" in tags
    call_events = [e for e in emitter.events if e.tag == "tool_call"]
    assert call_events
    call_payload = json.loads(call_events[-1].content)
    assert call_payload.get("tool_name") == "code.python"
    assert isinstance(call_payload.get("runtime"), dict)
    assert "allowed_imports" in (call_payload.get("runtime") or {})

    response_events = [e for e in emitter.events if e.tag == "tool_response"]
    assert response_events
    tag, rendered = render_trace_payload(trace_tag="tool_response", content=response_events[-1].content, run_id="run_test")
    assert tag == "tool_response"
    assert "```python" in rendered
    assert "exec_status=" in rendered


@pytest.mark.asyncio
async def test_code_python_blocks_unapproved_import(tmp_path):
    manager = DeepSearchToolManager(
        _tool_manager_configs(
            tmp_path,
            enabled_tools={
                "code.python": {
                    "params": {
                        "allowed_imports": ["math"],
                        "timeout_seconds": 2.0,
                        "max_memory_mb": 256,
                    }
                }
            },
        ),
        telemetry_client=None,
    )
    payload = {
        "question": "import should fail",
        "plan_step": "plan_01",
        "context_evidences": [],
        "adapter": None,
        "access_scope": None,
        "extra": {"code": "import os\nresult = 1\n"},
    }
    result = await manager.invoke("code.python", payload=payload)
    assert result.diagnostics.get("exec_status") == "failed"
    err = result.diagnostics.get("error")
    assert isinstance(err, dict)
    assert err.get("type") in {"import_blocked", "ImportError"}


@pytest.mark.asyncio
async def test_code_python_timeout(tmp_path):
    manager = DeepSearchToolManager(
        _tool_manager_configs(
            tmp_path,
            enabled_tools={
                "code.python": {
                    "params": {
                        "allowed_imports": ["math"],
                        "timeout_seconds": 0.2,
                        "max_memory_mb": 256,
                    }
                }
            },
        ),
        telemetry_client=None,
    )
    payload = {
        "question": "timeout",
        "plan_step": "plan_01",
        "context_evidences": [],
        "adapter": None,
        "access_scope": None,
        "extra": {"code": "while True:\n    pass\n"},
    }
    result = await manager.invoke("code.python", payload=payload)
    assert result.diagnostics.get("exec_status") == "failed"
    err = result.diagnostics.get("error")
    assert isinstance(err, dict)
    assert err.get("type") == "timeout"


@pytest.mark.asyncio
async def test_local_tool_exception_emits_tool_response(tmp_path):
    @dataclass
    class _BoomTool:
        descriptor: ToolDescriptor = ToolDescriptor(
            name="test.boom",
            channel="text",
            description="boom",
            namespace="test",
            mcp_callable=False,
        )

        async def run(self, request: ToolRunRequest) -> ToolResult:
            raise RuntimeError("boom")

    manager = DeepSearchToolManager(
        _tool_manager_configs(tmp_path),
        telemetry_client=None,
        local_tools={"test.boom": _BoomTool()},
    )
    emitter = _CaptureEmitter()
    token = set_trace_emitter(emitter)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            await manager.invoke(
                "test.boom",
                payload={
                    "question": "x",
                    "plan_step": "plan_01",
                    "context_evidences": [],
                    "adapter": None,
                    "access_scope": None,
                    "extra": {},
                },
            )
    finally:
        reset_trace_emitter(token)

    # Ensure failure still yields a tool_response trace.
    assert any(e.tag == "tool_call" for e in emitter.events)
    assert any(e.tag == "tool_response" for e in emitter.events)


@pytest.mark.asyncio
async def test_graph_reasoning_think_can_call_code_python_and_next_think_sees_result(tmp_path):
    think_round_1 = json.dumps(
        {
            "reasoning": "Need deterministic verification.",
            "tool_calls": [
                {
                    "tool_name": "code.python",
                    "tool_args": {
                        "purpose": "verify sqrt",
                        "code": "import math\nresult = math.sqrt(9)\nprint('sqrt', result)\n",
                    },
                    "rationale": "Deterministic check.",
                    "parallelizable": False,
                }
            ],
            "plan": [{"text": "Verify computation", "checked": False}],
        },
        ensure_ascii=False,
    )
    think_round_2 = json.dumps(
        {
            "reasoning": "Result observed; continue DeepSearch.",
            "tool_calls": [],
            "plan": [{"text": "Verify computation", "checked": True}],
        },
        ensure_ascii=False,
    )
    llm = _SequencedLLM([think_round_1, think_round_2])

    tool_manager = DeepSearchToolManager(
        {
            **_tool_manager_configs(tmp_path),
            "llm_connector": llm,
        },
        telemetry_client=None,
    )
    adapter = _StubAdapter()
    loop = GraphReasoningLoop(
        adapter=adapter,
        llm_connector=None,
        strategy_config=_strategy_config(
            think_overrides={
                "every_n_steps": 1,
                "min_coverage": 0.99,
                "enable_tool_calls": True,
                "max_tool_calls": 1,
                "tool_call_concurrency": 1,
                "tool_catalog_max_items": 6,
                "tool_catalog_allowlist": ["locate", "toc.tree", "read.pages", "web.search", "code.python"],
                "include_llm_tools": True,
                "max_rounds_per_checkpoint": 2,
            }
        ),
        tool_manager=tool_manager,
    )
    context = GraphQueryContext(
        adapter_name="hipporag",
        question="Need compute verification",
        access_scope=GraphAccessScope(scope_id="scope-1", scope_type="owner"),
        metadata={},
        seed_entities=[],
    )
    result = await loop.run_think_loop(
        "Need compute verification",
        graph_context=context,
    )

    assert any(run.get("tool_name") == "code.python" for run in result.get("tool_results") or [])
    assert any(ev.get("source") == "code.python" for ev in result.get("evidences") or [])

    think_payloads = llm.seen_user_payloads
    assert len(think_payloads) >= 2
    # Second think should see the code.python evidence via context_evidences.
    second = think_payloads[1]
    evidences = second.get("context_evidences") or []
    assert any((item.get("source") == "code.python") for item in evidences if isinstance(item, dict))
