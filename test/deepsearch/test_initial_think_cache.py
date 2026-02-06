import pytest

from application.rag_inference.deepsearch.runtime_cache import DeepSearchInitialThinkCache
from application.rag_inference.deepsearch.service_runtime.initial_think import DeepSearchServiceInitialThinkMixin
from core.graph_adapter.base import GraphAccessScope
from encapsulation.data_model.deepsearch import GraphQueryContext, ThinkNote


class _DummyToolResult:
    def __init__(self, *, think_notes):
        self.think_notes = think_notes


class _DummyToolManager:
    def __init__(self):
        self.calls = 0
        self.llm_fingerprint = "dummy-llm"

    def invoke(self, tool_name: str, *, payload):
        # Return an awaitable to match the real tool manager contract.
        async def _run():
            self.calls += 1
            note = ThinkNote(
                plan_step_id=None,
                reasoning="dummy reasoning",
                confidence_delta=None,
                coverage_delta=None,
                next_actions=[
                    "explore -> search.file",
                    "toc.tree / section.select",
                    "read.pages",
                ],
                metadata={
                    "raw": {
                        "report_needed": True,
                        "report_style": "deepsearch",
                        "plan": [{"text": "route -> navigate -> read.pages -> report", "checked": False}],
                        "reasoning": "dummy reasoning",
                        "tool_calls": [],
                    }
                },
            )
            return _DummyToolResult(think_notes=[note])

        return _run()


class _DummyService(DeepSearchServiceInitialThinkMixin):
    def __init__(self):
        self.tool_manager = _DummyToolManager()
        self.config = {
            "fingerprint": "svc-fp",
            "think_tool": "think",
            "coverage_expected_min_chunks": 3,
        }
        self._initial_think_cache = DeepSearchInitialThinkCache(max_entries=8, ttl_seconds=60.0)
        self._think_prompt_fingerprint = "prompt-fp"


@pytest.mark.asyncio
async def test_initial_think_cache_hits_for_same_question():
    svc = _DummyService()
    scope = GraphAccessScope(scope_id="owner-1")
    ctx = GraphQueryContext(adapter_name="test", owner_id="owner-1", question="Q", seed_entities=[], metadata={}, access_scope=scope)

    out1 = await svc._run_initial_think(
        question="What is X?",
        scope=scope,
        reasoning_context=ctx,
        context_evidences=[],
        plan_steps=[],
    )
    out2 = await svc._run_initial_think(
        question="What is X?",
        scope=scope,
        reasoning_context=ctx,
        context_evidences=[],
        plan_steps=[],
    )

    assert svc.tool_manager.calls == 1
    assert out1["report_needed"] is True
    assert out2["report_needed"] is True
    assert out2.get("cache", {}).get("hit") is True

