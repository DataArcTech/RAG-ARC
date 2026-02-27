from unittest.mock import AsyncMock, patch

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
        self.tool_configs = {"llm_connector": object()}

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
                    "explore -> locate",
                    "toc.tree / locate(file=X)",
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
async def test_initial_think_cache_hits_for_same_question(monkeypatch):
    monkeypatch.setenv("bench_mode", "0")
    svc = _DummyService()
    scope = GraphAccessScope(scope_id="owner-1")
    ctx = GraphQueryContext(adapter_name="test", owner_id="owner-1", question="Q", seed_entities=[], metadata={}, access_scope=scope)

    dummy_query_spec = {
        "report_needed": True,
        "report_style": "deepsearch",
        "bm25_terms": ["term1"],
        "regex_patterns": [],
        "bm25_terms_by_lang": {},
        "regex_patterns_by_lang": {},
        "reasoning": "dummy reasoning",
    }
    with patch(
        "application.rag_inference.deepsearch.service_runtime.initial_think.generate_query_spec",
        new_callable=AsyncMock,
        return_value=dummy_query_spec,
    ) as mock_gen:
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

    assert mock_gen.await_count == 1
    assert out1["report_needed"] is True
    assert out2["report_needed"] is True
    assert out2.get("cache", {}).get("hit") is True
    assert out2.get("query_spec") == dummy_query_spec


@pytest.mark.asyncio
async def test_initial_think_cache_normalization_strips_trailing_punctuation_only(monkeypatch):
    monkeypatch.setenv("bench_mode", "0")
    svc = _DummyService()
    scope = GraphAccessScope(scope_id="owner-1")
    ctx = GraphQueryContext(adapter_name="test", owner_id="owner-1", question="Q", seed_entities=[], metadata={}, access_scope=scope)

    dummy_query_spec = {
        "report_needed": True,
        "report_style": "deepsearch",
        "bm25_terms": ["term1"],
        "regex_patterns": [],
        "bm25_terms_by_lang": {},
        "regex_patterns_by_lang": {},
        "reasoning": "dummy reasoning",
    }
    with patch(
        "application.rag_inference.deepsearch.service_runtime.initial_think.generate_query_spec",
        new_callable=AsyncMock,
        return_value=dummy_query_spec,
    ) as mock_gen:
        out1 = await svc._run_initial_think(
            question="What is X?",
            scope=scope,
            reasoning_context=ctx,
            context_evidences=[],
            plan_steps=[],
        )
        out2 = await svc._run_initial_think(
            question="What is X",
            scope=scope,
            reasoning_context=ctx,
            context_evidences=[],
            plan_steps=[],
        )
    assert out1["report_needed"] is True
    assert out2.get("cache", {}).get("hit") is True
    assert mock_gen.await_count == 1

    # Interior punctuation should remain significant (avoid collisions).
    with patch(
        "application.rag_inference.deepsearch.service_runtime.initial_think.generate_query_spec",
        new_callable=AsyncMock,
        return_value=dummy_query_spec,
    ) as mock_gen2:
        out3 = await svc._run_initial_think(
            question="A-B",
            scope=scope,
            reasoning_context=ctx,
            context_evidences=[],
            plan_steps=[],
        )
        out4 = await svc._run_initial_think(
            question="AB",
            scope=scope,
            reasoning_context=ctx,
            context_evidences=[],
            plan_steps=[],
        )
    assert out3["report_needed"] is True
    assert out4.get("cache", {}).get("hit") is not True
    assert mock_gen2.await_count == 2


@pytest.mark.asyncio
async def test_bench_mode_skips_query_spec(monkeypatch):
    """In bench_mode, _run_initial_think should skip query_spec and return defaults."""
    monkeypatch.setenv("bench_mode", "1")
    svc = _DummyService()
    scope = GraphAccessScope(scope_id="owner-1")
    ctx = GraphQueryContext(adapter_name="test", owner_id="owner-1", question="Q", seed_entities=[], metadata={}, access_scope=scope)

    with patch(
        "application.rag_inference.deepsearch.service_runtime.initial_think.generate_query_spec",
        new_callable=AsyncMock,
    ) as mock_gen:
        out = await svc._run_initial_think(
            question="What is X?",
            scope=scope,
            reasoning_context=ctx,
        )

    # query_spec should NOT have been called
    mock_gen.assert_not_awaited()
    assert out["report_needed"] is True
    assert out["report_style"] == "deepsearch"
    assert out.get("cache", {}).get("bench_mode") is True
    assert out["query_spec"] is None
