"""Tests for the flattened agentic loop (P4) and report integration (P5)."""

import inspect

import pytest


# ---------------------------------------------------------------------------
# Structural deletion checks
# ---------------------------------------------------------------------------


def test_nav_bootstrap_deleted() -> None:
    from application.rag_inference.deepsearch.service_runtime.run import DeepSearchServiceRunMixin

    assert not hasattr(DeepSearchServiceRunMixin, "_run_navigation_bootstrap"), (
        "_run_navigation_bootstrap should be deleted (P4 flattening)"
    )
    assert not hasattr(DeepSearchServiceRunMixin, "_invoke_tool_for_bootstrap"), (
        "_invoke_tool_for_bootstrap should be deleted (P4 flattening)"
    )
    assert not hasattr(DeepSearchServiceRunMixin, "_resolve_navigation_bootstrap_config"), (
        "_resolve_navigation_bootstrap_config should be deleted (P4 flattening)"
    )


def test_final_think_deleted() -> None:
    from application.rag_inference.deepsearch.service_runtime.initial_think import (
        DeepSearchServiceInitialThinkMixin,
    )

    assert not hasattr(DeepSearchServiceInitialThinkMixin, "_run_final_think"), (
        "_run_final_think should be deleted (P4 flattening)"
    )


def test_initial_prompt_constants_deleted() -> None:
    from core.prompts.deepsearch import tools as tools_mod

    assert not hasattr(tools_mod, "THINK_TOOL_SYSTEM_PROMPT_INITIAL_EN"), (
        "INITIAL_EN prompt should be deleted (P4)"
    )
    assert not hasattr(tools_mod, "THINK_TOOL_SYSTEM_PROMPT_FINAL_EN"), (
        "FINAL_EN prompt should be deleted (P4)"
    )


def test_gate_prompt_deleted() -> None:
    """GATE_EN prompt should be deleted (gates removed)."""
    from core.prompts.deepsearch import tools as tools_mod

    assert not hasattr(tools_mod, "THINK_TOOL_SYSTEM_PROMPT_GATE_EN"), (
        "GATE_EN prompt should be deleted (gates removed)"
    )


def test_gates_removed_from_run() -> None:
    """Both compute gate and evidence gate should be removed from run.py."""
    from application.rag_inference.deepsearch.service_runtime.run import DeepSearchServiceRunMixin

    assert not hasattr(DeepSearchServiceRunMixin, "_count_read_pages_evidence"), (
        "_count_read_pages_evidence should be deleted (gates removed)"
    )


def test_gates_removed_from_graph_loop() -> None:
    """Gate helpers should be removed from graph_loop."""
    from core.deepsearch.reasoning.graph_loop import GraphReasoningLoop

    assert not hasattr(GraphReasoningLoop, "_should_require_primary_page_evidence")
    assert not hasattr(GraphReasoningLoop, "_has_primary_page_evidence")
    assert not hasattr(GraphReasoningLoop, "_should_require_compute")
    assert not hasattr(GraphReasoningLoop, "_has_compute_evidence")


# ---------------------------------------------------------------------------
# _extract_answer_from_trace
# ---------------------------------------------------------------------------


def test_extract_answer_from_trace_returns_reasoning() -> None:
    from application.rag_inference.deepsearch.service_runtime.run import DeepSearchServiceRunMixin

    trace = {
        "think_notes": [
            {"metadata": {"raw": {"reasoning": "First note."}}},
            {"metadata": {"raw": {"reasoning": "The capital of France is Paris."}}},
        ]
    }
    answer = DeepSearchServiceRunMixin._extract_answer_from_trace(trace)
    assert answer == "The capital of France is Paris."


def test_extract_answer_from_trace_empty() -> None:
    from application.rag_inference.deepsearch.service_runtime.run import DeepSearchServiceRunMixin

    assert DeepSearchServiceRunMixin._extract_answer_from_trace({}) == ""
    assert DeepSearchServiceRunMixin._extract_answer_from_trace({"think_notes": []}) == ""
    assert DeepSearchServiceRunMixin._extract_answer_from_trace({"think_notes": [{"metadata": {}}]}) == ""


# ---------------------------------------------------------------------------
# Prompt checks
# ---------------------------------------------------------------------------


def test_prompt_en_contains_report_needed_instruction() -> None:
    from core.prompts.deepsearch.tools import THINK_TOOL_SYSTEM_PROMPT_EN

    assert "report_needed=false" in THINK_TOOL_SYSTEM_PROMPT_EN
    assert "report_needed=true" in THINK_TOOL_SYSTEM_PROMPT_EN


def test_prompt_en_mentions_toc_cache() -> None:
    """Prompt should tell model that toc.tree results are cached."""
    from core.prompts.deepsearch.tools import THINK_TOOL_SYSTEM_PROMPT_EN

    assert "CACHED" in THINK_TOOL_SYSTEM_PROMPT_EN or "cached" in THINK_TOOL_SYSTEM_PROMPT_EN.lower()


def test_prompt_en_mentions_budget() -> None:
    """Prompt should mention budget awareness."""
    from core.prompts.deepsearch.tools import THINK_TOOL_SYSTEM_PROMPT_EN

    assert "budget" in THINK_TOOL_SYSTEM_PROMPT_EN.lower()


# ---------------------------------------------------------------------------
# P5: _report_stage is async
# ---------------------------------------------------------------------------


def test_report_stage_is_coroutine_function() -> None:
    """_report_stage should be a proper async def (not sync returning awaitable)."""
    from application.rag_inference.deepsearch.service_runtime.pipeline import (
        DeepSearchServicePipelineMixin,
    )

    assert inspect.iscoroutinefunction(DeepSearchServicePipelineMixin._report_stage)


# ---------------------------------------------------------------------------
# P5: _build_structured_report_stub
# ---------------------------------------------------------------------------


def test_structured_report_stub_schema() -> None:
    """Stub should produce format_version 2.0 with empty citations."""
    from application.rag_inference.deepsearch.service_runtime.run import DeepSearchServiceRunMixin

    stub = DeepSearchServiceRunMixin._build_structured_report_stub("test answer")
    assert stub["format_version"] == "2.0"
    assert stub["text"] == "test answer"
    assert stub["citations"] == []
    assert stub["evidence_index"] == []
    assert stub["source_key_map"] == {}


# ---------------------------------------------------------------------------
# P5: report_style default propagation
# ---------------------------------------------------------------------------


def test_report_style_defaults_to_deepsearch() -> None:
    """report_style should default to 'deepsearch' for all falsy initial_think values."""
    for initial_think in [None, {}, {"report_style": None}, {"report_style": ""}]:
        result = (
            (initial_think.get("report_style") if isinstance(initial_think, dict) else None)
            or "deepsearch"
        )
        assert result == "deepsearch", f"Expected 'deepsearch' for {initial_think!r}"


def test_report_style_respects_explicit_value() -> None:
    """An explicit report_style from initial_think should be preserved."""
    initial_think = {"report_style": "custom_style"}
    result = (
        (initial_think.get("report_style") if isinstance(initial_think, dict) else None)
        or "deepsearch"
    )
    assert result == "custom_style"


# ---------------------------------------------------------------------------
# P5: _report_stage accepts report_evidences
# ---------------------------------------------------------------------------


def test_report_stage_signature_has_report_evidences() -> None:
    """_report_stage should accept a report_evidences keyword argument."""
    from application.rag_inference.deepsearch.service_runtime.pipeline import (
        DeepSearchServicePipelineMixin,
    )

    sig = inspect.signature(DeepSearchServicePipelineMixin._report_stage)
    assert "report_evidences" in sig.parameters
    assert "external_logs" in sig.parameters
    # Both should have defaults (None)
    assert sig.parameters["report_evidences"].default is None
    assert sig.parameters["external_logs"].default is None
