import json

import pytest

from core.deepsearch.report.bench_answer import synthesize_benchmark_answer


class _StubLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def achat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append(messages)
        if self._responses:
            return self._responses.pop(0)
        return "ok"


@pytest.mark.asyncio
async def test_bench_answer_filters_non_primary_evidence_by_default():
    llm = _StubLLM(["answer"])
    trace = {
        "question": "Q",
        "evidences": [
            {"chunk_id": "c1", "source": "graph_adapter.query", "content": "primary text", "kind": "primary"},
            {"chunk_id": "d1", "source": "graph.llm_chain_explorer", "content": "derived text", "kind": "derived"},
            {"chunk_id": "x1", "source": "graph.neighbors", "content": "neighbors count=0", "kind": "diagnostic"},
        ],
    }
    bench_cfg = {
        "allowed_evidence_kinds": ["primary"],
        "heading_window_max_lines": 5,
        "default_policy": {"mode": "single_stage", "preference": "balanced", "max_evidence_items": 10, "max_evidence_chars": 1000, "snippet_chars": 200},
    }
    out = await synthesize_benchmark_answer(
        llm_connector=llm,
        question="What is asked?",
        reasoning_trace=trace,
        bench_answer_config=bench_cfg,
    )
    used = out.get("bench_evidence_used") or []
    assert len(used) == 1
    assert used[0]["chunk_id"] == "c1"
    assert "derived text" not in (out.get("bench_evidence_block") or "")
    assert "neighbors" not in (out.get("bench_evidence_block") or "")


@pytest.mark.asyncio
async def test_bench_answer_heading_window_preserves_bullets():
    llm = _StubLLM(["answer"])
    evidence = (
        "**Main Cancer Subtypes of Esophageal Cancer**:\n"
        "  - Adenocarcinoma: lower esophagus.\n"
        "  - Squamous cell carcinoma: upper and middle esophagus.\n"
    )
    trace = {"question": "Q", "evidences": [{"chunk_id": "c1", "source": "graph_adapter.query", "content": evidence, "kind": "primary"}]}
    bench_cfg = {
        "allowed_evidence_kinds": ["primary"],
        "heading_window_max_lines": 5,
        "default_policy": {"mode": "single_stage", "preference": "coverage", "max_evidence_items": 5, "max_evidence_chars": 2000, "snippet_chars": 200},
    }
    out = await synthesize_benchmark_answer(
        llm_connector=llm,
        question="What are the main cancer subtypes of esophageal cancer and their locations?",
        reasoning_trace=trace,
        bench_answer_config=bench_cfg,
    )
    block = out.get("bench_evidence_block") or ""
    assert "Adenocarcinoma" in block
    assert "Squamous" in block


@pytest.mark.asyncio
async def test_bench_answer_does_not_double_bulletize_first_line():
    llm = _StubLLM(["answer"])
    trace = {"question": "Q", "evidences": [{"chunk_id": "c1", "source": "graph_adapter.query", "content": "- Item A", "kind": "primary"}]}
    bench_cfg = {
        "allowed_evidence_kinds": ["primary"],
        "default_policy": {"mode": "single_stage", "preference": "balanced", "max_evidence_items": 5, "max_evidence_chars": 2000, "snippet_chars": 200},
    }
    out = await synthesize_benchmark_answer(
        llm_connector=llm,
        question="List items",
        reasoning_trace=trace,
        bench_answer_config=bench_cfg,
    )
    block = out.get("bench_evidence_block") or ""
    assert "- - " not in block


@pytest.mark.asyncio
async def test_bench_answer_two_stage_extract_then_answer():
    llm = _StubLLM([json.dumps({"points": [{"text": "p1", "evidence_chunk_ids": ["c1"]}]}), "final"])
    trace = {"question": "Q", "evidences": [{"chunk_id": "c1", "source": "graph_adapter.query", "content": "e1", "kind": "primary"}]}
    bench_cfg = {
        "allowed_evidence_kinds": ["primary"],
        "default_policy": {"mode": "two_stage", "preference": "correctness", "max_evidence_items": 5, "max_evidence_chars": 2000, "snippet_chars": 200},
    }
    out = await synthesize_benchmark_answer(
        llm_connector=llm,
        question="Complex question",
        reasoning_trace=trace,
        bench_answer_config=bench_cfg,
    )
    assert out.get("answer") == "final"
    assert len(llm.calls) == 2
    assert out.get("bench_extracted_points", {}).get("points")
