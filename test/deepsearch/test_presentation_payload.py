import pytest

from core.presentation.deepsearch_payload import trim_deepsearch_payload
from core.presentation.summary import DeepSearchReport


def _sample_result():
    return {
        "plan": {"plan": {"question": "Who runs RAG-ARC?", "steps": []}},
        "reasoning": {
            "reasoning_steps": [
                {
                    "step_id": "think_loop_01",
                    "description": "Think loop checkpoint",
                    "channel": "graph",
                    "status": "done",
                    "output_summary": "Planning file routing.",
                    "diagnostics": {"tool": "think", "latency_ms": 800},
                    "produced_evidence_ids": [],
                    "tool_logs": [],
                },
                {
                    "step_id": "think_loop_01_call_01",
                    "description": "Collect graph facts",
                    "channel": "graph",
                    "status": "done",
                    "output_summary": "Located 2 relevant files.",
                    "diagnostics": {"tool": "locate", "confidence": 0.72, "latency_ms": 120},
                    "produced_evidence_ids": ["rep-0"],
                    "tool_logs": [
                        {"tool_name": "locate", "latency_ms": 95, "arguments_snapshot": {"query": "RAG-ARC maintainers"}},
                        {"tool_name": "read.pages", "latency_ms": 25, "arguments_snapshot": {"file_id": "f-001", "pages": [1, 2]}},
                    ],
                },
            ],
            "evidences": [{"chunk_id": f"reason-{idx}", "content": f"reasoning {idx}"} for idx in range(4)],
            "think_notes": [
                {
                    "plan_step_id": "think_loop_01",
                    "reasoning": "Need to find which team maintains RAG-ARC by locating relevant docs.",
                    "next_actions": ["Locate files about RAG-ARC ownership"],
                },
            ],
            "coverage_metrics": {},
            "tool_results": [
                {
                    "plan_step_id": "think_loop_01_call_01",
                    "tool_name": "locate",
                    "channel": "graph",
                    "result": {
                        "summary": "Located 2 relevant files.",
                        "diagnostics": {"latency_ms": 120},
                        "think_notes": [],
                    },
                }
            ],
        },
        "report": {
            "question": "Who runs RAG-ARC?",
            "evidences": [{"chunk_id": f"rep-{idx}", "content": f"content {idx}"} for idx in range(6)],
            "highlights": [{"summary": "Highlight A"}, {"summary": "Highlight B"}],
            "structured_report": {
                "text": "Graph RAG is maintained by RAG-ARC.",
                "citations": [],
                "source_key_map": {},
                "evidence_index": [],
            },
        },
        "state": {"request_metadata": {"priority": "high"}},
    }


def test_trim_payload_attaches_evidence(monkeypatch):
    calls: list[tuple[dict, int | None]] = []

    def _fake_builder(payload, chunk_limit=None):
        calls.append((payload, chunk_limit))
        return {
            "chunks": [{"chunk_id": "trimmed"}],
            "seed_entities": ["Entity"],
            "graph": {"nodes": []},
            "graph_stats": {},
            "graph_chain": ["A -[related_to]-> B"],
        }

    monkeypatch.setattr("core.presentation.deepsearch_payload.build_deepsearch_evidence", _fake_builder)

    payload = trim_deepsearch_payload(_sample_result(), include_evidence=True, chunk_limit=2)

    assert payload["evidence"]["chunks"] == [{"chunk_id": "trimmed"}]
    assert payload["graph_chain"] == ["A -[related_to]-> B"]
    assert len(payload["report"]["evidences"]) == 2
    assert payload["reasoning"]["reasoning_steps"]
    assert payload["reasoning"]["think_notes"]
    assert payload["question"] == "Who runs RAG-ARC?"
    assert payload["tool_runs"]
    assert payload["tool_runs"][0]["tool_name"] == "locate"
    assert payload["request_metadata"] == {"priority": "high"}
    assert payload["overview"]["has_think_notes"] is True
    assert calls and calls[0][1] == 2


def test_trim_payload_skips_evidence_when_disabled(monkeypatch):
    def _unexpected_call(*args, **kwargs):
        raise AssertionError("build_deepsearch_evidence should not run when include_evidence=false")

    monkeypatch.setattr("core.presentation.deepsearch_payload.build_deepsearch_evidence", _unexpected_call)

    payload = trim_deepsearch_payload(_sample_result(), include_evidence=False, chunk_limit=3)

    assert len(payload["evidence"]["chunks"]) == 3
    assert payload["evidence"]["seed_entities"] == []
    assert payload["graph_chain"] == []
    assert len(payload["report"]["evidences"]) == 3


def test_trim_payload_includes_reasoning_summaries(monkeypatch):
    def _unexpected_call(*args, **kwargs):
        raise AssertionError("build_deepsearch_evidence should not run when include_evidence=false")

    monkeypatch.setattr("core.presentation.deepsearch_payload.build_deepsearch_evidence", _unexpected_call)

    payload = trim_deepsearch_payload(_sample_result(), include_evidence=False, chunk_limit=1)

    assert payload["reasoning"]["reasoning_steps"]
    step = payload["reasoning"]["reasoning_steps"][1]  # tool call step
    assert step["tool"] == "locate"
    assert step["output_summary"] == "Located 2 relevant files."
    assert step["diagnostics"] == {"confidence": 0.72, "latency_ms": 120, "tool": "locate"}


def test_trim_payload_preserves_structured_report(monkeypatch):
    def _unexpected_call(*args, **kwargs):
        raise AssertionError("build_deepsearch_evidence should not run when include_evidence=false")

    monkeypatch.setattr("core.presentation.deepsearch_payload.build_deepsearch_evidence", _unexpected_call)

    payload = trim_deepsearch_payload(_sample_result(), include_evidence=False, chunk_limit=2)

    structured = payload["report"].get("structured_report")
    assert structured
    assert structured["text"] == "Graph RAG is maintained by RAG-ARC."


def test_trim_payload_emits_hipporag_style_citations_and_sources(monkeypatch):
    def _unexpected_call(*args, **kwargs):
        raise AssertionError("build_deepsearch_evidence should not run when include_evidence=false")

    monkeypatch.setattr("core.presentation.deepsearch_payload.build_deepsearch_evidence", _unexpected_call)

    payload = _sample_result()
    payload["report"]["answer"] = "Claim A cites rep-0. <sup>1</sup> Claim B cites rep-1 and rep-2. <sup>2</sup><sup>3</sup>"
    payload["report"]["evidences"] = [
        {
            "chunk_id": "rep-0",
            "source": "hipporag",
            "content": "Alpha evidence snippet",
            "provenance": {"metadata": {"chunk_metadata": {"source_file_id": "file-0", "filename": "doc0.md"}}},
        },
        {
            "chunk_id": "rep-1",
            "source": "web.tavily",
            "content": "Web Title\nWeb Body",
            "provenance": {"url": "https://example.com/rep-1"},
        },
        {
            "chunk_id": "rep-2",
            "source": "hipporag",
            "content": "Gamma evidence snippet",
            "provenance": {"metadata": {"chunk_metadata": {"source_file_id": "file-2", "filename": "doc2.md"}}},
        },
    ]
    payload["report"]["structured_report"]["citations"] = [
        {"evidence_id": "rep-0", "source": "hipporag"},
        {"evidence_id": "rep-1", "source": "web.tavily"},
        {"evidence_id": "rep-2", "source": "hipporag"},
    ]
    payload["report"]["structured_report"]["source_key_map"] = {
        "1": "rep-0",
        "2": "rep-1",
        "3": "rep-2",
    }

    trimmed = trim_deepsearch_payload(payload, include_evidence=False, chunk_limit=10)

    answer = trimmed["report"]["answer"]
    assert "<sup>1</sup>" in answer
    assert "<sup>2</sup><sup>3</sup>" in answer
    assert "## Evidence Index" not in answer
    assert "## References" not in answer
    assert "(/knowledge/chunk/rep-0)" not in answer
    assert "(https://example.com/rep-1)" not in answer

    sources = trimmed["report"].get("sources")
    assert sources and [s["key"] for s in sources] == [1, 2, 3]
    assert [s["chunk_id"] for s in sources] == ["rep-0", "rep-1", "rep-2"]
    assert sources[0].get("file") == "/static/files/file-0/doc0.md"
    assert sources[0].get("file_id") == "file-0"
    assert sources[1].get("file") == "https://example.com/rep-1"
    assert trimmed["report"].get("citation_key_map") == {"rep-0": 1, "rep-1": 2, "rep-2": 3}


def test_deepsearch_report_falls_back_to_highlights():
    payload = _sample_result()
    payload["report"]["answer"] = ""
    payload["reasoning"]["final_answer"] = ""

    report = DeepSearchReport.from_payload(payload, graph_chain_builder=None)

    assert report.final_answer.startswith("Key findings"), "Final answer should fall back to highlights"


def test_deepsearch_report_prefers_reasoning_answer():
    payload = _sample_result()
    payload["report"]["answer"] = ""
    payload["reasoning"]["final_answer"] = "Graph RAG is managed by the RAG-ARC core team."

    report = DeepSearchReport.from_payload(payload, graph_chain_builder=None)

    assert report.final_answer == "Graph RAG is managed by the RAG-ARC core team."


def test_deepsearch_report_falls_back_to_highlights_when_answer_empty():
    payload = _sample_result()
    payload["report"]["answer"] = ""
    payload["reasoning"]["final_answer"] = ""

    report = DeepSearchReport.from_payload(payload, graph_chain_builder=None)

    # With no answer and no final_answer, should fall back to highlights.
    assert "Key findings" in report.final_answer


def test_execution_timeline_with_reasoning_and_args(monkeypatch):
    monkeypatch.setattr(
        "core.presentation.deepsearch_payload.build_deepsearch_evidence",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not call")),
    )

    payload = trim_deepsearch_payload(_sample_result(), include_evidence=False)
    timeline = payload.get("execution_timeline")
    assert isinstance(timeline, list) and len(timeline) == 2

    # Entry 0: think step — should have reasoning from think_notes
    think = timeline[0]
    assert think["step_id"] == "think_loop_01"
    assert think["tool"] == "think"
    assert think["latency_ms"] == 800
    assert think["reasoning"] == "Need to find which team maintains RAG-ARC by locating relevant docs."
    assert think["next_actions"] == ["Locate files about RAG-ARC ownership"]
    assert think.get("invocations") is None  # no tool_logs on think step

    # Entry 1: tool call — should have invocations with args, and inherited reasoning
    call = timeline[1]
    assert call["step_id"] == "think_loop_01_call_01"
    assert call["tool"] == "locate"
    assert call["latency_ms"] == 120
    assert call["status"] == "done"
    assert call["evidence_produced"] == 1
    assert call["summary"] == "Located 2 relevant files."
    # Tool call inherits parent think_loop_01's reasoning
    assert "RAG-ARC" in call["reasoning"]

    # Invocations should include arguments
    invocations = call.get("invocations")
    assert invocations and len(invocations) == 2
    assert invocations[0]["tool"] == "locate"
    assert invocations[0]["latency_ms"] == 95
    assert invocations[0]["arguments"] == {"query": "RAG-ARC maintainers"}
    assert invocations[1]["tool"] == "read.pages"
    assert invocations[1]["arguments"] == {"file_id": "f-001", "pages": [1, 2]}
