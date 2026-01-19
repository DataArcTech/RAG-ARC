import pytest

from core.presentation.deepsearch_payload import trim_deepsearch_payload
from core.presentation.summary import DeepSearchReport


def _sample_result():
    return {
        "plan": {"plan": {"question": "Who runs RAG-ARC?", "steps": []}},
        "reasoning": {
            "reasoning_steps": [
                {
                    "step_id": "plan_01",
                    "description": "Collect graph facts",
                    "channel": "graph",
                    "status": "done",
                    "output_summary": "Plan step completed.",
                    "diagnostics": {"tool": "search", "confidence": 0.72, "latency_ms": 120},
                    "produced_evidence_ids": ["rep-0"],
                }
            ],
            "evidences": [{"chunk_id": f"reason-{idx}", "content": f"reasoning {idx}"} for idx in range(4)],
            "think_notes": [{"plan_step_id": "plan_01", "reasoning": "pause"}],
            "coverage_metrics": {},
            "tool_results": [
                {
                    "plan_step_id": "plan_01",
                    "tool_name": "search",
                    "channel": "graph",
                    "result": {
                        "summary": "Plan step completed.",
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
                "title": "Who runs RAG-ARC?",
                "summary": "Graph RAG is maintained by RAG-ARC.",
                "sections": [
                    {"title": "Summary", "body": "Graph RAG is maintained by RAG-ARC."},
                ],
                "citations": [],
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
    assert payload["tool_runs"][0]["tool_name"] == "search"
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
    step = payload["reasoning"]["reasoning_steps"][0]
    assert step["tool"] == "search"
    assert step["output_summary"] == "Plan step completed."
    assert step["diagnostics"] == {"confidence": 0.72, "latency_ms": 120, "tool": "search"}


def test_trim_payload_preserves_structured_report(monkeypatch):
    def _unexpected_call(*args, **kwargs):
        raise AssertionError("build_deepsearch_evidence should not run when include_evidence=false")

    monkeypatch.setattr("core.presentation.deepsearch_payload.build_deepsearch_evidence", _unexpected_call)

    payload = trim_deepsearch_payload(_sample_result(), include_evidence=False, chunk_limit=2)

    structured = payload["report"].get("structured_report")
    assert structured
    assert structured["summary"] == "Graph RAG is maintained by RAG-ARC."


def test_trim_payload_emits_hipporag_style_citations_and_sources(monkeypatch):
    def _unexpected_call(*args, **kwargs):
        raise AssertionError("build_deepsearch_evidence should not run when include_evidence=false")

    monkeypatch.setattr("core.presentation.deepsearch_payload.build_deepsearch_evidence", _unexpected_call)

    payload = _sample_result()
    payload["report"]["answer"] = "Claim A cites [rep-0]. Claim B cites [rep-1, rep-2]."
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
    assert sources[0].get("file") == "/knowledge/chunk/rep-0"
    assert sources[0].get("file_id") == "file-0"
    assert sources[1].get("file") == "https://example.com/rep-1"
    assert trimmed["report"].get("citation_key_map") == {"rep-0": 1, "rep-1": 2, "rep-2": 3}


def test_trim_payload_converts_citations_without_structured_report(monkeypatch):
    def _unexpected_call(*args, **kwargs):
        raise AssertionError("build_deepsearch_evidence should not run when include_evidence=false")

    monkeypatch.setattr("core.presentation.deepsearch_payload.build_deepsearch_evidence", _unexpected_call)

    payload = _sample_result()
    payload["report"]["answer"] = "Claim A cites [rep-0]. Claim B cites 【rep-1】."
    payload["report"]["evidences"] = [
        {
            "chunk_id": "rep-0",
            "source": "hipporag",
            "content": "Alpha evidence snippet",
            "provenance": {"metadata": {"chunk_metadata": {"source_file_id": "file-0", "filename": "doc0.md"}}},
        },
        {
            "chunk_id": "rep-1",
            "source": "hipporag",
            "content": "Beta evidence snippet",
            "provenance": {"metadata": {"chunk_metadata": {"source_file_id": "file-1", "filename": "doc1.md"}}},
        },
    ]
    payload["report"].pop("structured_report", None)

    trimmed = trim_deepsearch_payload(payload, include_evidence=False, chunk_limit=10)

    answer = trimmed["report"]["answer"]
    assert "[rep-0]" not in answer
    assert "【rep-1】" not in answer
    assert "<sup>1</sup>" in answer
    assert "<sup>2</sup>" in answer

    sources = trimmed["report"].get("sources")
    assert sources and [s["key"] for s in sources] == [1, 2]
    assert [s["chunk_id"] for s in sources] == ["rep-0", "rep-1"]


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


def test_deepsearch_report_uses_llm_chain_explorer_when_available():
    payload = _sample_result()
    payload["report"]["answer"] = ""
    payload["reasoning"]["final_answer"] = ""
    payload["report"]["evidences"][0]["source"] = "graph.llm_chain_explorer"
    payload["report"]["evidences"][0]["content"] = "SAS 提供完善的 AP 课程与课外活动。"

    report = DeepSearchReport.from_payload(payload, graph_chain_builder=None)

    assert report.final_answer == "SAS 提供完善的 AP 课程与课外活动。"
