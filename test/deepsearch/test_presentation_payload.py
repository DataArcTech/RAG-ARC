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
                    "diagnostics": {"tool": "graph.pattern_scan", "confidence": 0.72, "latency_ms": 120},
                    "produced_evidence_ids": ["rep-0"],
                }
            ],
            "evidences": [{"chunk_id": f"reason-{idx}", "content": f"reasoning {idx}"} for idx in range(4)],
            "coverage_metrics": {},
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
        "state": {},
    }


def test_trim_payload_attaches_evidence(monkeypatch):
    calls: list[tuple[dict, int | None]] = []

    def _fake_builder(payload, chunk_limit=None):
        calls.append((payload, chunk_limit))
        return {
            "chunks": [{"chunk_id": "trimmed"}],
            "seed_entities": ["Entity"],
            "triples": [{"head": "A", "relation": "related_to", "tail": "B"}],
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
    assert calls and calls[0][1] == 2


def test_trim_payload_skips_evidence_when_disabled(monkeypatch):
    def _fake_builder(payload, chunk_limit=None):
        return {
            "chunks": [],
            "seed_entities": [],
            "triples": [],
            "graph": {},
            "graph_stats": {},
            "graph_chain": ["chain-item"],
        }

    monkeypatch.setattr("core.presentation.deepsearch_payload.build_deepsearch_evidence", _fake_builder)

    payload = trim_deepsearch_payload(_sample_result(), include_evidence=False, chunk_limit=3)

    assert "evidence" not in payload
    assert payload["graph_chain"] == ["chain-item"]
    assert len(payload["report"]["evidences"]) == 3


def test_trim_payload_includes_reasoning_summaries(monkeypatch):
    def _fake_builder(payload, chunk_limit=None):
        return {
            "chunks": [],
            "seed_entities": [],
            "triples": [],
            "graph": {},
            "graph_stats": {},
            "graph_chain": [],
        }

    monkeypatch.setattr("core.presentation.deepsearch_payload.build_deepsearch_evidence", _fake_builder)

    payload = trim_deepsearch_payload(_sample_result(), include_evidence=False, chunk_limit=1)

    assert payload["reasoning_steps"]
    step = payload["reasoning_steps"][0]
    assert step["tool"] == "graph.pattern_scan"
    assert step["output_summary"] == "Plan step completed."
    assert step["diagnostics"] == {"confidence": 0.72, "latency_ms": 120, "tool": "graph.pattern_scan"}


def test_trim_payload_preserves_structured_report(monkeypatch):
    def _fake_builder(payload, chunk_limit=None):
        return {"chunks": [], "seed_entities": [], "triples": [], "graph": {}, "graph_stats": {}, "graph_chain": []}

    monkeypatch.setattr("core.presentation.deepsearch_payload.build_deepsearch_evidence", _fake_builder)

    payload = trim_deepsearch_payload(_sample_result(), include_evidence=False, chunk_limit=2)

    structured = payload["report"].get("structured_report")
    assert structured
    assert structured["summary"] == "Graph RAG is maintained by RAG-ARC."


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


def test_deepsearch_report_uses_context_rollup_when_available():
    payload = _sample_result()
    payload["report"]["answer"] = ""
    payload["reasoning"]["final_answer"] = ""
    payload["report"]["evidences"][0]["source"] = "context_rollup"
    payload["report"]["evidences"][0]["content"] = "SAS 提供完善的 AP 课程与课外活动。"

    report = DeepSearchReport.from_payload(payload, graph_chain_builder=None)

    assert report.final_answer == "SAS 提供完善的 AP 课程与课外活动。"
