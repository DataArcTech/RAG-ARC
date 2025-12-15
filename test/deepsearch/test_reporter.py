import asyncio

from encapsulation.data_model.deepsearch import EvidenceChunk

from core.deepsearch.report import DeepSearchReporter


class _FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def chat(self, messages, **kwargs):  # noqa: ANN001
        if not self._responses:
            raise RuntimeError("No more fake responses configured")
        return self._responses.pop(0)


def _build_trace(include_final_answer: bool = True):
    trace = {
        "question": "Who partnered with OpenAI?",
        "plan_steps": [
            {"step_id": "plan_01", "description": "Collect graph facts", "channel": "graph", "metadata": {}},
        ],
        "reasoning_steps": [
            {
                "step_id": "plan_01",
                "description": "Collect graph facts",
                "channel": "graph",
                "status": "done",
                "output_summary": "Microsoft formed a strategic partnership with OpenAI in 2019.",
                "produced_evidence_ids": ["ev1"],
            }
        ],
        "graph_traversals": [
            {
                "visited_nodes": ["OpenAI", "Microsoft"],
                "visited_edges": ["OpenAI-works_with-Microsoft"],
            }
        ],
        "evidences": [
            {
                "chunk_id": "ev1",
                "source": "hipporag",
                "content": "OpenAI entered a partnership with Microsoft in 2019.",
            }
        ],
        "adapter_metadata": {"adapter_name": "hipporag"},
        "coverage_metrics": {"coverage_ratio": 0.75},
        "graph_context": {
            "adapter_name": "hipporag",
            "question": "Who partnered with OpenAI?",
            "metadata": {
                "request_metadata": {
                    "conversation_id": "conv-123",
                    "locale": "en-US",
                }
            },
        },
    }
    if include_final_answer:
        trace["final_answer"] = "Microsoft maintains the flagship partnership with OpenAI, enabling Azure-hosted deployments."
    return trace


def test_reporter_prefers_final_answer_and_merges_evidence():
    outline = """
[
  {"title": "Executive Summary", "purpose": "State the answer succinctly."},
  {"title": "Evidence-Based Findings", "purpose": "Explain what the evidence supports."},
  {"title": "Implications", "purpose": "Describe why the findings matter."},
  {"title": "Limitations", "purpose": "Clarify what cannot be concluded."},
  {"title": "Next Steps", "purpose": "Suggest how to improve coverage."}
]
""".strip()
    report_json = """
{
  "title": "Who partnered with OpenAI?",
  "summary": "Microsoft partnered with OpenAI and provides key infrastructure support. [ev1] [ev2]",
  "sections": [
    {
      "title": "Evidence-Based Findings",
      "body_markdown": "- Microsoft formed a strategic partnership with OpenAI in 2019. [ev1]\\n- Azure hosts OpenAI services. [ev2]"
    }
  ],
  "limitations": ["The trace contains limited evidence coverage beyond partnership and hosting."],
  "next_steps": ["Gather additional sources on partnership scope, timeline, and commercial terms."],
  "citations": [
    {"evidence_id": "ev1", "source": "hipporag", "used_for": "Partnership fact"},
    {"evidence_id": "ev2", "source": "press", "used_for": "Hosting detail"}
  ]
}
""".strip()
    reporter = DeepSearchReporter(
        template_store=None,
        config={"parallel_thinking_runs": 2},
        llm_connector=_FakeLLM([outline, report_json]),
    )
    trace = _build_trace(include_final_answer=True)
    external = [EvidenceChunk(chunk_id="ev2", source="press", content="Azure hosts OpenAI services.")]

    report = asyncio.run(reporter.compose(trace, external))

    assert report["answer"].startswith("# Who partnered with OpenAI?")
    assert "Microsoft partnered with OpenAI" in report["structured_report"]["summary"]
    assert len(report["evidences"]) == 2
    assert report["metadata"]["plan"]["completed"] == 1
    assert report["metadata"]["graph_summary"]["unique_nodes"] == 2
    assert report["metadata"]["parallel_thinking_runs"] == 2
    assert report["metadata"]["request_context"]["conversation_id"] == "conv-123"


def test_reporter_runs_consistency_check_when_enabled():
    outline = """
[
  {"title": "Executive Summary", "purpose": "State the answer succinctly."},
  {"title": "Evidence-Based Findings", "purpose": "Explain what the evidence supports."},
  {"title": "Limitations", "purpose": "Clarify what cannot be concluded."}
]
""".strip()
    report_json = """
{
  "title": "Who partnered with OpenAI?",
  "summary": "Microsoft partnered with OpenAI in 2019. [ev1]",
  "sections": [
    {
      "title": "Evidence-Based Findings",
      "body_markdown": "Microsoft formed a strategic partnership with OpenAI in 2019. [ev1]"
    }
  ],
  "limitations": [],
  "next_steps": [],
  "citations": [
    {"evidence_id": "ev1", "source": "hipporag", "used_for": "Partnership fact"}
  ]
}
""".strip()
    consistency_json = """
{
  "is_consistent": true,
  "confidence": 0.9,
  "issues": []
}
""".strip()
    reporter = DeepSearchReporter(
        template_store=None,
        config={"enable_consistency_check": True},
        llm_connector=_FakeLLM([outline, report_json, consistency_json]),
    )
    trace = _build_trace(include_final_answer=False)

    report = asyncio.run(reporter.compose(trace, external_evidence=[]))

    assert report["structured_report"]["consistency_check"]["is_consistent"] is True
    assert report["structured_report"]["consistency_check"]["issues"] == []

def test_reporter_citation_agent_fills_missing_citations():
    outline = """
[
  {"title": "Findings", "purpose": "Summarize what the evidence supports."}
]
""".strip()
    report_json = """
{
  "title": "Who partnered with OpenAI?",
  "summary": "Microsoft partnered with OpenAI in 2019. [ev1]",
  "sections": [
    {
      "title": "Findings",
      "body_markdown": "The evidence supports a partnership with Microsoft. [ev1]"
    }
  ],
  "limitations": [],
  "next_steps": [],
  "citations": []
}
""".strip()
    reporter = DeepSearchReporter(
        template_store=None,
        config={"enable_citation_agent": True},
        llm_connector=_FakeLLM([outline, report_json]),
    )
    trace = _build_trace(include_final_answer=False)

    report = asyncio.run(reporter.compose(trace, external_evidence=[]))

    citations = report["structured_report"]["citations"]
    assert isinstance(citations, list)
    assert any(entry.get("evidence_id") == "ev1" for entry in citations if isinstance(entry, dict))
    assert isinstance(report["structured_report"].get("evidence_index"), list)


def test_reporter_falls_back_without_llm_when_disabled():
    reporter = DeepSearchReporter(template_store=None, config={"enable_llm_report": False}, llm_connector=None)
    trace = _build_trace(include_final_answer=True)

    report = asyncio.run(reporter.compose(trace, external_evidence=[]))

    assert report["structured_report"]["generation"]["mode"] == "fallback"
    assert "## Answer" in report["answer"]


def test_reporter_raises_when_llm_output_invalid():
    reporter = DeepSearchReporter(template_store=None, config={}, llm_connector=_FakeLLM(["not-json"]))
    trace = _build_trace(include_final_answer=False)

    try:
        asyncio.run(reporter.compose(trace, external_evidence=[]))
    except Exception as exc:  # noqa: BLE001
        assert "outline" in str(exc).lower() or "json" in str(exc).lower()
    else:
        raise AssertionError("Expected the reporter to raise on invalid LLM output")


def test_reporter_includes_chunk_evidence_preview():
    """Verify the markdown output includes chunk evidence preview section."""
    reporter = DeepSearchReporter(
        template_store=None,
        config={"enable_llm_report": False},
        llm_connector=None,
    )
    trace = _build_trace(include_final_answer=True)
    trace["evidences"] = [
        {
            "chunk_id": "chunk_abc123",
            "source": "test-doc.pdf",
            "content": "This is a long piece of content that should be truncated to show only the first 100 characters in the preview section of the report.",
        },
        {
            "chunk_id": "chunk_def456",
            "source": "another.txt",
            "content": "Short content.",
        },
    ]

    report = asyncio.run(reporter.compose(trace, external_evidence=[]))

    answer = report["answer"]
    assert "## Appendix: Chunk Evidence" in answer
    assert "[chunk_abc123]" in answer
    assert "(test-doc.pdf)" in answer
    assert "This is a long piece of content that should be truncated to show only the first 100 characters" in answer
    assert "..." in answer
    assert "[chunk_def456]" in answer
    assert "Short content." in answer
