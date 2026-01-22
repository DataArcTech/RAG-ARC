import asyncio
import re

import pytest

from encapsulation.data_model.deepsearch import EvidenceChunk

from core.deepsearch.report import DeepSearchReporter


def _default_reporter_config(**overrides):  # noqa: ANN001
    config = {
        "max_highlights": 6,
        "include_graph_viz": True,
        "enable_custom_summary": False,
        "parallel_thinking_runs": 1,
        "enable_llm_report": True,
        "report_temperature": 0.2,
        "report_max_evidence_chars": 900,
        "max_evidence_items": 10,
        "report_max_graph_chain_items": 200,
        "report_max_seed_entities": 15,
        "sectionwise_writer": False,
        "sectionwise_retain_k": 5,
        "citation_aliases": False,
        "outline_evidence_summary_chars": 240,
        "methodology_summary_chars": 1200,
        "keep_tool_results": 8,
        "enable_citation_agent": False,
        "parallel_sections": False,
        "max_parallel_sections": 4,
        "synthesis_section_max_chars": 1200,
    }
    config.update(overrides)
    return config


class _FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def chat(self, messages, **kwargs):  # noqa: ANN001
        if not self._responses:
            raise RuntimeError("No more fake responses configured")
        return self._responses.pop(0)


class _PromptAwareLLM:
    def chat(self, messages, **kwargs):  # noqa: ANN001
        user_prompt = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_prompt = str(message.get("content") or "")
                break

        if "Return a JSON array of sections" in user_prompt:
            return """
[
  {"title": "Executive Summary", "section_type": "summary", "purpose": "State the answer succinctly.", "evidence_ids": ["ev1"]},
  {"title": "Evidence-Based Findings", "section_type": "analysis", "purpose": "Explain what the evidence supports.", "evidence_ids": ["ev1"]},
  {"title": "Next Steps", "section_type": "next_steps", "purpose": "Suggest how to improve coverage.", "evidence_ids": ["ev1"]}
]
""".strip()

        if "Section to write:" in user_prompt:
            match = re.search(r"- Title:\\s*(.+)", user_prompt)
            title = match.group(1).strip() if match else "Section"
            return (
                "{\n"
                f'  "title": "{title}",\n'
                '  "section_type": "analysis",\n'
                f'  "body_markdown": "Draft for {title}. <sup>1</sup>",\n'
                '  "citations": [{"evidence_id": "ev1", "used_for": "supporting detail"}]\n'
                "}\n"
            )

        if "Draft section bodies (JSON):" in user_prompt:
            return """
{
  "title": "Who partnered with OpenAI?",
  "short_answer": "Microsoft partnered with OpenAI. <sup>1</sup>",
  "summary": "Microsoft partnered with OpenAI. <sup>1</sup>",
  "limitations": ["Draft synthesized from limited evidence."],
  "next_steps": ["Collect additional sources and verify timeline details."]
}
""".strip()

        raise RuntimeError(f"Unrecognized prompt for fake LLM: {user_prompt[:120]}")


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
  {"title": "Executive Summary", "section_type": "summary", "purpose": "State the answer succinctly.", "evidence_ids": ["ev1", "ev2"]},
  {"title": "Evidence-Based Findings", "section_type": "analysis", "purpose": "Explain what the evidence supports.", "evidence_ids": ["ev1", "ev2"]},
  {"title": "Implications", "section_type": "analysis", "purpose": "Describe why the findings matter.", "evidence_ids": ["ev1", "ev2"]},
  {"title": "Limitations", "section_type": "limitations", "purpose": "Clarify what cannot be concluded.", "evidence_ids": ["ev1", "ev2"]},
  {"title": "Next Steps", "section_type": "next_steps", "purpose": "Suggest how to improve coverage.", "evidence_ids": ["ev1", "ev2"]}
]
""".strip()
    report_json = """
{
  "title": "Who partnered with OpenAI?",
  "short_answer": "Microsoft partnered with OpenAI and provides key infrastructure support. <sup>1</sup><sup>2</sup>",
  "summary": "Microsoft partnered with OpenAI and provides key infrastructure support. <sup>1</sup><sup>2</sup>",
  "sections": [
    {
      "title": "Evidence-Based Findings",
      "section_type": "analysis",
      "body_markdown": "- Microsoft formed a strategic partnership with OpenAI in 2019. <sup>1</sup>\\n- Azure hosts OpenAI services. <sup>2</sup>"
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
        config=_default_reporter_config(parallel_thinking_runs=2),
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


def test_reporter_raises_when_short_answer_missing_citations():
    outline = """
[
  {"title": "Evidence-Based Findings", "section_type": "analysis", "purpose": "Explain what the evidence supports.", "evidence_ids": ["ev1"]}
]
""".strip()
    report_json = """
{
  "title": "Who partnered with OpenAI?",
  "short_answer": "Microsoft partnered with OpenAI in 2019.",
  "summary": "Microsoft partnered with OpenAI in 2019.",
  "sections": [
    {
      "title": "Evidence-Based Findings",
      "section_type": "analysis",
      "body_markdown": "Microsoft formed a strategic partnership with OpenAI in 2019. <sup>1</sup>"
    }
  ],
  "limitations": [],
  "next_steps": [],
  "citations": []
}
""".strip()
    reporter = DeepSearchReporter(
        template_store=None,
        config=_default_reporter_config(),
        llm_connector=_FakeLLM([outline, report_json]),
    )
    trace = _build_trace(include_final_answer=False)

    with pytest.raises(ValueError):
        asyncio.run(reporter.compose(trace, external_evidence=None))

def test_reporter_citation_agent_fills_missing_citations():
    outline = """
[
  {"title": "Findings", "section_type": "analysis", "purpose": "Summarize what the evidence supports.", "evidence_ids": ["ev1"]}
]
""".strip()
    report_json = """
{
  "title": "Who partnered with OpenAI?",
  "short_answer": "Microsoft partnered with OpenAI in 2019. <sup>1</sup>",
  "summary": "Microsoft partnered with OpenAI in 2019. <sup>1</sup>",
  "sections": [
    {
      "title": "Findings",
      "section_type": "analysis",
      "body_markdown": "The evidence supports a partnership with Microsoft. <sup>1</sup>"
    }
  ],
  "limitations": [],
  "next_steps": [],
  "citations": []
}
""".strip()
    reporter = DeepSearchReporter(
        template_store=None,
        config=_default_reporter_config(enable_citation_agent=True),
        llm_connector=_FakeLLM([outline, report_json]),
    )
    trace = _build_trace(include_final_answer=False)

    report = asyncio.run(reporter.compose(trace, external_evidence=[]))

    citations = report["structured_report"]["citations"]
    assert isinstance(citations, list)
    assert any(entry.get("evidence_id") == "ev1" for entry in citations if isinstance(entry, dict))
    assert isinstance(report["structured_report"].get("evidence_index"), list)


def test_reporter_raises_when_llm_report_disabled():
    reporter = DeepSearchReporter(template_store=None, config=_default_reporter_config(enable_llm_report=False), llm_connector=None)
    trace = _build_trace(include_final_answer=True)

    with pytest.raises(RuntimeError):
        asyncio.run(reporter.compose(trace, external_evidence=[]))


def test_reporter_raises_when_llm_output_invalid():
    reporter = DeepSearchReporter(
        template_store=None,
        config=_default_reporter_config(),
        llm_connector=_FakeLLM(
            [
                "not-json",
                "not-json",
            ]
        ),
    )
    trace = _build_trace(include_final_answer=False)
    with pytest.raises(ValueError):
        asyncio.run(reporter.compose(trace, external_evidence=[]))


def test_reporter_includes_chunk_evidence_preview():
    """Verify the markdown output includes chunk evidence preview section."""
    outline = """\n[\n  {\"title\": \"Findings\", \"section_type\": \"analysis\", \"purpose\": \"Show evidence\", \"evidence_ids\": [\"chunk_abc123\"]}\n]\n""".strip()
    report_json = """\n{\n  \"title\": \"Report\",\n  \"short_answer\": \"ok. <sup>1</sup>\",\n  \"summary\": \"ok. <sup>1</sup>\",\n  \"sections\": [\n    {\"title\": \"Findings\", \"section_type\": \"analysis\", \"body_markdown\": \"Detail. <sup>1</sup>\"}\n  ],\n  \"limitations\": [],\n  \"next_steps\": [],\n  \"citations\": []\n}\n""".strip()
    reporter = DeepSearchReporter(
        template_store=None,
        config=_default_reporter_config(include_appendices_in_answer=True),
        llm_connector=_FakeLLM([outline, report_json]),
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
    assert "`chunk_abc123`" in answer
    assert "(test-doc.pdf)" in answer
    assert "This is a long piece of content that should be truncated to show only the first 100 characters" in answer
    assert "..." in answer
    assert "`chunk_def456`" in answer
    assert "Short content." in answer


def test_reporter_no_evidence_returns_deterministic_report_even_when_graph_viz_disabled():
    reporter = DeepSearchReporter(
        template_store=None,
        config=_default_reporter_config(include_graph_viz=False),
        llm_connector=_FakeLLM([]),
    )
    trace = {
        "question": "test question",
        "plan_steps": [],
        "reasoning_steps": [],
        "graph_traversals": [],
        "evidences": [],
        "graph_context": {"metadata": {}},
        "coverage_metrics": {"evidence_count": 0, "coverage_ratio": 0.0},
    }

    report = asyncio.run(reporter.compose(trace, external_evidence=[]))

    assert report["structured_report"]["generation"]["mode"] == "deterministic_no_evidence"


def test_reporter_comparison_avoids_false_missing_named_files_when_filename_is_nested():
    outline = """\n[\n  {\"title\": \"Findings\", \"section_type\": \"analysis\", \"purpose\": \"Compare\", \"evidence_ids\": [\"evA\", \"evB\"]}\n]\n""".strip()
    report_json = """\n{\n  \"title\": \"对比报告\",\n  \"short_answer\": \"两者均可对比。<sup>1</sup><sup>2</sup>\",\n  \"summary\": \"两者均可对比。<sup>1</sup><sup>2</sup>\",\n  \"sections\": [\n    {\"title\": \"Findings\", \"section_type\": \"analysis\", \"body_markdown\": \"对比细节。<sup>1</sup><sup>2</sup>\"}\n  ],\n  \"limitations\": [],\n  \"next_steps\": [],\n  \"citations\": [\n    {\"evidence_id\": \"evA\", \"used_for\": \"Plan A\"},\n    {\"evidence_id\": \"evB\", \"used_for\": \"Plan B\"}\n  ]\n}\n""".strip()
    reporter = DeepSearchReporter(
        template_store=None,
        config=_default_reporter_config(),
        llm_connector=_FakeLLM([outline, report_json]),
    )
    trace = {
        "question": "请对比《智盈匯聚(優越版)II壽險計劃》与《價值連承壽險計劃》的关键差异。",
        "final_answer": "",
        "plan_steps": [],
        "reasoning_steps": [],
        "graph_traversals": [],
        "adapter_metadata": {"adapter_name": "hipporag"},
        "coverage_metrics": {},
        "graph_context": {"adapter_name": "hipporag", "question": "Q", "metadata": {}},
        "evidences": [
            {
                "chunk_id": "evA",
                "source": "hipporag",
                "content": "A evidence",
                "provenance": {"metadata": {"chunk_metadata": {"filename": "智盈匯聚(優越版)II壽險計劃.pdf"}}},
            },
            {
                "chunk_id": "evB",
                "source": "hipporag",
                "content": "B evidence",
                "provenance": {"metadata": {"chunk_metadata": {"filename": "價值連承壽險計劃.pdf"}}},
            },
        ],
    }

    report = asyncio.run(reporter.compose(trace, external_evidence=[]))

    assert "Unable to complete the comparison" not in report["answer"]
    assert report["structured_report"]["generation"]["mode"] == "llm"


def test_reporter_parallel_sections_synthesizes_summary():
    reporter = DeepSearchReporter(
        template_store=None,
        config=_default_reporter_config(parallel_sections=True, max_parallel_sections=1, citation_aliases=False),
        llm_connector=_PromptAwareLLM(),
    )
    trace = _build_trace(include_final_answer=False)

    report = asyncio.run(reporter.compose(trace, external_evidence=[]))

    structured = report["structured_report"]
    assert structured["summary"], "Parallel writer should synthesize a non-empty summary"
    assert structured["limitations"], "Parallel writer should synthesize limitations"
    assert structured["next_steps"], "Parallel writer should synthesize next steps"


def test_reporter_includes_external_evidence_even_when_internal_is_full():
    outline = """\n[\n  {\"title\": \"Findings\", \"section_type\": \"analysis\", \"purpose\": \"Answer\", \"evidence_ids\": [\"ev00\"]}\n]\n""".strip()
    report_json = """\n{\n  \"title\": \"Q\",\n  \"short_answer\": \"A. <sup>1</sup>\",\n  \"summary\": \"A. <sup>1</sup>\",\n  \"sections\": [\n    {\"title\": \"Findings\", \"section_type\": \"analysis\", \"body_markdown\": \"Detail. <sup>1</sup>\"}\n  ],\n  \"limitations\": [],\n  \"next_steps\": [],\n  \"citations\": []\n}\n""".strip()
    reporter = DeepSearchReporter(
        template_store=None,
        config=_default_reporter_config(max_evidence_items=10),
        llm_connector=_FakeLLM([outline, report_json]),
    )
    trace = {
        "question": "Q",
        "final_answer": "A",
        "plan_steps": [],
        "reasoning_steps": [],
        "graph_traversals": [],
        "adapter_metadata": {"adapter_name": "hipporag"},
        "coverage_metrics": {},
        "graph_context": {"adapter_name": "hipporag", "question": "Q", "metadata": {}},
        "evidences": [
            {"chunk_id": f"ev{i:02d}", "source": "hipporag", "content": f"internal {i}"}
            for i in range(12)
        ],
    }
    external = [
        {"chunk_id": "tavily-1", "source": "web.tavily", "content": "external 1"},
        {"chunk_id": "tavily-2", "source": "web.tavily", "content": "external 2"},
        {"chunk_id": "tavily-3", "source": "web.tavily", "content": "external 3"},
    ]

    report = asyncio.run(reporter.compose(trace, external_evidence=external))

    assert len(report["evidences"]) == 10
    sources = {ev.get("source") for ev in report["evidences"]}
    assert "web.tavily" in sources
