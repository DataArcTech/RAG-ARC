import asyncio
import re

from encapsulation.data_model.deepsearch import EvidenceChunk

from core.deepsearch.report import DeepSearchReporter


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
  {"title": "Executive Summary", "purpose": "State the answer succinctly."},
  {"title": "Evidence-Based Findings", "purpose": "Explain what the evidence supports."},
  {"title": "Next Steps", "purpose": "Suggest how to improve coverage."}
]
""".strip()

        if "Section to write:" in user_prompt:
            match = re.search(r"- Title:\\s*(.+)", user_prompt)
            title = match.group(1).strip() if match else "Section"
            return (
                "{\n"
                f'  "title": "{title}",\n'
                f'  "body_markdown": "Draft for {title}. [ev1]",\n'
                '  "citations": [{"evidence_id": "ev1", "used_for": "supporting detail"}]\n'
                "}\n"
            )

        if "Draft section bodies (JSON):" in user_prompt:
            return """
{
  "title": "Who partnered with OpenAI?",
  "summary": "Microsoft partnered with OpenAI. [ev1]",
  "limitations": ["Draft synthesized from limited evidence."],
  "next_steps": ["Collect additional sources and verify timeline details."]
}
""".strip()

        raise RuntimeError(f"Unrecognized prompt for fake LLM: {user_prompt[:120]}")


class _AliasCitingLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        user_prompt = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_prompt = str(message.get("content") or "")
                break
        if "Return a JSON array of sections" in user_prompt:
            return """\n[\n  {\"title\": \"Findings\", \"purpose\": \"Summarize evidence.\"}\n]\n""".strip()
        if "Return a single JSON object with:" in user_prompt and "Evidence snippets" in user_prompt:
            return """\n{\n  \"title\": \"Report\",\n  \"summary\": \"Claim supported by evidence. [chunk_001]\",\n  \"sections\": [\n    {\"title\": \"Findings\", \"body_markdown\": \"Detail. [chunk_001]\"}\n  ],\n  \"limitations\": [],\n  \"next_steps\": [],\n  \"citations\": []\n}\n""".strip()
        if "Return the JSON result now." in user_prompt:
            return """\n{\"is_consistent\": true, \"confidence\": 0.9, \"issues\": []}\n""".strip()
        raise RuntimeError(f"Unexpected prompt: {user_prompt[:80]}")


class _AliasCitingCjkLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        user_prompt = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_prompt = str(message.get("content") or "")
                break
        if "Return a JSON array of sections" in user_prompt:
            return """\n[\n  {\"title\": \"Findings\", \"purpose\": \"Summarize evidence.\"}\n]\n""".strip()
        if "Return a single JSON object with:" in user_prompt and "Evidence snippets" in user_prompt:
            return (
                "{\n"
                '  "title": "Report",\n'
                '  "summary": "Claim supported by evidence. 【chunk_001】",\n'
                '  "sections": [\n'
                '    {"title": "Findings", "body_markdown": "Detail. 【chunk_001】"}\n'
                "  ],\n"
                '  "limitations": [],\n'
                '  "next_steps": [],\n'
                '  "citations": []\n'
                "}\n"
            )
        if "Return the JSON result now." in user_prompt:
            return """\n{\"is_consistent\": true, \"confidence\": 0.9, \"issues\": []}\n""".strip()
        raise RuntimeError(f"Unexpected prompt: {user_prompt[:80]}")


class _AliasVariantCitingLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        user_prompt = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_prompt = str(message.get("content") or "")
                break
        if "Return a JSON array of sections" in user_prompt:
            return """\n[\n  {\"title\": \"Findings\", \"purpose\": \"Summarize evidence.\"}\n]\n""".strip()
        if "Return a single JSON object with:" in user_prompt and "Evidence snippets" in user_prompt:
            return (
                "{\n"
                '  "title": "Report",\n'
                '  "summary": "Claim supported by evidence. [chunk 1] [chunk_1] [chunk_001, chunk 1]",\n'
                '  "sections": [\n'
                '    {"title": "Findings", "body_markdown": "Detail. [CHUNK_001]"}\n'
                "  ],\n"
                '  "limitations": [],\n'
                '  "next_steps": [],\n'
                '  "citations": [{"evidence_id": "chunk 1", "used_for": "support"}]\n'
                "}\n"
            )
        raise RuntimeError(f"Unexpected prompt: {user_prompt[:80]}")


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


def test_reporter_drops_uncited_summary_in_llm_report():
    outline = """
[
  {"title": "Evidence-Based Findings", "purpose": "Explain what the evidence supports."}
]
""".strip()
    report_json = """
{
  "title": "Who partnered with OpenAI?",
  "summary": "Microsoft partnered with OpenAI in 2019.",
  "sections": [
    {
      "title": "Evidence-Based Findings",
      "body_markdown": "Microsoft formed a strategic partnership with OpenAI in 2019. [ev1]"
    }
  ],
  "limitations": [],
  "next_steps": [],
  "citations": []
}
""".strip()
    reporter = DeepSearchReporter(
        template_store=None,
        config={"parallel_thinking_runs": 1, "enable_consistency_check": False},
        llm_connector=_FakeLLM([outline, report_json]),
    )
    trace = _build_trace(include_final_answer=False)

    report = asyncio.run(reporter.compose(trace, external_evidence=None))

    assert report["structured_report"]["summary"] == ""
    assert "## Answer" not in report["answer"]

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
    report = asyncio.run(reporter.compose(trace, external_evidence=[]))
    assert report["answer"].startswith("# Who partnered with OpenAI?")
    assert "## Highlights" in report["answer"]


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


def test_reporter_parallel_sections_synthesizes_summary(monkeypatch):
    monkeypatch.setenv("DEEPSEARCH_PARALLEL_SECTIONS", "true")
    monkeypatch.setenv("DEEPSEARCH_CITATION_ALIASES", "false")
    reporter = DeepSearchReporter(
        template_store=None,
        config={"parallel_sections": True, "max_parallel_sections": 1, "enable_consistency_check": False},
        llm_connector=_PromptAwareLLM(),
    )
    trace = _build_trace(include_final_answer=False)

    report = asyncio.run(reporter.compose(trace, external_evidence=[]))

    structured = report["structured_report"]
    assert structured["summary"], "Parallel writer should synthesize a non-empty summary"
    assert structured["limitations"], "Parallel writer should synthesize limitations"
    assert structured["next_steps"], "Parallel writer should synthesize next steps"


def test_reporter_rewrites_alias_citations_to_original_ids(monkeypatch):
    monkeypatch.setenv("DEEPSEARCH_CITATION_ALIASES", "true")
    reporter = DeepSearchReporter(
        template_store=None,
        config={"enable_consistency_check": True, "enable_citation_agent": False},
        llm_connector=_AliasCitingLLM(),
    )
    trace = _build_trace(include_final_answer=False)

    report = asyncio.run(reporter.compose(trace, external_evidence=[]))

    answer = report["answer"]
    assert "[chunk_001]" not in answer, "Alias citations should be rewritten before returning markdown"
    assert "[ev1]" in answer, "Alias should be rewritten back to original chunk IDs"


def test_reporter_rewrites_alias_citations_with_cjk_brackets(monkeypatch):
    monkeypatch.setenv("DEEPSEARCH_CITATION_ALIASES", "true")
    reporter = DeepSearchReporter(
        template_store=None,
        config={"enable_consistency_check": True, "enable_citation_agent": False},
        llm_connector=_AliasCitingCjkLLM(),
    )
    trace = _build_trace(include_final_answer=False)

    report = asyncio.run(reporter.compose(trace, external_evidence=[]))

    answer = report["answer"]
    assert "【chunk_001】" not in answer
    assert "[chunk_001]" not in answer
    assert "[ev1]" in answer


def test_reporter_rewrites_alias_citation_variants(monkeypatch):
    monkeypatch.setenv("DEEPSEARCH_CITATION_ALIASES", "true")
    reporter = DeepSearchReporter(
        template_store=None,
        config={"enable_consistency_check": False},
        llm_connector=_AliasVariantCitingLLM(),
    )
    trace = _build_trace(include_final_answer=False)

    report = asyncio.run(reporter.compose(trace, external_evidence=[]))

    answer = report["answer"]
    assert "chunk_001" not in answer.lower()
    assert "chunk 1" not in answer.lower()
    assert "[ev1]" in answer


def test_reporter_includes_external_evidence_even_when_internal_is_full():
    reporter = DeepSearchReporter(
        template_store=None,
        config={"enable_llm_report": False, "enable_consistency_check": False},
        llm_connector=None,
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
