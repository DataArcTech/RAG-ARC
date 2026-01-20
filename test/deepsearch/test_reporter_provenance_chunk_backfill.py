import asyncio

from core.deepsearch.report import DeepSearchReporter


def _default_reporter_config(**overrides):  # noqa: ANN001
    config = {
        "max_highlights": 2,
        "include_graph_viz": False,
        "enable_custom_summary": False,
        "parallel_thinking_runs": 1,
        "enable_llm_report": True,
        "report_temperature": 0.0,
        "report_max_evidence_chars": 900,
        "max_evidence_items": 10,
        "report_max_graph_chain_items": 50,
        "report_max_seed_entities": 10,
        "enable_consistency_check": False,
        "consistency_temperature": 0.0,
        "consistency_max_retries": 1,
        "sectionwise_writer": False,
        "sectionwise_retain_k": 3,
        "citation_aliases": False,
        "outline_evidence_summary_chars": 240,
        "methodology_summary_chars": 400,
        "keep_tool_results": 2,
        "enable_citation_agent": False,
        "parallel_sections": False,
        "max_parallel_sections": 2,
        "consistency_max_claims": 10,
        "synthesis_section_max_chars": 800,
        "provenance_chunk_attach_max": 5,
    }
    config.update(overrides)
    return config


class _ChunkCitingLLM:
    def chat(self, messages, **kwargs):  # noqa: ANN001
        user_prompt = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_prompt = str(message.get("content") or "")
                break

        if "Return a JSON array of sections" in user_prompt:
            return """
[
  {"title": "Answer", "section_type": "analysis", "purpose": "Answer with citations.", "evidence_ids": ["chunk_001"]}
]
""".strip()

        if "Return a single JSON object with:" in user_prompt and "Evidence Pack" in user_prompt:
            return """
{
  "title": "Report",
  "short_answer": "Supported by the policy text. [chunk_001]",
  "summary": "Supported by the policy text. [chunk_001]",
  "sections": [
    {"title": "Answer", "section_type": "analysis", "body_markdown": "Details. [chunk_001]"}
  ],
  "limitations": [],
  "next_steps": [],
  "citations": []
}
""".strip()

        raise RuntimeError(f"Unexpected prompt: {user_prompt[:120]}")


class _StubGraphStore:
    OWNER_GLOBAL_KEY = "__GLOBAL__"

    def _owner_key(self, owner_id):  # noqa: ANN001
        return str(owner_id or self.OWNER_GLOBAL_KEY)

    def _execute_query(self, cypher: str, params: dict):  # noqa: ANN001
        if "MATCH (c:Chunk)" not in cypher:
            return []
        requested = params.get("chunk_ids") or []
        rows = []
        for chunk_id in requested:
            if chunk_id == "chunk_001":
                rows.append(
                    {
                        "chunk_id": "chunk_001",
                        "content": "这是可引用的条款原文：免赔额为 5000 元。",
                        "metadata": '{"source_file_id":"file-1","source_file_name":"policy.pdf"}',
                    }
                )
        return rows


def test_reporter_backfills_provenance_chunks_into_evidence_pool() -> None:
    reporter = DeepSearchReporter(
        template_store=None,
        config=_default_reporter_config(),
        llm_connector=_ChunkCitingLLM(),
        graph_store=_StubGraphStore(),
    )

    trace = {
        "question": "免赔额是多少？",
        "evidences": [
            {
                "chunk_id": "graph.latest_truth:p1",
                "source": "graph.latest_truth",
                "content": "latest_truth: topic=免赔额 value=5000元",
                "provenance": {"fact_id": "fact-1", "source_chunk_ids": ["chunk_001"]},
            }
        ],
        "graph_context": {"access_scope": {"scope_id": "owner-1"}, "metadata": {}},
        "adapter_metadata": {},
        "plan_steps": [],
        "graph_traversals": [],
        "reasoning_steps": [],
        "tool_results": [],
        "coverage_metrics": {},
    }

    result = asyncio.run(reporter.compose(trace, external_evidence=[]))
    evidence_ids = {ev.get("chunk_id") for ev in (result.get("evidences") or []) if isinstance(ev, dict)}
    assert "chunk_001" in evidence_ids
    assert "[chunk_001]" in (result.get("structured_report") or {}).get("short_answer", "")
