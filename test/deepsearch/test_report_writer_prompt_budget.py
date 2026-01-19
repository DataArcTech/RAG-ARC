import json


class _BudgetLLM:
    def __init__(self, *, max_chars: int):
        self.max_chars = int(max_chars)
        self.calls: list[dict] = []

    async def achat(self, messages, **kwargs):  # noqa: ANN001
        total = 0
        for msg in messages:
            total += len(str((msg or {}).get("content") or ""))
        self.calls.append({"total_chars": total, "kwargs": kwargs, "messages": messages})
        if total > self.max_chars:
            raise RuntimeError(
                "This model's maximum context length is 128000 tokens. However, your messages resulted in "
                f"{total} tokens. Please reduce the length of the messages."
            )
        system = str((messages[0] or {}).get("content") or "")
        if "Return a JSON array of sections" in (messages[1] or {}).get("content", ""):
            return json.dumps(
                [
                    {"title": "Overview", "purpose": "Answer the question", "evidence_ids": ["ev1"]},
                    {"title": "Details", "purpose": "Give supporting details", "evidence_ids": ["ev2"]},
                    {"title": "Conclusion", "purpose": "Wrap up", "evidence_ids": ["ev1", "ev2"]},
                ],
                ensure_ascii=False,
            )
        if "Return a single JSON object with" in (messages[1] or {}).get("content", ""):
            return json.dumps(
                {
                    "title": "Test Report",
                    "short_answer": "ok[ev1]",
                    "summary": "ok[ev1]",
                    "sections": [{"title": "Body", "section_type": "analysis", "body_markdown": "Answer[ev1]."}],
                    "limitations": [],
                    "next_steps": [],
                    "citations": [],
                },
                ensure_ascii=False,
            )
        return json.dumps({}, ensure_ascii=False)


def test_report_writer_shrinks_prompts_on_context_limit():
    from core.deepsearch.report.llm_writer import DeepSearchLLMReportWriter

    llm = _BudgetLLM(max_chars=14_000)
    writer = DeepSearchLLMReportWriter(
        llm_connector=llm,
        max_retries=4,
        max_evidence_items=32,
        max_evidence_chars=4000,
        max_graph_chain_items=200,
        parallel_sections=False,
    )

    outline = [
        {"title": "Overview", "purpose": "overview", "evidence_ids": ["ev1"]},
        {"title": "Details", "purpose": "details", "evidence_ids": ["ev2"]},
        {"title": "Conclusion", "purpose": "conclusion", "evidence_ids": ["ev1", "ev2"]},
    ]
    evidences = [
        {"chunk_id": f"ev{i}", "source": "graph", "content": ("x" * 12000), "score": 1.0}
        for i in range(1, 60)
    ]
    graph_chain = [f"n{i} -[r]-> n{i+1}" for i in range(400)]
    context = {
        "highlights": [{"step_id": f"s{i}", "summary": "y" * 6000} for i in range(8)],
        "methodology": {
            "plan_steps": [{"step_id": f"p{i}", "description": "z" * 2000} for i in range(20)],
            "reasoning_steps": [{"step_id": f"r{i}", "output_summary": "w" * 4000} for i in range(30)],
            "tool_results": [{"tool_name": "search", "diagnostics": {"dump": "k" * 5000}} for _ in range(12)],
        },
        "graph_evidence": {"seed_entities": ["SAS"] * 50, "graph_stats": {"edges": 9999}},
        "coverage": {
            "coverage_metrics": {"evidence_count": 999, "missing_topics": ["x"] * 50},
            "gap_result": {"should_trigger_external": False, "missing_topics": ["x"] * 50},
            "pending_external": [{"task": "t", "payload": "p" * 2000} for _ in range(200)],
        },
        "evidences": evidences,
        "graph_chain": graph_chain,
    }

    import asyncio

    async def _run():
        result = await writer.write_report(question="Q", outline=outline, context=context)
        assert result["title"]
        assert result["sections"]
        assert len(llm.calls) >= 2
        assert llm.calls[-1]["total_chars"] <= llm.max_chars

    asyncio.run(_run())
