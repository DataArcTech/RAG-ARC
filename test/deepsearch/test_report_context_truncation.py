from core.deepsearch.report import DeepSearchReporter


def test_report_context_truncates_large_methodology_fields():
    reporter = DeepSearchReporter(
        template_store=None,
        config={
            "max_highlights": 6,
            "max_evidence_items": 10,
            "report_temperature": 0.0,
            "report_max_evidence_chars": 900,
            "report_max_graph_chain_items": 200,
            "report_max_seed_entities": 15,
            "enable_llm_report": False,
            "enable_consistency_check": False,
            "consistency_temperature": 0.0,
            "consistency_max_retries": 0,
            "consistency_max_claims": 0,
            "enable_citation_agent": False,
            "parallel_sections": False,
            "max_parallel_sections": 1,
            "sectionwise_writer": False,
            "sectionwise_retain_k": 0,
            "citation_aliases": False,
            "parallel_thinking_runs": 1,
            "include_graph_viz": False,
            "enable_custom_summary": False,
            "outline_evidence_summary_chars": 240,
            "methodology_summary_chars": 120,
            "keep_tool_results": 1,
            "synthesis_section_max_chars": 1200,
        },
        llm_connector=None,
    )

    trace = {
        "question": "q",
        "plan_steps": [{"step_id": "plan_01", "description": "d1", "channel": "graph"}],
        "reasoning_steps": [
            {
                "step_id": "plan_01",
                "description": "d1",
                "channel": "graph",
                "status": "done",
                "output_summary": "x" * 2000,
                "diagnostics": {"tool": "graph.ops"},
            }
        ],
        "tool_results": [
            {
                "plan_step_id": "plan_01",
                "tool_name": "search",
                "channel": "graph",
                "result": {
                    "summary": "ok",
                    "diagnostics": {
                        "big": "y" * 2000,
                        "n": 1,
                        "nested": {"k": "v"},
                    },
                },
            }
        ],
    }

    context = reporter._build_llm_context(
        trace=trace,
        highlights=[],
        evidences=[],
        coverage={},
        request_context={},
    )

    methodology = context.get("methodology") or {}
    reasoning_steps = methodology.get("reasoning_steps") or []
    assert reasoning_steps
    summary = reasoning_steps[0].get("output_summary") or ""
    assert isinstance(summary, str)
    assert len(summary) <= 123

    tool_results = methodology.get("tool_results") or []
    assert tool_results
    diagnostics = tool_results[0].get("diagnostics") or {}
    assert isinstance(diagnostics, dict)
    assert "n" in diagnostics
    assert "big" in diagnostics and isinstance(diagnostics["big"], str) and len(diagnostics["big"]) <= 243
    assert "nested" not in diagnostics
