from core.deepsearch.report import DeepSearchReporter


def test_report_context_truncates_large_methodology_fields():
    reporter = DeepSearchReporter(
        template_store=None,
        config={
            "enable_llm_report": False,
            "methodology_summary_chars": 120,
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
                "diagnostics": {"tool": "graph_adapter.query"},
            }
        ],
        "tool_results": [
            {
                "plan_step_id": "plan_01",
                "tool_name": "graph.chunk_scan",
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
        "pending_external": [],
    }

    context = reporter._build_llm_context(
        trace=trace,
        highlights=[],
        evidences=[],
        coverage={},
        gap_result={},
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

