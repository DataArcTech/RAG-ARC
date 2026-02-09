from application.rag_inference.deepsearch.service_runtime.reasoning_metrics import summarize_reasoning_trace


def test_summarize_reasoning_trace_counts_tools_and_cache_hits():
    trace = {
        "reasoning_steps": [
            {
                "tool_logs": [
                    {"tool_name": "think", "latency_ms": 10},
                    {"tool_name": "explore", "latency_ms": 20},
                ]
            },
            {"tool_logs": [{"tool_name": "read.pages", "latency_ms": 30}]},
        ],
        "tool_results": [
            {
                "tool_name": "explore",
                "result": {
                    "diagnostics": {
                        "actions": [
                            {"tool": "toc.tree", "diagnostics": {"cache": {"hit": False}}},
                            {"tool": "toc.tree", "diagnostics": {"cache": {"hit": True}}},
                        ]
                    }
                },
            },
            {"tool_name": "toc.tree", "result": {"diagnostics": {"cache": {"hit": False}}}},
            {"tool_name": "toc.tree", "result": {"diagnostics": {"cache": {"hit": True}}}},
        ],
        "tool_memoization": {"hits": 2, "misses": 3},
        "evidences": [{"source": "read.pages"}, {"source": "bm25"}],
    }
    out = summarize_reasoning_trace(trace)
    assert out["tool_calls_total"] == 3
    assert out["tool_calls_by_name"]["think"] == 1
    assert out["tool_latency_total_ms"] == 60
    assert out["cache_hits_total"] == 1
    assert out["cache_misses_total"] == 1
    assert out["explore_actions_total"] == 2
    assert out["explore_cache_hits_total"] == 1
    assert out["explore_cache_misses_total"] == 1
    assert out["tool_memoization"]["hits"] == 2
    assert out["primary_page_evidence_items"] == 1
