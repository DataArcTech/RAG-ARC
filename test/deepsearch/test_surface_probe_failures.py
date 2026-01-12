from core.deepsearch.state import DeepSearchState
from application.rag_inference.deepsearch.service_runtime.quality import DeepSearchServiceQualityMixin


def test_surface_worker_failures_includes_probe_errors() -> None:
    state = DeepSearchState(config_fingerprint="test")
    trace = {
        "coverage_metrics": {
            "probe_errors": [
                {
                    "agent_id": "agent-1",
                    "tool_name": "graph.pattern_scan",
                    "error": "probe boom",
                    "error_type": "RuntimeError",
                    "replay": {"tool_name": "graph.pattern_scan", "payload": {"question": "Q"}},
                }
            ]
        }
    }
    DeepSearchServiceQualityMixin._surface_worker_failures(state, trace)
    assert state.errors
    last = state.errors[-1]
    assert last.get("stage") == "graph_reasoning"
    assert "probe tool(s) failed" in str(last.get("message") or "")
    assert last.get("details", {}).get("probe_errors")

