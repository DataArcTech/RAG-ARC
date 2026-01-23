from core.deepsearch.state import DeepSearchState


def test_state_tracks_plan_reasoning_and_report():
    state = DeepSearchState(config_fingerprint="cfg-hash")
    plan = {
        "plan_id": "plan-run-1",
        "plan": {
            "created_at": "2024-01-01T00:00:00Z",
            "steps": [
                {"step_id": "plan_01", "description": "Graph search", "channel": "graph"},
                {"step_id": "plan_02", "description": "Summarize", "channel": "text"},
            ],
        },
    }
    state.record_plan(plan)

    assert state.stage == "planned"
    assert state.plan_metadata["plan_id"] == "plan-run-1"
    assert len(state.plan_steps) == 2

    reasoning_trace = {
        "plan_steps": state.plan_steps,
        "reasoning_steps": [
            {"step_id": "plan_01", "status": "done"},
            {"step_id": "plan_02", "status": "running"},
        ],
        "evidences": [
            {"chunk_id": "ev1", "content": "fact"},
        ],
    }
    state.record_reasoning(reasoning_trace)
    assert state.stage == "reasoned"
    assert state.reasoning_trace["evidences"][0]["chunk_id"] == "ev1"

    report = {
        "answer": "Graph summary",
        "evidences": reasoning_trace["evidences"],
    }
    state.record_report(report)
    assert state.stage == "reported"
    snapshot = state.snapshot()
    assert snapshot["report"]["answer"] == "Graph summary"
    assert snapshot["stage_history"][-1]["stage"] == "reported"


def test_state_handles_errors():
    state = DeepSearchState(config_fingerprint="cfg")
    state.append_error("missing chunk", stage="reasoned")
    state.mark_failed("gap timeout", details={"timeout_ms": 5000})
    assert state.stage == "failed"
    assert len(state.errors) == 2
    assert state.errors[-1]["reason"] == "gap timeout"

    snapshot = state.snapshot()
    assert snapshot["errors"][-1]["reason"] == "gap timeout"
    assert "kpis" in snapshot
    assert "error_summary" in snapshot
