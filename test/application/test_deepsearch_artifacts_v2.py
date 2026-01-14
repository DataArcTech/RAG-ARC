from application.rag_inference.deepsearch.service_runtime.artifact_views_v2 import build_v2_artifact_documents


def _sample_snapshot() -> dict:
    return {
        "run_id": "run_123",
        "config_fingerprint": "fp_abc",
        "stage": "done",
        "stage_history": [{"stage": "created"}, {"stage": "done"}],
        "plan_metadata": {"plan_id": "plan_1"},
        "plan_steps": [{"step_id": "s1"}, {"step_id": "s2"}],
        "reasoning_trace": {"tool_results": [{"x": "y"}], "reasoning_steps": [{"status": "done"}]},
        "report": {"answer": "final answer", "evidences": [{"chunk_id": "c1"}]},
        "external_calls": [{"tool": "web.search"}],
        "cost_telemetry": {"stage_timings": {"plan_ms": 1}},
        "quality_gates": [],
        "errors": [{"stage": "persist", "message": "x"}],
        "request_metadata": {"owner_id": "owner"},
        "kpis": {"evidence_count": 1},
        "error_summary": {"unknown": 1},
    }


def test_deepsearch_v2_artifacts_state_snapshot_is_manifest():
    snapshot = _sample_snapshot()
    stage_timings = {"plan_ms": 1, "total_ms": 2}
    artifacts_cfg = {
        "enabled": True,
        "version": 2,
        "profiles": ["dev", "public"],
        "state_snapshot_mode": "manifest",
        "refs": {"enabled": True},
        "public": {
            "include_final_report_in_json": False,
            "max_plan_steps": 12,
            "max_stage_history": 128,
            "max_external_calls": 12,
            "max_errors": 64,
        },
    }
    docs = build_v2_artifact_documents(
        snapshot=snapshot,
        stage_timings=stage_timings,
        artifacts_config=artifacts_cfg,
        artifacts_present={"plan_result": True, "reasoning": True, "report": True, "report_md": True, "stage_timings": True},
        trace_events=[
            {"tag": "think", "content": "hello", "meta": {}},
            {"tag": "write_outline", "content": "Plan ID: plan_1", "meta": {}},
            {"tag": "tool_call", "content": "{\"tool_name\":\"graph.entity_concepts\",\"call_id\":\"c1\"}", "meta": {}},
            {"tag": "tool_response", "content": "{\"tool_name\":\"graph.entity_concepts\",\"call_id\":\"c1\",\"result\":{\"evidences\":[]}}", "meta": {}},
            {"tag": "write", "content": "final report", "meta": {}},
        ],
    )

    assert set(docs) >= {"manifest.json", "dev.json", "public.json", "state_snapshot.json"}
    assert docs["state_snapshot.json"] == docs["manifest.json"]
    assert "reasoning_trace" not in docs["manifest.json"]
    assert "report" not in docs["manifest.json"]

    manifest = docs["manifest.json"]
    assert manifest["artifact_version"] == 2
    assert manifest["run_id"] == "run_123"
    assert manifest["artifacts"]["report"]["$ref"]["file"] == "report.json"
    assert manifest["artifacts"]["reasoning"]["$ref"]["file"] == "reasoning.json"
    assert manifest["profiles"]["dev"] == "dev.json"
    assert manifest["profiles"]["public"] == "public.json"

    dev = docs["dev.json"]
    assert dev["profile"] == "dev"
    assert "reasoning_trace" not in dev
    assert "report" not in dev
    assert "plan_steps" not in dev
    assert "plan_metadata" not in dev
    assert dev["plan_ref"]["$ref"]["file"] == "plan_result.json"
    assert dev["reasoning_ref"]["$ref"]["file"] == "reasoning.json"
    assert dev["report_ref"]["$ref"]["file"] == "report.json"

    public = docs["public.json"]
    assert public["dev_ref"]["$ref"]["file"] == "dev.json"
    assert [e["type"] for e in public["events"]] == ["think", "plan", "tool_call", "tool_response"]


def test_deepsearch_v2_artifacts_legacy_state_snapshot_mode_keeps_snapshot():
    snapshot = _sample_snapshot()
    docs = build_v2_artifact_documents(
        snapshot=snapshot,
        stage_timings={},
        artifacts_config={
            "enabled": True,
            "version": 2,
            "profiles": [],
            "state_snapshot_mode": "legacy",
            "refs": {"enabled": True},
            "public": {},
        },
        artifacts_present={"plan_result": False, "reasoning": False, "report": False, "report_md": False, "stage_timings": False},
    )
    assert "manifest.json" in docs
    assert docs["state_snapshot.json"] == snapshot
