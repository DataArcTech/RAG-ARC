import json

from application.rag_inference.deepsearch.service_runtime.llm_dump_metrics import summarize_llm_dump_metrics


def test_llm_dump_metrics_summarizes_per_run(tmp_path) -> None:
    path = tmp_path / "llm.jsonl"
    run_id = "run-1"
    other = "run-2"

    events = [
        {"event": "llm.request", "run_id": run_id, "model": "m1", "warn_context": "ctx.a", "llm_call_id": "c1"},
        {"event": "llm.response", "run_id": run_id, "model": "m1", "warn_context": "ctx.a", "llm_call_id": "c1", "elapsed_ms": 120},
        {"event": "llm.request", "run_id": run_id, "model": "m2", "warn_context": "ctx.b", "llm_call_id": "c2"},
        {"event": "llm.error", "run_id": run_id, "model": "m2", "warn_context": "ctx.b", "llm_call_id": "c2", "elapsed_ms": 250, "error": "timeout"},
        {"event": "llm.request", "run_id": other, "model": "m9", "warn_context": "ctx.z", "llm_call_id": "c9"},
        {"event": "llm.response", "run_id": other, "model": "m9", "warn_context": "ctx.z", "llm_call_id": "c9", "elapsed_ms": 5},
    ]
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8")

    out = summarize_llm_dump_metrics(path=str(path), run_id=run_id)
    assert out["enabled"] is True
    assert out["run_id"] == run_id
    assert out["requests"] == 2
    assert out["responses"] == 1
    assert out["errors"] == 1
    assert out["total_finished_calls"] == 2
    assert out["calls_by_model"]["m1"] == 1
    assert out["calls_by_model"]["m2"] == 1
    assert out["errors_by_model"]["m2"] == 1
    assert out["latency"]["count"] == 2
    assert out["latency"]["max_ms"] == 250

