from core.deepsearch.trace import TraceEvent


def test_save_trace_events_to_file_does_not_append_references_section(tmp_path, monkeypatch):
    # Keep test artifacts isolated from local/ workspace directories.
    monkeypatch.setenv("DEEPSEARCH_TRACE_STORAGE_PATH", f"io://deepsearch_traces/{tmp_path.name}")

    from api.routers.rag_inference_modules.stream_chat.deepsearch.deepsearch_handler import _save_trace_events_to_file

    deepsearch_result = {
        "report": {
            "answer": "Body cites ev1. <sup>1</sup>",
            "structured_report": {"citations": [{"evidence_id": "ev1"}], "source_key_map": {"1": "ev1"}},
            "evidences": [
                {
                    "chunk_id": "ev1",
                    "source": "hipporag",
                    "content": "Chunk 1 content",
                    "provenance": {"metadata": {"chunk_metadata": {"source_file_id": "file-1", "filename": "doc1.md"}}},
                }
            ],
        }
    }

    trace_events = [TraceEvent(tag="think", content="hello", meta={})]
    saved_path = _save_trace_events_to_file(
        trace_events,
        request_id="req-1",
        run_id="run-1",
        query="q",
        deepsearch_result=deepsearch_result,
    )

    assert saved_path
    from app_registration import registrator

    io_manager = registrator.get_object("io_manager")
    payload = io_manager.get_json(saved_path)
    assert isinstance(payload, dict)
    answer = (payload.get("deepsearch_result") or {}).get("report", {}).get("answer")
    assert isinstance(answer, str)

    # Product requirement: rely on <sup> anchors + sources mapping; avoid dumping a "References" section into text.
    assert "<sup>1</sup>" in answer
    assert "## References" not in answer
